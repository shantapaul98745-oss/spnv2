"""Prepare source queue proposals only. Never launch a task or publish live state."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

PLAN = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
SOURCE = PLAN / "source"
PYTHON = "/home/ps/anaconda3/envs/SPNv2/bin/python"
PY = [PYTHON, "-B", "-s", "-u"]
DOMAINS = [("synthetic", 11994), ("lightbox", 6740), ("sunlamp", 2791)]
CONFIGS = ["phi3_bn_ehs", "phi6_gn_ehs", "table2_h", "table2_e", "table2_eh",
           "table3_base", "table3_erase", "table3_flare", "table4_phi0_bn", "table4_phi3_gn"]
ENV = {
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
    "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "2021", "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2", "MPLBACKEND": "Agg",
    "TORCH_HOME": str(PLAN.parents[1] / "env/cache/torch"), "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
GPU = {"index": 0, "min_free_mb": 14500, "max_used_mb": 1200}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(name, value):
    path = DEST / name
    if path.exists():
        raise FileExistsError("Do not overwrite a proposal or audit: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def load_cfg(name, pilot=False):
    return yaml.safe_load((PLAN / ("pilots/configs" if pilot else "configs") / (name + ".yaml")).read_text())


SOURCE_ASSETS = {
    str(p.resolve()): sha(p)
    for folder in ("core", "tools") for p in sorted((SOURCE / folder).rglob("*")) if p.is_file()
}
HASH_CACHE = dict(SOURCE_ASSETS)


def assets(configs, helpers):
    value = dict(SOURCE_ASSETS)
    paths = [PLAN / "source_manifest.json"]
    paths.extend(PLAN / name for name in configs + helpers)
    for path in paths:
        key = str(path.resolve())
        if key not in HASH_CACHE:
            HASH_CACHE[key] = sha(path)
        value[key] = HASH_CACHE[key]
    # Pin the actual existing initialization weights for BN configurations.
    cfg = yaml.safe_load(paths[1].read_text())
    if not cfg["MODEL"]["USE_GROUPNORM_BACKBONE"]:
        phi = cfg["MODEL"]["EFFICIENTDET_PHI"]
        names = {0: "efficientnet_b0_rwightman-3dd342df.pth",
                 3: "efficientnet_b3_rwightman-cf984f9c.pth"}
        cached = PLAN.parents[1] / "env/cache/torch/hub/checkpoints" / names[phi]
        key = str(cached)
        if key not in HASH_CACHE:
            HASH_CACHE[key] = sha(cached)
        value[key] = HASH_CACHE[key]
    return value


def task_base(ident, kind, command, verify, cfgname, dependency, outputs, *, pilot=False):
    cfg = load_cfg(cfgname, pilot=pilot)
    config_path = ("pilots/configs/" if pilot else "configs/") + cfgname + ".yaml"
    helpers = (["pilots/gpu_smoke.py", "pilots/verify_pilot.py", "pilots/evidence.py"] if pilot
               else ["tasks/source_artifacts.py", "tasks/verify_heads.py"])
    return {
        "id": ident, "kind": kind, "command": command, "cwd": str(SOURCE),
        "requires": [dependency] if dependency else [],
        "session": "spnv2_" + ident.lower() + "_qv2_20261017",
        "log": str(PLAN / ("pilots/logs" if pilot else "logs") / (ident.lower() + "_qv2_20261017.log")),
        "env": dict(ENV), "gpu": dict(GPU), "verify_gpu": False,
        "verify_env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
        "verify_command": verify, "expected_outputs": outputs,
        "metadata": {
            "source": str(PLAN / config_path), "config_name": cfgname,
            "physical_batch": cfg["TRAIN"]["IMAGES_PER_GPU"],
            "accum_steps": cfg["TRAIN"]["ACCUM_STEPS"],
            "asset_sha256": assets([config_path], helpers),
            "cache_policy": "task_entry derives independent project-owned MPLCONFIGDIR/XDG_CACHE_HOME per task and phase",
        },
    }


def pilot_task(cfgname, dependency):
    ident = "pilot_" + ("phi6_EHS" if cfgname == "phi6_gn_ehs" else cfgname)
    stem = cfgname + "_queuev2_20261017"
    report = PLAN / "pilots/reports" / (stem + ".json")
    verified = PLAN / "pilots/reports" / (stem + "_verified.json")
    artifacts = PLAN / "pilots/artifacts" / stem
    config = PLAN / "pilots/configs" / (cfgname + ".yaml")
    command = PY + [str(PLAN / "pilots/gpu_smoke.py"), "--source", str(SOURCE),
                    "--cfg", str(config), "--report", str(report), "--artifacts", str(artifacts),
                    "--attempts", "32", "--checkpoint", "16"]
    verify = PY + [str(PLAN / "pilots/verify_pilot.py"), "--report", str(report), "--output", str(verified)]
    task = task_base(ident, "engineering_gpu_resume_pilot", command, verify, cfgname,
                     dependency, [str(verified)], pilot=True)
    task["metadata"].update(
        attempts=32, checkpoint_attempt=16,
        engineering_subset_rows=load_cfg(cfgname)["TRAIN"]["IMAGES_PER_GPU"] * 4 * 32,
        pilot_weights_for_formal=False, report=str(report), artifacts=str(artifacts))
    return task


def formal_tasks(cfgname, dependency):
    cfg = load_cfg(cfgname)
    label = "phi6_EHS" if cfgname == "phi6_gn_ehs" else cfgname
    ident = "train_" + label
    config = PLAN / "configs" / (cfgname + ".yaml")
    accepted = PLAN / "reports" / ("source_" + ("phi6" if cfgname == "phi6_gn_ehs" else cfgname) + "_verified.json")
    log = PLAN / "logs" / (ident.lower() + "_qv2_20261017.log")
    command = PY + [str(SOURCE / "tools/train.py"), "--cfg", str(config), "--gpu", "0"]
    verifier = PY + [str(PLAN / "tasks/source_artifacts.py"), "verify_train",
                     "--cfg", str(config), "--report", str(accepted), "--log", str(log)]
    task = task_base(ident, "source_training_" + str(cfg["TRAIN"]["END_EPOCH"]) + "_epochs",
                     command, verifier, cfgname, dependency, [str(accepted)])
    task["metadata"].update(
        epochs=cfg["TRAIN"]["END_EPOCH"], seed=2021, independently_initialized=True,
        pilot_weights_for_formal=False,
        initialization=("Independent seed2021, public GN random initialization; author initialization unverified"
                        if cfg["MODEL"]["USE_GROUPNORM_BACKBONE"] else
                        "Independent seed2021 + official cached ImageNet EfficientNet; no pilot weights"))
    result = [task]
    previous = ident
    for domain, samples in DOMAINS:
        eval_name = cfgname + "_eval_" + domain
        eval_config = PLAN / "configs" / (eval_name + ".yaml")
        eval_id = "eval_" + ("phi6" if cfgname == "phi6_gn_ehs" else cfgname) + "_" + domain
        report = PLAN / "reports" / (eval_id + ".json")
        verified = PLAN / "reports" / (eval_id + "_verified.json")
        output = PLAN / "evaluations" / cfgname / domain
        command = PY + [str(PLAN / "tasks/source_artifacts.py"), "evaluate", "--cfg", str(eval_config),
                        "--source-report", str(accepted), "--output", str(output),
                        "--report", str(report), "--expected-samples", str(samples)]
        verify = PY + [str(PLAN / "tasks/source_artifacts.py"), "verify_eval", "--cfg", str(eval_config),
                       "--source-report", str(report), "--report", str(verified),
                       "--expected-samples", str(samples)]
        item = task_base(eval_id, "full_source_evaluation", command, verify, eval_name,
                         previous, [str(verified)])
        item["metadata"].update(final_only=True, samples=samples, domain=domain,
                                source_config=cfgname, source_report=str(accepted))
        # Bind the source configuration whose final model this evaluation consumes.
        item["metadata"]["asset_sha256"].update(assets(["configs/" + cfgname + ".yaml"], []))
        result.append(item)
        previous = eval_id
    return result


def main():
    old = json.loads((PLAN / "queue.json").read_text())
    original = copy.deepcopy(old)
    if len(old["tasks"]) != 4:
        raise ValueError("Expected the unchanged four-task phase1 queue")
    phi6_pilot = pilot_task("phi6_gn_ehs", old["tasks"][-1]["id"])
    appended = [phi6_pilot] + formal_tasks("phi6_gn_ehs", phi6_pilot["id"])
    proposal = copy.deepcopy(old)
    proposal["scope"] = "phase2_phi3_phi6_sources"
    proposal["tasks"].extend(appended)
    proposal["backlog"] = [item for item in old["backlog"] if not item.startswith("φ6-GN EHS：")]
    pending = []
    groups = []
    previous = None
    for cfgname in CONFIGS[2:]:
        begin = len(pending)
        if cfgname.startswith("table2_"):
            pilot = pilot_task(cfgname, previous)
            pending.append(pilot)
            previous = pilot["id"]
        items = formal_tasks(cfgname, previous)
        pending.extend(items)
        previous = items[-1]["id"]
        groups.append({"config": cfgname, "first_task_id": pending[begin]["id"],
                       "last_task_id": previous, "task_count": len(pending) - begin})
    proposal_path = save("full_source_v2_proposal.json", proposal)
    pending_object = {
        "schema": "spnv2-pending-source-definitions-v1",
        "plan_id": PLAN.name, "execution_manifest": False, "activation_allowed": False,
        "reason": "Table2/3/4 must not run before accepted fixed 40 ODR tasks",
        "activation_requirements": [
            "All fixed40 ODR tasks (main20 plus phi3 B1/B2 repair20) independently accepted",
            "Root reviews actual ODR task IDs/evidence and explicitly adds a dependency on the true final accepted ODR task",
            "Root validates all hashes and task definitions, then publishes a separate append-only manifest revision",
        ],
        "first_task_requires_real_odr_dependency_at_activation": pending[0]["id"],
        "groups": groups, "pending_task_definitions": pending,
    }
    pending_path = save("pending_source_tasks.json", pending_object)
    index = []
    for name in CONFIGS:
        cfg = load_cfg(name)
        index.append({
            "config": name + ".yaml", "config_path": str(PLAN / "configs" / (name + ".yaml")),
            "sha256": sha(PLAN / "configs" / (name + ".yaml")), "epochs": cfg["TRAIN"]["END_EPOCH"],
            "phi": cfg["MODEL"]["EFFICIENTDET_PHI"],
            "heads": cfg["MODEL"]["HEAD"]["NAMES"], "physical_batch": cfg["TRAIN"]["IMAGES_PER_GPU"],
            "accum_steps": cfg["TRAIN"]["ACCUM_STEPS"], "seed": cfg["SEED"],
            "experiment": cfg["EXP_NAME"],
            "model_final": str(Path(cfg["OUTPUT_DIR"]) / cfg["MODEL"]["BACKBONE"]["NAME"] / cfg["EXP_NAME"] / "model_final.pth.tar"),
        })
    save("proposal_index.json", {
        "old_manifest": str(PLAN / "queue.json"), "old_manifest_file_sha256": sha(PLAN / "queue.json"),
        "old_manifest_canonical_sha256": canonical_hash(original),
        "proposal": str(proposal_path), "proposal_file_sha256": sha(proposal_path),
        "proposal_canonical_sha256": canonical_hash(proposal),
        "pending": str(pending_path), "pending_file_sha256": sha(pending_path),
        "original_tasks": 4, "appended_tasks": len(appended), "proposal_tasks": len(proposal["tasks"]),
        "pending_tasks": len(pending), "all_source_configurations": len(index),
        "all_formal_epochs": sum(x["epochs"] for x in index),
        "source_files_bound_per_new_task": len(SOURCE_ASSETS), "source_configurations": index,
    })
    print(json.dumps({"proposal_tasks": len(proposal["tasks"]), "appended": len(appended),
                      "pending": len(pending), "configs": len(index), "formal_epochs": sum(x["epochs"] for x in index)}))


if __name__ == "__main__":
    main()
