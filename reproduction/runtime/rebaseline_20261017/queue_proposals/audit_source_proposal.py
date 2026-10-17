"""Static CPU audit of proposals. No launches, signals, live status writes, or audits."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

PLAN = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN / "pipeline"))
import runner
from revision import extend_state


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def arg(command, flag):
    require(flag in command and command.index(flag) + 1 < len(command), "Missing CLI flag " + flag)
    return command[command.index(flag) + 1]


def audit():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("PYTHONNOUSERSITE") == "1",
            "CPU-only isolated environment required")
    require(os.environ.get("STY"), "Independent guarded screen required")
    old = json.loads((PLAN / "queue.json").read_text())
    proposal_path = DEST / "full_source_v2_proposal.json"
    pending_path = DEST / "pending_source_tasks.json"
    proposal = json.loads(proposal_path.read_text())
    pending = json.loads(pending_path.read_text())
    index = json.loads((DEST / "proposal_index.json").read_text())
    checks = {}
    runner.validate_manifest(proposal)
    checks["ordinary_manifest_validation"] = True
    require(proposal["tasks"][:4] == old["tasks"], "Original task structures changed")
    require(json.dumps(proposal["tasks"][:4], ensure_ascii=False) == json.dumps(old["tasks"], ensure_ascii=False),
            "Original task key ordering/structure changed")
    require(index["old_manifest_file_sha256"] == sha(PLAN / "queue.json"), "Original manifest file changed")
    require(index["proposal_file_sha256"] == sha(proposal_path), "Proposal file changed")
    require(index["pending_file_sha256"] == sha(pending_path), "Pending file changed")
    require(runner.manifest_hash(old) == index["old_manifest_canonical_sha256"], "Old binding hash mismatch")
    checks["original_four_tasks_exact_prefix"] = True
    current = json.loads((PLAN / "status.json").read_text())
    amended, revision_audit = extend_state(current, old, proposal, caller_context={"static_proposal_audit": True})
    state = runner.initialize_state(amended, proposal)
    require(state["tasks"][:4][0]["runtime"] == current["tasks"][0]["runtime"], "Active runtime lost")
    for key in current["current"]:
        if key != "process_state":
            require(state["current"][key] == current["current"][key], "Current progress lost: " + key)
    require(state["current"]["pid"] == 3111276 and str(state["current"]["starttime"]) == "517862167",
            "Unexpected active worker identity; refresh this audit for changed live context")
    checks["pure_append_state_compatibility_and_progress_preservation"] = True
    require(proposal["tasks"][4]["id"] == "pilot_phi6_EHS" and len(proposal["tasks"]) == 9,
            "Only phi6 five-task continuation may be proposed now")
    require(proposal["tasks"][4]["requires"] == ["eval_phi3_sunlamp"], "Phi6 starts before existing evaluations")
    require(all("table" not in task["id"] for task in proposal["tasks"][4:]), "Pending tables leaked into executable proposal")
    require(pending["execution_manifest"] is False and pending["activation_allowed"] is False,
            "Pending source definitions accidentally activated")
    require("tasks" not in pending and len(pending["pending_task_definitions"]) == 35,
            "Pending file must not be a runnable manifest")
    try:
        runner.validate_manifest(pending)
    except ValueError:
        pass
    else:
        raise AssertionError("Pending definition wrapper unexpectedly accepted as executable manifest")
    require("fixed40" in pending["activation_requirements"][0]
            and pending["first_task_requires_real_odr_dependency_at_activation"] == "pilot_table2_h",
            "Fixed40 ODR barrier missing")
    checks["table2_3_4_separate_and_not_executable_before_fixed40_odr"] = True
    definitions = pending["pending_task_definitions"]
    combined = copy.deepcopy(proposal)
    combined["tasks"].extend(definitions)
    runner.validate_manifest(combined)
    checks["all_definition_ids_sessions_logs_and_dependency_order_valid"] = True
    old_sessions = {task["session"] for task in old["tasks"]}
    all_tasks = proposal["tasks"] + definitions
    new_tasks = proposal["tasks"][4:] + definitions
    expected_source_assets = {
        str(p.resolve()): sha(p) for folder in ("core", "tools")
        for p in sorted((PLAN / "source" / folder).rglob("*")) if p.is_file()
    }
    require(len(expected_source_assets) == 48, "Frozen core/tools inventory changed")
    checked_assets = {}
    cfg_cache = {}
    cfg_rows = {}
    formal_by_config = {}
    pilot_by_config = {}
    eval_by_config = {}
    expected_counts = {"synthetic": 11994, "lightbox": 6740, "sunlamp": 2791}
    for task in new_tasks:
        require(task["verify_gpu"] is False and task["verify_env"]["CUDA_VISIBLE_DEVICES"] == "",
                "Verification is not CPU-only")
        require(task["env"]["PYTHONNOUSERSITE"] == "1"
                and task["env"]["OMP_NUM_THREADS"] == task["env"]["OPENBLAS_NUM_THREADS"] == "2"
                and task["env"]["PYTHONHASHSEED"] == "2021"
                and task["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
                and task["env"]["TORCH_HOME"] == str(PLAN.parents[1] / "env/cache/torch"),
                "Task environment mismatch")
        require(task["session"] not in old_sessions, "Session collision with original work")
        for target in (task["log"], task["log"] + ".exitcode",
                       str(PLAN / "events" / (task["id"] + ".main.config.json")),
                       str(PLAN / "events" / (task["id"] + ".main.process.json"))):
            require(not Path(target).exists(), "Proposed launch would overwrite existing evidence: " + target)
        require(not any(Path(p).exists() for p in task["expected_outputs"]), "Proposed output already exists")
        hashes = task["metadata"]["asset_sha256"]
        require(expected_source_assets.items() <= hashes.items(), "Incomplete source asset binding")
        require(str(PLAN / "source_manifest.json") in hashes, "Missing frozen manifest binding")
        for path, expected in hashes.items():
            require(Path(path).is_absolute(), "Asset key is not absolute")
            runner.inside(path)
            if path not in checked_assets:
                checked_assets[path] = sha(path)
            require(checked_assets[path] == expected, "Bound asset changed: " + path)
        cfg_path = Path(task["metadata"]["source"])
        require(str(cfg_path) in hashes, "Corresponding config is not bound")
        if str(cfg_path) not in cfg_cache:
            cfg_cache[str(cfg_path)] = yaml.safe_load(cfg_path.read_text())
        cfg = cfg_cache[str(cfg_path)]
        require(Path(cfg["ROOT"]) == PLAN / "source", "Task uses mutable code")
        require(cfg["MODEL"]["HEAD"]["NAMES"] == cfg["MODEL"]["HEAD"]["LOSS_HEADS"], "Head/loss mismatch")
        require(cfg["SEED"] == 2021 and cfg["DATASET"]["INPUT_SIZE"] == [768, 512],
                "Configuration protocol mismatch")
        require(str(PLAN / "tasks/source_artifacts.py") in hashes
                or str(PLAN / "pilots/gpu_smoke.py") in hashes, "Actual helper not bound")
        if task["kind"] == "engineering_gpu_resume_pilot":
            name = task["metadata"]["config_name"]
            pilot_by_config[name] = task
            require(arg(task["command"], "--attempts") == "32"
                    and arg(task["command"], "--checkpoint") == "16", "Pilot windows mismatch")
            require(arg(task["command"], "--source") == str(PLAN / "source"), "Pilot source mismatch")
            require(Path(arg(task["command"], "--cfg")).parent == PLAN / "pilots/configs", "Pilot config outside own directory")
            for value in (arg(task["command"], "--report"), arg(task["command"], "--artifacts"),
                          arg(task["verify_command"], "--output")):
                require(PLAN / "pilots" in Path(value).parents and not Path(value).exists(),
                        "Pilot evidence path exists or is not independent")
            require(task["verify_command"][4] == str(PLAN / "pilots/verify_pilot.py"), "Wrong CPU pilot verifier")
            require(arg(task["verify_command"], "--report") == arg(task["command"], "--report"),
                    "Pilot CPU verifier consumes wrong report")
            for name in ("gpu_smoke.py", "verify_pilot.py", "evidence.py"):
                require(str(PLAN / "pilots" / name) in hashes, "Pilot helper missing from binding")
        elif task["kind"].startswith("source_training_"):
            name = task["metadata"]["config_name"]
            require(name not in formal_by_config, "Duplicate formal source configuration")
            formal_by_config[name] = task
            require(task["command"][4] == str(PLAN / "source/tools/train.py")
                    and arg(task["command"], "--cfg") == str(cfg_path), "Formal command/config mismatch")
            require(task["verify_command"][5] == "verify_train"
                    and arg(task["verify_command"], "--log") == task["log"], "Formal verify command/log mismatch")
            require(cfg["TRAIN"]["RESUME_FILE"] == "" and cfg["AUTO_RESUME"] is False
                    and not cfg["MODEL"].get("PRETRAIN_FILE", "")
                    and not cfg["MODEL"]["BACKBONE"].get("PRETRAINED", ""), "Formal training inherits prior/pilot weights")
            require(PLAN / "pilots" not in Path(cfg["OUTPUT_DIR"]).parents
                    and task["metadata"]["pilot_weights_for_formal"] is False,
                    "Formal output/initialization aliases pilot")
            expected_batch = 1 if name == "phi6_gn_ehs" else 4
            require(cfg["TRAIN"]["IMAGES_PER_GPU"] == expected_batch
                    and cfg["TRAIN"]["ACCUM_STEPS"] == 4, "Formal physical/accumulation protocol mismatch")
        else:
            require(task["kind"] == "full_source_evaluation", "Unexpected task kind")
            name, domain = task["metadata"]["source_config"], task["metadata"]["domain"]
            eval_by_config.setdefault(name, {})[domain] = task
            require(task["command"][5] == "evaluate" and task["verify_command"][5] == "verify_eval",
                    "Evaluation helper CLI mismatch")
            require(arg(task["command"], "--expected-samples") == str(expected_counts[domain])
                    == arg(task["verify_command"], "--expected-samples"), "Evaluation denominator mismatch")
            require(cfg["TEST"]["IMAGES_PER_GPU"] == 1 and cfg["TRAIN"]["REPRODUCIBLE"] is False,
                    "Full evaluation protocol mismatch")
            csv_path = Path(cfg["DATASET"]["ROOT"]) / cfg["DATASET"]["DATANAME"] / cfg["TEST"]["TEST_CSV"]
            if str(csv_path) not in cfg_rows:
                with csv_path.open(newline="") as stream:
                    cfg_rows[str(csv_path)] = sum(1 for row in csv.reader(stream))
            require(cfg_rows[str(csv_path)] == expected_counts[domain], "Actual evaluation CSV count differs")
    for name, train in formal_by_config.items():
        require(set(eval_by_config[name]) == set(expected_counts), "A source is missing one full domain")
        cfg = cfg_cache[train["metadata"]["source"]]
        final = str(Path(cfg["OUTPUT_DIR"]) / cfg["MODEL"]["BACKBONE"]["NAME"] / cfg["EXP_NAME"] / "model_final.pth.tar")
        require(not Path(final).exists(), "Formal final model already exists; reconcile evidence instead of relaunch")
        previous = train["id"]
        for domain in ("synthetic", "lightbox", "sunlamp"):
            task = eval_by_config[name][domain]
            eval_cfg = cfg_cache[task["metadata"]["source"]]
            require(eval_cfg["TEST"]["MODEL_FILE"] == final, "Evaluation final-model pointer differs")
            require(task["requires"] == [previous], "Training/evaluation serial chain broken")
            previous = task["id"]
        if name in pilot_by_config:
            require(train["requires"] == [pilot_by_config[name]["id"]], "Formal source bypasses pilot CPU acceptance")
    checks["all_new_tasks_hash_bind_frozen_code_configs_and_helpers"] = True
    checks["new_outputs_unique_and_formal_sources_independent"] = True
    checks["formal_and_full_domain_evaluation_cli_and_denominators"] = True
    require(len(pilot_by_config) == 4 and len(formal_by_config) == 9 and len(eval_by_config) == 9,
            "Source task group count mismatch")
    source_index = index["source_configurations"]
    require(len(source_index) == 10 and len({x["config"] for x in source_index}) == 10
            and sum(x["epochs"] for x in source_index) == 190, "Full10 configuration/190-epoch count failed")
    require(sum(1 for x in source_index if x["epochs"] == 10) == 1
            and next(x for x in source_index if x["epochs"] == 10)["config"] == "table2_h.yaml",
            "Only H has10 formal epochs")
    require(not any(t["id"] == "train_phi3_EHS" for t in new_tasks), "Current phi3 duplicated")
    require(all(x["seed"] == 2021 and x["accum_steps"] == 4 for x in source_index),
            "Source index protocol mismatch")
    require(len({x["model_final"] for x in source_index}) == 10, "Formal output aliases another source")
    checks["complete10_source_configurations190_epochs_no_phi3_relaunch"] = True
    return {
        "status": "passed", "scope": "static proposal audit only", "checks": checks,
        "proposal_only": True, "manager_switched": False, "training_started": False,
        "proposal_tasks": len(proposal["tasks"]), "preserved_original_tasks": 4,
        "appended_phi6_tasks": len(proposal["tasks"]) - 4, "pending_source_tasks": len(definitions),
        "pilot_configs": sorted(pilot_by_config), "new_formal_source_configs": sorted(formal_by_config),
        "complete_formal_configurations": 10, "complete_formal_epochs": 190,
        "source_files_bound_per_task": len(expected_source_assets), "unique_assets_checked": len(checked_assets),
        "proposal_file_sha256": sha(proposal_path), "proposal_canonical_sha256": runner.manifest_hash(proposal),
        "pending_file_sha256": sha(pending_path), "original_manifest_file_sha256": sha(PLAN / "queue.json"),
        "original_manifest_canonical_sha256": runner.manifest_hash(old),
        "audited_current_worker": {"pid": state["current"]["pid"], "starttime": state["current"]["starttime"],
                                  "preserved_runtime_sha256": runner.manifest_hash(current["tasks"][0]["runtime"])},
        "code_sha256": {str(p): sha(p) for p in (Path(__file__), DEST / "build_source_proposal.py",
                                                PLAN / "pipeline/runner.py", PLAN / "pipeline/revision.py")},
        "screen": os.environ["STY"], "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "command": sys.argv, "finished": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    path = Path(args.report).resolve()
    require(DEST in path.parents and not path.exists(), "Use a new report inside queue_proposals/")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit()
    except Exception as error:
        result = {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                  "screen": os.environ.get("STY"), "command": sys.argv}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        raise
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
