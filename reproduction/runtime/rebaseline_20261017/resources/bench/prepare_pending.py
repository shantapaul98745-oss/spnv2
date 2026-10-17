"""Prepare one non-executable pending Table6 benchmark definition."""
import copy
import json
from pathlib import Path
import sys

from bench_contract import PLAN, BENCH, MODELS, code_assets, sha


def main():
    path = BENCH / "pending_task.json"
    if path.exists():
        raise FileExistsError("Do not overwrite a pending definition")
    base = json.loads((PLAN / "queue_v2.json").read_text())
    env = copy.deepcopy(base["tasks"][4]["env"])
    source_ids = []
    for name, train, stem, report in MODELS:
        source_ids += [train] + ["eval_" + stem + "_" + domain for domain in ("synthetic", "lightbox", "sunlamp")]
    odr = json.loads((PLAN / "odr/manifest.json").read_text())
    fixed = [t["id"] for t in odr["tasks"] if t["group"] == "fixed"]
    report = BENCH / "reports/table6_forward_four_models_20261017.json"
    verified = BENCH / "reports/table6_forward_four_models_20261017_verified.json"
    artifacts = BENCH / "artifacts/table6_forward_four_models_20261017"
    python = ["/home/ps/anaconda3/envs/SPNv2/bin/python", "-B", "-s", "-u"]
    task = {
        "id": "table6_forward_four_EHS_models", "kind": "table6_gpu_forward_resources",
        "command": python + [str(BENCH / "run_benchmark.py"), "--report", str(report), "--artifacts", str(artifacts)],
        "verify_command": python + [str(BENCH / "verify_benchmark.py"), "--report", str(report), "--output", str(verified)],
        "cwd": str(BENCH), "requires": source_ids + fixed,
        "session": "spnv2_table6_four_EHS_forward_20261017",
        "log": str(BENCH / "logs/table6_four_EHS_forward_20261017.log"),
        "env": env, "gpu": copy.deepcopy(base["tasks"][0]["gpu"]), "verify_gpu": False,
        "verify_env": {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
                       "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
        "expected_outputs": [str(verified)],
        "metadata": {
            "source": str(BENCH / "run_benchmark.py"), "source_acceptance_task_ids": source_ids,
            "fixed_odr_task_ids": fixed, "asset_sha256": code_assets(),
            "models": [x[0] for x in MODELS], "input_shape": [1, 3, 512, 768],
            "seed": 2021, "warmup_per_model": 10, "timed_forward_per_model": 30, "FP32": True,
            "workload": "Full network, all EHS heads, eval/no_grad, zero CUDA-resident input",
            "excluded": ["preprocessing", "data loading", "H2D", "pose/PnP postprocessing"],
            "no_accuracy_or_end_to_end_speed_claim": True, "new_ODR_experiments": 0,
        },
    }
    value = {
        "schema": "spnv2-pending-resource-benchmark-v1", "execution_manifest": False, "activation_allowed": False,
        "suggested_insert_after": "eval_table4_phi3_gn_sunlamp",
        "suggested_insert_before": next(t["id"] for t in odr["tasks"] if t["group"] == "Bscan"),
        "activation_requirements": [
            "Root reviews benchmark code and CPU evidence and explicitly integrates one serial parent task",
            "All four independent source finals and twelve domain evaluations accepted",
            "All fixed40 real ODR acceptance/audit records available; no additional phi0 or phi3GN ODR",
            "Parent global GPU gate succeeds before benchmark allocates its sole CUDA context",
        ],
        "pending_task_definition": task,
        "GPU_executed": False,
    }
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pending_only": True, "models": 4, "warmup_each": 10, "measured_each": 30,
                      "requires": len(task["requires"]), "file_sha256": sha(path)}))


if __name__ == "__main__":
    main()
