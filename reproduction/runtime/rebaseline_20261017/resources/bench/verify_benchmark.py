"""Independent CPU-only validation of saved Table6 GPU resource evidence."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

from bench_contract import (PLAN, BENCH, HEADS, atomic, cached_only, check_measurement,
                            code_assets, emit, inspect_parameters, require, sha, sources)
from odr_resources import collect, check_against_sources


def verify(path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("PYTHONNOUSERSITE") == "1"
            and os.environ.get("STY"), "Independent guarded CPU verifier required")
    import numpy as np
    import torch
    from types import SimpleNamespace
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    require(report["status"] == "passed" and report["exitcode"] == 0
            and Path(str(path) + ".exitcode").read_text().strip() == "0",
            "GPU benchmark lacks successful process/exit evidence")
    require(report["schema"] == "spnv2-table6-forward-benchmark-v1" and report["seed"] == 2021,
            "Benchmark schema/seed mismatch")
    require(report["accuracy_or_end_to_end_speed_claim"] is False, "Wrong speed/accuracy interpretation")
    require(report["assets_sha256"] == code_assets(), "Benchmark source/config/helper/static-parameter evidence changed")
    entries = sources()
    require(report["sources"] == entries, "Accepted source/domain/final evidence changed")
    require(report["odr_resources"] == collect(), "Real fixed40 ODR resource evidence changed")
    check_against_sources(report["odr_resources"], entries)
    require(report["parent_gpu_gate"]["passed"] is True
            and report["parent_gpu_gate"]["runner_sha256"] == sha(PLAN / "pipeline/runner.py"),
            "Reviewed parent GPU gate proof changed")
    require(report["runtime"]["torch_home"] == str(PLAN.parents[1] / "env/cache/torch")
            and report["runtime"]["OMP_NUM_THREADS"] == report["runtime"]["OPENBLAS_NUM_THREADS"] == "2"
            and report["runtime"]["screen"], "Benchmark environment mismatch")
    require(type(report["device"]["total_memory_bytes"]) is int and report["device"]["total_memory_bytes"] > 0
            and report["device"]["name"], "GPU device evidence missing")
    require(len(report["measurements"]) == 4 and [x["name"] for x in report["measurements"]] == [x["name"] for x in entries],
            "Four-architecture sequential measurement set mismatch")
    torch.set_num_threads(2)
    torch.hub.set_dir(str(PLAN.parents[1] / "env/cache/torch/hub"))
    sys.path.insert(0, str(PLAN / "source/core"))
    from config import cfg as defaults, update_config
    from nets import build_spnv2
    import nets.backbones.efficientnet as efficientnet
    from utils.training_state import model_digest
    cached_only(efficientnet, torch)
    results = []
    for item, entry in zip(report["measurements"], entries):
        artifact = Path(item["path"]).resolve()
        require(artifact == Path(report["artifacts"]).resolve() / (entry["name"] + ".json")
                and BENCH in artifact.parents and sha(artifact) == item["sha256"], "Raw per-model file/path/hash mismatch")
        value = json.loads(artifact.read_text())
        check_measurement(value)
        require(value["name"] == entry["name"] and value["source_final"] == entry["final"]
                and value["source_final_sha256"] == entry["final_sha256"]
                and value["parameter_counts"] == entry["parameter_counts"], "Model/file/parameter denominator mismatch")
        samples = np.asarray(value["latencies_ms"], dtype=np.float64)
        reference = {"mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
                     "p05_ms": float(np.quantile(samples, .05, method="linear")),
                     "p90_ms": float(np.quantile(samples, .9, method="linear")),
                     "p95_ms": float(np.quantile(samples, .95, method="linear")),
                     "population_std_ms": float(np.std(samples, ddof=0))}
        for key, expected in reference.items():
            require(math.isclose(value["statistics"][key], expected, rel_tol=1e-12, abs_tol=1e-12),
                    "Independent NumPy statistics mismatch: " + key)
        require(value["TF32"] is False and value["deterministic_algorithms"] is True
                and value["runtime_config_changes"] == {"TEST.HEAD": HEADS, "FP16": False},
                "Actual FP32/EHS inference selection mismatch")
        cfg = defaults.clone()
        update_config(cfg, SimpleNamespace(cfg=entry["cfg"], opts=[]))
        cfg.defrost(); cfg.TEST.HEAD = list(HEADS); cfg.FP16 = False; cfg.CUDA = False; cfg.freeze()
        torch.random.default_generator.manual_seed(2021)
        model = build_spnv2(cfg).float()
        saved = torch.load(entry["final"], map_location="cpu", weights_only=True)
        model.load_state_dict(saved, strict=True)
        for name, tensor in model.state_dict().items():
            require(torch.equal(tensor, saved[name]) and tensor.dtype == saved[name].dtype,
                    "Accepted source CPU strict reload differs")
        require(model_digest(model) == value["model_digest_before"], "Timed model did not equal accepted final")
        inventory = json.loads(Path(entry["parameter_inventory"]).read_text())
        inspect_parameters(model, inventory)
        results.append({"name": entry["name"], "status": "passed", "raw_measurement_sha256": item["sha256"],
                        "source_final_sha256": entry["final_sha256"], "statistics": value["statistics"],
                        "parameter_counts": value["parameter_counts"], "memory_bytes": value["memory_bytes"]})
        del model, saved; gc.collect()
    return {
        "status": "passed", "schema": "spnv2-table6-benchmark-cpu-verification-v1",
        "report": str(path), "report_sha256": sha(path),
        "exitcode_file_sha256": sha(str(path) + ".exitcode"),
        "verifier_sha256": sha(__file__), "models": results,
        "scope": "Accepted final strict CPU reload, raw30 event samples/statistics, parameter denominators and real fixed40 ODR audit resources",
        "claim": "Pure all-EHS GPU forward workload only; no end-to-end speed or scientific accuracy claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(BENCH in output.parents and not output.exists(), "Use new benchmark-owned CPU acceptance report")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = verify(args.report)
    except Exception as error:
        atomic(output, {"status": "failed", "error": type(error).__name__ + ": " + str(error), "verifier_sha256": sha(__file__)})
        emit("error", error=str(error), message="Independent CPU resource verification failed")
        raise
    atomic(output, result)
    emit("resource_verification_complete", output=str(output), message="Independent CPU verification passed")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
