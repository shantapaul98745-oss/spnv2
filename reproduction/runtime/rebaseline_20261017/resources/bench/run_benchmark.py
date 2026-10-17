"""Pending Table6 full-EHS inference benchmark. Execute only via parent guarded queue."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

from bench_contract import (PLAN, BENCH, HEADS, atomic, cached_only, check_measurement,
                            code_assets, emit, inspect_parameters, latency_stats, require, sha, sources)
from odr_resources import collect, check_against_sources


def run(args, report):
    require(os.environ.get("STY") and os.environ.get("PYTHONNOUSERSITE") == "1", "Guarded screen and no user-site required")
    require(os.environ.get("TORCH_HOME") == str(PLAN.parents[1] / "env/cache/torch"),
            "Use fixed project TORCH_HOME")
    require(os.environ.get("OMP_NUM_THREADS") == os.environ.get("OPENBLAS_NUM_THREADS") == "2",
            "Two CPU threads required")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Fixed CUBLAS workspace required")
    entries = sources()
    odr = collect()  # Accepted fixed evidence only, before any CUDA allocation.
    check_against_sources(odr, entries)
    assets = code_assets()
    sys.path.insert(0, str(PLAN / "pipeline"))
    import runner
    gate_manifest = json.loads((PLAN / "queue.json").read_text())
    spec = next(t["gpu"] for t in gate_manifest["tasks"] if isinstance(t.get("gpu"), dict))
    ready, reason = runner.gpu_available(spec)
    require(ready, "Parent global GPU gate blocked benchmark: " + str(reason))
    # Gate runs once before this sole serial GPU task initializes its own CUDA context.
    # Rechecking between models would incorrectly treat this same context as another job.
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(PLAN / "source/core"))
    from config import cfg as defaults, update_config
    from nets import build_spnv2
    import nets.backbones.efficientnet as efficientnet
    from utils.utils import set_seeds_cudnn
    from utils.training_state import model_digest, assert_finite_weights
    cached_only(efficientnet, torch)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    output = Path(args.artifacts).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report.update(
        schema="spnv2-table6-forward-benchmark-v1", seed=2021, sources=entries, assets_sha256=assets,
        runtime={"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "cudnn": torch.backends.cudnn.version(),
                 "torch_home": os.environ["TORCH_HOME"], "screen": os.environ["STY"],
                 "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"], "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"]},
        device={"name": props.name, "total_memory_bytes": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(device)), "logical_index": 0},
        parent_gpu_gate={"function": str(PLAN / "pipeline/runner.py") + ":gpu_available",
                         "runner_sha256": sha(PLAN / "pipeline/runner.py"), "spec": spec, "passed": True},
        workload_scope="Whole SPNv2 backbone and all EHS heads, fixed CUDA-resident zero input; excludes preprocessing, H2D transfer, PnP/pose postprocessing and data loading",
        accuracy_or_end_to_end_speed_claim=False, odr_resources=odr, measurements=[],
        parameter_scope="Counts reused from verified resources parameter evidence; actual source model inventory rechecked",
    )
    emit("resource_benchmark_start", source="four accepted independent EHS finals", models=4)
    for entry in entries:
        emit("resource_model_start", source=entry["final"], config=entry["cfg"], message=entry["name"])
        cfg = defaults.clone()
        update_config(cfg, SimpleNamespace(cfg=entry["cfg"], opts=[]))
        # Architecture/weights are unchanged; select every existing EHS head for timing.
        cfg.defrost(); cfg.TEST.HEAD = list(HEADS); cfg.FP16 = False; cfg.freeze()
        set_seeds_cudnn(cfg, seed=2021)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model = build_spnv2(cfg).float()
        weights = torch.load(entry["final"], map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        for name, value in model.state_dict().items():
            require(value.dtype == weights[name].dtype and torch.equal(value.cpu(), weights[name].cpu()),
                    "Strictly reloaded final differs: " + name)
        del weights
        inventory = json.loads(Path(entry["parameter_inventory"]).read_text())
        counts = inspect_parameters(model, inventory)
        model.eval(); model.requires_grad_(False)
        before = model_digest(model)
        require(list(model.test_h_idx) == [0, 1, 2] and list(model.head_names) == HEADS,
                "Actual full EHS inference membership mismatch")
        model = model.to(device)
        tensor = torch.zeros((1, 3, 512, 768), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            for _ in range(10):
                out = model(tensor, gpu=device)
                del out
            torch.cuda.synchronize(device)
            emit("resource_warmup_complete", source=entry["final"], samples_seen=10)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(30)]
            latencies = []
            for index, (start, end) in enumerate(events):
                start.record()
                out = model(tensor, gpu=device)
                end.record()
                torch.cuda.synchronize(device)
                latencies.append(float(start.elapsed_time(end)))
                if index != 29:
                    del out
            # Capture peaks before output finite checks or CPU integrity copies.
            memory = {
                "baseline_allocated": baseline_allocated, "baseline_reserved": baseline_reserved,
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            }
            def finite(value):
                if torch.is_tensor(value):
                    return not value.is_floating_point() or bool(torch.isfinite(value).all())
                if isinstance(value, (list, tuple)):
                    return all(finite(x) for x in value)
                if isinstance(value, dict):
                    return all(finite(x) for x in value.values())
                return True
            require(finite(out), "Nonfinite actual forward output")
            del out
        after = model_digest(model)
        assert_finite_weights(model)
        require(before == after and sha(entry["final"]) == entry["final_sha256"], "Benchmark changed accepted final/model")
        item = {
            "name": entry["name"], "source_final": entry["final"], "source_final_sha256": entry["final_sha256"],
            "input_shape": [1, 3, 512, 768], "input_kind": "fixed_zero_tensor_cuda_resident",
            "warmup_count": 10, "forward_count": 40, "dtype": "torch.float32",
            "eval_mode": not model.training, "no_grad": True, "autocast": False, "all_EHS_heads": list(HEADS),
            "timing_method": "CUDA events with synchronize after every forward",
            "latencies_ms": latencies, "statistics": latency_stats(latencies), "memory_bytes": memory,
            "strict_reload": True, "finite_output_verified": True, "weights_unchanged": before == after,
            "model_digest_before": before, "model_digest_after": after, "parameter_counts": counts,
            "runtime_config_changes": {"TEST.HEAD": list(HEADS), "FP16": False},
            "cudnn_benchmark": False, "deterministic_algorithms": True, "TF32": False,
            "peak_scope": "Model/input resident baseline plus measured whole-EHS forward, captured before finite/integrity checks",
        }
        check_measurement(item)
        artifact = output / (entry["name"] + ".json")
        atomic(artifact, item)
        report["measurements"].append({"path": str(artifact), "sha256": sha(artifact), "name": entry["name"]})
        atomic(Path(args.report), report)
        emit("resource_model_complete", source=entry["final"], output=str(artifact), message=entry["name"])
        del model, tensor, events
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    require(assets == code_assets(), "Benchmark implementation/source/config/parameter evidence changed during run")
    report["status"] = "passed"
    emit("resource_benchmark_complete", output=args.report, message="Four-model forward resources measured; independent CPU verification pending")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    path, output = Path(args.report).resolve(), Path(args.artifacts).resolve()
    require(BENCH in path.parents and BENCH in output.parents, "Outputs must stay under resources/bench")
    require(not output.exists() and not Path(str(path) + ".exitcode").exists(), "Use independent fresh output paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents a second invocation from taking over the same evidence.
    with path.open("x") as stream:
        stream.write("{}\n")
    report = {"status": "failed", "command": sys.argv, "started": time.time(),
              "artifacts": str(output), "pid": os.getpid()}
    rc = 1
    with (BENCH / "gpu_serial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args, report); rc = 0
        except Exception as error:
            report.update(status="failed", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
            emit("error", error=report["error"], message="Resource benchmark failed; preserve evidence")
            print(report["traceback"], file=sys.stderr, flush=True)
        finally:
            report.update(exitcode=rc, finished=time.time())
            atomic(path, report)
            with Path(str(path) + ".exitcode").open("x") as stream:
                stream.write(str(rc) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
