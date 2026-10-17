"""Actual full-resolution engineering pilots; launch only in a guarded GPU screen.

Fresh independent initialization, 32 complete update attempts, boundary16 restore.
This runs one engineering subset epoch and never provides weights to formal runs.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from evidence import (ARTIFACT_NAMES, exact_tree, model_sha256, parameter_changes,
                      protocol, sha256_file, write_json)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--checkpoint", type=int, default=16)
    return parser.parse_args()


def run(args, report):
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not os.environ.get("STY"):
        raise RuntimeError("A guarded independent screen and PYTHONNOUSERSITE=1 are required")
    plan = Path(__file__).resolve().parents[1]
    source = Path(args.source).resolve()
    if source != plan / "source":
        raise ValueError("Use this plan's frozen source")
    fixed_cache = str(plan.parents[1] / "env/cache/torch")
    if os.environ.get("TORCH_HOME") != fixed_cache:
        raise ValueError("TORCH_HOME must be the fixed project cache: " + fixed_cache)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
    import torch
    import yaml
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(source / "core"))
    from config import cfg as defaults, update_config
    from dataset.build import build_dataset, _seed_worker
    from nets import build_spnv2
    import nets.backbones.efficientnet as efficientnet
    from solver import get_optimizer, get_scaler, adjust_learning_rate
    import engine.trainer as trainer
    from utils.utils import set_seeds_cudnn
    from utils.training_state import (TrainingState, EpochBatchSampler, SeededDataset,
        build_manifest, save_training_checkpoint, load_training_checkpoint,
        atomic_torch_save, capture_rng_state, assert_finite_weights, emit_event)

    cfg = defaults.clone()
    update_config(cfg, argparse.Namespace(cfg=args.cfg, opts=[]))
    effective = yaml.safe_load(cfg.dump())
    report["protocol_checks"] = protocol(effective, args.attempts, args.checkpoint)
    if Path(cfg.ROOT).resolve() != source:
        raise ValueError("Config ROOT must point to frozen source")
    pilots = plan / "pilots"
    for value in (args.cfg, args.report, args.artifacts, cfg.OUTPUT_DIR, cfg.LOG_DIR):
        if pilots not in Path(value).resolve().parents:
            raise ValueError("Pilot-owned configuration and outputs must stay under pilots/")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual GPU pilot requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    heads = list(cfg.MODEL.HEAD.NAMES)
    count = int(cfg.TRAIN.IMAGES_PER_GPU * cfg.TRAIN.ACCUM_STEPS * args.attempts)
    report.update(
        schema="spnv2-engineering-pilot-v1", source=str(source),
        cfg=str(Path(args.cfg).resolve()), cfg_file_sha256=sha256_file(args.cfg),
        effective_config=cfg.dump(), effective_config_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(),
        attempts=args.attempts, checkpoint_attempt=args.checkpoint, subset_rows=count,
        physical_batch=int(cfg.TRAIN.IMAGES_PER_GPU), accumulation=int(cfg.TRAIN.ACCUM_STEPS),
        heads=heads, seed=int(cfg.SEED), clip_norm=1.0, loss_normalization="local_mean_divided_by_accumulation",
        subset_selection="first CSV rows followed by epoch-keyed permutation",
        pilot_weights_for_formal=False, test_sha256=sha256_file(__file__),
        evidence_helper_sha256=sha256_file(Path(__file__).with_name("evidence.py")),
        device={"name": props.name, "total_memory": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(device))},
        runtime={"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                 "cudnn": torch.backends.cudnn.version(), "torch_home": os.environ["TORCH_HOME"],
                 "screen": os.environ["STY"]},
        initialization_disclosure=(
            "GN public code random initialization with residual GN zero initialization; author initialization unverified"
            if cfg.MODEL.USE_GROUPNORM_BACKBONE else
            "BN EfficientNet official ImageNet weights from fixed local cache only; local forward batch statistics"),
    )
    # The production BN builder is unchanged; its requested URL must already be cached.
    def cached_only(url, *unused, **kwargs):
        cached = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
        if not cached.is_file():
            raise FileNotFoundError("Missing official BN cache; network download forbidden: " + str(cached))
        return torch.load(cached, map_location="cpu", weights_only=True)
    efficientnet.load_state_dict_from_url = cached_only

    artifacts = Path(args.artifacts).resolve()
    if artifacts.exists() and any(artifacts.iterdir()):
        raise FileExistsError("Use a new empty artifact directory")
    artifacts.mkdir(parents=True, exist_ok=True)
    paths = {name: artifacts / name for name in ARTIFACT_NAMES}
    def construct():
        set_seeds_cudnn(cfg, seed=cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model = build_spnv2(cfg).to(device)
        if list(model.head_names) != heads or len(model.heads) != len(heads) or list(model.loss_h_idx) != list(range(len(heads))):
            raise AssertionError("Actual model head/loss membership differs from effective config")
        def verify_actual_input(module, inputs):
            shape = list(inputs[0].shape)
            expected = [cfg.TRAIN.IMAGES_PER_GPU, 3, 512, 768]
            if shape != expected:
                raise AssertionError("Actual input shape mismatch: " + str(shape))
            observed = report.setdefault("observed_training_shapes", [])
            if shape not in observed:
                observed.append(shape)
            report["observed_forward_count"] = report.get("observed_forward_count", 0) + 1
        model.register_forward_pre_hook(verify_actual_input)
        raw = build_dataset(cfg, split="train", load_labels=True)
        original_count = len(raw)
        if original_count < count:
            raise ValueError("Insufficient original CSV rows")
        raw.csv = raw.csv.iloc[:count].copy().reset_index(drop=True)
        sampler = EpochBatchSampler(len(raw), cfg.TRAIN.IMAGES_PER_GPU,
                                    cfg.TRAIN.ACCUM_STEPS, cfg.SEED, cfg.TRAIN.SHUFFLE)
        expected_ids = [str(raw.csv.iloc[i, 0]) for i in sampler.indices()]
        data = SeededDataset(raw, cfg.SEED)
        loader = DataLoader(data, batch_sampler=sampler,
            num_workers=min(cfg.TRAIN.IMAGES_PER_GPU, cfg.TRAIN.WORKERS),
            pin_memory=cfg.PIN_MEMORY, worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(int(cfg.SEED) + 1))
        val = build_dataset(cfg, split="val", load_labels=True)
        optimizer, scaler = get_optimizer(cfg, model), get_scaler(cfg)
        if not isinstance(optimizer, torch.optim.AdamW) or scaler is None or not scaler.is_enabled():
            raise AssertionError("Actual AdamW and enabled CUDA GradScaler required")
        adjust_learning_rate(optimizer, 0, cfg)
        manifest = build_manifest(cfg, model, datasets={"train": data, "val": val})
        return model, optimizer, scaler, sampler, loader, manifest, original_count, expected_ids

    emit_event("pilot_start", phase="engineering_pilot", samples=count, attempts=args.attempts,
               physical_batch=cfg.TRAIN.IMAGES_PER_GPU, accumulation_steps=cfg.TRAIN.ACCUM_STEPS)
    report["stage"] = "construct"
    model, optimizer, scaler, sampler, loader, manifest, original_count, expected_ids = construct()
    report.update(manifest=manifest, code_sha256=manifest["code"], original_training_rows=original_count,
                  expected_sample_ids=expected_ids,
                  trainable_parameter_names=[name for name, p in model.named_parameters() if p.requires_grad])
    write_json(paths["manifest.json"], manifest)
    atomic_torch_save(model.state_dict(), paths["model_initial.pth.tar"])
    save_training_checkpoint(paths["initial_checkpoint.pth.tar"], model, optimizer, scaler,
                             TrainingState(), sampler, manifest)
    continuous_ids = []
    original_emit = trainer.emit_event
    def collect_continuous(event, state=None, **fields):
        if event == "update":
            continuous_ids.extend(str(value) for value in fields.get("sample_ids", []))
        return original_emit(event, state, **fields)
    checkpoint_times = []
    def boundary(current):
        if current.update_attempts == args.checkpoint:
            before = time.monotonic()
            save_training_checkpoint(paths["boundary_checkpoint.pth.tar"], model, optimizer, scaler,
                                     current, sampler, manifest)
            checkpoint_times.append(time.monotonic() - before)
            emit_event("checkpoint", current, phase="engineering_pilot",
                       path=str(paths["boundary_checkpoint.pth.tar"]), reason="pilot_resume_boundary")
    trainer.emit_event = collect_continuous
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    report["stage"] = "continuous_training"
    try:
        whole = trainer.do_train(0, cfg, model, loader, optimizer, device=device,
                                scaler=scaler, state=TrainingState(), on_boundary=boundary)
    finally:
        trainer.emit_event = original_emit
    torch.cuda.synchronize()
    elapsed = time.monotonic() - began
    if whole.update_attempts != args.attempts or whole.samples_seen != count or whole.optimizer_updates < 3:
        raise AssertionError("Pilot update/sample/success counters failed")
    if continuous_ids != expected_ids:
        raise AssertionError("Actual continuous ID stream differs from sampler/CSV")
    assert_finite_weights(model)
    initial = torch.load(paths["model_initial.pth.tar"], map_location="cpu", weights_only=True)
    report["component_parameter_changes"] = parameter_changes(
        initial, model.state_dict(), report["trainable_parameter_names"], heads)
    del initial
    report["continuous"] = {
        "seconds": elapsed, "images_per_second": count / elapsed,
        "checkpoint_seconds": sum(checkpoint_times), "state": whole.state_dict(),
        "memory_allocated_peak": torch.cuda.max_memory_allocated(device),
        "memory_reserved_peak": torch.cuda.max_memory_reserved(device),
        "sample_ids": continuous_ids,
    }
    atomic_torch_save(model.state_dict(), paths["model_final.pth.tar"])
    save_training_checkpoint(paths["continuous_final_checkpoint.pth.tar"], model, optimizer, scaler,
                             whole, sampler, manifest)
    whole_rng = capture_rng_state()
    whole_state = whole.state_dict()
    del model, optimizer, scaler, sampler, loader
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report["allocated_after_continuous_delete"] = torch.cuda.memory_allocated(device)
    report["stage"] = "fresh_reconstruction_and_restore"
    model, optimizer, scaler, sampler, loader, rebuilt_manifest, _, rebuilt_ids = construct()
    if rebuilt_ids != expected_ids or rebuilt_manifest != manifest:
        raise AssertionError("Fresh same-seed initialization/manifest differs")
    initial = torch.load(paths["model_initial.pth.tar"], map_location="cpu", weights_only=True)
    exact_tree(initial, model.state_dict(), "fresh_initialization")
    del initial
    report["fresh_initialization_equal"] = True
    resumed, _ = load_training_checkpoint(paths["boundary_checkpoint.pth.tar"], model, optimizer,
                                          scaler, sampler, rebuilt_manifest, device=device)
    if resumed.update_attempts != args.checkpoint or resumed.next_batch != args.checkpoint * cfg.TRAIN.ACCUM_STEPS:
        raise AssertionError("Restored checkpoint is not the requested complete boundary")
    boundary_samples = resumed.samples_seen
    resumed_ids = []
    def collect_resumed(event, state=None, **fields):
        if event == "update":
            resumed_ids.extend(str(value) for value in fields.get("sample_ids", []))
        return original_emit(event, state, **fields)
    emit_event("resume", resumed, phase="engineering_pilot", path=str(paths["boundary_checkpoint.pth.tar"]))
    trainer.emit_event = collect_resumed
    torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    report["stage"] = "resumed_training"
    try:
        trainer.do_train(0, cfg, model, loader, optimizer, device=device, scaler=scaler, state=resumed)
    finally:
        trainer.emit_event = original_emit
    torch.cuda.synchronize()
    resumed_rng = capture_rng_state()
    expected = torch.load(paths["continuous_final_checkpoint.pth.tar"], map_location="cpu", weights_only=True)
    exact_tree(expected["state_dict"], model.state_dict(), "model")
    exact_tree(expected["optimizer"], optimizer.state_dict(), "optimizer")
    exact_tree(expected["scaler"], scaler.state_dict(), "scaler")
    exact_tree(whole_state, resumed.state_dict(), "counters")
    exact_tree(whole_rng, resumed_rng, "rng")
    if continuous_ids[:boundary_samples] + resumed_ids != continuous_ids:
        raise AssertionError("Restored actual ID stream differs")
    del expected
    assert_finite_weights(model)
    atomic_torch_save(model.state_dict(), paths["model_resumed_final.pth.tar"])
    save_training_checkpoint(paths["resumed_final_checkpoint.pth.tar"], model, optimizer, scaler,
                             resumed, sampler, rebuilt_manifest)
    report["resume"] = {
        "exact": True, "model_equal": True, "optimizer_equal": True, "scaler_equal": True,
        "counters_equal": True, "rng_equal": True, "sample_ids_equal": True,
        "checkpoint_attempt": args.checkpoint, "seconds": time.monotonic() - began,
        "state": resumed.state_dict(), "sample_ids": resumed_ids,
        "memory_allocated_peak": torch.cuda.max_memory_allocated(device),
        "memory_reserved_peak": torch.cuda.max_memory_reserved(device),
    }
    current_code = {str(path.relative_to(source)): sha256_file(path)
                    for folder in ("core", "tools") for path in sorted((source / folder).rglob("*.py"))}
    if current_code != manifest["code"]:
        raise AssertionError("Frozen source changed during pilot")
    report["artifacts"] = {name: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
                           for name, path in paths.items()}
    report["status"] = "passed"
    report["stage"] = "complete"
    emit_event("pilot_complete", resumed, phase="engineering_pilot", exact_resume=True,
               report=str(Path(args.report).resolve()))


def main():
    args = arguments()
    report_path = Path(args.report).resolve()
    pilots = Path(__file__).resolve().parent
    if pilots not in report_path.parents:
        raise ValueError("Report must stay under pilots/")
    if report_path.exists():
        raise FileExistsError("Pilot report already exists")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "started": time.time(), "command": sys.argv,
              "artifacts_directory": str(Path(args.artifacts).resolve())}
    exitcode = 1
    try:
        run(args, report)
        exitcode = 0
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error),
                      traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        report.update(finished=time.time(), exitcode=exitcode)
        write_json(report_path, report)
    return exitcode


if __name__ == "__main__":
    sys.exit(main())
