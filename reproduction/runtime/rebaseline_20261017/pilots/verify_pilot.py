"""Independent CPU evidence verifier; never construct a network or initialize CUDA."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

from evidence import (ARTIFACT_NAMES, exact_tree, model_sha256, parameter_changes,
                      protocol, sha256_file, tree_sha256, write_json)

PLAN = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_order(size, seed, shuffle):
    import torch
    if not shuffle:
        return list(range(size))
    # Reimplement the persisted epoch-keyed-v1 protocol from its documented key.
    key = ("spnv2-v1:" + str(seed) + ":0:0:order").encode()
    order_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
    return torch.randperm(size, generator=torch.Generator().manual_seed(order_seed)).tolist()


def validate_counters(state, attempts, physical, accumulation, *, successful_required=True):
    require(isinstance(state, dict), "Missing training counters")
    for field in ("epoch", "next_batch", "update_attempts", "optimizer_updates", "amp_skips",
                  "consecutive_amp_skips", "samples_seen", "microbatches_seen"):
        require(type(state.get(field)) is int and state[field] >= 0, "Invalid counter " + field)
    require(state["epoch"] == 0, "Pilot must be a single engineering subset epoch")
    require(state["update_attempts"] == attempts, "Update attempt count mismatch")
    require(state["next_batch"] == attempts * accumulation, "Not a complete update boundary")
    require(state["microbatches_seen"] == attempts * accumulation, "Microbatch count mismatch")
    require(state["samples_seen"] == attempts * accumulation * physical, "Actual sample count mismatch")
    require(state["optimizer_updates"] + state["amp_skips"] == attempts, "Successful/skipped update accounting mismatch")
    require(state["consecutive_amp_skips"] <= state["amp_skips"], "Consecutive skip accounting mismatch")
    if successful_required:
        require(state["optimizer_updates"] >= 3, "Fewer than three successful updates")
        require(isinstance(state.get("last_loss"), (int, float)), "Missing actual final loss")
    tree_sha256(state)


def validate_checkpoint(payload, manifest, attempts, physical, accumulation, count, order):
    import torch
    require(payload.get("format") == "spnv2-boundary-v1", "Invalid checkpoint format")
    require(set(payload) == {"format", "state_dict", "optimizer", "scaler", "training_state",
                             "rng", "sampler", "manifest"}, "Incomplete checkpoint evidence")
    exact_tree(payload["manifest"], manifest, "checkpoint.manifest")
    validate_counters(payload["training_state"], attempts, physical, accumulation,
                      successful_required=(attempts == 32))
    sampler = payload["sampler"]
    expected_order_hash = hashlib.sha256(
        __import__("numpy").asarray(order, dtype="<i8").tobytes()).hexdigest()
    expected_sampler = {
        "version": 1, "size": count, "batch_size": physical, "accum_steps": accumulation,
        "seed": 2021, "shuffle": bool(__import__("yaml").safe_load(manifest["config"])["TRAIN"]["SHUFFLE"]),
        "epoch": 0, "next_batch": attempts * accumulation,
        "total_batches": count // physical, "order_sha256": expected_order_hash,
    }
    exact_tree(sampler, expected_sampler, "checkpoint.sampler")
    require(isinstance(payload["state_dict"], dict) and payload["state_dict"], "Missing model state")
    optimizer = payload["optimizer"]
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}, "Missing optimizer state")
    require(isinstance(optimizer["param_groups"], list) and len(optimizer["param_groups"]) == 1,
            "Unexpected optimizer parameter grouping")
    group = optimizer["param_groups"][0]
    require(group["lr"] == .001 and group["weight_decay"] == .0001
            and tuple(group["betas"]) == (.9, .999) and group["eps"] == .0001,
            "Actual optimizer protocol mismatch")
    require(isinstance(payload["scaler"], dict)
            and {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"} <= set(payload["scaler"]),
            "Missing actual AMP scaler")
    require(payload["scaler"]["scale"] > 0, "Invalid AMP scale")
    rng = payload["rng"]
    require(isinstance(rng, dict) and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"},
            "Incomplete RNG state")
    require(isinstance(rng["torch_cpu"], torch.Tensor) and rng["torch_cpu"].dtype == torch.uint8,
            "Missing CPU RNG bytes")
    require(isinstance(rng["torch_cuda"], list) and len(rng["torch_cuda"]) == 1
            and isinstance(rng["torch_cuda"][0], torch.Tensor) and rng["torch_cuda"][0].dtype == torch.uint8,
            "Pilot requires one saved CUDA RNG state")
    if attempts:
        require(optimizer["state"], "Actual successful updates have no optimizer state")
    require(len(group["params"]) == len(set(group["params"])), "Duplicate optimizer parameter IDs")
    require(set(optimizer["state"]) <= set(group["params"]), "Optimizer state references unknown parameters")
    for item in optimizer["state"].values():
        require(set(item) == {"step", "exp_avg", "exp_avg_sq"}, "Incomplete AdamW state")
        require(float(item["step"]) == payload["training_state"]["optimizer_updates"],
                "AdamW actual step counter differs from successful updates")
        require(item["exp_avg"].shape == item["exp_avg_sq"].shape, "AdamW moment shapes differ")
    return {key: tree_sha256(payload[key]) for key in
            ("state_dict", "optimizer", "scaler", "training_state", "rng", "sampler", "manifest")}


def verify(report_path):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU verifier requires CUDA_VISIBLE_DEVICES=''")
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "CPU verifier requires PYTHONNOUSERSITE=1")
    import torch
    import yaml
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    require(report.get("schema") == "spnv2-engineering-pilot-v1"
            and report.get("status") == "passed" and report.get("exitcode") == 0,
            "Pilot is not a successful actual GPU report")
    source = Path(report["source"]).resolve()
    require(source == PLAN / "source", "Use this plan's frozen source")
    cfg_path = Path(report["cfg"]).resolve()
    require(PLAN / "pilots" in cfg_path.parents, "Config must be pilot-owned")
    require(sha256_file(cfg_path) == report["cfg_file_sha256"], "Config file hash mismatch")
    require(sha256_file(Path(__file__).with_name("gpu_smoke.py")) == report["test_sha256"],
            "GPU pilot implementation hash mismatch")
    require(sha256_file(Path(__file__).with_name("evidence.py")) == report["evidence_helper_sha256"],
            "Evidence helper implementation hash mismatch")
    require(hashlib.sha256(report["effective_config"].encode()).hexdigest()
            == report["effective_config_sha256"], "Effective config hash mismatch")
    cfg = yaml.safe_load(report["effective_config"])
    sys.path.insert(0, str(source / "core"))
    from config import cfg as defaults, update_config
    independently_loaded = defaults.clone()
    update_config(independently_loaded, argparse.Namespace(cfg=str(cfg_path), opts=[]))
    exact_tree(yaml.safe_load(independently_loaded.dump()), cfg, "effective_config_from_actual_file")
    checks = protocol(cfg, report["attempts"], report["checkpoint_attempt"])
    exact_tree(report["protocol_checks"], checks, "protocol_checks")
    require(Path(cfg["ROOT"]).resolve() == source, "Effective source ROOT mismatch")
    require(report["fresh_initialization_equal"] is True and report["pilot_weights_for_formal"] is False,
            "Fresh initialization/no inheritance evidence missing")
    physical, accumulation = cfg["TRAIN"]["IMAGES_PER_GPU"], cfg["TRAIN"]["ACCUM_STEPS"]
    count, boundary_count = physical * accumulation * 32, physical * accumulation * 16
    require(report["physical_batch"] == physical and report["accumulation"] == accumulation
            and report["subset_rows"] == count and report["seed"] == 2021, "Reported protocol counters differ")
    require(report["heads"] == cfg["MODEL"]["HEAD"]["NAMES"], "Head list mismatch")
    require(report["clip_norm"] == 1.0 and report["loss_normalization"] == "local_mean_divided_by_accumulation",
            "Gradient algorithm evidence mismatch")
    require(report["observed_training_shapes"] == [[physical, 3, 512, 768]],
            "Actual input shape evidence mismatch")
    require(report["observed_forward_count"] == 48 * accumulation, "Actual forward count mismatch")
    require(report["runtime"]["torch_home"] == str(PLAN.parents[1] / "env/cache/torch"),
            "Initialization cache is not fixed")
    require(report["runtime"].get("screen"), "Missing original screen evidence")
    disclosure = report["initialization_disclosure"]
    if cfg["MODEL"]["USE_GROUPNORM_BACKBONE"]:
        require("random initialization" in disclosure and "author initialization unverified" in disclosure,
                "GN initialization limitation missing")
    else:
        require("local cache only" in disclosure, "BN cache-only initialization missing")
    paths = {}
    require(set(report["artifacts"]) == set(ARTIFACT_NAMES), "Required artifact set mismatch")
    artifacts = Path(report["artifacts_directory"]).resolve()
    require(PLAN / "pilots" in artifacts.parents, "Artifact directory must be pilot-owned")
    for name in ARTIFACT_NAMES:
        item = report["artifacts"][name]
        path = Path(item["path"]).resolve()
        require(path == artifacts / name and path.is_file(), "Unexpected artifact path: " + name)
        require(path.stat().st_size == item["size"] and sha256_file(path) == item["sha256"],
                "Artifact size/hash mismatch: " + name)
        paths[name] = path
    manifest = json.loads(paths["manifest.json"].read_text())
    exact_tree(report["manifest"], manifest, "report.manifest")
    require(manifest["config"] == report["effective_config"]
            and manifest["config_sha256"] == report["effective_config_sha256"], "Manifest config mismatch")
    require(manifest["sampler"] == "epoch-keyed-v1" and manifest["augmentation"] == "sample-keyed-v1",
            "Dataset reproducibility protocol mismatch")
    code = {str(p.relative_to(source)): sha256_file(p)
            for folder in ("core", "tools") for p in sorted((source / folder).rglob("*.py"))}
    require(code == manifest["code"] == report["code_sha256"], "Frozen source hash mismatch")
    # Check the source whose hash was just verified actually implements local/K and clip1.
    trainer = (source / "core/engine/trainer.py").read_text()
    require("loss / accumulation" in trainer and "clip_grad_norm_(model.parameters(), 1.0)" in trainer,
            "Frozen trainer accumulation/clipping protocol missing")
    require(manifest["csv"], "No original CSV evidence")
    for name, digest in manifest["csv"].items():
        require(sha256_file(name) == digest, "Original CSV hash mismatch")
    csv_path = Path(cfg["DATASET"]["ROOT"]) / cfg["DATASET"]["DATANAME"] / cfg["TRAIN"]["TRAIN_CSV"]
    require(str(csv_path) in manifest["csv"], "Training CSV is not bound")
    with csv_path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    require(len(rows) == report["original_training_rows"] and len(rows) >= count, "Original training row count mismatch")
    order = independent_order(count, 2021, cfg["TRAIN"]["SHUFFLE"])
    ids = [str(rows[i][0]) for i in order]
    require(report["expected_sample_ids"] == ids == report["continuous"]["sample_ids"],
            "Actual continuous IDs differ from independent CSV/sampler order")
    require(report["resume"]["sample_ids"] == ids[boundary_count:], "Resumed actual ID suffix mismatch")
    require(all(report["resume"].get(k) is True for k in
                ("exact", "model_equal", "optimizer_equal", "scaler_equal", "counters_equal",
                 "rng_equal", "sample_ids_equal")), "Resume evidence incomplete")
    require(report["resume"]["checkpoint_attempt"] == 16, "Resume boundary mismatch")
    inventory = manifest.get("data_inventory")
    require(inventory and inventory["raw_content_hashed"] is False
            and inventory["hash_kind"] == "sha256-of-path-size-mtime_ns-records", "Data inventory missing")
    encoded = json.dumps(inventory["records"], ensure_ascii=False, separators=(",", ":")).encode()
    require(hashlib.sha256(encoded).hexdigest() == inventory["sha256"], "Data inventory digest mismatch")
    require(inventory["coverage"]["train"]["samples"] == count, "Training subset inventory mismatch")
    for split, row, path, size, mtime in inventory["records"]:
        stat = Path(path).stat()
        require(stat.st_size == size and stat.st_mtime_ns == mtime, "Referenced input asset changed")
    expected_records, expected_coverage = [], {}
    for split, csv_field in (("train", "TRAIN_CSV"), ("val", "VAL_CSV")):
        data_csv = Path(cfg["DATASET"]["ROOT"]) / cfg["DATASET"]["DATANAME"] / cfg["TRAIN"][csv_field]
        require(str(data_csv) in manifest["csv"], "Input split CSV is not bound")
        with data_csv.open(newline="") as stream:
            split_rows = list(csv.reader(stream))
        if split == "train":
            split_rows = split_rows[:count]
        folders = ["images_768x512_RGB"]
        active_heads = report["heads"] if split == "train" else cfg["TEST"]["HEAD"]
        if "segmentation" in active_heads:
            folders.append("masks_192x128")
        if split == "train" and cfg["AUGMENT"]["APPLY_TEXTURE_RANDOMIZATION"]:
            folders.append("styles_768x512_RGB")
        domain = cfg["TRAIN"][csv_field].split("/")[0]
        for row, values in enumerate(split_rows):
            for folder in folders:
                asset = Path(cfg["DATASET"]["ROOT"]) / cfg["DATASET"]["DATANAME"] / domain / folder / values[0]
                stat = asset.stat()
                expected_records.append([split, row, str(asset.absolute()), stat.st_size, stat.st_mtime_ns])
        expected_coverage[split] = {"samples": len(split_rows), "folders": folders,
                                    "files": len(split_rows) * len(folders)}
    exact_tree(inventory["records"], expected_records, "complete_input_inventory")
    exact_tree(inventory["coverage"], expected_coverage, "complete_input_coverage")
    geometry = {}
    for field in ("CAMERA", "KEYPOINTS", "CADMODEL"):
        asset = Path(cfg["DATASET"][field])
        geometry[field] = {"path": str(asset.absolute()), "exists": asset.is_file(),
                           "sha256": sha256_file(asset) if asset.is_file() else None}
    exact_tree(manifest["auxiliary_geometry"], geometry, "auxiliary_geometry")
    if cfg["MODEL"]["USE_GROUPNORM_BACKBONE"]:
        require(not manifest["initialization"], "GN random initialization unexpectedly binds pretrained weights")
    else:
        init = manifest["initialization"]["imagenet_backbone"]
        require(sha256_file(init["path"]) == init["sha256"], "Cached official BN initialization changed")
        require(Path(init["path"]).resolve().parent == Path(report["runtime"]["torch_home"]) / "hub/checkpoints",
                "BN weights outside fixed torch cache")
    checkpoints = {}
    initial_model = torch.load(paths["model_initial.pth.tar"], map_location="cpu", weights_only=True)
    require(model_sha256(initial_model) == manifest["initial_state_sha256"], "Initial state digest mismatch")
    for name, attempts in (("initial_checkpoint.pth.tar", 0), ("boundary_checkpoint.pth.tar", 16),
                           ("continuous_final_checkpoint.pth.tar", 32),
                           ("resumed_final_checkpoint.pth.tar", 32)):
        payload = torch.load(paths[name], map_location="cpu", weights_only=True)
        fingerprints = validate_checkpoint(payload, manifest, attempts, physical, accumulation, count, order)
        checkpoints[name] = fingerprints
        if attempts == 0:
            exact_tree(initial_model, payload["state_dict"], "initial_snapshot")
        if name == "continuous_final_checkpoint.pth.tar":
            require(len(payload["optimizer"]["param_groups"][0]["params"]) == len(report["trainable_parameter_names"]),
                    "Trainable parameter evidence differs from optimizer grouping")
            final_model = payload["state_dict"]
            changes = parameter_changes(initial_model, final_model,
                                        report["trainable_parameter_names"], report["heads"])
            exact_tree(changes, report["component_parameter_changes"], "actual_parameter_changes")
            exact_tree(payload["training_state"], report["continuous"]["state"], "continuous.counters")
        if name == "resumed_final_checkpoint.pth.tar":
            exact_tree(payload["training_state"], report["resume"]["state"], "resume.counters")
        del payload
        gc.collect()
    require(checkpoints["continuous_final_checkpoint.pth.tar"] == checkpoints["resumed_final_checkpoint.pth.tar"],
            "Full checkpoint state differs after resume")
    del initial_model, final_model
    gc.collect()
    for name in ("model_final.pth.tar", "model_resumed_final.pth.tar"):
        snapshot = torch.load(paths[name], map_location="cpu", weights_only=True)
        require(tree_sha256(snapshot) == checkpoints["continuous_final_checkpoint.pth.tar"]["state_dict"],
                "Final model snapshot differs from full checkpoint")
        del snapshot
    return {
        "status": "passed", "schema": "spnv2-pilot-cpu-verification-v1",
        "report": str(report_path), "report_sha256": sha256_file(report_path),
        "verifier_sha256": sha256_file(__file__), "tests_real_gpu_claim": False,
        "verified": ["artifact_hashes", "frozen_source", "configuration", "input_shapes",
                     "csv_and_inventory", "actual_sample_id_stream", "initial_model",
                     "every_active_head_and_backbone_changed", "all_complete_boundary_counters",
                     "full_model_optimizer_scaler_rng_sampler_and_manifest_resume_equality"],
        "checkpoint_tree_sha256": checkpoints, "component_parameter_changes": changes,
        "scope": "32 update attempts on engineering subset; no formal accuracy result",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(PLAN / "pilots" in output.parents, "Verifier output must be pilot-owned")
    require(not output.exists(), "Verifier output already exists")
    try:
        value = verify(args.report)
    except Exception as error:
        write_json(output, {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                            "report": str(Path(args.report).resolve()),
                            "verifier_sha256": sha256_file(__file__)})
        raise
    write_json(output, value)
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
