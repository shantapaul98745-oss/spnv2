"""Table6 benchmark contract; imports never query or initialize CUDA."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

PLAN = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
PARAMS = PLAN / "resources/reports/table6_success_20261017"
MODELS = [
    ("table4_phi0_bn", "train_table4_phi0_bn", "table4_phi0_bn", "table4_phi0_bn"),
    ("phi3_bn_ehs", "train_phi3_EHS", "phi3", "phi3"),
    ("table4_phi3_gn", "train_table4_phi3_gn", "table4_phi3_gn", "table4_phi3_gn"),
    ("phi6_gn_ehs", "train_phi6_EHS", "phi6", "phi6"),
]
HEADS = ["heatmap", "efficientpose", "segmentation"]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, obj):
    path = Path(path)
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def emit(event, **fields):
    import time
    print("SPNV2_EVENT " + json.dumps(dict(event=event, phase="resource_benchmark", timestamp=time.time(), **fields),
                                    allow_nan=False, sort_keys=True), flush=True)


def percentile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    lower, upper = math.floor(point), math.ceil(point)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


def latency_stats(values):
    require(isinstance(values, list) and len(values) == 30, "Exactly30 measured latency samples required")
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values),
            "Latency samples must be finite positive milliseconds")
    mean = statistics.mean(values)
    return {"sample_count": 30, "mean_ms": mean, "median_ms": statistics.median(values),
            "p05_ms": percentile(values, .05), "p90_ms": percentile(values, .9),
            "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values),
            "population_std_ms": statistics.pstdev(values), "images_per_second": 1000.0 / mean,
            "percentile_method": "linear interpolation at (n-1)*q"}


def check_measurement(item):
    require(item["input_shape"] == [1, 3, 512, 768] and item["input_kind"] == "fixed_zero_tensor_cuda_resident",
            "Input workload mismatch")
    require(item["warmup_count"] == 10 and item["forward_count"] == 40, "Wrong warmup/forward count")
    require(item["timing_method"] == "CUDA events with synchronize after every forward", "Wrong timer")
    require(item["dtype"] == "torch.float32" and item["eval_mode"] is True and item["no_grad"] is True
            and item["all_EHS_heads"] == HEADS and item["autocast"] is False, "Inference mode/head mismatch")
    require(item["weights_unchanged"] is True and item["strict_reload"] is True
            and item["finite_output_verified"] is True, "Model integrity evidence missing")
    require(item["model_digest_before"] == item["model_digest_after"], "Benchmark mutated source model")
    expected = latency_stats(item["latencies_ms"])
    require(item["statistics"] == expected, "Reported statistics differ from raw30 samples")
    memory = item["memory_bytes"]
    for key in ("baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved"):
        require(type(memory[key]) is int and memory[key] >= 0, "Invalid GPU memory value")
    require(memory["peak_allocated"] >= memory["baseline_allocated"] > 0
            and memory["peak_reserved"] >= memory["baseline_reserved"]
            and memory["peak_reserved"] >= memory["peak_allocated"], "GPU peak memory accounting mismatch")
    require(item["parameter_counts"]["whole_spnv2"] > 0 and item["parameter_counts"]["odr_heads"] == 0,
            "Invalid parameter denominator")
    return expected


def check_source_report(report, config_path, final):
    require(report["status"] == "passed" and report["cfg"] == str(config_path)
            and report["final"] == str(final) and report["final_sha256"] == sha(final),
            "Accepted final/config/hash mismatch")
    require(report["strict_reload"] is True and report["final_matches_checkpoint"] is True,
            "Source has no strict final/checkpoint acceptance")


def sources():
    import yaml
    state = json.loads((PLAN / "status.json").read_text())
    completed = set(state["completed"])
    task_states = {t["id"]: t["status"] for t in state["tasks"]}
    frozen = json.loads((PLAN / "source_manifest.json").read_text())["core_tools_sha256"]
    actual_code = {str(path.relative_to(PLAN / 'source')): sha(path) for folder in ('core', 'tools') for path in sorted((PLAN / 'source' / folder).rglob('*.py'))}
    require(actual_code == frozen, 'Frozen source changed before benchmark')
    provenance = json.loads((PLAN / "resources/manifest_success_20261017.json").read_text())
    acceptance = json.loads((PARAMS / "parameter_acceptance.json").read_text())
    require(acceptance["status"] == "passed_parameters_only_all_four_models", "Four-model parameter acceptance missing")
    require(sha(PARAMS / "parameter_acceptance.json") == provenance["success_reports_sha256"]["parameter_acceptance.json"], "Parameter acceptance provenance changed")
    require(sha(PARAMS / "table6_parameters.json") == acceptance["table_report_sha256"],
            "Parameter table evidence changed")
    entries = []
    for name, train_id, eval_stem, report_stem in MODELS:
        cfg_path = PLAN / "configs" / (name + ".yaml")
        cfg = yaml.safe_load(cfg_path.read_text())
        final = Path(cfg["OUTPUT_DIR"]) / cfg["MODEL"]["BACKBONE"]["NAME"] / cfg["EXP_NAME"] / "model_final.pth.tar"
        proof_path = PLAN / "reports" / ("source_" + report_stem + "_verified.json")
        ids = [train_id] + ["eval_" + eval_stem + "_" + domain for domain in ("synthetic", "lightbox", "sunlamp")]
        require(all(key in completed and task_states.get(key) == "accepted" for key in ids),
                "Source training/three-domain task acceptance missing: " + name)
        proof = json.loads(proof_path.read_text())
        check_source_report(proof, cfg_path, final)
        require(proof["code_sha256"] == frozen and proof["accum_steps"] == 4
                and proof["physical_batch"] == (1 if name == "phi6_gn_ehs" else 4), "Wrong source protocol")
        require(cfg["MODEL"]["HEAD"]["NAMES"] == HEADS == cfg["MODEL"]["HEAD"]["LOSS_HEADS"],
                "Benchmark only supports EHS sources")
        domain_proofs = {}
        for domain, count in (("synthetic", 11994), ("lightbox", 6740), ("sunlamp", 2791)):
            path = PLAN / "reports" / ("eval_" + eval_stem + "_" + domain + "_verified.json")
            value = json.loads(path.read_text())
            require(value["status"] == "passed" and value["source_sha256"] == proof["final_sha256"]
                    and value["summary"]["n_samples"] == value["metrics"]["n_samples"] == count,
                    "Source domain proof/hash/denominator mismatch")
            domain_proofs[domain] = {"path": str(path), "sha256": sha(path), "samples": count}
        inventory_path = PARAMS / (name + "_parameters.json")
        require(sha(inventory_path) == provenance["success_reports_sha256"][inventory_path.name],
                "Verified parameter inventory changed")
        inventory = json.loads(inventory_path.read_text())
        expected = next(x["parameter_counts"] for x in acceptance["checks"] if x["config"] == name)
        require(inventory["parameter_counts"] == expected and inventory["config_sha256"] == sha(cfg_path)
                and inventory["code_sha256"] == frozen, "Parameter denominator/config/source mismatch")
        entries.append({
            "name": name, "cfg": str(cfg_path), "cfg_sha256": sha(cfg_path),
            "source_report": str(proof_path), "source_report_sha256": sha(proof_path),
            "final": str(final), "final_sha256": proof["final_sha256"], "accepted_task_ids": ids,
            "domain_proofs": domain_proofs, "parameter_inventory": str(inventory_path),
            "parameter_inventory_sha256": sha(inventory_path), "parameter_counts": expected,
        })
    return entries


def inspect_parameters(model, inventory):
    import torch
    actual = {name: parameter for name, parameter in model.named_parameters()}
    expected = {item["name"]: item for item in inventory["full_parameter_inventory"]}
    require(set(actual) == set(expected), "Actual parameter names differ from verified inventory")
    for name, p in actual.items():
        item = expected[name]
        require(list(p.shape) == item["shape"] and p.numel() == item["numel"]
                and str(p.dtype) == item["dtype"] and torch.isfinite(p).all(), "Actual parameter schema/finite mismatch")
    counts = inventory["parameter_counts"]
    require(sum(p.numel() for p in actual.values()) == counts["whole_spnv2"], "Whole parameter denominator mismatch")
    require(sum(p.numel() for n, p in actual.items() if n.startswith("backbone.efficientnet."))
            == counts["efficientnet_encoder"], "Encoder parameter denominator mismatch")
    require(sum(p.numel() for n, p in actual.items() if n.startswith("backbone."))
            == counts["efficientdet_backbone"], "Backbone parameter denominator mismatch")
    require(sum(actual[name].numel() for name in inventory["odr_allowed_names"])
            == counts["odr_updated_backbone_norm_affine"], "Static ODR-affine parameter denominator mismatch")
    return counts


def cached_only(efficientnet, torch):
    def load(url, *args, **kwargs):
        path = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
        require(path.is_file(), "Required official initialization cache missing; network forbidden")
        return torch.load(path, map_location="cpu", weights_only=True)
    efficientnet.load_state_dict_from_url = load


def code_assets():
    files = [path for folder in ("core", "tools") for path in (PLAN / "source" / folder).rglob("*") if path.is_file()]
    files += list(BENCH.glob("*.py"))
    files += [PLAN / "pipeline/runner.py", PLAN / "source_manifest.json",
              PLAN / "resources/manifest_success_20261017.json", PARAMS / "parameter_acceptance.json",
              PARAMS / "table6_parameters.json", PLAN / "odr/manifest.json", PLAN / "odr/implementation_manifest.json"]
    files += [PLAN / "configs" / (name + ".yaml") for name, *_ in MODELS]
    files += [PARAMS / (name + "_parameters.json") for name, *_ in MODELS]
    return {str(path): sha(path) for path in sorted(files)}
