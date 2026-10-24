"""Independent acceptance of one saved N1024 round-two run, without a forward.

The entry is invoked under guarded_entry's shared-host admission.  It reads only
the frozen developer labels, reconstructs metrics from saved poses, replays the
scientific transaction trace, and checks the latest complete durable generation.
A main-completed report is deliberately not an independently accepted result.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from artifact_binding import require_same_artifact
import os
import sys

from frozen_backend import sha256, canonical_sha, import_bound_module
from guarded_entry import PLAN
from round2_protocol import frozen_protocol, build_complete_matrix, build_baseline_matrix, derive_controls
from deterministic_runtime import validate_saved_receipt
from verify_engineering_phase import (_binding, _code_bindings, _attempt_evidence,
                                      _attempt_costs, _new_or_identical, require, load_json)


METRIC_FIELDS = ("valid", "error", "original_valid", "original_error", "rotation_deg",
                 "translation_m", "heat_rotation_HIL", "effi_translation_HIL", "heat_error", "effi_error")
PROOF_SCHEMA = "odr003-round2-complete-run-proof-v2"


class DeveloperOnlyReads:
    """Enforce the one developer-label file; forbid full/reserve target GT."""
    def __init__(self, allowed):
        self.allowed = Path(allowed).resolve(strict=True)
        self.reads = 0
        self.denied = []
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        data_root = Path("/home/DATA/Downloads/DataSet/speedplus")
        forbidden = (path.name == "test.json" or
                     (path.is_relative_to(data_root) and ("labels" in path.parts or path.suffix.lower() == ".csv")))
        if forbidden or (path.name == "dev_labels.json" and path.resolve() != self.allowed):
            self.denied.append(str(path))
            raise PermissionError("complete verifier forbids full/reserved/other developer GT")
        if path.name == "dev_labels.json": self.reads += 1


def _finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite saved JSON value")
    elif isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _finite(item)


def _close(actual, expected, label):
    if expected is None or type(expected) is bool:
        require(type(actual) is type(expected) and actual == expected, label + " differs")
    else:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label + " differs")


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        require(bool(line.strip()), "blank saved prediction/risk row")
        row = json.loads(line)
        require(isinstance(row, dict), "saved row is not a mapping")
        _finite(row)
        rows.append(row)
    return rows


def checked_developer_rows(document, labels, developer_ids, *, recompute, cfg, flat=False):
    """Recalculate all declared metrics, retaining unavailable predictions as None."""
    rows = document if flat else document["rows"]
    require(len(developer_ids) == len(set(developer_ids)) == 512
            and len(rows) == len(labels) == 512, "developer coverage changed")
    require([row["sample_id"] for row in rows] == developer_ids
            and [row["sample_id"] for row in labels] == developer_ids, "developer IDs/order differ")
    if not flat:
        require(document.get("next_index") == 512, "evaluation cursor is not complete")
    result = []
    import numpy as np
    for row, label in zip(rows, labels):
        prediction = row if flat else row["prediction"]
        actual = row if flat else row["metrics"]
        _finite(prediction)
        require(type(prediction.get("h_available")) is bool
                and type(prediction.get("e_available")) is bool
                and type(prediction.get("original_h_rejected")) is bool,
                "pose availability/rejection must be explicit")
        for field, shape in (("heat_q", (4,)), ("heat_t", (3,)), ("effi_R", (3,3)), ("effi_t", (3,))):
            available = prediction["h_available" if field.startswith("heat_") else "e_available"]
            value = prediction.get(field)
            if available:
                array = np.asarray(value)
                require(array.shape == shape and bool(np.isfinite(array).all()), "unavailable/malformed declared pose: "+field)
            else:
                require(value is None, "unavailable pose must remain None: "+field)
        if prediction["h_available"]:
            require(abs(float(np.linalg.norm(prediction["heat_q"]))-1) < 1e-6, "saved H quaternion is not unit")
        else:
            require(prediction["original_h_rejected"] is True, "missing H cannot be original accepted")
        if prediction["e_available"]:
            rotation = np.asarray(prediction["effi_R"], dtype=np.float32)
            require(abs(float(np.linalg.det(rotation))-1) < 1e-6
                    and float(np.max(np.abs(rotation@rotation.T-np.eye(3)))) <= 1e-6, "saved E rotation validity differs")
        metric = recompute(prediction, dict(quaternion=label["quaternion_wxyz"],
                                           translation=label["translation_m"]), cfg)
        require(set(metric) == set(METRIC_FIELDS), "frozen developer metric schema differs")
        for field in METRIC_FIELDS:
            _close(actual.get(field), metric[field], row["sample_id"] + "." + field)
        _finite(metric)
        require(metric["valid"] == (metric["error"] is not None)
                and metric["original_valid"] == (metric["original_error"] is not None),
                "metric availability/count differs")
        result.append(dict(sample_id=row["sample_id"], **metric,
                           original_rejected=prediction["original_h_rejected"]))
    return result


def evaluate_fixed_reference(developer_ids, offline, original_a, methods, *, expected_reference_count,
                             fixed_reference_ids):
    """Same fixed O/A denominator and ceil(5%) tail as the first exploration.

    Includes zero degradations.  Any new missing reference prediction makes all
    fixed-reference means unavailable; no method may shrink its own denominator.
    """
    require(len(developer_ids) == len(set(developer_ids)) == 512, "expected512 developer IDs")
    indexed = {}
    for name, rows in {"O": offline, "A": original_a, **methods}.items():
        require([row["sample_id"] for row in rows] == developer_ids, name + " IDs/order differ")
        for row in rows:
            require(type(row["valid"]) is bool and row["valid"] == (row["error"] is not None), "invalid explicit missingness")
            _finite(row)
            for field in ("error", "rotation_deg", "translation_m"):
                value = row.get(field)
                require(value is None or (type(value) in (int, float) and value >= 0), "negative/non-numeric metric")
        indexed[name] = {row["sample_id"]: row for row in rows}
    o, a = indexed["O"], indexed["A"]
    reference = list(fixed_reference_ids)
    require(len(reference) == len(set(reference)) == expected_reference_count
            and [sid for sid in developer_ids if sid in set(reference)] == reference,
            "frozen reference IDs/order/count changed")
    reference_set = set(reference)
    tail_count = math.ceil(.05 * len(reference))
    def mean(values):
        return None if not values or any(value is None for value in values) else math.fsum(value / len(values) for value in values)
    output = {}
    for name, rows in indexed.items():
        missing = [sid for sid in reference if not rows[sid]["valid"] or not o[sid]["valid"]]
        complete = not missing
        details, degradations = [], []
        for sid in developer_ids:
            row = rows[sid]
            d = max(row["error"] - o[sid]["error"], 0.0) if sid in reference_set and row["valid"] and o[sid]["valid"] else None
            if d is not None: degradations.append(d)
            details.append(dict(row, in_reference=sid in reference_set,
                           offline_error=o[sid]["error"], a_error=a[sid]["error"],
                           positive_degradation=d,
                           delta_vs_A=None if not (row["valid"] and a[sid]["valid"]) else row["error"]-a[sid]["error"],
                           valid_to_invalid_vs_A=a[sid]["valid"] and not row["valid"],
                           invalid_to_valid_vs_A=not a[sid]["valid"] and row["valid"]))
        metrics = dict(mean_error=None, mean_D=None, tail_D=None,
                       mean_rotation_deg=None, mean_translation_m=None, negative_transfer_rate=None)
        if complete:
            metrics.update(mean_error=mean([rows[sid]["error"] for sid in reference]),
                           mean_D=mean(degradations), tail_D=mean(sorted(degradations, reverse=True)[:tail_count]),
                           mean_rotation_deg=mean([rows[sid].get("rotation_deg") for sid in reference]),
                           mean_translation_m=mean([rows[sid].get("translation_m") for sid in reference]),
                           negative_transfer_rate=sum(value > 0 for value in degradations)/len(reference))
        coverage = sum(row["valid"] for row in rows.values())
        a_coverage = sum(row["valid"] for row in a.values())
        output[name] = dict(metrics=metrics, coverage=coverage, A_coverage=a_coverage,
                            reference_missing=len(missing), missing_reference_ids=missing,
                            coverage_pass=complete and coverage >= a_coverage,
                            status="complete" if complete else "missing_reference_predictions",
                            per_sample=details)
    return dict(reference_ids=reference, reference_count=len(reference), developer_count=512,
                tail_count=tail_count, tail_includes_zeros=True, methods=output,
                metric_units=dict(error="H rotation HIL + E translation HIL", mean_D="mixed score",
                                  tail_D="mixed score", mean_rotation_deg="degrees", mean_translation_m="metres"))


def derive_verified_controls(report, proof, *, report_binding):
    require(proof.get("status") == "passed" and proof.get("report_sha256") == report_binding["sha256"],
            "paired P lacks independent report acceptance")
    definition = report["definition"]
    require(definition["group"] == "P" and report["stage"] == "complete"
            and report["source_sha256"] == frozen_protocol()["source_sha256"]
            and report["protocol_sha256"] == frozen_protocol()["protocol_sha256"], "paired P identity differs")
    require(proof.get("counts") == report["counts"], "paired P independent counts differ")
    return derive_controls(definition["domain"], definition["stream_seed"],
                           report["counts"]["candidates"], report["counts"]["commits"])


def verify_original_baseline_state(state, *, expected_stream, identity, method, calibration, gate_decider):
    """Replay original A/G50 decisions/counts; never use protected update logic."""
    from verify_round2 import _learning_contract, _same, _json_hash
    require(method in ("A", "G50") and len(expected_stream) == 1024, "Only the full fixed original baselines are admissible")
    require(state.get("schema") == "odr003-round2-original-g50-reference-v1"
            and state.get("finished") is True and state.get("next_index") == 1024,
            "Original baseline lacks its complete N1024 checkpoint")
    _same(state["identity"], identity, "baseline_identity")
    _same(state["stream_records"], expected_stream, "baseline_stream")
    _learning_contract(state["learning"])
    require(all(state["learning"][key] == {} for key in ("bn_mean", "bn_mean_sq")),
            "Original baseline has unresolved statistic caches")
    trace = state["observation_trace"]
    require(len(trace) == 1024, "Original baseline did not observe all1024 inputs")
    accepted = 0
    for index, (event, record) in enumerate(zip(trace, expected_stream)):
        require(event["id"] == f"observation:{index:04d}" and event["kind"] == "observation"
                and event["observation_index"] == index and event["sample_id"] == record["sample_id"],
                "Original baseline observation identity/order differs")
        expected = (True, "injected_gate", None) if method == "A" else gate_decider("G50", event["arrival"], calibration)
        _same((event["accepted"], event["gate_reason"], event["gate_score"]), expected, "original_baseline_gate")
        accepted += int(expected[0])
    counts = dict(observed=1024, accepted=accepted, candidates=accepted//4, commits=accepted//4,
                  rollbacks=0, tail_discarded=accepted%4, consecutive_rollbacks=0, max_consecutive_rollbacks=0)
    _same(state["learning"]["counters"], dict(observed=1024, accepted=accepted, updates=accepted//4, tail=accepted%4),
          "original_adapter_counts")
    _same(state["finish_counts"], dict(observed=1024, accepted=accepted, rejected=1024-accepted,
                                     updates=accepted//4, used=4*(accepted//4), tail=accepted%4), "original_finish_counts")
    optimizer_states = state["learning"]["optimizer"]["state"]
    updates = accepted//4
    require(bool(optimizer_states) == bool(updates), "Original optimizer state contradicts successful update count")
    for value in optimizer_states.values():
        step = value.get("step")
        if hasattr(step, "item"): step = step.item()
        require(type(step) in (int, float) and step == updates, "Original optimizer step count differs from committed windows")
    return dict(passed=True, kind="independent_original_"+method, counters=counts,
                observation_trace_sha256=_json_hash(trace), no_new_model_forward=True)


def validate_offline_checkpoint(state):
    require(state.get("schema") == "odr003-round2-offline-source-v1"
            and state.get("finished_source_checkpoint") is True and state.get("next_index") == 0,
            "Offline source checkpoint is incomplete or adapted")
    expected = dict(observed=0, accepted=0, candidates=0, commits=0, rollbacks=0, tail_discarded=0,
                    consecutive_rollbacks=0, max_consecutive_rollbacks=0)
    require(state.get("counters") == expected and all(type(v) is int for v in state["counters"].values()),
            "Offline source cannot contain observation/update counts")
    from verify_round2 import _learning_contract, _same
    _learning_contract(state["learning"])
    learning = state["learning"]
    _same(learning["counters"], dict(observed=0, accepted=0, updates=0, tail=0), "offline_learning_counts")
    require(learning["optimizer"]["state"] == {} and
            all(learning[k] == {} for k in ("bn_features", "bn_mean", "bn_mean_sq")),
            "Offline source contains optimizer steps or a pending adaptation window")
    _same(learning["buffers"], learning["boundary_buffers"], "offline_buffer_boundary")


def verify_offline_source_state(state, *, identity, source_binding, official_receipt, cache):
    validate_offline_checkpoint(state)
    from verify_round2 import _same
    _same(state["identity"], identity, "offline_identity")
    source_path = _binding(source_binding, cache)
    receipt = load_json(_binding(official_receipt, cache))
    require(receipt.get("status") == "passed" and receipt.get("strict_model_load_performed") is True
            and receipt.get("source_offline_identity_verified") is True
            and receipt.get("source_sha256") == source_binding["sha256"] == frozen_protocol()["source_sha256"],
            "Offline source lacks its accepted official identity")
    import torch
    raw = torch.load(source_path, map_location="cpu", weights_only=True)
    actual = raw if receipt.get("state_dict_key") is None else raw[receipt["state_dict_key"]]
    _same(state["learning"]["model"], actual, "offline_actual_source_tensors")
    return dict(passed=True, kind="independent_offline_unadapted_source", counters=state["counters"],
                all_model_tensors_equal_official_source=True, no_new_model_forward=True)


def checked_runtime_receipt(report, output, *, cache):
    saved = validate_saved_receipt(report.get("deterministic_runtime"))
    binding = report.get("deterministic_runtime_binding")
    require(isinstance(binding, dict), "Main report lacks actual persisted deterministic runtime receipt")
    path = _binding(binding, cache)
    require(path == Path(output)/"deterministic_runtime.json" and load_json(path) == saved,
            "Main backend receipt differs from persisted runtime policy/versions")
    return saved


def _real_receipt(receipt, task, task_path, cache):
    output = Path(task["output_dir"]).resolve(strict=True)
    require(output.is_relative_to(PLAN/"runs") and output.name == task["run_id"], "run output escapes plan")
    report_path = _binding(dict(path=receipt["report_path"], sha256=receipt["report_sha256"]), cache)
    require(report_path == output/"report.json" and receipt["run_id"] == task["run_id"], "receipt/run differs")
    log = Path(receipt["log_path"]).resolve(strict=True)
    exit_path = Path(receipt["exit_code_path"]).resolve(strict=True)
    require(log.is_file() and exit_path.read_text(encoding="ascii").strip() == "0", "actual main exit is not0")
    launch_path = Path(receipt["launch_record_path"]).resolve(strict=True)
    launch = load_json(launch_path)
    require(canonical_sha(launch) == receipt["launch_record_sha256"], "launch record changed")
    require(launch["run_id"] == task["run_id"] and launch["project_id"] == "odr003_round2_20261023"
            and launch["role"] == "worker" and len(launch["run_nonce"]) >= 16
            and launch["pid"] > 0 and launch["starttime"] > 0 and launch["boot_id"], "launch identity incomplete")
    argv = launch["argv"]
    require("--task-json" in argv and argv[argv.index("--task-json")+1] == str(task_path)
            and Path(launch["log_path"]).resolve() == log, "actual process/task/log differs")
    outcome_path = Path(receipt["outcome_path"]).resolve(strict=True)
    require(outcome_path.parent == launch_path.parent, "outcome belongs to a different attempt")
    outcome = load_json(outcome_path)
    require(outcome.get("status") == "main_complete" and outcome.get("exit_code") == 0
            and Path(outcome["report"]).resolve() == report_path, "actual outcome is incomplete")
    report = load_json(report_path)
    require(report.get("schema") == "odr003-round2-run-report-v1"
            and report.get("status") == "main_completed_pending_independent_verification"
            and report["run_id"] == task["run_id"] and report["stage"] == task["stage"]
            and report["definition"] == task["definition"] and report["task_sha256"] == sha256(task_path)
            and report["implementation_sha256"] == task["implementation_sha256"]
            and report["protocol_sha256"] == task["protocol"]["protocol_sha256"]
            and report["source_sha256"] == task["protocol"]["source_sha256"]
            and report["stream_binding"] == task["baseline_task"]["bindings"]["stream_manifest"]
            and report["launch_record_sha256"] == canonical_sha(launch), "main report binding differs")
    require(report["strict_source_load"] is True and report["strict_final_reload"] is True
            and report["gt_denied_attempts"] == [] and not (output/"failure.json").exists(), "load/GT/failure evidence not accepted")
    require(load_json(output/"task.json") == task, "installed science task differs")
    return output, report


def _assets(run_task, cache):
    """Verify the frozen first-round implementation and manifests; no image reads."""
    b = run_task["baseline_task"]["bindings"]
    allowed = {"calibration", "calibration_proof", "camera", "config", "data_prepared", "data_verified",
               "developer_manifest", "developer_labels", "keypoints", "official_receipt", "source",
               "source_core_manifest", "stream_manifest"}
    require(set(b) == allowed, "unexpected asset/GT binding")
    paths = {name: _binding(b[name], cache) for name in sorted(allowed)}
    baseline = import_bound_module("run_exploration", run_task["frozen_modules"]["run_exploration"])
    frozen_paths, receipt, core, prepared = baseline.verify_bindings(run_task["baseline_task"])
    require({name: Path(path).resolve() for name, path in frozen_paths.items()} == paths, "asset path resolution differs")
    cfg, _ = baseline.configure(run_task["baseline_task"], paths, core)
    definition = run_task["definition"]
    domain, seed = definition["domain"], definition["stream_seed"]
    for name, artifact in (("stream_manifest", f"{domain}/stream_{seed}"),
                           ("developer_manifest", domain+"/developer"),
                           ("developer_labels", domain+"/developer_labels")):
        require_same_artifact(b[name], prepared["artifacts"][artifact], "frozen split/stream binding differs: "+artifact)
    stream, dev, labels = (load_json(paths[name]) for name in ("stream_manifest", "developer_manifest", "developer_labels"))
    require(stream["seed"] == seed and len(stream["items"]) == len({row["sample_id"] for row in stream["items"]}) == 1024,
            "observation stream identity/size differs")
    ids = [row["sample_id"] for row in dev["items"]]
    require(labels["domain"] == domain and labels["image_manifest_sha256"] == b["developer_manifest"]["sha256"], "developer label boundary differs")
    bundle, calibration_proof = load_json(paths["calibration"]), load_json(paths["calibration_proof"])
    require(calibration_proof["status"] == "passed" and calibration_proof["recomputed"] is True
            and calibration_proof["bundle_sha256"] == b["calibration"]["sha256"]
            and bundle["status"] == "passed" and bundle["source_sha256"] == b["source"]["sha256"], "calibration proof differs")
    preprocess = canonical_sha(dict(config=b["config"], core=b["source_core_manifest"], data_code=run_task["frozen_modules"]["data_boundary"]))
    records = [dict(sample_id=row["sample_id"], image_sha256=row["image_sha256"], preprocess_sha256=preprocess) for row in stream["items"]]
    for name, binding in run_task["frozen_modules"].items():
        require(binding["sha256"] == run_task["baseline_task"]["code_sha256"][name+".py"], "frozen code hash differs")
        _binding(binding, cache)
    prediction = import_bound_module("prediction", run_task["frozen_modules"]["prediction"])
    gate = import_bound_module("gate_core", run_task["frozen_modules"]["gate_core"])
    return records, bundle["calibration"], ids, labels["items"], cfg, prediction, gate


def checked_fixed_reference(binding, *, domain, seed, developer_ids, developer_manifest, cache):
    doc = load_json(_binding(binding, cache))
    require(doc.get("schema") == "odr003-round2-fixed-reference-v1" and doc.get("domain") == domain
            and doc.get("stream_seed") == seed, "Frozen reference document belongs to another pair")
    require(doc["developer_manifest"] == developer_manifest, "Frozen reference developer boundary changed")
    _binding(developer_manifest, cache)
    require(doc["developer_ids_sha256"] == canonical_sha(developer_ids), "Frozen reference developer ID order changed")
    expected = frozen_protocol()["data"]["fixed_reference_count"][domain]
    ids = doc["reference_ids"]
    require(doc["reference_count"] == len(ids) == len(set(ids)) == expected
            and [sid for sid in developer_ids if sid in set(ids)] == ids, "Frozen reference IDs/count/order differ")
    origin = load_json(_binding(doc["origin_summary"], cache))
    proof = load_json(_binding(doc["origin_proof"], cache))
    require(proof.get("status") == "passed" and proof.get("summary_sha256") == doc["origin_summary"]["sha256"],
            "Original frozen reference summary lacks independent acceptance")
    keys = doc["origin_reference_path"]
    require(keys == ["domains", domain, "streams", str(seed), "reference_ids"], "Reference origin path changed")
    current = origin
    for key in keys: current = current[key]
    require(current == ids, "Reference IDs differ from the original immutable reference")
    return ids


def _reused(specs, run_task, labels, ids, cfg, prediction, cache, *, backend):
    require(set(specs) == {"O", "A", "G50"}, "reuse exactly O/A/G50")
    result, evidence = {}, {}
    b = run_task["baseline_task"]["bindings"]
    domain, seed = run_task["definition"]["domain"], run_task["definition"]["stream_seed"]
    for method, spec in specs.items():
        paths = {name: _binding(spec[name], cache) for name in ("task", "report", "proof", "predictions")}
        oldtask, report, proof = (load_json(paths[name]) for name in ("task", "report", "proof"))
        require(report["status"] == "main_completed_pending_independent_verification" and proof["status"] == "passed"
                and proof.get("schema") == PROOF_SCHEMA
                and proof["report_sha256"] == spec["report"]["sha256"]
                and report["task_sha256"] == spec["task"]["sha256"]
                and report["definition"] == oldtask["definition"] == proof["definition"]
                and report["definition"]["group"] == method and report["definition"]["domain"] == domain
                and (method == "O" or report["definition"]["stream_seed"] == seed),
                "Reused deterministic result lacks exact accepted identity")
        require(report["source_sha256"] == b["source"]["sha256"] == frozen_protocol()["source_sha256"]
                and report["protocol_sha256"] == proof["protocol_sha256"] == frozen_protocol()["protocol_sha256"]
                and oldtask["protocol"] == frozen_protocol()
                and report["implementation_sha256"] == proof["implementation_sha256"] == run_task["implementation_sha256"]
                and oldtask["implementation_sha256"] == run_task["implementation_sha256"],
                "Reused baseline comes from another source/backend/protocol/code revision")
        require(checked_runtime_receipt(report, paths["report"].parent, cache=cache) == backend
                and proof["deterministic_runtime"] == backend, "Reused baseline deterministic runtime differs")
        require(proof["task_binding"] == spec["task"] and proof["predictions"] == spec["predictions"],
                "Reused baseline proof task/predictions binding differs")
        for name in ("source", "config", "source_core_manifest", "developer_manifest", "developer_labels", "camera", "keypoints"):
            require_same_artifact(oldtask["baseline_task"]["bindings"][name], b[name], "Reuse asset/split/protocol differs: "+name)
        if method != "O": require_same_artifact(oldtask["baseline_task"]["bindings"]["stream_manifest"], b["stream_manifest"], "Reuse observation stream differs")
        if method == "G50":
            require_same_artifact(oldtask["baseline_task"]["bindings"]["calibration"], b["calibration"], "Reuse calibration differs")
        output_prediction = report["outputs"]["predictions"]
        require(output_prediction["path"] == spec["predictions"]["path"]
                and output_prediction["sha256"] == spec["predictions"]["sha256"], "reuse prediction bytes differ")
        rows = checked_developer_rows(read_jsonl(paths["predictions"]), labels, ids,
                                     recompute=prediction.developer_metrics, cfg=cfg, flat=True)
        result[method] = rows
        evidence[method] = dict(bindings=copy.deepcopy(spec), source_reused_without_forward=True,
                                deterministic_baseline_executed_in_this_protocol=True, metric_rows_recomputed=512)
    return result, evidence


def execute(task_path, *, admission, launch_record, conflict_check):
    if not admission.get("allow_start") or not admission.get("boundary_verified"):
        raise RuntimeError("shared-host admission required before CPU verification")
    task_path = Path(task_path).resolve(strict=True)
    task = load_json(task_path)
    require(task.get("schema") == "odr003-round2-task-v1" and task.get("action") == "verify_complete"
            and task.get("stage") == "verification" and task["protocol"] == frozen_protocol(), "wrong complete-verification task")
    require(task_path.is_relative_to(PLAN/"tasks") and launch_record["run_id"] == task["run_id"], "verification launch/task differs")
    _code_bindings(task, Path(__file__).resolve().parent)
    cache = {}
    run_path = _binding(task["run_task"], cache)
    run_task = load_json(run_path)
    require(run_task["stage"] in ("complete", "offline")
            and run_task["definition"] in build_baseline_matrix()+build_complete_matrix()
            and run_task["protocol"] == task["protocol"]
            and run_task["implementation_sha256"] == task["implementation_sha256"], "science task is outside fixed complete matrix")
    admission_path = _binding(run_task["engineering_admission"], cache)
    engineering = load_json(admission_path)
    require(engineering.get("status") == "passed" and engineering.get("allow_complete_matrix") is True
            and engineering["implementation_sha256"] == task["implementation_sha256"]
            and engineering["protocol_sha256"] == task["protocol"]["protocol_sha256"], "engineering did not admit this code/protocol")
    output, report = _real_receipt(task["execution_receipt"], run_task, run_path, cache)
    backend = checked_runtime_receipt(report, output, cache=cache)
    rows = task["all_attempts"]
    require(all(row["exit_code"] in (0, 75) for row in rows), "complete run cannot silently retry a failed/planned engineering attempt")
    attempts = _attempt_evidence(dict(all_attempts=rows, queue_wait_seconds=task.get("queue_wait_seconds", 0)),
                  {run_task["run_id"]: dict(task_path=str(run_path), task_sha256=sha256(run_path))})[run_task["run_id"]]
    if conflict_check(): return dict(exit_code=75, status="waiting_resources", model_forwards=0, proof_published=False)
    boundary = DeveloperOnlyReads(run_task["baseline_task"]["bindings"]["developer_labels"]["path"])
    records, calibration, ids, labels, cfg, prediction, gate = _assets(run_task, cache)
    group = run_task["definition"]["group"]
    control = None
    if group in ("RW", "RP"):
        p_report_path = _binding(run_task["paired_P_report"], cache)
        p_proof_path = _binding(run_task["paired_P_proof"], cache)
        p_report, p_proof = load_json(p_report_path), load_json(p_proof_path)
        require(p_report["definition"]["domain"] == run_task["definition"]["domain"]
                and p_report["definition"]["stream_seed"] == run_task["definition"]["stream_seed"]
                and p_report["stream_binding"] == report["stream_binding"], "control paired P differs")
        control = derive_verified_controls(p_report, p_proof, report_binding=run_task["paired_P_report"])
        require(control == run_task["control"], "random control differs from frozen P-derived probability/mask")
    import_bound_module("state_adapter", run_task["frozen_modules"]["state_adapter"])
    from durable_state import DurableRunStore
    from protected_engine import validate_engine_state
    from reference_driver import validate_reference_state
    from verify_round2 import verify_final_state, _same
    identity = dict(run_id=run_task["run_id"], source_sha256=task["protocol"]["source_sha256"],
                    config_sha256=sha256(run_path), manifest_sha256=report["stream_binding"]["sha256"])
    validator = validate_offline_checkpoint if group == "O" else validate_reference_state if group in ("A", "G50") else validate_engine_state
    store = DurableRunStore(output/"durable", identity, state_validator=validator)
    recovered = store.recover()
    require(recovered is not None and recovered.info.generation == report["final_generation"]
            and recovered.info.metadata["state_sha256"] == report["final_state_sha256"], "report is not latest complete recoverable checkpoint")
    state = recovered.state
    seeds = run_task["definition"]["rng_seeds"]
    if group == "O":
        independent = verify_offline_source_state(state, identity=identity,
            source_binding=run_task["baseline_task"]["bindings"]["source"],
            official_receipt=run_task["baseline_task"]["bindings"]["official_receipt"], cache=cache)
    elif group in ("A", "G50"):
        independent = verify_original_baseline_state(state, expected_stream=records, identity=identity,
                         method=group, calibration=calibration, gate_decider=gate.decide)
    else:
        independent = verify_final_state(state, expected_stream=records, calibration=calibration,
                        pool_seeds=(seeds["uniform_pool"], seeds["vulnerable_pool"]), expected_identity=identity,
                        expected_mode=group, rw_control=None if control is None else control["RW"],
                        rp_control=None if control is None else control["RP"], gate_decider=gate.decide)
    _same(report["counts"], independent["counters"], "main_vs_independent_counts")
    observed = 0 if group == "O" else 1024
    _same(recovered.scientific_cursors, dict(observed=observed, candidates=report["counts"]["candidates"],
          commits=report["counts"]["commits"], rollbacks=report["counts"]["rollbacks"], next_index=observed), "durable_scientific_cursors")
    journal, torn = store.read_journal()
    require(not torn, "unconfirmed journal suffix in completed run")
    physical = store.work_summary()
    _same(report["physical_work"], physical, "physical_work_vs_durable_ledger")
    require(not physical["inflight_unknown_attempt_ids"], "completed run has unresolved physical work")
    require(physical["forwards_confirmed_completed"].get("developer_evaluation", 0) >= 512, "evaluation work not accounted")
    if group == "O":
        require(all(value == 0 for category, value in physical["forwards_reserved_attempt_upper_bound"].items()
                    if category != "developer_evaluation"), "Offline source performed adaptation/gate/protection model work")
    elif group in ("A", "G50"):
        require(physical["forwards_confirmed_completed"].get("adaptation", 0) >= report["counts"]["accepted"],
                "Original baseline accepted updates lack corresponding actual adaptation forwards")
        if group == "G50":
            require(physical["forwards_confirmed_completed"].get("arrival_scoring", 0) >= 1024,
                    "G50 did not account for scoring all observed inputs")
    prediction_binding = dict(path=report["evaluation"]["prediction_path"], sha256=report["evaluation"]["prediction_sha256"])
    prediction_path = _binding(prediction_binding, cache)
    require(prediction_path == output/"developer_predictions.json", "unexpected developer prediction path")
    document = load_json(prediction_path)
    _same(document, state["evaluation"], "complete_evaluation_vs_durable_checkpoint")
    new_rows = checked_developer_rows(document, labels, ids, recompute=prediction.developer_metrics, cfg=cfg)
    checkpoint_prediction_binding = prediction_binding
    flat_binding = report.get("outputs", {}).get("predictions")
    flat_path = _binding(flat_binding, cache)
    require(flat_path == output/"predictions.jsonl", "Flat saved prediction output differs")
    flat_rows = checked_developer_rows(read_jsonl(flat_path), labels, ids,
                     recompute=prediction.developer_metrics, cfg=cfg, flat=True)
    _same(flat_rows, new_rows, "flat_predictions_vs_complete_developer_checkpoint")
    prediction_binding = flat_binding
    require(report["evaluation"]["count"] == 512
            and report["evaluation"]["extended_valid"] == sum(row["valid"] for row in new_rows)
            and report["evaluation"]["original_valid"] == sum(row["original_valid"] for row in new_rows), "evaluation counts differ")
    # Newly executed O/A/G50 are independently accepted before they can be
    # paired. Their accuracy proof never waits for another baseline's risk.
    if group in ("O", "A", "G50"):
        risk, reused_evidence, fixed_binding = None, {}, None
    else:
        reused, reused_evidence = _reused(task["reusable_baselines"], run_task, labels, ids, cfg, prediction, cache, backend=backend)
        fixed_binding = task["fixed_reference_binding"]
        fixed_ids = checked_fixed_reference(fixed_binding, domain=run_task["definition"]["domain"],
                    seed=run_task["definition"]["stream_seed"], developer_ids=ids,
                    developer_manifest=run_task["baseline_task"]["bindings"]["developer_manifest"], cache=cache)
        risk = evaluate_fixed_reference(ids, reused["O"], reused["A"], {"G50": reused["G50"], group: new_rows},
                   expected_reference_count=task["protocol"]["data"]["fixed_reference_count"][run_task["definition"]["domain"]],
                   fixed_reference_ids=fixed_ids)
    control_derivation = derive_controls(run_task["definition"]["domain"], run_task["definition"]["stream_seed"],
                                        report["counts"]["candidates"], report["counts"]["commits"]) if group == "P" else control
    proof = dict(schema=PROOF_SCHEMA, status="passed", run_id=run_task["run_id"],
                 definition=run_task["definition"], report_sha256=task["execution_receipt"]["report_sha256"],
                 report_path=str(output/"report.json"), task_binding=task["run_task"], verification_task_sha256=sha256(task_path),
                 protocol_sha256=task["protocol"]["protocol_sha256"], source_sha256=task["protocol"]["source_sha256"],
                 implementation_sha256=task["implementation_sha256"], counts=report["counts"],
                 deterministic_runtime=backend, deterministic_runtime_binding=report["deterministic_runtime_binding"],
                 stream_binding=report["stream_binding"], asset_bindings=run_task["baseline_task"]["bindings"],
                 engineering_admission=run_task["engineering_admission"], independent_state=independent,
                 final_generation=report["final_generation"], final_state_sha256=report["final_state_sha256"],
                 predictions=prediction_binding, checkpoint_predictions=checkpoint_prediction_binding,
                 developer_metrics_recomputed_from_saved_poses=True,
                 evaluation_bound_to_complete_checkpoint=True, risk=risk, reused_baselines=reused_evidence,
                 fixed_reference_binding=fixed_binding,
                 evaluation_summary=dict(count=512, extended_valid=sum(row["valid"] for row in new_rows),
                                         original_valid=sum(row["original_valid"] for row in new_rows)),
                 controls_derived_from_verified_counts=control_derivation,
                 physical_work=physical, all_attempts=attempts, all_attempt_costs=_attempt_costs(attempts),
                 queue_wait_seconds=task.get("queue_wait_seconds", 0), queue_wait_excluded_from_compute_time=True,
                 models_constructed=0, model_forwards_added_by_verification=0, reserved_GT_read=False,
                 developer_label_open_count=boundary.reads, forbidden_GT_attempts=boundary.denied,
                 interpretation="independent engineering/metric acceptance; not a positive efficacy result",
                 formal_efficacy_claim=False)
    _finite(proof)
    if conflict_check(): return dict(exit_code=75, status="waiting_resources", model_forwards=0, proof_published=False)
    proof_path = Path(task.get("proof_path", str(output/"proof.json"))).resolve()
    require(proof_path == output/"proof.json", "proof path escapes run")
    _new_or_identical(proof_path, proof)
    return dict(exit_code=0, status="complete_run_verified", run_id=run_task["run_id"],
                report=str(output/"report.json"), proof=str(proof_path),
                proof_sha256=sha256(proof_path), model_forwards=0)

