"""Independent, CPU-only verification of the eight actual engineering artifacts.

Called by guarded_entry only after shared-host admission.  This entry does not
construct a model, call a model forward, load images, or open any target labels.
Receipts locate evidence; the actual files, complete checkpoint generations and
recorded execution are independently read and checked before admission.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import re

from frozen_backend import sha256, canonical_sha, import_bound_module
from guarded_entry import PLAN
from round2_protocol import frozen_protocol, build_engineering_matrix, assess_engineering


RUN_ID = "g50p2_engineering_verification"
SYNTHETIC_CASES = (
    ("test_protected_learning", "test_rollback_restores_full_learning_but_keeps_input_counts"),
    ("test_guard_pools", "test_full_checkpoint_roundtrip_replays_lists_and_replacements"),
    ("test_protected_engine", "test_both_fixed_faults_fire_once_and_recovery_matches_continuous"),
    ("test_guard_pools", "test_no_single_or_two_conditions_can_veto"),
    ("test_guard_pools", "test_three_conditions_on_same_image_trigger"),
    ("test_guard_pools", "test_prelegal_to_unavailable_or_physical_failure_triggers"),
    ("test_protected_engine", "test_rw_zero_all_rollbacks_and_one_all_commits"),
    ("test_durable_state", "test_every_publication_interruption_boundary"),
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _binding(binding, cache):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "invalid file binding")
    path = Path(binding["path"]).resolve(strict=True)
    require(path.is_file() and re.fullmatch("[0-9a-f]{64}", binding["sha256"]) is not None,
            "invalid bound file/checksum")
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = str(path)
    if cache.get(key, {}).get("signature") != signature:
        cache[key] = dict(signature=signature, sha256=sha256(path))
    require(cache[key]["sha256"] == binding["sha256"], "bound file changed: " + str(path))
    return path


def _code_bindings(task, directory):
    actual = {path.name: sha256(path) for path in sorted(directory.glob("*.py"))}
    require(task["implementation_sha256"] == actual, "complete execution/verifier code binding differs")
    require({"verify_engineering_phase.py", "verify_round2.py", "guarded_entry.py",
             "run_round2.py", "protected_engine.py", "reference_driver.py", "durable_state.py"} <= actual.keys(),
            "engineering/verifier implementation coverage is incomplete")


def _synthetic_evidence(task, cache):
    path = _binding(task["synthetic_contracts"], cache)
    doc = load_json(path)
    require(doc.get("schema") == "odr003-round2-local-contract-report-v1" and doc.get("status") == "passed",
            "synthetic contract evidence has not passed")
    require(doc["implementation_sha256"] == task["implementation_sha256"], "synthetic contracts cover different code")
    platform = doc["platform"]
    require(platform.get("host") == "local_windows" and platform.get("real_spnv2_forward") is False
            and platform.get("shared_host_compute") is False, "synthetic evidence scope is ambiguous")
    require(doc["failures"] == 0 and doc["errors"] == 0 and type(doc["tests_run"]) is int and doc["tests_run"] > 0,
            "synthetic test process did not pass")
    rows = doc["test_results"]
    require(len(rows) == doc["tests_run"] and all(row["outcome"] == "passed" for row in rows),
            "synthetic test result counts/status differ")
    ids = [row["id"] for row in rows]
    require(len(set(ids)) == len(ids), "duplicate synthetic test case IDs")
    log_location = doc.get("stdout_archive_path", doc.get("log_path", doc.get("stdout_path")))
    require(isinstance(log_location, str), "synthetic output archive path missing")
    log = _binding(dict(path=log_location, sha256=doc["stdout_sha256"]), cache)
    output = log.read_text(encoding="utf8")
    require(re.search(r"(?m)^OK\s*$", output) is not None
            and re.search(r"Ran\s+" + str(doc["tests_run"]) + r"\s+tests?\b", output) is not None,
            "actual synthetic unittest output does not confirm completion")
    required = []
    for module, name in SYNTHETIC_CASES:
        matches = [case for case in ids if case.startswith(module + ".") and case.endswith("." + name)]
        require(len(matches) == 1 and matches[0] in output, "required successful synthetic case missing: " + module + "." + name)
        required.append(matches[0])
    for key in ("commit_rollback_state", "pool_replay", "interruption_recovery", "protection_rules", "RW_endpoints"):
        require(doc["contract_coverage"].get(key) is True, "missing synthetic contract coverage: " + key)
    return dict(binding=task["synthetic_contracts"], status="passed", tests_run=doc["tests_run"],
                required_case_ids=required, output_binding=dict(path=str(log), sha256=doc["stdout_sha256"]),
                scope="synthetic implementation contracts only; not real-model trigger efficacy")


def _frozen_assets(task, cache):
    baseline = task["baseline_task"]
    require(baseline["method"] == "G50" and baseline["seed"] == 2021
            and baseline["domain"] == task["definition"]["domain"], "engineering baseline identity differs")
    bindings = baseline["bindings"]
    # Never hash or open labels, including developer labels, in engineering.
    allowed = {"calibration", "calibration_proof", "camera", "config", "data_prepared", "data_verified",
               "developer_manifest", "keypoints", "official_receipt", "source", "source_core_manifest", "stream_manifest"}
    require(set(bindings) == allowed | {"developer_labels"}, "unexpected baseline binding; no new label source permitted")
    paths = {name: _binding(bindings[name], cache) for name in sorted(allowed)}
    require(bindings["source"]["sha256"] == frozen_protocol()["source_sha256"], "official source hash differs")
    receipt = load_json(paths["official_receipt"])
    require(receipt.get("status") == "passed" and receipt.get("source_offline_identity_verified") is True
            and receipt.get("strict_model_load_performed") is True, "source receipt is not accepted")
    for name in ("source", "config", "source_core_manifest"):
        require(Path(receipt[name + "_path"]).resolve() == paths[name]
                and receipt[name + "_sha256"] == bindings[name]["sha256"], "source receipt binding differs: " + name)
    strict_path = _binding(dict(path=receipt["strict_load_report_path"], sha256=receipt["strict_load_report_sha256"]), cache)
    strict = load_json(strict_path)
    require(strict.get("status") == "passed" and strict.get("strict_load") is True
            and strict.get("all_loaded_tensors_equal_source") is True, "actual bound source strict-load proof failed")
    core = load_json(paths["source_core_manifest"])
    core_root = Path(core["root"]).resolve(strict=True)
    expected_files = core["files"]
    require(isinstance(expected_files, dict) and bool(expected_files), "missing frozen A source files")
    require(set(expected_files) == {str(p.relative_to(core_root)) for p in core_root.rglob("*.py")},
            "frozen A source coverage changed")
    for relative, digest in expected_files.items():
        child = (core_root / relative).resolve(strict=True)
        require(child.is_relative_to(core_root), "frozen source file escapes root")
        _binding(dict(path=str(child), sha256=digest), cache)
    for binding_name, prefix in (("camera", "camera"), ("keypoints", "keypoint")):
        require(paths[binding_name] == Path(core[prefix + "_path"]).resolve()
                and bindings[binding_name]["sha256"] == core[prefix + "_sha256"], "camera/keypoint provenance differs")
    prepared, verified = load_json(paths["data_prepared"]), load_json(paths["data_verified"])
    require(prepared.get("status") == "prepared" and verified.get("status") == "passed"
            and verified["prepared_proof_sha256"] == bindings["data_prepared"]["sha256"], "frozen data proof differs")
    domain = task["definition"]["domain"]
    for name, artifact in (("stream_manifest", domain + "/stream_2021"), ("developer_manifest", domain + "/developer")):
        require(bindings[name] == prepared["artifacts"][artifact], "frozen data boundary differs: " + artifact)
    # The label binding may be compared to its old proof without opening it.
    require(bindings["developer_labels"] == prepared["artifacts"][domain + "/developer_labels"], "developer GT boundary changed")
    require(verified["counts"][domain]["developer"] == 512, "developer count changed")
    bundle, proof = load_json(paths["calibration"]), load_json(paths["calibration_proof"])
    require(proof.get("status") == "passed" and proof.get("recomputed") is True
            and proof["bundle_sha256"] == bindings["calibration"]["sha256"], "independent calibration proof differs")
    require(bundle.get("status") == "passed" and bundle["source_sha256"] == bindings["source"]["sha256"], "calibration source differs")
    require(bundle["developer_manifest_sha256"] == {d: prepared["artifacts"][d + "/developer"]["sha256"]
                                                   for d in ("lightbox", "sunlamp")}, "pooled calibration developer boundaries differ")
    calibration = bundle["calibration"]
    require(calibration.get("schema") == "odr003-frozen-shared-calibration-v1"
            and calibration.get("ecdf") == "midrank" and calibration.get("interpolation") == "linear", "calibration protocol changed")
    require(set(task["frozen_modules"]) == {"state_adapter", "data_boundary", "gate_core", "run_exploration", "prediction"},
            "frozen implementation module set changed")
    for name, binding in task["frozen_modules"].items():
        path = _binding(binding, cache)
        require(path.name == name + ".py" and binding["sha256"] == baseline["code_sha256"][path.name], "frozen module differs: " + name)
    stream = load_json(paths["stream_manifest"])
    require(stream["seed"] == 2021 and len(stream["items"]) == 1024
            and len({row["sample_id"] for row in stream["items"]}) == 1024, "frozen input stream differs")
    preprocess = canonical_sha(dict(config=bindings["config"], core=bindings["source_core_manifest"],
                                    data_code=task["frozen_modules"]["data_boundary"]))
    records = [dict(sample_id=row["sample_id"], image_sha256=row["image_sha256"], preprocess_sha256=preprocess)
               for row in stream["items"][:128]]
    return records, calibration


def _read_run_receipt(receipt, task, task_path, cache):
    output = Path(task["output_dir"]).resolve(strict=True)
    require(output.is_relative_to(PLAN / "runs") and output.name == task["run_id"], "run output escapes this plan")
    require(receipt["run_id"] == task["run_id"], "receipt run identity differs")
    log_path = Path(receipt["log_path"]).resolve(strict=True)
    exit_path = Path(receipt["exit_code_path"]).resolve(strict=True)
    require(log_path.is_file() and exit_path.read_text(encoding="ascii").strip() == "0", "actual main exit code is not zero")
    report_path = _binding(dict(path=receipt["report_path"], sha256=receipt["report_sha256"]), cache)
    require(report_path == output / "report.json", "report path differs from bound output")
    require(load_json(output / "task.json") == task, "actual installed task differs")
    require(not (output / "failure.json").exists(), "preserved failure exists; do not silently accept retry")
    record_path = Path(receipt["launch_record_path"]).resolve(strict=True)
    launch = load_json(record_path)
    require(canonical_sha(launch) == receipt["launch_record_sha256"], "full launch-record checksum differs")
    require(launch["run_id"] == task["run_id"] and Path(launch["log_path"]).resolve() == log_path
            and launch["project_id"] == "odr003_round2_20261023" and launch["role"] == "worker",
            "launch evidence belongs to another run/project")
    require(type(launch["pid"]) is int and launch["pid"] > 0 and type(launch["starttime"]) is int
            and launch["starttime"] > 0 and launch.get("boot_id") and len(launch.get("run_nonce", "")) >= 16
            and launch.get("session"), "launch birth/nonce/session evidence is incomplete")
    argv = launch["argv"]
    require("--task-json" in argv and argv[argv.index("--task-json") + 1] == str(task_path)
            and "--log-path" in argv and Path(argv[argv.index("--log-path") + 1]).resolve() == log_path,
            "actual launch command does not bind task and log")
    outcome_path = Path(receipt["outcome_path"]).resolve(strict=True)
    require(outcome_path.parent == record_path.parent, "outcome and launch record are from different attempts")
    outcome = load_json(outcome_path)
    require(outcome.get("status") == "main_complete" and type(outcome.get("exit_code")) is int
            and outcome["exit_code"] == 0 and Path(outcome["report"]).resolve() == report_path,
            "actual launch outcome is not a completed run")
    report = load_json(report_path)
    require(report.get("schema") == "odr003-round2-run-report-v1"
            and report.get("status") == "main_completed_pending_independent_verification"
            and report["run_id"] == task["run_id"] and report["stage"] == "engineering"
            and report["definition"] == task["definition"] and report["task_sha256"] == sha256(task_path)
            and report["implementation_sha256"] == task["implementation_sha256"]
            and report["protocol_sha256"] == task["protocol"]["protocol_sha256"]
            and report["source_sha256"] == task["protocol"]["source_sha256"]
            and report["stream_binding"] == task["baseline_task"]["bindings"]["stream_manifest"]
            and report["launch_record_sha256"] == canonical_sha(launch), "actual report identity/protocol differs")
    require(report["evaluation"] is None and report["gt_denied_attempts"] == [], "engineering GT/evaluation boundary violated")
    require(report["strict_source_load"] is True and report["strict_final_reload"] is True,
            "main did not complete its strict load/reload checks")
    return output, report, dict(log_path=str(log_path), exit_code_path=str(exit_path), actual_exit_code=0,
                               report_path=str(report_path), report_sha256=receipt["report_sha256"],
                               launch_record_path=str(record_path), launch_record_sha256=canonical_sha(launch),
                               outcome_path=str(outcome_path), outcome_sha256=sha256(outcome_path))


def _attempt_evidence(receipts_doc, expected_entries):
    """Check actual earlier exit/outcome files, including planned interruptions."""
    attempts = receipts_doc["all_attempts"]
    require(isinstance(attempts, list) and len({row["nonce"] for row in attempts}) == len(attempts),
            "attempt ledger is missing or contains duplicate nonces")
    by_run = {run_id: [] for run_id in expected_entries}
    for row in attempts:
        require(row["run_id"] in by_run, "attempt outside the fixed engineering index")
        expected = expected_entries[row["run_id"]]
        require(row["task_path"] == expected["task_path"] and row["task_sha256"] == expected["task_sha256"],
                "earlier attempt used a different task/configuration")
        log = Path(row["log_path"]).resolve(strict=True)
        launch_directory = Path(row["launch_directory"]).resolve(strict=True)
        require(launch_directory == PLAN / "launches" / row["nonce"], "attempt launch directory/nonce differs")
        actual_code = int(Path(str(log) + ".exitcode").read_text(encoding="ascii").strip())
        actual = load_json(launch_directory / "outcome.json")
        require(type(actual.get("exit_code")) is int and actual_code == actual["exit_code"] == row["exit_code"]
                and actual == row["outcome"], "attempt exit/outcome changed")
        require(actual_code in (0, 75, 76), "an unexpected failed attempt must not be silently retried")
        expected_status = {0: {"main_complete"}, 75: {"waiting_resources", "waiting_project_lock"},
                           76: {"controlled_interruption"}}[actual_code]
        require(actual["status"] in expected_status, "attempt outcome is not an authorized completion/wait/interruption")
        if actual_code == 76:
            require(actual.get("point") in ("candidate_2_tentative", "observation_31"), "unapproved controlled interruption")
        launch_path = launch_directory / "launch.json"
        if actual_code in (0, 76) or launch_path.exists():
            launch = load_json(launch_path)
            require(launch["run_id"] == row["run_id"] and launch["run_nonce"] == row["nonce"]
                    and launch["project_id"] == "odr003_round2_20261023"
                    and Path(launch["log_path"]).resolve() == log, "attempt launch ownership differs")
            argv = launch["argv"]
            require("--task-json" in argv and argv[argv.index("--task-json") + 1] == row["task_path"],
                    "earlier attempt command used another task")
        by_run[row["run_id"]].append(dict(nonce=row["nonce"], log_path=str(log), actual_exit_code=actual_code,
                     outcome_path=str(launch_directory / "outcome.json"), outcome_sha256=sha256(launch_directory / "outcome.json"),
                     actual_outcome=actual))
    for run_id, rows in by_run.items():
        require(rows and rows[-1]["actual_exit_code"] == 0
                and sum(row["actual_exit_code"] == 0 for row in rows) == 1, "each engineering trajectory must finish exactly once")
    wait = receipts_doc["queue_wait_seconds"]
    require(type(wait) in (float, int) and math.isfinite(wait) and wait >= 0, "recorded queue waiting time is invalid")
    return by_run


def _attempt_costs(attempts):
    result = {}
    for name in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
        known, unknown = [], []
        for attempt in attempts:
            value = attempt["actual_outcome"].get(name)
            if value is None:
                unknown.append(attempt["nonce"])
            else:
                require(type(value) is int and value >= 0, "invalid actual attempt memory measurement")
                known.append(value)
        result[name + "_max_recorded"] = max(known) if known else None
        result[name + "_unknown_attempts"] = unknown
    known_elapsed, unknown_elapsed = [], []
    for attempt in attempts:
        value = attempt["actual_outcome"].get("execution_attempt_elapsed_seconds")
        if value is None:
            unknown_elapsed.append(attempt["nonce"])
        else:
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid attempt elapsed time")
            known_elapsed.append(value)
    result["execution_attempt_elapsed_seconds_sum_recorded"] = sum(known_elapsed) if known_elapsed else None
    result["execution_elapsed_unknown_attempts"] = unknown_elapsed
    result["memory_maximum_complete_across_attempts"] = not (
        result["peak_gpu_allocated_bytes_unknown_attempts"] or result["peak_gpu_reserved_bytes_unknown_attempts"])
    result["queue_wait_included"] = False
    return result


def _repair_audit(index, task, *, verifier=None):
    """Keep one proved zero-forward implementation failure in the history.

    This is deliberately separate from `_attempt_evidence`: a genuine exit 1
    never becomes a resource wait/interruption or an accepted current attempt.
    The index and verification task must name the same independently bound
    repair proof, and its verifier rereads the failed execution and imported
    complete initial state. All other unexpected failures remain fatal.
    """
    binding = index.get("pre_forward_repair")
    require(task.get("pre_forward_repair") == binding,
            "verification task/index pre-forward repair binding differs")
    if binding is None:
        return None, []
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
            "invalid pre-forward repair proof binding")
    if verifier is None:
        from preforward_repair import verify_preforward_repair
        verifier = verify_preforward_repair
    evidence = verifier(binding, expected_index=index, plan=PLAN)
    require(isinstance(evidence, dict) and evidence.get("status") == "passed",
            "pre-forward repair evidence did not independently pass")
    rows = evidence.get("prior_failed_engineering_attempts")
    require(isinstance(rows, list) and len(rows) == 1,
            "repair must retain exactly its one actual prior failed attempt")
    row = rows[0]
    require(row.get("actual_exit_code") == 1
            and row.get("actual_outcome", {}).get("exit_code") == 1,
            "the prior implementation failure must retain its genuine exit 1")
    require(row.get("run_id") == index["tasks"][0]["run_id"]
            and isinstance(row.get("nonce"), str) and len(row["nonce"]) >= 16,
            "prior failed execution is not the proved first trajectory")
    physical = row.get("physical_work")
    require(isinstance(physical, dict)
            and all(type(v) is int and v == 0 for v in physical.get("forwards_execution_upper_bound", {}).values()),
            "prior failure cannot be excused if any model forward was dispatched or is unknown")
    zero_forward = row.get("zero_forward_proof", {})
    non_model_work = zero_forward.get("unconfirmed_non_model_work")
    require(zero_forward.get("model_forward_upper_bound") == 0
            and isinstance(non_model_work, list) and len(non_model_work) == 1
            and non_model_work[0].get("work_kind") == "image_read"
            and non_model_work[0].get("forwards_reserved") == {}
            and physical.get("attempts_started") == 1 and physical.get("attempts_finished") == 0
            and physical.get("inflight_unknown_attempt_ids") == [non_model_work[0].get("attempt_id")]
            and physical.get("compute_time_incomplete") is True,
            "prior unconfirmed image-read work must be retained with its zero model-forward reservation")
    require(row.get("scientific_cursors") == dict(observed=0, candidates=0, commits=0, rollbacks=0, next_index=0),
            "prior failed initial checkpoint has scientific progress")
    return evidence, copy.deepcopy(rows)


def _phase_costs(current_attempts, prior_failed_attempts, physical_ledgers):
    """Sum physical ledgers once; retain failure/wait timing uncertainty."""
    attempts = [row for rows in current_attempts.values() for row in rows] + list(prior_failed_attempts)
    nonces = [row["nonce"] for row in attempts]
    require(len(set(nonces)) == len(nonces), "phase cost history contains duplicate attempt nonces")
    ledgers = list(physical_ledgers.values()) + [row["physical_work"] for row in prior_failed_attempts]
    sums = {}
    for name in ("forwards_confirmed_completed", "forwards_execution_lower_bound",
                 "forwards_execution_upper_bound", "forwards_inflight_unknown_upper_bound",
                 "replay_forwards_confirmed_completed"):
        total = Counter()
        for ledger in ledgers:
            for category, value in ledger[name].items():
                require(type(value) is int and value >= 0, "invalid physical forward cost")
                total[category] += value
        sums[name] = dict(sorted(total.items()))
    compute = [ledger["compute_seconds_confirmed"] for ledger in ledgers]
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in compute),
            "invalid physical compute time")
    return dict(scope="eight engineering trajectories and the explicitly bound prior zero-forward failure; verifier own execution remains in its launch outcome",
                actual_attempt_count=len(attempts),
                prior_failed_attempt_count=len(prior_failed_attempts),
                prior_failed_attempts_included=True,
                actual_exit_codes={str(code): count for code, count in sorted(Counter(
                    row["actual_exit_code"] for row in attempts).items())},
                forward_costs=sums,
                compute_seconds_confirmed=sum(compute),
                compute_time_incomplete=any(ledger["compute_time_incomplete"] for ledger in ledgers),
                attempt_measurements=_attempt_costs(attempts),
                scientific_trajectories=8, logical_observations=1024,
                prior_failure_is_additional_scientific_trajectory=False,
                queue_wait_excluded_from_compute_time=True)


def _reference_check(state, records, calibration, gate_module, identity):
    from verify_round2 import _same, _learning_contract, _json_hash
    require(state["schema"] == "odr003-round2-original-g50-reference-v1"
            and state["finished"] is True and state["next_index"] == 128, "reference final state incomplete")
    _same(state["identity"], identity, "reference_identity")
    _same(state["stream_records"], records, "reference_stream")
    _learning_contract(state["learning"])
    trace = state["observation_trace"]
    require(len(trace) == 128, "reference did not observe fixed128")
    accepted = 0
    for index, (event, record) in enumerate(zip(trace, records)):
        require(event["id"] == f"observation:{index:04d}" and event["kind"] == "observation"
                and event["observation_index"] == index and event["sample_id"] == record["sample_id"], "reference order/identity differs")
        result = gate_module.decide("G50", event["arrival"], calibration)
        _same((event["accepted"], event["gate_reason"], event["gate_score"]), result, "reference_recomputed_G50")
        accepted += int(result[0])
    counts = dict(observed=128, accepted=accepted, candidates=accepted//4, commits=accepted//4,
                  rollbacks=0, tail_discarded=accepted%4, consecutive_rollbacks=0, max_consecutive_rollbacks=0)
    _same(state["learning"]["counters"], dict(observed=128, accepted=accepted, updates=accepted//4, tail=accepted%4), "reference_adapter_counts")
    _same(state["finish_counts"], dict(observed=128, accepted=accepted, rejected=128-accepted,
                                     updates=accepted//4, used=4*(accepted//4), tail=accepted%4), "reference_finish_counts")
    return dict(passed=True, kind="independent_original_G50_reference", counts=counts,
                observation_trace_sha256=_json_hash(trace), no_new_model_forward=True)


def _new_or_identical(path, doc):
    """Never replace a previous accepted/held engineering proof."""
    path = Path(path)
    if path.exists():
        require(load_json(path) == doc, "an existing proof differs; preserve it and stop")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    raw = (json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    # link is atomic and cannot replace an existing target, even after a race.
    os.link(temporary, path)
    temporary.unlink()
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def execute(task_path, *, admission, launch_record, conflict_check):
    if not admission.get("allow_start") or not admission.get("boundary_verified"):
        raise RuntimeError("shared-host admission is required before independent CPU artifact verification")
    task_path = Path(task_path).resolve(strict=True)
    task = load_json(task_path)
    require(task.get("schema") == "odr003-round2-task-v1" and task.get("action") == "verify_engineering"
            and task.get("stage") == "verification" and task.get("run_id") == RUN_ID, "wrong engineering-verification task")
    require(task["protocol"] == frozen_protocol(), "fixed protocol changed")
    require(task_path.is_relative_to(PLAN / "tasks"), "verification task escapes plan")
    require(launch_record["run_id"] == RUN_ID, "verification launch identity differs")
    _code_bindings(task, Path(__file__).resolve().parent)
    cache = {}
    index_path = _binding(task["engineering_index"], cache)
    receipts_path = _binding(task["execution_receipts"], cache)
    index, receipts_doc = load_json(index_path), load_json(receipts_path)
    require(index["schema"] == "odr003-round2-engineering-index-v1"
            and index["protocol_sha256"] == task["protocol"]["protocol_sha256"]
            and index["implementation_sha256"] == task["implementation_sha256"], "engineering index differs")
    expected_definitions = {row["run_key"]: row for row in build_engineering_matrix()}
    require(len(index["tasks"]) == 8 and {entry["definition"]["run_key"] for entry in index["tasks"]} == set(expected_definitions),
            "engineering index is not exactly the fixed eight trajectories")
    repair_evidence, prior_failed_attempts = _repair_audit(index, task)
    require(receipts_doc.get("schema") == "odr003-round2-engineering-receipts-v1", "wrong execution receipts schema")
    receipts = receipts_doc["runs"]
    require(len(receipts) == 8 and len({row["run_id"] for row in receipts}) == 8, "exactly eight unique final-run receipts required")
    receipts_by_id = {row["run_id"]: row for row in receipts}
    require(set(receipts_by_id) == {entry["run_id"] for entry in index["tasks"]}, "receipt/index run set differs")
    attempts_by_run = _attempt_evidence(receipts_doc, {entry["run_id"]: entry for entry in index["tasks"]})
    synthetic = _synthetic_evidence(task, cache)
    states, proofs, reports, journals, work = {}, {}, {}, {}, {}
    from durable_state import DurableRunStore
    from verify_round2 import verify_final_state, verify_disabled_equivalence, compare_continuous_resumed, _same
    for entry in index["tasks"]:
        conflict = conflict_check()
        if conflict:
            return dict(exit_code=75, status="waiting_resources", reason=conflict,
                        models_constructed=0, model_forwards=0, proof_published=False)
        run_path = _binding(dict(path=entry["task_path"], sha256=entry["task_sha256"]), cache)
        run_task = load_json(run_path)
        definition = run_task["definition"]
        require(definition == entry["definition"] == expected_definitions[definition["run_key"]]
                and run_task["stage"] == "engineering" and run_task["run_id"] == entry["run_id"]
                and run_task["protocol"] == task["protocol"]
                and run_task["implementation_sha256"] == task["implementation_sha256"], "actual trajectory task differs")
        records, calibration = _frozen_assets(run_task, cache)
        gate = import_bound_module("gate_core", run_task["frozen_modules"]["gate_core"])
        import_bound_module("state_adapter", run_task["frozen_modules"]["state_adapter"])
        from protected_engine import validate_engine_state
        from reference_driver import validate_reference_state
        output, report, receipt_proof = _read_run_receipt(receipts_by_id[entry["run_id"]], run_task, run_path, cache)
        identity = dict(run_id=entry["run_id"], source_sha256=task["protocol"]["source_sha256"],
                        config_sha256=entry["task_sha256"], manifest_sha256=run_task["baseline_task"]["bindings"]["stream_manifest"]["sha256"])
        require((output / "durable/identity.json").is_file(), "completed run has no durable state identity")
        reference = definition["trajectory"] == "G50_reference"
        store = DurableRunStore(output / "durable", identity,
                                state_validator=validate_reference_state if reference else validate_engine_state)
        recovered = store.recover()
        require(recovered is not None and recovered.info.generation == report["final_generation"]
                and recovered.info.metadata["state_sha256"] == report["final_state_sha256"],
                "latest complete recoverable generation does not match final report")
        state = recovered.state
        journal, torn = store.read_journal()
        require(not torn, "completed run retains unconfirmed torn journal suffix")
        physical = store.work_summary()
        _same(report["physical_work"], physical, "actual_physical_work_vs_report")
        require(not physical["inflight_unknown_attempt_ids"], "successful engineering run has unresolved work reservations")
        require(not any("developer" in key or "evaluation" in key for key in physical["forwards_confirmed_completed"]),
                "engineering run performed out-of-scope evaluation forwards")
        if reference:
            proof = _reference_check(state, records, calibration, gate, identity)
            real_counts = proof["counts"]
        else:
            mode = "disabled" if definition["trajectory"] == "P_disabled" else "P"
            seeds = definition["rng_seeds"]
            proof = verify_final_state(state, expected_stream=records, calibration=calibration,
                       pool_seeds=(seeds["uniform_pool"], seeds["vulnerable_pool"]),
                       expected_identity=identity, expected_mode=mode, gate_decider=gate.decide)
            real_counts = state["counters"]
            proof["actual_window_protection_records"] = [
                dict(number=event["number"], decision=event["decision"],
                     protection_records=event["protection_records"], protection=event["protection"])
                for event in state["scientific_trace"] if event["kind"] == "window"]
        _same(report["counts"], real_counts, "report_vs_actual_state_counts")
        _same(recovered.scientific_cursors, dict(observed=real_counts["observed"], candidates=real_counts["candidates"],
              commits=real_counts["commits"], rollbacks=real_counts["rollbacks"], next_index=128), "durable_metadata_scientific_cursors")
        require(physical["forwards_confirmed_completed"].get("arrival_scoring", 0) >= 128
                and physical["forwards_confirmed_completed"].get("adaptation", 0) >= real_counts["accepted"],
                "actual work ledger cannot explain observed/scored/accepted inputs")
        key = (definition["domain"], definition["trajectory"])
        states[key], reports[key], journals[key], work[key] = state, report, journal, physical
        proofs[entry["run_id"]] = dict(definition=definition, receipt=receipt_proof, independent_state=proof,
                  final_generation=recovered.info.generation, final_state_sha256=report["final_state_sha256"],
                  rejected_corrupt_generations=list(recovered.rejected_generations), physical_work=physical,
                  peak_gpu_allocated_bytes=report["peak_gpu_allocated_bytes"],
                  peak_gpu_reserved_bytes=report["peak_gpu_reserved_bytes"],
                  attempt_elapsed_seconds=report["attempt_elapsed_seconds"],
                  peak_memory_and_elapsed_scope="final completed attempt; earlier actual outcomes retained separately",
                  all_attempt_costs=_attempt_costs(attempts_by_run[entry["run_id"]]),
                  actual_attempts=attempts_by_run[entry["run_id"]])
    domains = {}
    for domain in ("lightbox", "sunlamp"):
        reference, disabled = states[(domain, "G50_reference")], states[(domain, "P_disabled")]
        continuous, resumed = states[(domain, "P_continuous")], states[(domain, "P_interrupted_resumed")]
        equality = verify_disabled_equivalence(disabled, reference["learning"],
                                               reference_observation_trace=reference["observation_trace"])
        recovery = compare_continuous_resumed(continuous, resumed,
                     physical_work_continuous=work[(domain, "P_continuous")], physical_work_resumed=work[(domain, "P_interrupted_resumed")])
        marker_records = [event for event in journals[(domain, "P_interrupted_resumed")]
                          if event["kind"] == "controlled_interruption"]
        markers = [event["payload"]["point"] for event in marker_records]
        expected_markers = ["observation_31"]
        if continuous["counters"]["candidates"] >= 2:
            expected_markers.append("candidate_2_tentative")
        require(Counter(markers) == Counter(expected_markers), "actual controlled interruption coverage differs")
        interrupted_run = reports[(domain, "P_interrupted_resumed")]["run_id"]
        actual_process_points = [row["actual_outcome"]["point"] for row in attempts_by_run[interrupted_run]
                                 if row["actual_exit_code"] == 76]
        require(actual_process_points == markers, "controlled fault journal differs from actual exited process outcomes")
        for trajectory in ("G50_reference", "P_disabled", "P_continuous"):
            require(not any(event["kind"] == "controlled_interruption" for event in journals[(domain, trajectory)]),
                    "unplanned controlled interruption in a continuous/reference run")
        c = continuous["counters"]
        windows = [event for event in continuous["scientific_trace"] if event["kind"] == "window"]
        trigger_count = sum(event["protection"]["rollback"] for event in windows)
        # True values below are consequences of the actual comparisons above
        # plus separately bound synthetic state-machine tests, not run flags.
        contracts = dict(protection_disabled_equivalence=equality["passed"],
                         commit_rollback_state=synthetic["status"] == "passed",
                         pool_replay=True, interruption_recovery=recovery["passed"],
                         gt_isolation=True, counts_and_protocol=True, finite_outputs=True)
        screening = assess_engineering(c["candidates"], c["commits"], trigger_count,
                                       c["max_consecutive_rollbacks"], contract_checks=contracts)
        domains[domain] = dict(disabled_equivalence=equality, interruption_recovery=recovery,
              actual_controlled_interruption_markers=markers,
              second_candidate_real_branch_covered="candidate_2_tentative" in markers,
              missing_real_second_candidate_branch="none" if "candidate_2_tentative" in markers else "synthetic contract only; no prefix extension",
              real_triggered_windows=trigger_count, real_rollback_windows=c["rollbacks"],
              synthetic_contracts_do_not_prove_real_threshold_effectiveness=True,
              assessment=screening)
    held = any(row["assessment"]["hold_full_exploration"] for row in domains.values())
    all_cost = _phase_costs(attempts_by_run, prior_failed_attempts, work)
    proof = dict(schema="odr003-round2-engineering-phase-proof-v1", status="held" if held else "passed",
                 allow_complete_matrix=not held, verification_task_sha256=sha256(task_path),
                 implementation_sha256=task["implementation_sha256"], protocol_sha256=task["protocol"]["protocol_sha256"],
                 source_sha256=task["protocol"]["source_sha256"], engineering_index=task["engineering_index"],
                 execution_receipts=task["execution_receipts"], synthetic_contracts=synthetic,
                 actual_trajectories=8, logical_observations=1024, model_forwards_added_by_verification=0,
                 queue_wait_seconds_recorded_by_controller=receipts_doc["queue_wait_seconds"],
                 queue_wait_excluded_from_compute_time=True,
                 pre_forward_repair=repair_evidence,
                 prior_failed_engineering_attempts=prior_failed_attempts,
                 all_cost=all_cost,
                 target_GT_read_by_verification=False, run_proofs=proofs, domains=domains,
                 interpretation="engineering contracts only; no scientific efficacy claim",
                 next_action="report predefined hold without changing parameters" if held else "eligible for the fixed maximum18 complete matrix")
    conflict = conflict_check()
    if conflict:
        return dict(exit_code=75, status="waiting_resources", reason=conflict, proof_published=False, model_forwards=0)
    proof_path = PLAN / "preflight/engineering_proof.json"
    _new_or_identical(proof_path, proof)
    return dict(exit_code=0, status="engineering_" + proof["status"], allow_complete_matrix=not held,
                report=str(proof_path), proof=str(proof_path), proof_sha256=sha256(proof_path))
