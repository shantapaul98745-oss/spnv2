"""Independent saved-evidence acceptance for one paired-window diagnosis.

Only saved poses, the fixed developer labels, original complete checkpoints and
budget/process receipts are read. This entry never constructs a model, loads an
image or invokes a model forward. It cannot promote a main-completed claim alone.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import pickle
from collections import Counter

from pair_runner import (SCHEMA, OUTPUT_SCHEMA, atomic_json, binding, canonical_sha,
                         derive_branch_snapshots, load_json, load_producer,
                         read_generation, require, science_state, sha256,
                         tree_digest, _metrics_summary, _offline_rows, reference_trace)
from pair_runner import configure_frozen_modules


METRICS = ("valid", "error", "original_valid", "original_error", "rotation_deg", "translation_m",
           "heat_rotation_HIL", "effi_translation_HIL", "heat_error", "effi_error")


def _close(actual, expected, label):
    if expected is None or type(expected) is bool:
        require(type(actual) is type(expected) and actual == expected, label+" differs")
    else:
        require(type(actual) in (int,float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label+" differs")


def verify_branch_rows(rows, ids, labels, cfg, recompute):
    require(len(ids) == len(set(ids)) == 128 and len(rows) == 128
            and [row["sample_id"] for row in rows] == ids, "branch must retain exactly the fixed128 panel IDs/order")
    result = []
    for row in rows:
        sid = row["sample_id"]
        require(set(row) == {"sample_id", "prediction", "metrics"} and sid in labels, "unexpected row or non-developer GT")
        p = row["prediction"]
        for flag in ("h_available", "e_available", "original_h_rejected"):
            require(type(p.get(flag)) is bool, "pose availability must be explicit")
        label = labels[sid]
        expected = recompute(p, dict(quaternion=label["quaternion_wxyz"], translation=label["translation_m"]), cfg)
        require(set(expected) == set(METRICS) and set(row["metrics"]) == set(METRICS), "saved metric schema differs")
        for key in METRICS: _close(row["metrics"][key], expected[key], sid+"."+key)
        result.append(dict(sample_id=sid, prediction=copy.deepcopy(p), metrics=expected))
    return result


def verify_budget(report):
    before, after = report["budget_before"], report["budget"]
    for snapshot in (before, after):
        require(snapshot.get("schema") == "g50p-diagnostic-budget-snapshot-v1", "unrecognized actual budget receipt")
    counts = report["forward_categories"]
    require(isinstance(counts, dict) and all(type(v) is int and v >= 0 for v in counts.values()), "invalid model forward categories")
    require(counts.get("panel_accept", 0) == counts.get("panel_rollback", 0) == 128,
            "fixed two-branch panel does not have exactly256 measured forwards")
    total = sum(counts.values())
    attempts = report["attempt_costs"]
    require(attempts and len({row["run_nonce"] for row in attempts}) == len(attempts)
            and sum(row["status"] == "pair_main_complete" for row in attempts) == 1,
            "actual diagnostic cost history is incomplete/duplicated")
    summed, reserved, confirmed = Counter(), 0, 0
    for row in attempts:
        b, a = row["budget_before"], row["budget"]
        n = sum(row["forward_categories"].values())
        r = a["physical_forwards_reserved_upper"]-b["physical_forwards_reserved_upper"]
        c = a["physical_forwards_confirmed"]-b["physical_forwards_confirmed"]
        require(r == c == n and a["physical_forwards_unknown_upper"] == 0
                and not a["unknown_forward_tokens"], "attempt's actual global ledger does not explain every backbone call")
        summed.update(row["forward_categories"]); reserved += r; confirmed += c
    require(dict(summed) == counts and reserved == confirmed == total and after["physical_forwards_unknown_upper"] == 0,
            "all actual attempt counts differ from the completed paired cost")
    require(not after["unknown_forward_tokens"], "completed paired run has unconfirmed dispatched work")
    return dict(physical_model_forwards=total, forward_categories=copy.deepcopy(counts), actual_attempt_count=len(attempts),
                reservation_difference=reserved, confirmed_difference=confirmed, unknown_upper=0,
                active_attempt_time_reported=report["actual_attempt_elapsed_seconds"],
                global_budget_before=before, global_budget_at_main_report=after,
                timing_note=("Time hard stop removed; measured execution and historical conservative charges remain separate in the continuous ledger" if report["budget"].get("time_hard_stop_enabled") is False else "running snapshots reserve the attempt deadline; final settled active time is in its actual process receipt"))


def verify_saved(report, *, task, candidate, actual_match, expected_match, original_trace,
                 ids, labels, cfg, recompute, offline, equal):
    """Pure independent reconstruction; usable with synthetic saved records."""
    require(report.get("schema") == OUTPUT_SCHEMA
            and report.get("status") == "pair_main_complete_pending_independent_verification",
            "main result is incomplete or masquerades as independently passed")
    require(report["selection"] == task["selection"] and report["producer"] == task["producer"]
            and report["panel"] == task["panel"] and report["run_id"] == task["run_id"], "saved paired identity changed")
    require(report["candidate_learner_sha256"] == tree_digest(candidate), "candidate learner state digest changed")
    expected_branches = derive_branch_snapshots(candidate)
    require(report["expected_branch_learning_sha256"] == {name:tree_digest(value) for name,value in expected_branches.items()},
            "branch derivation is not commit vs complete-window rollback")
    require(equal(actual_match, science_state(expected_match))
            and report["matched_scientific_sha256"] == tree_digest(actual_match)
            and report["expected_scientific_sha256"] == tree_digest(science_state(expected_match)),
            "full actual producer checkpoint did not exactly rejoin saved scientific state")
    require(report["excluded_match_fields"] == ["ledger_boundary"]
            and report["original_producer_files_modified"] is False, "scientific fields were excluded or source modified")
    target = [event for event in original_trace if event["kind"] == "window"
              and event["number"] == task["selection"]["candidate_number"]]
    require(len(target) == 1 and equal(target[0], report["original_target"]), "actual target trace differs from original")
    recomputed = {}
    for name in ("accept", "rollback"):
        saved = report["branch_results"][name]
        rows = verify_branch_rows(saved["rows"], ids, labels, cfg, recompute)
        digest = tree_digest(expected_branches[name])
        require(saved["before_learning_sha256"] == saved["after_learning_sha256"] == digest
                and saved["learning_unchanged"] is True and saved["producer_provider_unchanged"] is True,
                "branch evaluation lacks exact complete learning/RNG/provider isolation")
        summary = _metrics_summary(rows, offline)
        require(saved["summary"] == summary, "fixed128/tail7 or missingness result differs")
        recomputed[name] = dict(rows=rows, summary=summary)
    expected_changes = []
    for a, r, o in zip(recomputed["accept"]["rows"], recomputed["rollback"]["rows"], offline):
        am, rm = a["metrics"], r["metrics"]
        expected_changes.append(dict(sample_id=a["sample_id"], accept=am, rollback=rm, offline=o,
               error_change=None if not(am["valid"] and rm["valid"]) else am["error"]-rm["error"],
               rotation_change_deg=None if am["rotation_deg"] is None or rm["rotation_deg"] is None else am["rotation_deg"]-rm["rotation_deg"],
               translation_change_m=None if am["translation_m"] is None or rm["translation_m"] is None else am["translation_m"]-rm["translation_m"],
               accept_new_missing=rm["valid"] and not am["valid"], rollback_new_missing=am["valid"] and not rm["valid"]))
    require(report["branch_results"]["per_image_changes"] == expected_changes, "per-image improvements/regressions/missingness were omitted or changed")
    differences = {metric: (None if recomputed["accept"]["summary"][metric] is None or recomputed["rollback"]["summary"][metric] is None
                           else recomputed["accept"]["summary"][metric]-recomputed["rollback"]["summary"][metric])
                   for metric in ("mean_error", "mean_D", "tail_D")}
    require(report["branch_results"]["paired_summary"] == differences, "paired continuous differences changed")
    budget = verify_budget(report)
    return dict(branch_summaries={name:value["summary"] for name,value in recomputed.items()},
                paired_summary=differences, actual_decision=target[0]["decision"],
                complete_state_match=True, fixed_denominator=128, fixed_tail=7,
                candidate_sha256=tree_digest(candidate), matched_state_sha256=tree_digest(actual_match),
                per_image_changes=expected_changes, budget=budget,
                model_forwards_added_by_verification=0, efficacy_claim=False)


def _receipt(spec, run_task, run_path):
    report_path = binding(spec["report"])
    require(spec["run_id"] == run_task["run_id"] and report_path == Path(run_task["output_dir"])/"report.json"
            and spec["task"] == dict(path=str(run_path), sha256=sha256(run_path)), "run receipt identity/path differs")
    log = Path(spec["log_path"]).resolve(strict=True)
    require(log.is_file() and Path(spec["exit_code_path"]).read_text(encoding="ascii").strip() == "0", "actual paired main exit is not0")
    outcome = load_json(spec["outcome_path"])
    require(outcome.get("status") == "pair_main_complete" and outcome.get("exit_code") == 0
            and Path(outcome["report"]).resolve() == report_path and outcome["report_sha256"] == spec["report"]["sha256"], "actual main outcome is incomplete")
    launch = load_json(spec["launch_record_path"])
    require(launch["run_id"] == run_task["run_id"] and len(launch["run_nonce"]) >= 16
            and launch["pid"] > 0 and launch["starttime"] > 0 and launch["boot_id"], "main process ownership identity incomplete")
    argv = launch["argv"]
    require("--task-json" in argv and Path(argv[argv.index("--task-json")+1]).resolve() == run_path,
            "actual main command used another diagnostic task")
    report = load_json(report_path)
    require(report["launch_record_sha256"] == canonical_sha(launch)
            and report["task_sha256"] == sha256(run_path), "actual launch/task differs from paired report")
    attempts = spec["attempts"]
    require(attempts and attempts[-1]["exit_code"] == 0
            and sum(row["exit_code"] == 0 for row in attempts) == 1
            and all(row["exit_code"] in (0,75) for row in attempts), "failed paired attempt was silently retried")
    by_nonce = {}
    for row in attempts:
        require(row["run_id"] == run_task["run_id"] and row["task_sha256"] == sha256(run_path)
                and Path(row["task_path"]).resolve() == run_path and row["nonce"] not in by_nonce,
                "earlier actual attempt belongs to another task or was duplicated")
        log_path = Path(row["log_path"])
        code = int(Path(str(log_path)+".exitcode").read_text(encoding="ascii").strip())
        attempt_dir = Path(row["launch_directory"])
        actual = load_json(attempt_dir/"outcome.json")
        require(code == row["exit_code"] == actual["exit_code"] and row["outcome"] == actual,
                "actual earlier exit/outcome changed")
        by_nonce[row["nonce"]] = (attempt_dir, actual)
    for row in report["attempt_costs"]:
        require(row["run_nonce"] in by_nonce, "cost receipt has no actual owned process")
        attempt_dir, actual = by_nonce[row["run_nonce"]]
        actual_launch = load_json(attempt_dir/"launch.json")
        require(row["status"] == actual["status"]
                and row["launch_record_sha256"] == canonical_sha(actual_launch), "cost identity differs from actual owned attempt")
    return report_path, report


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    if not admission.get("allow_start") or not admission.get("boundary_verified"):
        raise RuntimeError("shared-host admission required for CPU-only paired verification")
    task_path = Path(task_path).resolve(strict=True); task = load_json(task_path)
    require(task.get("schema") == SCHEMA and task.get("action") == "verify_pair"
            and launch_record["run_id"] == task["run_id"], "wrong independent pair verifier task")
    budget.check_time("read paired verification inputs")
    run_path = binding(task["run_task"]); run_task = load_json(run_path)
    frozen = load_json(binding(run_task["index_binding"]))
    require(run_task["selection"] in frozen["windows"] and run_task["producer"] == frozen["producer"], "run outside frozen48 diagnostic scope")
    report_path, report = _receipt(task["execution_receipt"], run_task, run_path)
    require([load_json(binding(b)) for b in report["attempt_cost_bindings"]] == report["attempt_costs"], "actual per-attempt cost receipt changed")
    if conflict_check(): return dict(status="waiting_resources", exit_code=75, model_forwards=0)
    selection = run_task["selection"]
    budget.check_time("read complete original matching checkpoint")
    expected_match, _ = read_generation(selection["match_checkpoint"])
    trace = reference_trace(run_task, expected_match["identity"], budget=budget)
    with binding(report["candidate_learner"]).open("rb") as handle: candidate = pickle.load(handle)
    budget.check_time("read recorded actual matched state")
    with binding(report["matched_scientific_state"]).open("rb") as handle: actual_match = pickle.load(handle)
    old = load_producer(run_task)
    backend = old("frozen_backend")
    original_task = load_json(binding(selection["original_task"]))
    budget.check_time("CPU verify fixed source/configuration bindings")
    frozen_modules, boundary, paths, receipt, core, prepared, cfg = configure_frozen_modules(
        backend, original_task, output_dir=run_task["output_dir"], run_id=run_task["run_id"], phase="evaluation")
    baseline, prediction = frozen_modules["run_exploration"], frozen_modules["prediction"]
    panel_doc = load_json(binding(run_task["panel"]["manifest"]))
    ids = [row["sample_id"] for row in panel_doc["items"]]
    labels = {row["sample_id"]:row for row in load_json(binding(run_task["panel"]["labels"]))["items"]}
    require(set(ids) <= labels.keys(), "verifier attempted non-developer labels")
    offline = _offline_rows(run_task, ids)
    result = verify_saved(report, task=run_task, candidate=candidate, actual_match=actual_match,
                          expected_match=expected_match, original_trace=trace, ids=ids, labels=labels,
                          cfg=cfg, recompute=prediction.developer_metrics, offline=offline,
                          equal=frozen_modules["state_adapter"].equal_tree)
    require(boundary.denied == [] and report["gt_boundary"]["forbidden_attempts"] == []
            and report["gt_boundary"]["restored_phase"] == "adaptation", "GT boundary evidence differs")
    original_record = frozen["runs"][f"{selection['domain']}:{selection['stream_seed']}"]
    original_report = load_json(binding(original_record["original_report"]))
    require(report["deterministic_runtime"] == original_report["deterministic_runtime"], "diagnostic backend/software differs from original")
    proof = dict(schema="g50p-risk-diagnostic-pair-proof-v1", status="passed", run_id=run_task["run_id"],
                 report_sha256=sha256(report_path), task_binding=task["run_task"],
                 verifier_task_sha256=sha256(task_path), selection=selection, panel=run_task["panel"],
                 producer=run_task["producer"], independent_result=result,
                 actual_main_exit_verified=True, model_forwards_added_by_verification=0,
                 reserved_GT_read=False, original_runs_modified=False,
                 interpretation="paired-window diagnostic evidence only; not a new ODR run or threshold selection")
    proof_path = Path(task["proof_path"]).resolve()
    require(proof_path == Path(run_task["output_dir"]).resolve()/"proof.json", "pair proof escapes its own isolated output")
    budget.check_time("publish independent paired verification")
    if proof_path.exists(): require(load_json(proof_path) == proof, "existing accepted proof differs; preserve and stop")
    else: atomic_json(proof_path, proof)
    return dict(status="pair_verified", exit_code=0, run_id=run_task["run_id"], proof=str(proof_path),
                proof_sha256=sha256(proof_path), report=str(report_path), model_forwards=0)

