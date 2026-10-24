"""Bounded final aggregation of accepted round-two evidence; no model forward.

Each observation stream remains one descriptive unit. RW is probability matched,
RP is count matched, and no result or failed research criterion adds new runs.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
from pathlib import Path
import statistics

from frozen_backend import sha256
from guarded_entry import PLAN
from round2_protocol import (DOMAINS, STREAM_SEEDS, frozen_protocol, build_complete_matrix,
                             build_baseline_matrix, derive_controls, assess_exploration)
from verify_engineering_phase import _binding, _code_bindings, _new_or_identical, require, load_json
from verify_complete import _finite, PROOF_SCHEMA, checked_fixed_reference
from deterministic_runtime import validate_saved_receipt


def _description(values):
    complete = bool(values) and all(type(v) in (int, float) and math.isfinite(v) for v in values)
    return dict(values=list(values), mean=math.fsum(v/len(values) for v in values) if complete else None,
                sample_sd=statistics.stdev(values) if complete and len(values)>1 else 0.0 if complete else None,
                minimum=min(values) if complete else None, maximum=max(values) if complete else None,
                complete=complete)


def validate_matrix(proofs, skipped_controls):
    """Require new deterministic O2/A6/G506 plus the approved experimental18."""
    expected = {row["run_key"]: row for row in build_baseline_matrix()+build_complete_matrix()}
    by_key = {}
    backend = None
    for proof in proofs:
        key = proof["definition"]["run_key"]
        require(key in expected and proof["definition"] == expected[key] and key not in by_key,
                "duplicate/unknown/changed complete run definition")
        require(proof.get("schema") == PROOF_SCHEMA and proof.get("status") == "passed"
                and proof["source_sha256"] == frozen_protocol()["source_sha256"]
                and proof["protocol_sha256"] == frozen_protocol()["protocol_sha256"], "unaccepted scientific proof")
        receipt = validate_saved_receipt(proof.get("deterministic_runtime"))
        if backend is None: backend = receipt
        require(receipt == backend, "Runtime/backend versions differ across paired runs")
        c, group = proof["counts"], proof["definition"]["group"]
        require(set(c) == {"observed","accepted","candidates","commits","rollbacks","tail_discarded", "consecutive_rollbacks","max_consecutive_rollbacks"}
                and all(type(v) is int and v >= 0 for v in c.values())
                and 0 <= c["commits"] <= c["candidates"] <= 256, "Complete scientific budget differs")
        if group == "O":
            require(all(value == 0 for value in c.values()), "Offline O cannot count adaptations or observations")
        else:
            require(c["observed"] == 1024, "Complete ODR must observe1024 distinct frozen inputs")
        if group in ("A", "G50"):
            require(c["rollbacks"] == 0 and c["commits"] == c["candidates"] == c["accepted"]//4
                    and c["tail_discarded"] == c["accepted"]%4, "Original baseline acquired a rollback or changed its update math")
            if group == "A": require(c["accepted"] == 1024 and c["commits"] == 256, "Original A must update all1024 inputs")
        by_key[key] = proof
    require(26 <= len(by_key) <= 32, "Incomplete/excess approved baseline+ODR matrix")
    require(len({p["run_id"] for p in proofs}) == len(proofs), "scientific run ID was reused for multiple slots")
    skips = {}
    for row in skipped_controls:
        key = row.get("run_key", row.get("definition", {}).get("run_key"))
        require(key in expected and expected[key]["group"] == "RW" and key not in skips and key not in by_key,
                "only one missing RW per Qp0 pair may be skipped")
        if "definition" in row: require(row["definition"] == expected[key], "skipped control definition changed")
        d, seed = expected[key]["domain"], expected[key]["stream_seed"]
        p = by_key.get(f"P:{d}:{seed}")
        require(p is not None and p["counts"]["candidates"] == 0
                and row.get("reason", row.get("skip_reason")) == "P_candidates_zero_probability_undefined",
                "RW skip is not supported by accepted zero P candidates")
        require(row.get("paired_P_run_id", p["run_id"]) == p["run_id"], "RW skip paired P identity differs")
        skips[key] = row
    require(set(by_key) | set(skips) == set(expected), "missing complete runs cannot be silently dropped")
    for d in DOMAINS:
        for seed in STREAM_SEEDS:
            p = by_key[f"P:{d}:{seed}"]
            for method in ("O", "A", "G50"):
                baseline = by_key[f"O:{d}:offline" if method == "O" else f"{method}:{d}:{seed}"]
                spec = p["reused_baselines"][method]["bindings"]
                require(spec["task"] == baseline["task_binding"]
                        and spec["report"] == dict(path=baseline["report_path"], sha256=baseline["report_sha256"])
                        and spec["predictions"] == baseline["predictions"],
                        "P risk did not use these newly executed deterministic baselines")
            control = derive_controls(d, seed, p["counts"]["candidates"], p["counts"]["commits"])
            require(p["controls_derived_from_verified_counts"] == control, "P-derived control settings changed")
            for group in ("RW", "RP"):
                row = by_key.get(f"{group}:{d}:{seed}")
                if row is None: continue
                require(row["controls_derived_from_verified_counts"] == control,
                        "paired control is not derived exclusively from verified P counts")
                require(row["stream_binding"] == p["stream_binding"]
                        and row["fixed_reference_binding"] == p["fixed_reference_binding"], "Control observation stream or fixed reference differs")
                if group == "RP":
                    require(row["counts"]["commits"] == p["counts"]["commits"]
                            and row["counts"]["accepted"] == 4*p["counts"]["commits"]
                            and row["counts"]["rollbacks"] == 0, "RP actual budget must exactly equal P")
    return by_key, skips


def _risk_identity(left, right):
    require(left["reference_ids"] == right["reference_ids"]
            and left["reference_count"] == right["reference_count"]
            and left["developer_count"] == right["developer_count"] == 512
            and left["tail_count"] == right["tail_count"], "fixed reference denominator changed across controls")
    for method in ("O", "A", "G50"):
        require(left["methods"][method] == right["methods"][method], "reused baseline evidence differs across paired methods")


def build_summary(proofs, skipped_controls, *, queue_wait_seconds=0):
    by_key, skips = validate_matrix(proofs, skipped_controls)
    require(type(queue_wait_seconds) in (int, float) and math.isfinite(queue_wait_seconds)
            and queue_wait_seconds >= 0, "invalid queue wait")
    domains, screening, all_budgets, resources = {}, {}, [], {}
    for domain in DOMAINS:
        streams = {}
        for seed in STREAM_SEEDS:
            p = by_key[f"P:{domain}:{seed}"]
            risk = p["risk"]
            require(risk["reference_count"] == frozen_protocol()["data"]["fixed_reference_count"][domain],
                    "frozen per-domain reference count changed")
            methods = copy.deepcopy(risk["methods"])
            budget = dict(Q_P=p["counts"]["candidates"], K_P=p["counts"]["commits"],
                          M_P=4*p["counts"]["commits"], Q_RW=None, K_RW=None, M_RW=None,
                          RW_commit_difference=None, RW_used_image_difference=None,
                          RW_exact_actual_commit_budget_match=None, RP_exact_actual_commit_budget_match=None,
                          RW_matching="probability only; later G50 acceptance trajectories can differ")
            for group in ("RW", "RP"):
                proof = by_key.get(f"{group}:{domain}:{seed}")
                if proof is None:
                    budget["RW_skipped"] = copy.deepcopy(skips[f"RW:{domain}:{seed}"])
                    continue
                _risk_identity(risk, proof["risk"])
                methods[group] = copy.deepcopy(proof["risk"]["methods"][group])
                if group == "RW":
                    k = proof["counts"]["commits"]
                    budget.update(Q_RW=proof["counts"]["candidates"], K_RW=k, M_RW=4*k,
                                  RW_commit_difference=k-budget["K_P"], RW_used_image_difference=4*(k-budget["K_P"]),
                                  RW_exact_actual_commit_budget_match=k == budget["K_P"])
                else:
                    budget["RP_exact_actual_commit_budget_match"] = proof["counts"]["commits"] == budget["K_P"]
            deltas = {}
            for group in ("P", "RW", "RP"):
                if group not in methods: continue
                deltas[group] = {}
                for baseline in ("A", "G50", "P"):
                    if baseline == group: continue
                    deltas[group][baseline] = {metric: (None if value is None or methods[baseline]["metrics"][metric] is None
                                                       else value-methods[baseline]["metrics"][metric])
                                               for metric, value in methods[group]["metrics"].items()}
            streams[str(seed)] = dict(reference_ids=risk["reference_ids"], reference_count=risk["reference_count"],
                                     tail_count=risk["tail_count"], methods=methods,
                                     budget=budget, paired_differences= deltas)
            all_budgets.append(dict(domain=domain, stream_seed=seed, **budget))
        summaries = {}
        for method in ("O", "A", "G50", "P", "RW", "RP"):
            per_stream = [streams[str(seed)]["methods"].get(method) for seed in STREAM_SEEDS]
            metric_names = streams[str(STREAM_SEEDS[0])]["methods"]["P"]["metrics"]
            summaries[method] = dict(metrics={metric: _description([None if row is None else row["metrics"][metric]
                                                                        for row in per_stream]) for metric in metric_names},
                                     stream_seeds=list(STREAM_SEEDS), stream_count=sum(row is not None for row in per_stream),
                                     coverage=[None if row is None else row["coverage"] for row in per_stream],
                                     reference_missing=[None if row is None else row["reference_missing"] for row in per_stream],
                                     aggregation="equal weight over the three paired streams; missing stays None",
                                     no_new_independent_unit_from_images_or_random_coins=True)
        domains[domain] = dict(streams=streams, method_summaries=summaries)
        data = {"stream_seeds": list(STREAM_SEEDS)}
        for method in ("O", "A", "G50", "P"):
            data[method] = {metric: desc["mean"] for metric, desc in summaries[method]["metrics"].items()}
            data[method]["coverage"] = summaries[method]["coverage"]
            data[method]["reference_missing"] = summaries[method]["reference_missing"]
        screening[domain] = data
    for key, proof in by_key.items():
        resources[key] = dict(physical_work=proof["physical_work"], all_attempt_costs=proof["all_attempt_costs"],
                              window_diagnostics=proof["independent_state"], counts=proof["counts"],
                              run_id=proof["run_id"], queue_wait_seconds=proof["queue_wait_seconds"])
    categories = ("forwards_confirmed_completed", "forwards_execution_lower_bound", "forwards_execution_upper_bound",
                  "forwards_inflight_unknown_upper_bound", "replay_forwards_confirmed_completed")
    forwards = {}
    for category in categories:
        total = Counter()
        for p in by_key.values(): total.update(p["physical_work"][category])
        forwards[category] = dict(sorted(total.items()))
    complete_odr = [p for p in by_key.values() if p["definition"]["group"] != "O"]
    experimental = [p for p in complete_odr if p["definition"]["group"] in ("P", "RW", "RP")]
    baseline = [p for p in by_key.values() if p["definition"]["group"] in ("O", "A", "G50")]
    result = dict(schema="odr003-round2-summary-v2", status="completed_pending_independent_summary_acceptance",
                  protocol_sha256=frozen_protocol()["protocol_sha256"], source_sha256=frozen_protocol()["source_sha256"],
                  complete_ODR_executed=len(complete_odr), complete_ODR_authorized_upper_bound=30,
                  experimental_ODR_executed=len(experimental), experimental_ODR_authorized_upper_bound=18,
                  newly_executed_baseline_ODR=12, offline_O_evaluations=2,
                  new_baseline_groups=dict(Counter(p["definition"]["group"] for p in baseline)),
                  deterministic_runtime=next(iter(by_key.values()))["deterministic_runtime"],
                  actual_groups=dict(Counter(p["definition"]["group"] for p in by_key.values())),
                  skipped_controls=list(skips.values()), no_omitted_or_added_run_slots=True,
                  domains=domains, exploration_screen=assess_exploration(screening), budgets=all_budgets,
                  resources_by_run=resources,
                  total_resources=dict(forward_costs=forwards,
                      compute_seconds_confirmed=math.fsum(p["physical_work"]["compute_seconds_confirmed"] for p in by_key.values()),
                      compute_time_incomplete=any(p["physical_work"]["compute_time_incomplete"] for p in by_key.values()),
                      queue_wait_seconds=queue_wait_seconds, queue_wait_excluded_from_compute_time=True,
                      engineering_and_verifier_costs="separate engineering proof / actual verification launch receipts; not counted as science ODR"),
                  study_scope="developer-only second-round exploration; no reserved GT, full-domain comparison, or formal efficacy",
                  control_interpretation=dict(RW="dynamic G50 plus random full-window rollback; probability matched, report actual K/Q differences",
                                              RP="random input positions with exactly4K_P accepted images; exact successful-update budget match",
                                              causal_limit="RW rollbacks alter subsequent model, G50 acceptance and candidate trajectory; actual equal K alone does not equalize history"),
                  all_improvements_and_regressions_retained=True, formal_efficacy_claim=False,
                  automatic_parameter_changes_or_extra_runs=False,
                  next_action="deliver all exploratory evidence; any further study requires a new research decision")
    _finite(result)
    return result


def _verification_processes(receipts, proofs, cache):
    """Proof files alone do not certify that the independent verifier exited0."""
    rows = receipts.get("verification_attempts", [])
    by_id = {}
    for row in rows:
        if row.get("exit_code") != 0: continue
        directory = Path(row["launch_directory"]).resolve(strict=True)
        log = Path(row["log_path"]).resolve(strict=True)
        require(directory == PLAN/"launches"/row["nonce"] and log.is_file(), "verifier attempt location differs")
        code = int(Path(str(log)+".exitcode").read_text(encoding="ascii").strip())
        outcome = load_json(directory/"outcome.json")
        launch = load_json(directory/"launch.json")
        require(code == outcome["exit_code"] == 0 and outcome == row["outcome"], "independent verifier actual exit/outcome differs")
        if outcome.get("status") != "complete_run_verified": continue
        run_id = outcome["run_id"]
        require(run_id not in by_id and launch["project_id"] == "odr003_round2_20261023"
                and launch["run_nonce"] == row["nonce"] and launch["run_id"] == row["run_id"], "duplicate/foreign verifier receipt")
        task_path = _binding(dict(path=row["task_path"], sha256=row["task_sha256"]), cache)
        vt = load_json(task_path)
        require(vt["action"] == "verify_complete" and vt["run_id"] == launch["run_id"], "verifier task/action differs")
        by_id[run_id] = dict(outcome=outcome, verification_task=vt, actual_exit_code=code,
                            verification_task_binding=dict(path=str(task_path), sha256=sha256(task_path)),
                            launch_binding=dict(path=str(directory/"launch.json"), sha256=sha256(directory/"launch.json")),
                            outcome_binding=dict(path=str(directory/"outcome.json"), sha256=sha256(directory/"outcome.json")))
    require(set(by_id) == {p["run_id"] for p in proofs}, "not all complete runs have actual successful independent verifier exits")
    for proof in proofs:
        process = by_id[proof["run_id"]]
        require(proof["verification_task_sha256"] == process["verification_task_binding"]["sha256"],
                "actual verifier process used another task")
        require(process["outcome"]["proof"] == process["verification_task"]["proof_path"], "verifier result proof path differs")
        _binding(dict(path=process["outcome"]["proof"], sha256=process["outcome"]["proof_sha256"]), cache)
    return by_id


def execute(task_path, *, admission, launch_record, conflict_check):
    if not admission.get("allow_start") or not admission.get("boundary_verified"):
        raise RuntimeError("shared-host admission required for CPU summary verification")
    task_path = Path(task_path).resolve(strict=True)
    task = load_json(task_path)
    require(task.get("schema") == "odr003-round2-task-v1" and task.get("action") == "summarize_round2"
            and task.get("stage") == "verification" and task["protocol"] == frozen_protocol()
            and task_path.is_relative_to(PLAN/"tasks") and launch_record["run_id"] == task["run_id"], "wrong summary task")
    _code_bindings(task, Path(__file__).resolve().parent)
    cache = {}
    index_path = _binding(task["complete_index"], cache)
    index = load_json(index_path)
    require(index["schema"] == "odr003-round2-complete-index-v2"
            and index["implementation_sha256"] == task["implementation_sha256"]
            and index["protocol_sha256"] == task["protocol"]["protocol_sha256"], "complete index differs")
    require(index.get("skipped_controls", []) == task["skipped_controls"], "skip records differ from index")
    engineering = load_json(_binding(index["engineering_admission"], cache))
    require(engineering["status"] == "passed" and engineering["allow_complete_matrix"] is True, "engineering not accepted")
    proofs = [load_json(_binding(binding, cache)) for binding in task["run_proofs"]]
    by_key, _ = validate_matrix(proofs, task["skipped_controls"])
    baseline_bindings = task["baseline_proofs"]
    baseline_proofs = [load_json(_binding(binding, cache)) for binding in baseline_bindings]
    require(len(baseline_proofs) == 14
            and {p["definition"]["run_key"] for p in baseline_proofs} == {d["run_key"] for d in build_baseline_matrix()}
            and all(by_key[p["definition"]["run_key"]] == p for p in baseline_proofs),
            "Summary must explicitly bind all fourteen newly executed baselines")
    reuse = load_json(_binding(task["baseline_reuse"], cache))
    require(reuse.get("schema") == "odr003-round2-reused-baselines-v2"
            and reuse.get("protocol_sha256") == task["protocol"]["protocol_sha256"]
            and reuse.get("source_sha256") == task["protocol"]["source_sha256"]
            and reuse.get("implementation_sha256") == task["implementation_sha256"],
            "Summary baseline reuse document comes from another backend/protocol")
    require(reuse.get("baseline_proofs") == baseline_bindings
            and reuse.get("historical_results_reused_as_new_backend") is False,
            "Summary reuse artifact did not bind these fourteen new deterministic baseline proofs")
    require(len(index["tasks"]) == len(by_key)
            and {entry["run_id"] for entry in index["tasks"]} == {p["run_id"] for p in proofs}, "index/proof run set differs")
    by_id = {p["run_id"]: p for p in proofs}
    for entry in index["tasks"]:
        p = by_id[entry["run_id"]]
        require(p["definition"] == entry["definition"] and p["implementation_sha256"] == task["implementation_sha256"]
                and p["task_binding"] == dict(path=entry["task_path"], sha256=entry["task_sha256"]), "index/task/proof binding differs")
        _binding(p["task_binding"], cache)
        report = load_json(_binding(dict(path=p["report_path"], sha256=p["report_sha256"]), cache))
        require(report["counts"] == p["counts"] and report["status"] == "main_completed_pending_independent_verification", "accepted report changed")
        require(validate_saved_receipt(report["deterministic_runtime"]) == p["deterministic_runtime"],
                "Accepted report deterministic backend changed")
        _binding(p["predictions"], cache)
    for domain in DOMAINS:
        for seed in STREAM_SEEDS:
            p = by_key[f"P:{domain}:{seed}"]
            manifest = p["asset_bindings"]["developer_manifest"]
            dev = load_json(_binding(manifest, cache))
            frozen = checked_fixed_reference(p["fixed_reference_binding"], domain=domain, seed=seed,
                       developer_ids=[item["sample_id"] for item in dev["items"]],
                       developer_manifest=manifest, cache=cache)
            require(frozen == p["risk"]["reference_ids"], "Summary P risk changed its original frozen IDs")
            pair = reuse["pairs"][f"{domain}:{seed}"]
            for method in ("O", "A", "G50"):
                require(pair[method] == p["reused_baselines"][method]["bindings"],
                        "Summary pairing document differs from independently accepted risk baselines")
    receipts = load_json(_binding(task["execution_receipts"], cache))
    require(receipts.get("schema") == "odr003-round2-complete-receipts-v1", "wrong complete receipt schema")
    verifier_evidence = _verification_processes(receipts, proofs, cache)
    if conflict_check(): return dict(exit_code=75, status="waiting_resources", model_forwards=0, proof_published=False)
    summary = build_summary(proofs, task["skipped_controls"], queue_wait_seconds=receipts["queue_wait_seconds"])
    summary["actual_independent_verifier_processes"] = verifier_evidence
    summary["accepted_run_proofs"] = task["run_proofs"]
    summary["new_baseline_proofs"] = baseline_bindings
    summary["baseline_reuse"] = task["baseline_reuse"]
    summary["complete_index"] = task["complete_index"]
    summary_path, proof_path = Path(task["summary_path"]).resolve(), Path(task["proof_path"]).resolve()
    require(summary_path == PLAN/"summaries/round2/summary.json" and proof_path == summary_path.parent/"proof.json", "summary output escapes plan")
    _new_or_identical(summary_path, summary)
    # Reconstruct from actual accepted evidence again, rather than accept a
    # caller-provided successful screening flag or an existing summary claim.
    repeated = build_summary(proofs, task["skipped_controls"], queue_wait_seconds=receipts["queue_wait_seconds"])
    saved_summary = load_json(summary_path)
    for key, value in repeated.items(): require(saved_summary[key] == value, "summary independently reconstructed field differs: "+key)
    final_proof = dict(schema="odr003-round2-summary-proof-v2", status="passed", summary_sha256=sha256(summary_path),
                       summary_path=str(summary_path), complete_ODR_executed=summary["complete_ODR_executed"], authorized_upper_bound=30,
                       experimental_ODR_executed=summary["experimental_ODR_executed"], experimental_upper_bound=18,
                       newly_executed_baseline_ODR=12, offline_O_evaluations=2,
                       deterministic_runtime=summary["deterministic_runtime"],
                       baseline_proofs=baseline_bindings, baseline_reuse=task["baseline_reuse"],
                       independently_reconstructed_from_all_accepted_runs=True, actual_verifier_processes_checked=True,
                       protocol_sha256=task["protocol"]["protocol_sha256"], source_sha256=task["protocol"]["source_sha256"],
                       complete_index=task["complete_index"], execution_receipts=task["execution_receipts"],
                       verification_task_sha256=sha256(task_path), model_forwards_added=0, reserved_GT_read=False,
                       exploration_screen=summary["exploration_screen"], formal_efficacy_claim=False)
    if conflict_check(): return dict(exit_code=75, status="waiting_resources", model_forwards=0, proof_published=False)
    _new_or_identical(proof_path, final_proof)
    return dict(exit_code=0, status="round2_delivered", summary=str(summary_path), proof=str(proof_path),
                proof_sha256=sha256(proof_path), model_forwards=0)

