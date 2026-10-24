"""Frozen G50-P round-two protocol: pure configuration and decision functions.

No filesystem access, model evaluation, GPU calls, labels, or process creation.
This module codifies the confirmed revised plan; it does not tune its rules.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any, Mapping


DOMAINS = ("lightbox", "sunlamp")
STREAM_SEEDS = (2021, 2022, 2023)
SOURCE_SHA256 = "d8f9c281d23adaeb77d03023aee8d40ae1cb9e18dd503539c4719bbe719514cc"
ROLE_OFFSETS = {"uniform_pool": 11, "vulnerable_pool": 12, "rw_coin": 21, "rp_permutation": 31}
RNG_SEED_BASE = 2026102300
ENGINEERING_CONTRACTS = (
    "protection_disabled_equivalence", "commit_rollback_state", "pool_replay",
    "interruption_recovery", "gt_isolation", "counts_and_protocol", "finite_outputs",
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _validate_pair(domain: str, stream_seed: int) -> None:
    if domain not in DOMAINS or stream_seed not in STREAM_SEEDS:
        raise ValueError("only the two frozen domains and 2021/2022/2023 streams are authorized")


def _rng_seeds(domain: str, stream_seed: int) -> dict[str, int]:
    _validate_pair(domain, stream_seed)
    pair_base = RNG_SEED_BASE + DOMAINS.index(domain) * 10000 + (stream_seed - 2021) * 100
    return {role: pair_base + offset for role, offset in ROLE_OFFSETS.items()}


def frozen_protocol() -> dict:
    """Return an independent JSON-serializable copy of the approved fixed rules."""
    payload = {
        "protocol_version": "G50P-round2-deterministic-approved-v2",
        "source_sha256": SOURCE_SHA256,
        "architecture": "official_phi3_BN_EHS_source",
        "domains": list(DOMAINS), "stream_seeds": list(STREAM_SEEDS),
        "observations_per_complete_run": 1024, "accepted_images_per_window": 4,
        "max_new_complete_odr_runs": 30,
        "complete_groups": {"A": 6, "G50": 6, "P": 6, "RW": 6, "RP": 6},
        "offline_O_evaluations": {"lightbox": 512, "sunlamp": 512},
        "reuse_groups": [],
        "historical_nondeterministic_results": "preserved; never used as current same-protocol baselines",
        "reuse_requirement": "identical identity and complete protocol with accepted evidence",
        "original_A_G50_PnP_decode_evaluation_unchanged": True,
        "update": {"precision": "FP32", "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.0},
        "deterministic_execution": {
            "schema": "odr003-strict-deterministic-execution-v1",
            "cudnn_benchmark": False, "cudnn_deterministic": True,
            "cudnn_allow_tf32": False, "cuda_matmul_allow_tf32": False,
            "torch_deterministic_algorithms": True, "warn_only": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
            "unsupported_operation": "stop; no automatic nondeterministic fallback",
            "required_on_every_attempt_and_restore": True,
            "backend_receipt_and_actual_versions_recorded": True,
        },
        "data": {"developer_count_each_domain": 512,
                 "fixed_reference_count": {"lightbox": 512, "sunlamp": 511},
                 "frozen_splits_and_streams_reused": True, "reserved_GT_access": False,
                 "gate_protection_update_GT_access": False,
                 "protection_inputs": "already observed unlabeled images only"},
        "pools": {
            "uniform_capacity": 4, "vulnerable_capacity": 4,
            "vulnerable_point_count_max": 6, "vulnerable_peak_threshold_distance_max": 0.05,
            "replacement": "independent uniform reservoir, jth candidate enters with probability 4/j",
            "candidate_population": "all historical observations including G50 rejections",
            "insertion": "after window decision, pending observations in arrival order",
            "selection_order": ["current_window", "uniform_slots", "vulnerable_slots"],
            "duplicate_ids": "first occurrence retained", "maximum_unique_checks_per_window": 12,
        },
        "protection": {
            "raw_nonfinite": "engineering failure, preserve evidence and block downstream",
            "rule1": "previously usable and original-geometrically-valid H becomes unusable or invalid",
            "rule2_join": "all three conditions on the same checked image",
            "h_rotation_change_degrees_gt": 15.0,
            "he_rotation_disagreement_increase_degrees_gt": 10.0,
            "all_11_point_reprojection_increase_gt": {
                "relative_to_pre": 0.25,
                "absolute_pixels_over_original_image_diagonal": 1.0,
                "combine": "max",
            },
            "large_prediction_change_alone_rejects": False,
            "keypoint_set_change_alone_rejects": False,
            "extra_translation_quality_gate": False,
        },
        "random_controls": {
            "RW": {"keep_dynamic_G50": True, "commit_probability": "P.commits/P.candidates",
                   "Q0": "skip paired RW; probability undefined",
                   "exact_update_budget_guaranteed": False, "quota_filling": False,
                   "repeat_seed_to_match_budget": False,
                   "required_budget_reporting": ["Q_P", "K_P", "Q_RW", "K_RW", "M_P", "M_RW", "actual_differences"],
                   "equal_budget_label": "only when actual K_RW equals actual K_P"},
            "RP": {"population": "same 1024 input positions",
                   "selected_count": "4*P.commits", "order": "original stream order",
                   "permutation_prefix": True, "exact_successful_update_budget": True},
        },
        "randomness": {
            "seed_formula": "2026102300 + domain_index*10000 + (stream_seed-2021)*100 + role_offset",
            "domain_indices": {d: i for i, d in enumerate(DOMAINS)},
            "role_offsets": dict(ROLE_OFFSETS),
            "algorithm": "stdlib random.Random; persist actual lists and exact RNG states",
            "seeds": {f"{d}:{s}": _rng_seeds(d, s) for d in DOMAINS for s in STREAM_SEEDS},
            "dependent_on_GT_or_scores": False,
            "paired_continuous_and_recovery_pool_seeds_identical": True,
        },
        "engineering": {
            "stream_seed": 2021, "prefix_observations": 128,
            "max_trajectories_per_domain": 4, "max_trajectories_total": 8,
            "max_total_logical_observations": 1024, "developer_accuracy_evaluation": False,
            "interruptions": ["after temporary update of second full candidate window",
                              "safe point at observation index 31"],
            "missing_second_window": "record uncovered real branch; synthetic contract only; never extend prefix",
            "required_contracts": list(ENGINEERING_CONTRACTS),
            "required_measurements": [
                "per_rule_and_subcondition_evaluable_and_triggered_counts",
                "actual_protection_IDs_order_sources_and_pre_post_signals",
                "pool_sizes_candidate_replacement_counts_and_unique_coverage",
                "rejected_G50_images_checked_and_cold_start_coverage",
                "candidate_commit_rollback_tail_and_longest_rollback_counts",
                "forward_counts_recomputations_component_time_peak_memory_and_queue_wait",
            ],
            "Q0_holds_full_exploration": True,
            "near_frozen": {"minimum_candidates": 8, "maximum_commit_fraction": 0.1,
                            "holds_full_exploration": True},
            "long_rollback_warning": 8,
            "zero_real_trigger_when_Q_positive": "not a hold if all engineering contracts pass",
        },
        "cost": {"protection_forwards_per_candidate_max": 24,
                 "complete_run_protection_forwards_max": 6144,
                 "short_trajectory_protection_forwards_max": 768,
                 "recomputed_forwards_count_as_actual_work": True,
                 "recomputed_scientific_observations_or_updates_count_twice": False,
                 "queue_wait_separate_from_compute": True,
                 "wall_clock_success_cap": None},
        "recovery": {
            "authority": "latest complete recoverable checkpoint; log commit alone is insufficient",
            "unpersisted_work": "may recompute; charge actual work; do not double-count science",
            "persisted_complete_commit": "never execute again",
            "normal_rollback": "learning state rolls back; observation and decision histories advance",
        },
        "exploration": {
            "sunlamp": {"tail_D": "P < G50", "mean_D": "P <= A"},
            "lightbox_risk_retention": 0.8,
            "average_error_relative_to_A_max": 1.05,
            "A_improvement_over_O_retention": 0.8,
            "A_without_improvement": "P mean error <= O mean error",
            "coverage": "per stream no worse than A and no new fixed-reference missing predictions",
            "minimum_risk_improvement_percent": None,
            "aggregation": "three paired observation streams, equal weights; retain all per-stream outcomes",
            "formal_efficacy_claim": False,
        },
        "automatic_parameter_changes_or_extra_runs": False,
    }
    payload["protocol_sha256"] = _canonical_hash(payload)
    return payload


def build_engineering_matrix() -> list[dict]:
    """Exactly eight bounded trajectories; interrupted resumes share one budget."""
    rows = []
    for domain in DOMAINS:
        for trajectory in ("G50_reference", "P_disabled", "P_continuous", "P_interrupted_resumed"):
            rows.append({
                "run_key": f"engineering:{domain}:2021:{trajectory}",
                "domain": domain, "stream_seed": 2021, "trajectory": trajectory,
                "observations": 128, "observation_prefix": [0, 128],
                "logical_observation_budget": 128,
                "protection_enabled": trajectory in ("P_continuous", "P_interrupted_resumed"),
                "rng_seeds": _rng_seeds(domain, 2021),
                "developer_accuracy_evaluation": False,
                "interruptions": (["candidate_2_after_temporary_update", "observation_31_safe_point"]
                                  if trajectory == "P_interrupted_resumed" else []),
                "resume_recomputations": "charged work; same logical trajectory and budget",
            })
    return rows


def build_complete_matrix() -> list[dict]:
    """Upper-bound templates, not permission to bypass engineering or P counts."""
    rows = []
    for group in ("P", "RW", "RP"):
        for domain in DOMAINS:
            for stream_seed in STREAM_SEEDS:
                rows.append({
                    "run_key": f"{group}:{domain}:{stream_seed}", "group": group,
                    "domain": domain, "stream_seed": stream_seed, "observations": 1024,
                    "source_sha256": SOURCE_SHA256, "rng_seeds": _rng_seeds(domain, stream_seed),
                    "engineering_all_pairs_required": True,
                    "paired_P_dependency": None if group == "P" else f"P:{domain}:{stream_seed}",
                    "counts_must_be_bound_before_start": group != "P",
                    "run_limit_slot": len(rows) + 13,
                })
    return rows


def build_baseline_matrix() -> list[dict]:
    """Two offline O evaluations and twelve same-backend A/G50 ODR runs.

    These are the explicitly approved additions, not aliases of historical
    nondeterministic results. O never observes or adapts to an input stream.
    """
    rows = []
    for domain in DOMAINS:
        rows.append(dict(run_key=f"O:{domain}:offline", group="O", domain=domain,
                         stream_seed=2021, observations=0, developer_observations=512,
                         source_sha256=SOURCE_SHA256, rng_seeds=_rng_seeds(domain, 2021),
                         engineering_all_pairs_required=True, paired_P_dependency=None,
                         counts_must_be_bound_before_start=False, run_limit_slot=None))
    for group in ("A", "G50"):
        for domain in DOMAINS:
            for stream_seed in STREAM_SEEDS:
                rows.append(dict(run_key=f"{group}:{domain}:{stream_seed}", group=group,
                                 domain=domain, stream_seed=stream_seed, observations=1024,
                                 developer_observations=512, source_sha256=SOURCE_SHA256,
                                 rng_seeds=_rng_seeds(domain, stream_seed),
                                 engineering_all_pairs_required=True, paired_P_dependency=None,
                                 counts_must_be_bound_before_start=False,
                                 run_limit_slot=len(rows)-1))
    return rows


def derive_controls(domain: str, stream_seed: int, candidates: int, commits: int) -> dict:
    """Freeze a paired RW probability and RP permutation using P's GT-free counts.

    Q/K are scientific counts from an accepted complete P run, not provisional
    journal events.  The caller must bind that report's identity and proof hash.
    """
    _validate_pair(domain, stream_seed)
    q, k = _count(candidates, "candidates"), _count(commits, "commits")
    if not 0 <= k <= q <= 256:
        raise ValueError("P must satisfy 0 <= commits <= candidates <= 1024/4")
    seeds = _rng_seeds(domain, stream_seed)
    permutation = list(range(1024))
    random.Random(seeds["rp_permutation"]).shuffle(permutation)
    m = 4 * k
    selected = sorted(permutation[:m])
    rw = {
        "enabled": q > 0, "skip_reason": "P_candidates_zero_probability_undefined" if q == 0 else None,
        "commit_probability": k / q if q else None,
        "probability_numerator": k, "probability_denominator": q,
        "rng_seed": seeds["rw_coin"], "dynamic_G50": True,
        "budget_kind": "commit_probability_matched_not_update_count_matched",
        "target_P_commits": k, "target_P_used_images": m,
        "quota_filling": False, "seed_retry": False,
        "deterministic_endpoint": ("all_rollback" if q and k == 0 else "all_commit" if q and k == q else None),
        "endpoint_reuse": "only with full identity and protocol equivalence proof",
    }
    rp = {
        "enabled": True, "rng_seed": seeds["rp_permutation"],
        "permutation": permutation, "permutation_sha256": _canonical_hash(permutation),
        "selected_positions": selected, "selected_positions_sha256": _canonical_hash(selected),
        "used_images": m, "commits": k,
        "budget_kind": "exact_successful_update_count_matched",
        "selection_order": "original_observation_order",
        "zero_budget_reuse": "only with full identity and protocol equivalence proof",
    }
    return {"domain": domain, "stream_seed": stream_seed, "P_candidates": q, "P_commits": k,
            "P_used_images": m, "RW": rw, "RP": rp,
            "requires_verified_P_report_identity_binding": True}


def assess_engineering(candidates: int, commits: int, triggered_windows: int,
                       max_consecutive_rollbacks: int, *,
                       contract_checks: Mapping[str, bool] | None = None) -> dict:
    """Apply only the fixed P short-test hold/warning rules; never retune.

    Missing or false required engineering checks fail closed.  Real zero
    triggering is distinct from synthetic wiring checks and is not a hold by
    itself.  This function assesses P, not random-control low-commit behavior.
    """
    q = _count(candidates, "candidates")
    k = _count(commits, "commits")
    triggered = _count(triggered_windows, "triggered_windows")
    longest = _count(max_consecutive_rollbacks, "max_consecutive_rollbacks")
    if q > 32 or k > q or triggered > q or longest > q - k:
        raise ValueError("short-test counts violate the fixed 128-observation, B4 limits")
    checks = dict(contract_checks or {})
    holds, warnings = [], []
    failed_contracts = [name for name in ENGINEERING_CONTRACTS if checks.get(name) is not True]
    if failed_contracts:
        holds.append("required_engineering_contract_missing_or_failed")
    if triggered != q - k:
        holds.append("P_rollback_count_does_not_match_distinct_triggered_windows")
    if q == 0:
        holds.append("no_complete_P_candidate_window")
    near_frozen = q >= 8 and 10 * k <= q
    if near_frozen:
        holds.append("P_near_frozen_commit_fraction_at_most_10_percent")
    if longest >= 8:
        warnings.append("at_least_8_consecutive_rollbacks")
    if q > 0 and triggered == 0:
        warnings.append("no_real_trigger_in_fixed_prefix_not_evidence_of_efficacy_or_uselessness")
    if k == 0:
        warnings.append("zero_commits_report_separately")
    return {
        "candidates": q, "commits": k, "rollbacks": q - k,
        "triggered_windows": triggered,
        "commit_fraction": k / q if q else None,
        "trigger_fraction": triggered / q if q else None,
        "max_consecutive_rollbacks": longest,
        "near_frozen": near_frozen, "hold_full_exploration": bool(holds),
        "engineering_contracts_passed": not failed_contracts and triggered == q - k,
        "failed_contracts": failed_contracts, "hold_reasons": holds, "warnings": warnings,
        "allowed_response": "preserve fixed parameters and report; never extend prefix, alter seed, or add runs",
    }


def _finite_nonnegative(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _metric(method: Any, name: str):
    return method.get(name) if isinstance(method, Mapping) else None


def _comparison(name: str, left: Any, right: Any, *, strict: bool = False) -> dict:
    available = _finite_nonnegative(left) and _finite_nonnegative(right)
    return {"criterion": name, "passed": bool(available and (left < right if strict else left <= right)),
            "value": _json_safe(left), "limit": _json_safe(right), "comparison": "<" if strict else "<=",
            "available": available, "reason": None if available else "missing_nonfinite_or_negative_metric",
            "invalid_input_representations": (None if available else {"value": repr(left), "limit": repr(right)})}


def _json_safe(value: Any):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return copy.deepcopy(value)


def _coverage_check(a: Mapping, p: Mapping) -> dict:
    a_coverage, p_coverage = _metric(a, "coverage"), _metric(p, "coverage")
    missing = _metric(p, "reference_missing")
    valid = all(isinstance(values, (list, tuple)) and len(values) == 3
                for values in (a_coverage, p_coverage, missing))
    if valid:
        valid = all(not isinstance(v, bool) and isinstance(v, int) and v >= 0
                    for values in (a_coverage, p_coverage, missing) for v in values)
    if valid:
        valid = all(v <= 512 for values in (a_coverage, p_coverage) for v in values)
    per_stream = ([{"stream_seed": s, "A_coverage": a_coverage[i], "P_coverage": p_coverage[i],
                    "P_reference_missing": missing[i],
                    "passed": p_coverage[i] >= a_coverage[i] and missing[i] == 0}
                   for i, s in enumerate(STREAM_SEEDS)] if valid else [])
    return {"criterion": "coverage_and_fixed_reference", "passed": valid and all(x["passed"] for x in per_stream),
            "available": valid, "per_stream": per_stream,
            "reason": None if valid else "three_stream_coverage_or_missing_counts_absent_or_invalid"}


def assess_exploration(domains: Mapping[str, Mapping]) -> dict:
    """Screen verified three-stream aggregates without reading data or GT.

    Input per domain::

      {"stream_seeds": [2021, 2022, 2023],
       "O": {"mean_error": ...},
       "A": {"mean_error": ..., "mean_D": ..., "tail_D": ...,
             "coverage": [512,512,512]},
       "G50": {"mean_D": ..., "tail_D": ...},
       "P": {"mean_error": ..., "mean_D": ..., "tail_D": ...,
             "coverage": [...], "reference_missing": [0,0,0]}}

    Scalar metrics must already be verified equal-weight means of the three
    paired streams on the frozen O/A reference sets.  Coverage is assessed per
    stream, so one stream's missing predictions cannot be offset by another.
    Unknown/None metrics fail explicitly; they are never filled with zero.
    """
    if not isinstance(domains, Mapping) or set(domains) != set(DOMAINS):
        raise ValueError("both frozen domains must be present; none may be selected or dropped")
    reports = {}
    for domain in DOMAINS:
        data = domains[domain]
        if not isinstance(data, Mapping):
            raise ValueError("domain summary must be a mapping")
        stream_order_valid = data.get("stream_seeds") == list(STREAM_SEEDS)
        methods = {name: data.get(name, {}) for name in ("O", "A", "G50", "P")}
        o, a, g, p = (methods[name] for name in ("O", "A", "G50", "P"))
        criteria = [{"criterion": "all_three_frozen_paired_streams", "passed": stream_order_valid,
                     "available": stream_order_valid,
                     "reason": None if stream_order_valid else "missing_or_changed_stream_order"}]
        if domain == "sunlamp":
            criteria.append(_comparison("tail_D_below_original_G50", _metric(p, "tail_D"), _metric(g, "tail_D"), strict=True))
            criteria.append(_comparison("mean_D_no_worse_than_original_A", _metric(p, "mean_D"), _metric(a, "mean_D")))
        else:
            for metric in ("mean_D", "tail_D"):
                av, gv = _metric(a, metric), _metric(g, metric)
                benefit_defined = _finite_nonnegative(av) and _finite_nonnegative(gv) and gv < av
                limit = av - 0.8 * (av - gv) if benefit_defined else None
                criterion = _comparison(f"retain_80_percent_G50_{metric}_benefit", _metric(p, metric), limit)
                if not benefit_defined:
                    criterion["reason"] = "original_G50_risk_benefit_missing_or_not_positive"
                criteria.append(criterion)
        av, ov = _metric(a, "mean_error"), _metric(o, "mean_error")
        criteria.append(_comparison("mean_error_at_most_105_percent_of_A", _metric(p, "mean_error"),
                                    1.05 * av if _finite_nonnegative(av) else None))
        gain_defined = _finite_nonnegative(av) and _finite_nonnegative(ov)
        a_improves = gain_defined and av < ov
        limit = ov - 0.8 * (ov - av) if a_improves else ov if gain_defined else None
        criteria.append(_comparison("retain_80_percent_A_mean_error_benefit" if a_improves else "A_no_gain_P_mean_error_no_worse_than_O",
                                    _metric(p, "mean_error"), limit))
        criteria.append(_coverage_check(a, p))
        passed = all(item["passed"] for item in criteria)
        reports[domain] = {"passed": passed, "criteria": criteria,
                           "failed_criteria": [item["criterion"] for item in criteria if not item["passed"]],
                           "input_metrics_retained": _json_safe(dict(data))}
    all_passed = all(report["passed"] for report in reports.values())
    sp, sa = domains["sunlamp"].get("P", {}), domains["sunlamp"].get("A", {})
    st, at = _metric(sp, "tail_D"), _metric(sa, "tail_D")
    if all_passed:
        interpretation = ("exploratory_candidate_sunlamp_tail_no_worse_than_A"
                          if _finite_nonnegative(st) and _finite_nonnegative(at) and st <= at
                          else "partial_progress_sunlamp_tail_still_above_A")
    else:
        interpretation = "predeclared_criteria_not_all_met_keep_all_improvements_and_regressions"
    return {"exploratory_screen_passed": all_passed, "domains": reports,
            "interpretation": interpretation, "formal_efficacy_claim": False,
            "control_attribution": "assess RW actual budget differences and RP exact budget separately; this screening does not establish a pure causal effect",
            "automatic_extra_experiments_or_parameter_changes": False}
