"""Developer-only, pure CPU metrics for the confirmed ODR003 exploration v2.

This module never opens data/model files and never reads ground truth. Its caller
must provide only errors already computed on the frozen developer ID lists, and
must verify source/protocol hashes before comparing methods. It does not infer
asset identity from numerical results.

Public API: evaluate_exploration(records, development_ids, ...). ``records`` is a
flat JSON array following INPUT_ROW_SCHEMA. Repeat the same Offline O rows for
each of the three streams; equality is checked. Every method/mask must contain
exactly one explicit row per developer ID. A missing prediction is a row with
valid=False and error=None, not an absent row. R, when present, needs all three
predeclared mask IDs in every domain/stream. It is averaged *after* computing
each mask's metrics, and then the three streams receive equal weight.

Output contains per-image evidence, per-stream results, descriptive summaries,
paired differences against A, and the predeclared G50/G75 selection. None is
preserved for incomplete fixed-reference metrics. No statistical efficacy claim
or inference from the engineering tests is produced.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "odr003-risk-metrics-v1"
DOMAINS = ("lightbox", "sunlamp")
BASE_METHODS = ("O", "A", "V", "E50", "E75", "G50", "G75")
METRIC_NAMES = (
    "mean_error", "mean_positive_degradation", "worst_5pct_degradation",
    "negative_transfer_rate", "negative_transfer_rate_above_tolerance",
    "mean_rotation_deg", "mean_translation_m",
)
INPUT_ROW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["domain", "stream", "method", "sample_id", "error", "valid"],
    "properties": {
        "domain": {"enum": list(DOMAINS)},
        "stream": {"type": "string"},
        "method": {"enum": [*BASE_METHODS, "R"]},
        "sample_id": {"type": "string", "minLength": 1},
        "error": {"type": ["number", "null"], "minimum": 0},
        "valid": {"type": "boolean"},
        "mask": {"type": "string"},
        "original_rejected": {"type": ["boolean", "null"]},
        "rotation_deg": {"type": ["number", "null"], "minimum": 0},
        "translation_m": {"type": ["number", "null"], "minimum": 0},
    },
    "allOf": [
        {"if": {"properties": {"valid": {"const": True}}},
         "then": {"properties": {"error": {"type": "number"}}},
         "else": {"properties": {"error": {"type": "null"}}}},
        {"if": {"properties": {"method": {"const": "R"}}},
         "then": {"required": ["mask"]},
         "else": {"not": {"required": ["mask"]}}},
    ],
    "additionalProperties": True,
    "$comment": "Runtime validation also rejects NaN/Inf, bool-as-number and duplicates.",
}


class MetricContractError(ValueError):
    """An incomplete or invalid evidence table must block downstream selection."""


def _number(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricContractError(f"{label}: expected a finite nonnegative number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise MetricContractError(f"{label}: not representable as a finite float") from exc
    if not math.isfinite(result) or result < 0:
        raise MetricContractError(f"{label}: expected a finite nonnegative number")
    return result


def _ids(value: Sequence[str], label: str, count: int | None = None) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MetricContractError(f"{label}: expected a sequence of IDs")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise MetricContractError(f"{label}: IDs must be nonempty strings")
    if len(set(result)) != len(result) or (count is not None and len(result) != count):
        raise MetricContractError(f"{label}: duplicate IDs or incorrect count")
    return result


def _mean(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    # Division before summation avoids overflow for individually finite values.
    return math.fsum(value / len(values) for value in values)


def _description(values: Sequence[float | None]) -> dict[str, Any]:
    avg = _mean(values)
    if avg is None:
        return {"values": list(values), "mean": None, "sample_sd": None,
                "minimum": None, "maximum": None, "complete": False}
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    if not math.isfinite(deviation):
        raise MetricContractError("Descriptive dispersion overflow; preserve evidence and stop")
    return {"values": list(values), "mean": avg, "sample_sd": deviation,
            "minimum": min(values), "maximum": max(values), "complete": True}


def _delta(value: float | None, baseline: float | None) -> float | None:
    return None if value is None or baseline is None else value - baseline


def _normalize(row: Mapping[str, Any]) -> dict[str, Any]:
    valid = row.get("valid")
    if type(valid) is not bool:
        raise MetricContractError("valid must be an explicit boolean")
    if "error" not in row:
        raise MetricContractError("error must be explicit, including None for missing predictions")
    error = _number(row["error"], "error", nullable=True)
    if valid != (error is not None):
        raise MetricContractError("valid=True requires finite error; valid=False requires error=None")
    rejected = row.get("original_rejected")
    if rejected is not None and type(rejected) is not bool:
        raise MetricContractError("original_rejected must be bool or None")
    return {"valid": valid, "error": error, "original_rejected": rejected,
            "rotation_deg": _number(row.get("rotation_deg"), "rotation_deg", nullable=True),
            "translation_m": _number(row.get("translation_m"), "translation_m", nullable=True)}


def _one_method(ids, reference_ids, offline, original_a, rows, tolerance):
    ref_missing = [sid for sid in reference_ids if not rows[sid]["valid"]]
    complete = bool(reference_ids) and not ref_missing
    valid_ids = [sid for sid in ids if rows[sid]["valid"]]
    a_valid_ids = [sid for sid in ids if original_a[sid]["valid"]]
    lost_a = [sid for sid in ids if original_a[sid]["valid"] and not rows[sid]["valid"]]
    gained_a = [sid for sid in ids if not original_a[sid]["valid"] and rows[sid]["valid"]]
    rejected_known = all(rows[sid]["original_rejected"] is not None for sid in ids)
    coverage_pass = complete and len(valid_ids) >= len(a_valid_ids)
    tail_count = math.ceil(len(reference_ids) * 0.05) if reference_ids else 0
    metrics = dict.fromkeys(METRIC_NAMES)
    per_sample = []
    degradations = []
    raw_deltas = []
    reference_set = set(reference_ids)
    for sid in ids:
        row, o, a = rows[sid], offline[sid], original_a[sid]
        on_reference = sid in reference_set
        d = max(row["error"] - o["error"], 0.0) if on_reference and row["valid"] else None
        if d is not None:
            degradations.append(d)
            raw_deltas.append(row["error"] - o["error"])
        per_sample.append({"sample_id": sid, "in_reference": on_reference,
                           **row, "offline_error": o["error"], "a_error": a["error"],
                           "positive_degradation": d,
                           "delta_vs_a": _delta(row["error"], a["error"]),
                           "valid_to_invalid_vs_a": a["valid"] and not row["valid"],
                           "invalid_to_valid_vs_a": not a["valid"] and row["valid"]})
    if complete:
        metrics = {
            "mean_error": _mean([rows[sid]["error"] for sid in reference_ids]),
            "mean_positive_degradation": _mean(degradations),
            "worst_5pct_degradation": _mean(sorted(degradations, reverse=True)[:tail_count]),
            "negative_transfer_rate": sum(d > 0 for d in raw_deltas) / len(reference_ids),
            "negative_transfer_rate_above_tolerance": sum(d > tolerance for d in raw_deltas) / len(reference_ids),
            "mean_rotation_deg": _mean([rows[sid]["rotation_deg"] for sid in reference_ids]),
            "mean_translation_m": _mean([rows[sid]["translation_m"] for sid in reference_ids]),
        }
    return {
        "metrics": metrics, "per_sample": per_sample,
        "coverage": {"developer_count": len(ids), "valid_count": len(valid_ids),
                     "a_valid_count": len(a_valid_ids),
                     "original_rejected_count": sum(rows[sid]["original_rejected"] for sid in ids) if rejected_known else None,
                     "newly_missing_reference_ids": ref_missing,
                     "valid_to_invalid_vs_a_ids": lost_a, "invalid_to_valid_vs_a_ids": gained_a,
                     "reference_complete": complete, "pass": coverage_pass},
        "reference_count": len(reference_ids), "tail_count": tail_count,
        "status": "complete" if complete else "empty_reference" if not reference_ids else "missing_reference_predictions",
    }


def _average_masks(mask_reports):
    """Mask dispersion remains nested; a mask never counts as a fourth stream."""
    reports = list(mask_reports.values())
    return {
        "metrics": {key: _mean([report["metrics"][key] for report in reports]) for key in METRIC_NAMES},
        "coverage": {"pass": all(report["coverage"]["pass"] for report in reports),
                     "reference_complete": all(report["coverage"]["reference_complete"] for report in reports),
                     "valid_count_by_mask": {mask: report["coverage"]["valid_count"] for mask, report in mask_reports.items()},
                     "original_rejected_count_by_mask": {mask: report["coverage"]["original_rejected_count"] for mask, report in mask_reports.items()}},
        "reference_count": reports[0]["reference_count"], "tail_count": reports[0]["tail_count"],
        "mask_count": len(reports), "masks": mask_reports,
        "status": "complete" if all(report["status"] == "complete" for report in reports) else "incomplete_mask_evidence",
        "independent_unit": "observation_stream",
    }


def _risk_direction(value, baseline):
    if value is None or baseline is None:
        return {"pass": False, "baseline_zero": None, "delta_vs_a": None, "relative_reduction": None}
    zero = baseline == 0.0
    return {"pass": value <= baseline if zero else value < baseline,
            "baseline_zero": zero, "delta_vs_a": value - baseline,
            "relative_reduction": None if zero else (baseline - value) / baseline}


def _screen_domain(summaries, method):
    offline = summaries["O"]["metrics"]["mean_error"]["mean"]
    baseline = summaries["A"]["metrics"]["mean_error"]["mean"]
    candidate = summaries[method]["metrics"]["mean_error"]["mean"]
    available = all(v is not None for v in (offline, baseline, candidate))
    improves = baseline < offline if available else None
    accuracy = {"pass": False, "a_improves_offline": improves,
                "offline_mean": offline, "a_mean": baseline, "candidate_mean": candidate,
                "five_percent_bound": None, "eighty_percent_bound": None,
                "retained_gain_fraction": None, "fallback_offline_bound": None}
    if available and improves:
        bound5 = baseline * 1.05
        bound80 = offline - 0.8 * (offline - baseline)
        accuracy.update({"pass": candidate <= bound5 and candidate <= bound80,
                         "five_percent_bound": bound5, "eighty_percent_bound": bound80,
                         "retained_gain_fraction": (offline - candidate) / (offline - baseline)})
    elif available:
        accuracy.update({"pass": candidate <= offline, "fallback_offline_bound": offline})
    risk = {key: _risk_direction(summaries[method]["metrics"][key]["mean"], summaries["A"]["metrics"][key]["mean"])
            for key in ("mean_positive_degradation", "worst_5pct_degradation")}
    coverage = summaries[method]["coverage_pass"]
    passed = accuracy["pass"] and coverage and all(value["pass"] for value in risk.values())
    reasons = []
    if not coverage:
        reasons.append("coverage_constraint_failed")
    if not accuracy["pass"]:
        reasons.append("accuracy_constraint_failed_or_unavailable")
    if not all(value["pass"] for value in risk.values()):
        reasons.append("risk_direction_failed_or_unavailable")
    return {"pass": passed, "coverage_pass": coverage, "accuracy": accuracy,
            "risk": risk, "reasons": reasons,
            "has_risk_improvement_space": any(value["baseline_zero"] is False for value in risk.values())}


def _select(domains):
    candidates = {}
    for method in ("G50", "G75"):
        domain_results = {domain: _screen_domain(domains[domain]["method_summaries"], method) for domain in DOMAINS}
        space = any(result["has_risk_improvement_space"] for result in domain_results.values())
        eligible = all(result["pass"] for result in domain_results.values()) and space
        tails = [domains[d]["method_summaries"][method]["metrics"]["worst_5pct_degradation"]["mean"] for d in DOMAINS]
        means = [domains[d]["method_summaries"][method]["metrics"]["mean_positive_degradation"]["mean"] for d in DOMAINS]
        candidates[method] = {
            "eligible_for_random_control": eligible, "domains": domain_results,
            "has_risk_improvement_space": space,
            "maximum_domain_tail": max(tails) if all(v is not None for v in tails) else None,
            "equal_domain_mean_degradation": _mean(means),
        }
    eligible = [method for method in candidates if candidates[method]["eligible_for_random_control"]]
    selected = min(eligible, key=lambda method: (candidates[method]["maximum_domain_tail"],
                                                candidates[method]["equal_domain_mean_degradation"],
                                                0 if method == "G75" else 1)) if eligible else None
    return {
        "selected_method": selected, "run_random_controls": selected is not None,
        "candidates": candidates, "status": "favorable_exploratory_trend" if selected else "no_eligible_shared_setting",
        "scope": "exploration_only_not_formal_confirmation",
        "rank_order": ["maximum_domain_tail", "equal_domain_mean_degradation", "prefer_G75_on_exact_tie"],
        "comparison_tolerance": 0.0,
        "note": "No minimum improvement percentage; zero baseline risk has no percentage improvement and cannot alone justify extra runs.",
    }


def evaluate_exploration(
    records: Sequence[Mapping[str, Any]],
    development_ids: Mapping[str, Sequence[str]],
    *,
    expected_streams: Sequence[str] = ("2021", "2022", "2023"),
    random_mask_ids: Sequence[str] = ("0", "1", "2"),
    negative_transfer_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Validate complete developer evidence and compute the frozen v2 rules.

    All grouping IDs are strings. Error is the already-computed nonnegative HIL
    hybrid error; optional raw rotation/translation must use degrees/metres.
    ``original_rejected`` is independent of extended finite-pose validity.
    R mask IDs must be frozen by the caller before random-control execution.
    No finite value is silently dropped, no method intersection is used, and
    unavailable metrics propagate as None through mask/stream summaries.
    """
    streams = _ids(expected_streams, "expected_streams", count=3)
    masks = _ids(random_mask_ids, "random_mask_ids", count=3)
    tolerance = _number(negative_transfer_tolerance, "negative_transfer_tolerance")
    if not isinstance(development_ids, Mapping) or set(development_ids) != set(DOMAINS):
        raise MetricContractError("development_ids must contain exactly lightbox and sunlamp")
    ids_by_domain = {domain: _ids(development_ids[domain], f"{domain} development_ids") for domain in DOMAINS}
    id_sets = {domain: set(ids) for domain, ids in ids_by_domain.items()}
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise MetricContractError("records must be a JSON row sequence")
    grouped = defaultdict(dict)
    has_random = False
    for row in records:
        if not isinstance(row, Mapping):
            raise MetricContractError("Each input row must be an object")
        domain, stream, method, sid = (row.get(k) for k in ("domain", "stream", "method", "sample_id"))
        if not all(isinstance(v, str) for v in (domain, stream, method, sid)):
            raise MetricContractError("domain/stream/method/sample_id must be explicit strings")
        if domain not in DOMAINS or stream not in streams or method not in (*BASE_METHODS, "R"):
            raise MetricContractError("Unexpected domain, stream or method")
        if sid not in id_sets[domain]:
            raise MetricContractError(f"{domain}: ID outside frozen developer set: {sid}")
        if method == "R":
            mask = row.get("mask")
            if not isinstance(mask, str) or mask not in masks:
                raise MetricContractError("R rows require a predeclared random mask ID")
            has_random = True
        else:
            if "mask" in row:
                raise MetricContractError("Only R rows may have a mask ID")
            mask = ""
        key = (domain, stream, method, mask)
        if sid in grouped[key]:
            raise MetricContractError(f"Duplicate row: {key} / {sid}")
        grouped[key][sid] = _normalize(row)
    methods = (*BASE_METHODS, "R") if has_random else BASE_METHODS
    for domain in DOMAINS:
        for stream in streams:
            for method in methods:
                for mask in masks if method == "R" else ("",):
                    key = (domain, stream, method, mask)
                    if set(grouped[key]) != id_sets[domain]:
                        raise MetricContractError(f"Incomplete explicit developer rows: {key}")
        offline_first = grouped[(domain, streams[0], "O", "")]
        if any(grouped[(domain, stream, "O", "")] != offline_first for stream in streams[1:]):
            raise MetricContractError(f"{domain}: Offline O must be identical across streams")

    domains = {}
    for domain in DOMAINS:
        ids = ids_by_domain[domain]
        stream_reports = {}
        for stream in streams:
            offline = grouped[(domain, stream, "O", "")]
            original_a = grouped[(domain, stream, "A", "")]
            reference = [sid for sid in ids if offline[sid]["valid"] and original_a[sid]["valid"]]
            report = {"reference_ids": reference, "reference_count": len(reference), "methods": {}}
            for method in methods:
                variants = {
                    mask: _one_method(ids, reference, offline, original_a,
                                      grouped[(domain, stream, method, mask)], tolerance)
                    for mask in (masks if method == "R" else ("",))
                }
                report["methods"][method] = _average_masks(variants) if method == "R" else variants[""]
            stream_reports[stream] = report
        summaries = {}
        for method in methods:
            per_stream = [stream_reports[stream]["methods"][method] for stream in streams]
            metrics = {key: _description([result["metrics"][key] for result in per_stream]) for key in METRIC_NAMES}
            deltas = {
                key: _description([_delta(stream_reports[stream]["methods"][method]["metrics"][key],
                                           stream_reports[stream]["methods"]["A"]["metrics"][key]) for stream in streams])
                for key in METRIC_NAMES
            }
            summaries[method] = {"metrics": metrics, "paired_difference_vs_a": deltas,
                                 "coverage_pass": all(result["coverage"]["pass"] for result in per_stream),
                                 "stream_count": len(streams), "stream_ids": list(streams),
                                 "independent_unit": "observation_stream", "stream_weight": 1 / len(streams)}
        domains[domain] = {"developer_ids": list(ids), "streams": stream_reports, "method_summaries": summaries}
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"reference": "O_and_A_valid_intersection_only",
                     "positive_degradation": "max(method_error-offline_error,0)",
                     "tail": "largest_ceil_0.05_times_reference_count_including_zeros",
                     "missing_rule": "explicit_None_and_propagate_incomplete_fixed_reference",
                     "stream_aggregation": "three_stream_equal_weight_not_pooled_images",
                     "random_mask_aggregation": "metrics_per_mask_then_equal_mean_within_stream",
                     "negative_transfer_tolerance": tolerance,
                     "strict_rate_uses_tolerance": False,
                     "selection_uses_tolerance": False,
                     "formal_inference": False},
        "domains": domains, "selection": _select(domains),
    }
    # Finite inputs can still overflow a derived ratio/bound. Such evidence is
    # an engineering failure, never a NaN/Infinity JSON value or silent None.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError) as exc:
        raise MetricContractError("Nonfinite derived metric; preserve evidence and stop") from exc
    return result
