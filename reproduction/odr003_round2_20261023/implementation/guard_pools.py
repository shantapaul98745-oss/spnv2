"""Label-free G50-P historical reservoirs and fixed protection predicates.

This module has no model, filesystem, GT, or GPU access. Normal learning-window
rollback must NOT roll back this object's observation history. Full checkpoint
recovery restores ``state_dict`` together with the learning state and cursor.
All IDs below belong to the already observed frozen stream.
"""
from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "odr003-round2-guard-pools-v1"
RULE_SCHEMA = "odr003-round2-protection-rules-v1"
SOURCE_IMAGE_DIAGONAL = math.hypot(1920, 1200)
FIXED_RULES = {
    "h_rotation_change_degrees": 15.0,
    "he_disagreement_increase_degrees": 10.0,
    "reprojection_relative_increase": 0.25,
    "reprojection_absolute_increase": 1.0 / SOURCE_IMAGE_DIAGONAL,
}


def _plain(value: Any) -> Any:
    """A fresh JSON-safe tree; reject non-finite numerical metadata."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite protection metadata")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Protection metadata keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    raise TypeError(f"Unsupported protection metadata type: {type(value).__name__}")


def _rng_tuple(value: Any) -> Any:
    return tuple(_rng_tuple(v) for v in value) if isinstance(value, list) else value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _reject_gt_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(key.lower() in {"gt", "ground_truth", "groundtruth", "quaternion_gt",
                               "translation_gt", "developer_label", "target_label"}
               for key in value):
            raise ValueError("GT is forbidden in protection records")
        for child in value.values():
            _reject_gt_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_gt_keys(child)


def decode_arrival_fragility(peaks: Sequence[float] | None, *,
                             threshold: float = 0.5,
                             minimum_points: int = 5,
                             max_points: int = 11) -> dict:
    """Replay only the original float32 visibility selection, never solve PnP.

    The original decoder compares ``maxVal > threshold`` while fewer than five
    points exist and the Python-float threshold is >= 0, then subtracts 0.1.
    The scalar comparison itself occurs in float32. We retain the *last used*
    threshold, not the decremented next threshold. Enough visible points do not
    establish that PnP solved. Proximity also uses explicit float32 arithmetic.
    Missing peaks / no actual comparison have None, never synthetic zero counts.
    """
    if minimum_points != 5 or max_points != 11:
        raise ValueError("The frozen original protocol uses five of eleven points")
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)):
        raise ValueError("Threshold must be finite")
    initial = float(threshold)
    if initial > 1.0:
        raise ValueError("Unsupported heatmap threshold; frozen threshold is 0.5")
    result = dict(initial_threshold=initial, effective_threshold=None,
                  effective_threshold_float32=None, next_threshold=initial,
                  comparison_dtype="float32", comparison="strict_gt",
                  proximity_dtype="float32", proximity_limit=0.05,
                  used_indices=None, used_count=None, enough_points=None,
                  pnp_solved=None, near_threshold_indices=None,
                  fragile=None, fragility_comparable=False, reasons=[])
    if peaks is None:
        result["reasons"] = ["heat_peaks_unavailable"]
        return result
    raw = np.asarray(peaks)
    if raw.shape != (11,):
        raise ValueError("Exactly eleven arrival heatmap peaks required")
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite raw heatmap peaks")
    values = raw.astype(np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("Heatmap peak overflow on float32 conversion")
    visible: np.ndarray | None = None
    current = initial
    while (visible is None or len(visible) < 5) and current >= 0:
        effective = current
        visible = np.flatnonzero(values > np.float32(effective))
        current -= 0.1
        result["effective_threshold"] = effective
        result["effective_threshold_float32"] = float(np.float32(effective))
    result["next_threshold"] = current
    if visible is None:
        result["reasons"] = ["no_effective_threshold"]
        return result
    near = np.flatnonzero(
        np.abs(values - np.float32(result["effective_threshold"])) <= np.float32(0.05)
    )
    count = len(visible)
    reasons = []
    if count <= 6:
        reasons.append("used_keypoints_le_6")
    if len(near):
        reasons.append("peak_near_effective_threshold")
    if count < 5:
        reasons.append("not_solvable_fewer_than_5_points")
    result.update(used_indices=visible.tolist(), used_count=count,
                  enough_points=count >= 5,
                  pnp_solved=False if count < 5 else None,
                  near_threshold_indices=near.tolist(),
                  fragile=bool(count <= 6 or len(near)),
                  fragility_comparable=True, reasons=reasons)
    return result


class _Reservoir:
    def __init__(self, seed: int, capacity: int = 4):
        self.seed = _integer(seed, "reservoir seed")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.slots: list[str] = []
        self.candidate_count = 0
        self.replacement_count = 0

    def add(self, sample_id: str) -> dict:
        self.candidate_count += 1
        count = self.candidate_count
        replaced = None
        slot = None
        draw = None
        if len(self.slots) < self.capacity:
            slot = len(self.slots)
            self.slots.append(sample_id)
        else:
            draw = self.rng.randrange(count)
            if draw < self.capacity:
                slot = draw
                replaced = self.slots[slot]
                self.slots[slot] = sample_id
                self.replacement_count += 1
        return dict(sample_id=sample_id, candidate_index=count,
                    selected=slot is not None, slot=slot, draw=draw,
                    replaced_sample_id=replaced)

    def state_dict(self) -> dict:
        return _plain(dict(seed=self.seed, capacity=self.capacity,
                           slots=self.slots, candidate_count=self.candidate_count,
                           replacement_count=self.replacement_count,
                           rng_state=self.rng.getstate()))

    @classmethod
    def from_state(cls, state: Mapping) -> "_Reservoir":
        if state["capacity"] != 4:
            raise ValueError("The frozen reservoir capacity is four")
        out = cls(state["seed"], state["capacity"])
        out.candidate_count = _integer(state["candidate_count"], "candidate_count")
        out.replacement_count = _integer(state["replacement_count"], "replacement_count")
        slots = list(state["slots"])
        if (len(slots) != min(out.capacity, out.candidate_count)
                or len(set(slots)) != len(slots)
                or out.replacement_count > max(0, out.candidate_count - out.capacity)):
            raise ValueError("Inconsistent reservoir checkpoint counters")
        out.slots = slots
        out.rng.setstate(_rng_tuple(state["rng_state"]))
        return out


class GuardPools:
    """Independent uniform/fragile historical reservoirs with deferred insertion.

    ``observe`` records each arrival once. ``protection_records`` is pure and
    uses only the existing historical slots plus four current accepted IDs.
    Call ``finish_window`` after either commit or normal rollback. It inserts
    every pending observation, including G50 rejects, in arrival order.

    Required record fields: sample_id, observation_index, image_sha256,
    preprocess_sha256, g50_accepted (bool), arrival (label-free prediction dict).
    ``arrival.heat_peaks`` supplies the arrival-time fragile qualification.
    """
    def __init__(self, uniform_seed: int, fragile_seed: int, *,
                 capacity: int = 4, heatmap_threshold: float = 0.5):
        if capacity != 4 or heatmap_threshold != 0.5:
            raise ValueError("This protocol freezes pool size four and threshold 0.5")
        if uniform_seed == fragile_seed:
            raise ValueError("Historical pools require distinct independent seeds")
        self.heatmap_threshold = float(heatmap_threshold)
        self.uniform = _Reservoir(uniform_seed, capacity)
        self.fragile = _Reservoir(fragile_seed, capacity)
        self.records: dict[str, dict] = {}
        self.pending: list[str] = []
        self.last_observation_index = -1
        self.completed_decisions = 0
        self.check_events: dict[str, list[str]] = {}

    def observe(self, record: Mapping) -> dict:
        value = _plain(record)
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Nonempty sample_id required")
        if sample_id in self.records:
            raise ValueError("Same frozen-stream image cannot be observed twice")
        position = _integer(value.get("observation_index"), "observation_index")
        if position <= self.last_observation_index:
            raise ValueError("Observation positions must increase strictly")
        for name in ("image_sha256", "preprocess_sha256"):
            if not isinstance(value.get(name), str) or not re.fullmatch("[0-9a-f]{64}", value[name]):
                raise ValueError(f"A bound lowercase SHA256 is required for {name}")
        if type(value.get("g50_accepted")) is not bool:
            raise ValueError("Arrival G50 decision must be an explicit bool")
        arrival = value.get("arrival")
        if not isinstance(arrival, dict):
            raise ValueError("Arrival label-free prediction record required")
        _reject_gt_keys(value)
        value["fragility"] = decode_arrival_fragility(
            arrival.get("heat_peaks"), threshold=self.heatmap_threshold
        )
        self.records[sample_id] = value
        self.pending.append(sample_id)
        self.last_observation_index = position
        return copy.deepcopy(value)

    def protection_records(self, current_window: Sequence[str]) -> list[dict]:
        ids = list(current_window)
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("Protection is checked only for four distinct accepted arrivals")
        for sample_id in ids:
            if sample_id not in self.pending:
                raise ValueError("Current-window ID must already be observed and still pending")
            if not self.records[sample_id]["g50_accepted"]:
                raise ValueError("A G50 rejected arrival cannot enter the learning window")
        positions = [self.records[sample_id]["observation_index"] for sample_id in ids]
        if positions != sorted(positions):
            raise ValueError("Current-window IDs must retain observation order")
        accepted_pending = [sample_id for sample_id in self.pending
                            if self.records[sample_id]["g50_accepted"]]
        if ids != accepted_pending:
            raise ValueError("A protection window cannot omit pending accepted arrivals")
        chosen: dict[str, dict] = {}
        for origin, source in (("current_window", ids), ("uniform", self.uniform.slots),
                               ("fragile", self.fragile.slots)):
            for slot, sample_id in enumerate(source):
                if sample_id not in self.records:
                    raise ValueError("Future or unknown image in a protection reservoir")
                if sample_id not in chosen:
                    chosen[sample_id] = copy.deepcopy(self.records[sample_id])
                    chosen[sample_id]["protection_sources"] = []
                chosen[sample_id]["protection_sources"].append(dict(pool=origin, slot=slot))
        return list(chosen.values())

    def record_check(self, event_id: str, records: Sequence[Mapping]) -> None:
        """Idempotent bookkeeping after an actual pair check, not during selection."""
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Stable check event_id required")
        ids = [record["sample_id"] for record in records]
        if (not 1 <= len(ids) <= 12 or len(ids) != len(set(ids))
                or any(sample_id not in self.records for sample_id in ids)):
            raise ValueError("Protection check list has duplicate or future images")
        if event_id in self.check_events:
            if self.check_events[event_id] != ids:
                raise ValueError("A persisted check ID cannot acquire a different list")
            return
        self.check_events[event_id] = ids

    def finish_window(self) -> dict:
        """Advance history after either decision; never restore this on rollback."""
        if not self.pending:
            raise ValueError("Cannot finalize the same empty observation window twice")
        events = []
        for sample_id in self.pending:
            value = self.records[sample_id]
            events.append(dict(pool="uniform", **self.uniform.add(sample_id)))
            if value["fragility"]["fragile"] is True:
                events.append(dict(pool="fragile", **self.fragile.add(sample_id)))
        inserted = list(self.pending)
        self.pending.clear()
        self.completed_decisions += 1
        return dict(inserted_observation_ids=inserted, pool_events=events,
                    decision_index=self.completed_decisions)

    def diagnostics(self) -> dict:
        checked = [sample_id for ids in self.check_events.values() for sample_id in ids]
        unique = set(checked)
        return dict(observed_count=len(self.records), pending_count=len(self.pending),
                    uniform_size=len(self.uniform.slots), fragile_size=len(self.fragile.slots),
                    uniform_candidates=self.uniform.candidate_count,
                    fragile_candidates=self.fragile.candidate_count,
                    uniform_replacements=self.uniform.replacement_count,
                    fragile_replacements=self.fragile.replacement_count,
                    check_events=len(self.check_events), checked_images=len(checked),
                    checked_unique_images=len(unique), repeated_checks=len(checked) - len(unique),
                    checked_unique_g50_rejected=sum(not self.records[x]["g50_accepted"] for x in unique),
                    fragility_unavailable=sum(not x["fragility"]["fragility_comparable"]
                                              for x in self.records.values()),
                    no_future_inputs=True)

    def state_dict(self) -> dict:
        return _plain(dict(schema=SCHEMA, heatmap_threshold=self.heatmap_threshold,
                           uniform=self.uniform.state_dict(), fragile=self.fragile.state_dict(),
                           records=self.records, pending=self.pending,
                           last_observation_index=self.last_observation_index,
                           completed_decisions=self.completed_decisions,
                           check_events=self.check_events))

    def load_state_dict(self, state: Mapping) -> None:
        """Validate a complete state first, then replace atomically in memory."""
        value = _plain(state)
        if value.get("schema") != SCHEMA or value.get("heatmap_threshold") != 0.5:
            raise ValueError("Guard pool checkpoint schema/protocol mismatch")
        candidate = GuardPools(value["uniform"]["seed"], value["fragile"]["seed"])
        records = value["records"]
        if not isinstance(records, dict):
            raise ValueError("Checkpoint records must be a map")
        ordered = sorted(records.values(), key=lambda record: record["observation_index"])
        for record in ordered:
            restored = candidate.observe(record)
            if restored != record or records.get(record["sample_id"]) != record:
                raise ValueError("Arrival/fragility checkpoint metadata mismatch")
        pending = value["pending"]
        if not isinstance(pending, list) or len(set(pending)) != len(pending):
            raise ValueError("Pending checkpoint contains duplicates")
        expected_tail = [record["sample_id"] for record in ordered][-len(pending):] if pending else []
        if pending != expected_tail:
            raise ValueError("Pending observations must be an ordered suffix")
        candidate.pending = pending
        candidate.uniform = _Reservoir.from_state(value["uniform"])
        candidate.fragile = _Reservoir.from_state(value["fragile"])
        processed = {record["sample_id"] for record in ordered} - set(pending)
        if (candidate.uniform.candidate_count != len(processed)
                or candidate.fragile.candidate_count != sum(
                    candidate.records[x]["fragility"]["fragile"] is True for x in processed)):
            raise ValueError("Reservoir candidate counts disagree with processed arrivals")
        if any(x not in processed for x in candidate.uniform.slots + candidate.fragile.slots):
            raise ValueError("Checkpoint reservoir includes a pending/future observation")
        if any(candidate.records[x]["fragility"]["fragile"] is not True for x in candidate.fragile.slots):
            raise ValueError("A non-fragile arrival entered the fragile pool")
        if candidate.last_observation_index != value["last_observation_index"]:
            raise ValueError("Checkpoint observation cursor mismatch")
        candidate.completed_decisions = _integer(value["completed_decisions"], "completed_decisions")
        for event_id, ids in value["check_events"].items():
            candidate.record_check(event_id, [candidate.records[sample_id] for sample_id in ids])
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)


def _optional_finite(value: Any, name: str, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a scalar, not bool")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"Nonfinite protection signal: {name}")
    if nonnegative and scalar < 0:
        raise ValueError(f"Negative protection signal: {name}")
    return scalar


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be bool or None")
    return value


def _quaternion_alignment(first: Any, second: Any) -> float | None:
    quaternions = []
    for value in (first, second):
        if value is None:
            quaternions.append(None)
            continue
        q = np.asarray(value, dtype=np.float64)
        if q.shape != (4,):
            raise ValueError("Quaternion shape must be (4,)")
        if not np.isfinite(q).all():
            raise FloatingPointError("Nonfinite H quaternion")
        norm = float(np.linalg.norm(q))
        if norm <= 1e-12 or not math.isfinite(norm):
            raise ValueError("Invalid H quaternion norm")
        quaternions.append(q / norm)
    if any(q is None for q in quaternions):
        return None
    dot = abs(float(np.dot(quaternions[0], quaternions[1])))
    return min(1.0, max(0.0, dot))


def quaternion_distance_degrees(first: Any, second: Any) -> float | None:
    """Sign-invariant shortest SO(3) angle for scalar-first quaternions."""
    dot = _quaternion_alignment(first, second)
    return None if dot is None else math.degrees(2.0 * math.acos(dot))


def protection_signals(prediction: Mapping) -> dict:
    """Explicit adapter for existing original-A prediction records.

    Canonical keys are also accepted. No effective pose is substituted for H.
    A failed E-only availability does not silently become a PnP failure.
    """
    legal = prediction.get("legal", prediction.get("pose_legal"))
    pnp = prediction.get("pnp_solved", prediction.get("h_available"))
    if "physical_valid" in prediction:
        physical = prediction["physical_valid"]
    elif prediction.get("original_h_rejected") is not None:
        physical = not _optional_bool(prediction["original_h_rejected"], "original_h_rejected")
    else:
        physical = None
    return dict(legal=_optional_bool(legal, "legal"),
                pnp_solved=_optional_bool(pnp, "pnp_solved"),
                physical_valid=_optional_bool(physical, "physical_valid"),
                h_quaternion_wxyz=prediction.get("h_quaternion_wxyz", prediction.get("heat_q")),
                he_rotation_deg=_optional_finite(
                    prediction.get("he_rotation_deg", prediction.get("rotation_disagreement")),
                    "he_rotation_deg", nonnegative=True),
                reprojection_normalized=_optional_finite(
                    prediction.get("reprojection_normalized", prediction.get("reprojection")),
                    "reprojection_normalized", nonnegative=True))


def _condition(value: float | None, threshold: float) -> dict:
    return dict(value=value, threshold=threshold, operator="strict_gt",
                comparable=value is not None,
                triggered=None if value is None else bool(value > threshold))


def assess_protection_pair(sample_id: str, pre: Mapping, post: Mapping) -> dict:
    """Evaluate both fixed rules on one pre/post pair without state changes."""
    before, after = protection_signals(pre), protection_signals(post)
    prelegal = before["legal"]
    post_invalid = (after["pnp_solved"] is False or after["physical_valid"] is False)
    post_valid = (after["pnp_solved"] is True and after["physical_valid"] is True)
    comparable1 = (prelegal is False or (prelegal is True and (post_invalid or post_valid)))
    rule1 = bool(prelegal is True and post_invalid) if comparable1 else None
    reasons = []
    if rule1:
        if after["pnp_solved"] is False:
            reasons.append("previously_legal_pnp_unavailable")
        if after["physical_valid"] is False:
            reasons.append("previously_legal_physical_rejection")
    rotation_change = quaternion_distance_degrees(before["h_quaternion_wxyz"], after["h_quaternion_wxyz"])
    disagreement_delta = None if (before["he_rotation_deg"] is None or after["he_rotation_deg"] is None) else after["he_rotation_deg"] - before["he_rotation_deg"]
    reprojection_delta = None if (before["reprojection_normalized"] is None or after["reprojection_normalized"] is None) else after["reprojection_normalized"] - before["reprojection_normalized"]
    reprojection_limit = None if before["reprojection_normalized"] is None else max(
        FIXED_RULES["reprojection_relative_increase"] * before["reprojection_normalized"],
        FIXED_RULES["reprojection_absolute_increase"])
    conditions = dict(h_rotation_change=_condition(rotation_change, 15.0),
                      he_disagreement_increase=_condition(disagreement_delta, 10.0),
                      reprojection_increase=dict(value=reprojection_delta, threshold=reprojection_limit,
                          operator="strict_gt", comparable=reprojection_delta is not None and reprojection_limit is not None,
                          triggered=None if reprojection_delta is None or reprojection_limit is None else bool(reprojection_delta > reprojection_limit)))
    # Compare equivalent native quantities at the boundary. Inverting cosine
    # can print an exact 15-degree construction as 15.000000000000027; subtracting
    # .0125 - .01 can similarly move a mathematical 25% equality above the limit.
    # These strict comparisons add no tolerance or modified research threshold.
    alignment = _quaternion_alignment(before["h_quaternion_wxyz"], after["h_quaternion_wxyz"])
    cosine_limit = math.cos(math.radians(15.0) / 2.0)
    conditions["h_rotation_change"].update(
        operator="abs_normalized_quaternion_dot < cos(threshold/2)",
        compared_value=alignment, compared_threshold=cosine_limit,
        triggered=None if alignment is None else alignment < cosine_limit)
    for key, source_key, limit in (("he_disagreement_increase", "he_rotation_deg", 10.0),
                                  ("reprojection_increase", "reprojection_normalized", reprojection_limit)):
        before_value, after_value = before[source_key], after[source_key]
        post_limit = None if before_value is None or limit is None else before_value + limit
        conditions[key].update(operator="post > pre + threshold", compared_value=after_value,
                               compared_threshold=post_limit,
                               triggered=None if after_value is None or post_limit is None else after_value > post_limit)
    comparable2 = all(item["comparable"] for item in conditions.values())
    rule2 = all(item["triggered"] for item in conditions.values()) if comparable2 else None
    if rule2:
        reasons.append("rotation_jump_with_disagreement_and_reprojection_worsening")
    # Explanatory records may be absent; they never independently veto an update.
    pre_points = pre.get("used_indices", pre.get("pnp_indices"))
    post_points = post.get("used_indices", post.get("pnp_indices"))
    return _plain(dict(schema=RULE_SCHEMA, sample_id=sample_id,
                       rollback=bool(rule1 or rule2), reasons=reasons,
                       rules=dict(new_unavailable=dict(comparable=comparable1, triggered=rule1,
                           pre_legal=prelegal, post_pnp_solved=after["pnp_solved"],
                           post_physical_valid=after["physical_valid"]),
                           joint_geometry_worsening=dict(comparable=comparable2, triggered=rule2,
                                                        subconditions=conditions)),
                       denominators={"new_unavailable": int(comparable1),
                                     "joint_geometry_worsening": int(comparable2),
                                     **{name: int(item["comparable"]) for name, item in conditions.items()}},
                       pre=before, post=after,
                       point_set_changed=None if pre_points is None or post_points is None else set(pre_points) != set(post_points)))


def assess_protection_window(pairs: Sequence[Mapping]) -> dict:
    """pairs: ordered dictionaries containing sample_id, pre, post."""
    ids = [pair["sample_id"] for pair in pairs]
    if len(ids) != len(set(ids)) or not 1 <= len(ids) <= 12:
        raise ValueError("A protection window requires one to twelve unique observed IDs")
    rows = [assess_protection_pair(pair["sample_id"], pair["pre"], pair["post"]) for pair in pairs]
    denominators: Counter = Counter()
    triggered: Counter = Counter()
    for row in rows:
        denominators.update(row["denominators"])
        for name, rule in row["rules"].items():
            triggered[name] += int(rule["triggered"] is True)
            for subname, condition in rule.get("subconditions", {}).items():
                triggered[subname] += int(condition["triggered"] is True)
    reasons = list(dict.fromkeys(reason for row in rows for reason in row["reasons"]))
    return dict(schema=RULE_SCHEMA, rollback=any(row["rollback"] for row in rows),
                checked_image_count=len(rows), rollback_image_count=sum(row["rollback"] for row in rows),
                rollback_window_count=int(any(row["rollback"] for row in rows)),
                rules=dict(FIXED_RULES), reasons=reasons, rows=rows,
                comparable_image_counts=dict(denominators), triggered_image_counts=dict(triggered))
