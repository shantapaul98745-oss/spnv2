"""Independent CPU checks of saved round-two artifacts; no model forward.

The engine's validator and its self-reported success flags are not used here.
We recompute frozen G50 decisions, protection rules, reservoir transitions and
random controls from the saved records.  This verifies recorded execution; it
does not manufacture evidence about unsaved intermediate model outputs.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from guard_pools import GuardPools, assess_protection_window


ENGINE_SCHEMA = "odr003-round2-protected-engine-v1"
LEARNER_SCHEMA = "odr003-round2-window-learner-v1"
LEARNING_FIELDS = {
    "model", "buffers", "optimizer", "optimizer_type", "optimizer_parameter_names",
    "modes", "requires_grad", "gradients", "bn_features", "bn_mean", "bn_mean_sq",
    "boundary_buffers", "rng", "scaler", "counters",
}
COUNT_FIELDS = ("observed", "accepted", "candidates", "commits", "rollbacks",
                "tail_discarded", "consecutive_rollbacks", "max_consecutive_rollbacks")


class VerificationError(ValueError):
    pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _is_tensor(value: Any) -> bool:
    # Avoid importing torch until comparing an actual serialized tensor.
    return type(value).__module__.startswith("torch") and hasattr(value, "detach")


def _first_difference(a: Any, b: Any, path: str = "root") -> str | None:
    if _is_tensor(a) or _is_tensor(b):
        if not (_is_tensor(a) and _is_tensor(b)):
            return path + ": tensor/type differs"
        import torch
        if a.device.type != "cpu" or b.device.type != "cpu":
            return path + ": verifier requires CPU-loaded tensors"
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return path + ": tensor dtype/shape/value differs"
        return None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return path + ": ndarray/type differs"
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            return path + ": ndarray dtype/shape/value differs"
        return None
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)) or a.keys() != b.keys():
            return path + ": mapping keys/type differ"
        if hasattr(a, "_metadata") != hasattr(b, "_metadata"):
            return path + ": state_dict version metadata presence differs"
        if hasattr(a, "_metadata"):
            difference = _first_difference(a._metadata, b._metadata, path + "._metadata")
            if difference:
                return difference
        for key in a:
            difference = _first_difference(a[key], b[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return path + ": sequence type/length differs"
        for index, (left, right) in enumerate(zip(a, b)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if type(a) is not type(b) or a != b:
        return path + ": scalar type/value differs"
    return None


def _same(a: Any, b: Any, label: str) -> None:
    difference = _first_difference(a, b, label)
    if difference:
        raise VerificationError(difference)


def _finite_tree(value: Any, path: str) -> None:
    if _is_tensor(value):
        _require(value.device.type == "cpu", f"{path}: load checkpoint onto CPU for independent verification")
        import torch
        _require(bool(torch.isfinite(value).all()), f"{path}: nonfinite tensor")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            _require(bool(np.isfinite(value).all()), f"{path}: nonfinite array")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)):
        _require(math.isfinite(float(value)), f"{path}: nonfinite scalar")


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _numeric_cost_differences(first: Any, second: Any):
    """Only subtract values recorded on both sides; missing costs stay missing."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        result = {}
        for key in first.keys() & second.keys():
            difference = _numeric_cost_differences(first[key], second[key])
            if difference is not None and difference != {}:
                result[key] = difference
        return result
    if (type(first) in (int, float) and type(second) in (int, float)
            and math.isfinite(first) and math.isfinite(second)):
        return second - first
    return None


def _learning_contract(learning: Any) -> None:
    _require(isinstance(learning, Mapping) and LEARNING_FIELDS <= learning.keys(),
             "learning snapshot must include model, buffers, optimizer, gradients, modes, caches, RNG and counters")
    _require(learning["scaler"] is None, "the fixed FP32 protocol cannot acquire an AMP scaler")
    _require(isinstance(learning["model"], Mapping) and bool(learning["model"]), "model state is empty")
    _require(learning["gradients"].keys() == learning["requires_grad"].keys(), "gradient/parameter identities differ")
    _require(set(learning["gradients"]) <= set(learning["model"]), "gradient identity missing from model state")
    _require(all(value is None for value in learning["gradients"].values()), "final learning state has unresolved gradients")
    _require(all(type(value) is bool for value in learning["requires_grad"].values()), "requires_grad state is not boolean")
    _require(bool(learning["modes"]) and all(type(value) is bool for value in learning["modes"].values()),
             "complete module modes are required")
    _require(learning["buffers"].keys() == learning["boundary_buffers"].keys(), "buffer/boundary identities differ")
    parameter_groups = learning["optimizer_parameter_names"]
    names = [name for group in parameter_groups for name in group]
    _require(len(names) == len(set(names)) and set(names) == {name for name, active in learning["requires_grad"].items() if active},
             "optimizer does not contain exactly the permitted trainable parameters")
    optimizer_groups = learning["optimizer"]["param_groups"]
    _require(len(parameter_groups) == len(optimizer_groups)
             and all(len(names) == len(group["params"]) for names, group in zip(parameter_groups, optimizer_groups)),
             "optimizer group sizes differ from parameter identities")
    _finite_tree(learning, "learning")


def _bound_gate_decider(gate_decider):
    if gate_decider is None:
        # The caller must put the code-hash-bound first-round implementation on
        # sys.path.  No discovery, downloading, model loading, or forward occurs.
        from gate_core import decide
        return decide
    return gate_decider


def verify_final_state(state: Mapping, *, expected_stream: Sequence[Mapping],
                       calibration: Mapping | None, pool_seeds: tuple[int, int],
                       expected_identity: Mapping, expected_mode: str,
                       rw_control: Mapping | None = None,
                       rp_control: Mapping | None = None,
                       gate_decider: Callable | None = None) -> dict:
    """Verify one complete final state using externally bound protocol inputs.

    ``rw_control`` is derive_controls(...)["RW"], and ``rp_control`` its RP
    counterpart.  It is the caller's responsibility to bind them to accepted P
    counts and the frozen protocol/seed hashes; do not take them from this run.
    Expected stream is exactly the 128-prefix or all 1024 unique asset records.
    Raises VerificationError on any mismatch; never silently repairs evidence.
    """
    try:
        return _verify_final_state(state, expected_stream=expected_stream, calibration=calibration,
                                   pool_seeds=pool_seeds, expected_identity=expected_identity,
                                   expected_mode=expected_mode, rw_control=rw_control,
                                   rp_control=rp_control, gate_decider=gate_decider)
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"invalid or incomplete saved evidence: {type(exc).__name__}: {exc}") from exc


def _verify_final_state(state, *, expected_stream, calibration, pool_seeds,
                        expected_identity, expected_mode, rw_control, rp_control, gate_decider):
    _require(expected_mode in ("P", "disabled", "RW", "RP"), "unknown bound mode")
    _require(isinstance(state, Mapping), "final state must be a mapping")
    _require(state["schema"] == ENGINE_SCHEMA and state["mode"] == expected_mode, "engine schema/mode differs")
    _same(state["identity"], dict(expected_identity), "bound_identity")
    expected = _plain(list(expected_stream))
    n = len(expected)
    _require(n in (128, 1024), "only the frozen 128-prefix and N1024 are authorized")
    ids = [record["sample_id"] for record in expected]
    _require(len(set(ids)) == n, "expected stream contains duplicate observations")
    _same(state["stream_records"], expected, "frozen_stream_records")
    _require(state["finished"] is True and state["transaction"] is None and state["window_ids"] == [],
             "a final artifact must have no unresolved learning transaction")
    _require(state["next_index"] == n and state["checkpoint_every"] == 64, "final sampling/checkpoint schedule differs")
    _require(len(pool_seeds) == 2 and all(type(seed) is int and seed >= 0 for seed in pool_seeds)
             and pool_seeds[0] != pool_seeds[1], "two independent externally frozen pool seeds required")
    pools = GuardPools(*pool_seeds)
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    window = []
    seen_events = set()
    window_results = []
    comparable = Counter()
    triggered = Counter()
    triggered_windows = Counter()
    observed_index = 0
    rng = None
    p = None
    selected_mask = None
    if expected_mode == "RW":
        _require(isinstance(rw_control, Mapping) and rw_control["enabled"] is True,
                 "RW needs a defined externally bound P budget")
        q, k = rw_control["probability_denominator"], rw_control["probability_numerator"]
        _require(type(q) is int and type(k) is int and 0 <= k <= q <= 256 and q > 0,
                 "RW probability requires actual positive P candidate count")
        p = k / q
        _same(rw_control["commit_probability"], p, "bound_RW_probability_ratio")
        _same(state["rw_probability"], p, "RW_probability")
        _same(state["rw_seed"], rw_control["rng_seed"], "RW_seed")
        _require(state["rp_mask"] is None, "RW cannot acquire an RP mask")
        rng = random.Random(rw_control["rng_seed"])
    else:
        _require(state["rw_probability"] is None and state["rw_seed"] is None
                 and state["rw_rng_state"] is None, "non-RW run contains random-window state")
    if expected_mode == "RP":
        _require(isinstance(rp_control, Mapping), "RP requires an external exact-budget mask")
        positions = rp_control["selected_positions"]
        permutation = rp_control["permutation"]
        _require(len(permutation) == 1024 and set(permutation) == set(range(1024)), "invalid RP full permutation")
        permutation_rng = random.Random(rp_control["rng_seed"])
        independently_shuffled = list(range(1024))
        permutation_rng.shuffle(independently_shuffled)
        _same(permutation, independently_shuffled, "RP_frozen_permutation")
        used = rp_control["used_images"]
        _require(type(used) is int and 0 <= used <= n and used % 4 == 0,
                 "RP used images must fit an exact complete-window budget")
        _same(positions, sorted(permutation[:used]), "RP_original_order_prefix_selection")
        _require(all(index < n for index in positions), "RP selection exceeds available fixed stream")
        _require(rp_control["commits"] == used // 4, "RP commit/image budgets disagree")
        position_set = set(positions)
        selected_mask = [index in position_set for index in range(n)]
        _same(state["rp_mask"], selected_mask, "RP_actual_mask")
    elif expected_mode != "RW":
        _require(state["rp_mask"] is None, "non-RP run contains random-position state")
    if expected_mode != "RP":
        _require(isinstance(calibration, Mapping), "the frozen G50 calibration is required")
        decider = _bound_gate_decider(gate_decider)
    trace = state["scientific_trace"]
    _require(isinstance(trace, list), "scientific trace must preserve event order")
    _finite_tree(trace, "scientific_trace")
    for event in trace:
        _require(event["id"] not in seen_events, "duplicate scientific event ID")
        seen_events.add(event["id"])
        if event["kind"] == "observation":
            _require(len(window) < 4, "observation occurred before resolving a full learning window")
            _require(observed_index < n, "too many observations")
            record = expected[observed_index]
            _require(event["id"] == f"observation:{observed_index:04d}"
                     and event["observation_index"] == observed_index
                     and event["sample_id"] == record["sample_id"], "observation identity/order differs")
            _require(type(event["accepted"]) is bool, "accepted flag must be an explicit boolean")
            if expected_mode == "RP":
                decision, reason, score = selected_mask[observed_index], "random_budget_mask", None
                _same(event["arrival"], {"heat_peaks": None, "scoring_skipped": "frozen_random_position_mask"},
                      "RP_scoring_skipped_record")
            else:
                decision, reason, score = decider("G50", event["arrival"], calibration)
                decision, score = bool(decision), _plain(score)
            _same(event["accepted"], decision, f"observation[{observed_index}].recomputed_G50_or_RP")
            _same(event["gate_reason"], reason, f"observation[{observed_index}].gate_reason")
            _same(event["gate_score"], score, f"observation[{observed_index}].gate_score")
            counts["observed"] += 1
            counts["accepted"] += int(decision)
            if expected_mode != "RP":
                pools.observe(dict(sample_id=record["sample_id"], observation_index=observed_index,
                                   image_sha256=record["image_sha256"], preprocess_sha256=record["preprocess_sha256"],
                                   g50_accepted=decision, arrival=event["arrival"]))
            if decision:
                window.append(record["sample_id"])
            observed_index += 1
        elif event["kind"] == "window":
            number = counts["candidates"] + 1
            _require(len(window) == 4, "window lacks exactly four sequential accepted images")
            _require(event["id"] == f"candidate:{number:04d}" and event["number"] == number,
                     "candidate identity/sequence differs")
            _same(event["sample_ids"], window, f"candidate[{number}].accepted_input_order")
            _require(isinstance(event["fourth_loss"], (float, int)) and not isinstance(event["fourth_loss"], bool)
                     and math.isfinite(event["fourth_loss"]), "adaptation loss must remain finite")
            if expected_mode == "P":
                protected = pools.protection_records(window)
                _same(event["protection_records"], protected, f"candidate[{number}].replayed_protection_records")
                pairs = event["protection_pairs"]
                _same([pair["sample_id"] for pair in pairs], [row["sample_id"] for row in protected],
                      f"candidate[{number}].protection_pair_order")
                assessment = assess_protection_window(pairs)
                _same(event["protection"], assessment, f"candidate[{number}].recomputed_protection_rules")
                pools.record_check(event["id"], protected)
                commit = not assessment["rollback"]
                comparable.update(assessment["comparable_image_counts"])
                triggered.update(assessment["triggered_image_counts"])
                for name in assessment["triggered_image_counts"]:
                    triggered_windows[name] += int(assessment["triggered_image_counts"][name] > 0)
                _require(event["rw_coin"] is None and event["rw_probability"] is None, "P cannot use random-window decisions")
            else:
                _require(event["protection_records"] == [] and event["protection_pairs"] is None
                         and event["protection"] is None, "non-P run unexpectedly used post-update protection")
                if expected_mode == "RW":
                    coin = rng.random()
                    _same(event["rw_coin"], coin, f"candidate[{number}].independent_RW_draw")
                    _same(event["rw_probability"], p, f"candidate[{number}].RW_probability")
                    commit = coin < p
                else:
                    _require(event["rw_coin"] is None and event["rw_probability"] is None,
                             "disabled/RP must not acquire random-window decisions")
                    commit = True
            _same(event["decision"], "commit" if commit else "rollback", f"candidate[{number}].decision")
            counts["candidates"] += 1
            counts["commits" if commit else "rollbacks"] += 1
            counts["consecutive_rollbacks"] = 0 if commit else counts["consecutive_rollbacks"] + 1
            counts["max_consecutive_rollbacks"] = max(counts["max_consecutive_rollbacks"], counts["consecutive_rollbacks"])
            _same(event["counters"], counts, f"candidate[{number}].recomputed_counters")
            transition = pools.finish_window() if expected_mode != "RP" else None
            _same(event["pool_transition"], transition, f"candidate[{number}].replayed_pool_transition")
            window_results.append({"number": number, "decision": event["decision"],
                                   "checked_images": len(event["protection_records"]),
                                   "rules_triggered": event["protection"]["reasons"] if expected_mode == "P" else []})
            window = []
        else:
            raise VerificationError("unknown scientific event kind")
    _require(observed_index == n, "final trace does not cover every frozen input")
    _require(len(window) < 4, "final artifact contains an unresolved full candidate")
    counts["tail_discarded"] = len(window)
    _require(counts["candidates"] == counts["accepted"] // 4
             and len(window) == counts["accepted"] % 4, "tail/window count mismatch")
    _same(state["counters"], counts, "final_recomputed_counters")
    _same(state["pools"], pools.state_dict(), "final_exact_replayed_pool_state")
    if expected_mode == "RW":
        _same(state["rw_rng_state"], rng.getstate(), "final_RW_RNG_state")
    if expected_mode == "RP":
        _require(counts["commits"] == rp_control["commits"] and counts["rollbacks"] == 0
                 and counts["accepted"] == rp_control["used_images"] and counts["tail_discarded"] == 0,
                 "RP final exact budget mismatch")
    finish = dict(observed=n, accepted=counts["accepted"], rejected=n-counts["accepted"],
                  updates=counts["commits"], used=4*counts["commits"], tail=counts["tail_discarded"])
    _same(state["finish_counts"], finish, "final_finish_counts")
    learner = state["learner"]
    _require(learner["schema"] == LEARNER_SCHEMA and learner["batch"] == 4
             and learner["phase"] == "finished" and learner["window_size"] == 0
             and learner["window_base"] is None, "final learning state is not complete")
    _same(learner["finish_mode"], "original_adapter_finish" if expected_mode == "disabled" else "rollback_partial_window",
          "final_learning_finish_mode")
    _learning_contract(learner["learning"])
    _same(learner["learning"]["counters"], {"accepted": counts["accepted"], "observed": n,
                                           "updates": counts["commits"], "tail": counts["tail_discarded"]},
          "actual_adapter_counters")
    provider = state["provider_state"]
    _require(provider is None or isinstance(provider, Mapping), "provider state must be explicit mapping or stateless None")
    if isinstance(provider, Mapping) and "tensor_sha256" in provider:
        _require(set(provider["tensor_sha256"]) == set(ids), "provider reconstruction history includes missing/future images")
        _require(all(isinstance(value, str) and len(value) == 64 for value in provider["tensor_sha256"].values()),
                 "provider reconstruction checksum missing")
        _same(provider["manifest_sha256"], expected_identity["manifest_sha256"], "provider_manifest_identity")
    result = {
        "schema": "odr003-round2-independent-state-proof-v1", "passed": True,
        "mode": expected_mode, "identity": copy.deepcopy(dict(expected_identity)),
        "observations": n, "stream_sha256": _json_hash(expected),
        "calibration_sha256": _json_hash(calibration) if calibration is not None else None,
        "scientific_trace_sha256": _json_hash(trace), "counters": counts,
        "observations_unique_and_ordered": True, "gate_decisions_recomputed": expected_mode != "RP",
        "random_position_mask_verified": expected_mode == "RP", "pool_replay_exact": True,
        "protection_rule_evidence_recomputed": expected_mode == "P",
        "rule_comparable_image_counts": dict(comparable), "rule_triggered_image_counts": dict(triggered),
        "rule_triggered_window_counts": dict(triggered_windows),
        "pool_diagnostics": pools.diagnostics(), "window_decisions": window_results,
        "no_new_model_forward": True,
        "verification_limit": "checks saved execution records and final state; continuous/recovery and disabled/reference state comparisons are separate required proofs",
    }
    if expected_mode == "RW":
        kp, kr = rw_control["probability_numerator"], counts["commits"]
        result["RW_budget"] = dict(Q_P=rw_control["probability_denominator"], K_P=kp,
                                   Q_RW=counts["candidates"], K_RW=kr, M_P=4*kp, M_RW=4*kr,
                                   commit_difference=kr-kp, used_image_difference=4*(kr-kp),
                                   exact_commit_budget_match=kr == kp,
                                   probability_match_does_not_guarantee_budget_match=True)
    return result


def compare_continuous_resumed(continuous: Mapping, resumed: Mapping, *,
                               physical_work_continuous: Mapping | None = None,
                               physical_work_resumed: Mapping | None = None) -> dict:
    """Compare full scientific state while preserving separate physical costs.

    Each input must separately pass verify_final_state with its own bound run
    identity.  Different run IDs and checkpoint/journal boundaries are expected.
    Every other top-level field, including unknown future fields, is compared.
    """
    _require(continuous.get("finished") is True and resumed.get("finished") is True,
             "continuous/resumed artifacts must both be final")
    _learning_contract(continuous["learner"]["learning"])
    _learning_contract(resumed["learner"]["learning"])
    excluded = {"identity", "ledger_boundary"}
    left = {key: value for key, value in continuous.items() if key not in excluded}
    right = {key: value for key, value in resumed.items() if key not in excluded}
    _same(left, right, "continuous_vs_resumed_complete_scientific_state")
    return {"schema": "odr003-round2-resume-comparison-v1", "passed": True,
            "exact_fields_compared": sorted(left), "excluded_fields": sorted(excluded),
            "full_learning_tree_equal": True, "pool_state_equal": True,
            "scientific_trace_equal": True, "provider_state_equal": True,
            "identity_requirement": "each artifact separately verified against its bound identity",
            "physical_work_continuous": copy.deepcopy(physical_work_continuous),
            "physical_work_resumed": copy.deepcopy(physical_work_resumed),
            "recorded_numeric_cost_resumed_minus_continuous": _numeric_cost_differences(
                physical_work_continuous, physical_work_resumed),
            "physical_cost_equality_required": False, "no_new_model_forward": True}


def verify_disabled_equivalence(disabled_state: Mapping, reference_learning: Mapping, *,
                                reference_observation_trace: Sequence[Mapping] | None = None) -> dict:
    """Compare actual complete original-adapter learning snapshot, not a flag.

    reference_learning must be captured from the untouched original-adapter
    reference run with all LEARNING_FIELDS, not only its smaller checkpoint().
    """
    _require(disabled_state.get("mode") == "disabled" and disabled_state.get("finished") is True,
             "protection-disabled final state required")
    actual = disabled_state["learner"]["learning"]
    _learning_contract(reference_learning)
    _learning_contract(actual)
    _same(actual, reference_learning, "disabled_vs_original_reference_full_learning_tree")
    if reference_observation_trace is not None:
        actual_trace = [row for row in disabled_state["scientific_trace"] if row["kind"] == "observation"]
        _same(actual_trace, list(reference_observation_trace), "disabled_vs_reference_observation_trace")
    return {"schema": "odr003-round2-disabled-equivalence-v1", "passed": True,
            "full_learning_tree_equal": True, "learning_fields_compared": sorted(actual),
            "observation_trace_compared": reference_observation_trace is not None,
            "success_flags_used_as_evidence": False, "no_new_model_forward": True}
