"""Independent-verifier contracts over synthetic saved records; no forward."""
from __future__ import annotations

import copy
from collections import OrderedDict
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

BASELINE = Path(__file__).resolve().parents[2] / "odr003_exploratory_20261023" / "implementation"
sys.path.insert(0, str(BASELINE))

from gate_core import decide
from guard_pools import GuardPools, assess_protection_window
from round2_protocol import derive_controls
from verify_round2 import (
    VerificationError, compare_continuous_resumed, verify_disabled_equivalence,
    verify_final_state,
)


IDENTITY = dict(run_id="synthetic-artifact-only", source_sha256="a"*64,
                config_sha256="b"*64, manifest_sha256="c"*64)
CALIBRATION = {"references": [[0., .5, 1.], [-1., -.1, 0.], [-20., -10., 0.], [-1., -.1, 0.]],
               "thresholds": {"G50": .5}}
POOL_SEEDS = (2026102311, 2026102312)


def signal(index=1):
    peaks = [.51]*5+[.2]*6 if index % 5 == 0 else [.9]*11
    return dict(geometry_scoreable=True, geometry_reason="ok", confidence=0. if index % 7 == 0 else .9,
                reprojection=.01, rotation_disagreement=2., translation_disagreement=.01,
                pose_legal=True, h_available=True, e_available=True,
                original_h_rejected=False, heat_q=[1., 0., 0., 0.],
                heat_peaks=peaks, used_indices=list(range(5 if index % 5 == 0 else 11)),
                used_count=5 if index % 5 == 0 else 11, effective_heat_threshold=.5)


def complete_learning(counts):
    return {
        "model": {"weight": np.array([1., 2.], dtype=np.float32)},
        "buffers": {"scratch": np.array([3.], dtype=np.float32)},
        "optimizer": {"state": {0: {"step": counts["commits"], "exp_avg": np.array([.1, .2])}},
                      "param_groups": [{"lr": 1e-4, "params": [0]}]},
        "optimizer_type": "synthetic.AdamW", "optimizer_parameter_names": [["weight"]],
        "modes": {"": False}, "requires_grad": {"weight": True}, "gradients": {"weight": None},
        "bn_features": {}, "bn_mean": {}, "bn_mean_sq": {},
        "boundary_buffers": {"scratch": np.array([3.], dtype=np.float32)},
        "rng": {"python": random.Random(29).getstate(), "numpy": np.random.RandomState(41).get_state(),
                "cpu": np.array([1, 2, 3], dtype=np.uint8), "cuda": []},
        "scaler": None,
        "counters": {"accepted": counts["accepted"], "observed": counts["observed"],
                     "updates": counts["commits"], "tail": counts["tail_discarded"]},
    }


def synthetic_final(mode="P", n=128, *, rw_control=None, rp_control=None):
    """Build records/state only.  It does not construct or call any model."""
    stream = [dict(sample_id=f"s{i:04d}", image_sha256=f"{i:064x}", preprocess_sha256="d"*64)
              for i in range(n)]
    pools = GuardPools(*POOL_SEEDS)
    counts = dict(observed=0, accepted=0, candidates=0, commits=0, rollbacks=0,
                  tail_discarded=0, consecutive_rollbacks=0, max_consecutive_rollbacks=0)
    trace, window = [], []
    rng = random.Random(rw_control["rng_seed"]) if mode == "RW" else None
    p = rw_control["commit_probability"] if mode == "RW" else None
    mask = [i in set(rp_control["selected_positions"]) for i in range(n)] if mode == "RP" else None
    for index, row in enumerate(stream):
        if mode == "RP":
            arrival = {"heat_peaks": None, "scoring_skipped": "frozen_random_position_mask"}
            accepted, reason, score = mask[index], "random_budget_mask", None
        else:
            arrival = signal(index)
            accepted, reason, score = decide("G50", arrival, CALIBRATION)
        counts["observed"] += 1
        counts["accepted"] += int(accepted)
        if mode != "RP":
            pools.observe(dict(**row, observation_index=index, g50_accepted=accepted, arrival=arrival))
        trace.append(dict(id=f"observation:{index:04d}", kind="observation", sample_id=row["sample_id"],
                          observation_index=index, accepted=accepted, gate_reason=reason,
                          gate_score=score, arrival=arrival))
        if not accepted:
            continue
        window.append(row["sample_id"])
        if len(window) < 4:
            continue
        counts["candidates"] += 1
        number = counts["candidates"]
        key = f"candidate:{number:04d}"
        records, pairs, assessment, coin = [], None, None, None
        if mode == "P":
            records = pools.protection_records(window)
            pairs = []
            for position, record in enumerate(records):
                before, after = signal(1), signal(1)
                if position == 0 and number in (2, 5):
                    after.update(pose_legal=False, h_available=False, original_h_rejected=True,
                                 heat_q=None, rotation_disagreement=None, reprojection=None)
                if position == 0 and number == 7:
                    angle = math.radians(45.)/2
                    after.update(heat_q=[math.cos(angle), math.sin(angle), 0., 0.],
                                 rotation_disagreement=30., reprojection=.02)
                pairs.append(dict(sample_id=record["sample_id"], pre=before, post=after))
            assessment = assess_protection_window(pairs)
            pools.record_check(key, records)
            commit = not assessment["rollback"]
        elif mode == "RW":
            coin = rng.random()
            commit = coin < p
        else:
            commit = True
        counts["commits" if commit else "rollbacks"] += 1
        counts["consecutive_rollbacks"] = 0 if commit else counts["consecutive_rollbacks"]+1
        counts["max_consecutive_rollbacks"] = max(counts["max_consecutive_rollbacks"], counts["consecutive_rollbacks"])
        transition = pools.finish_window() if mode != "RP" else None
        trace.append(dict(id=key, kind="window", number=number, sample_ids=list(window),
                          decision="commit" if commit else "rollback", fourth_loss=.1,
                          protection_records=records, protection_pairs=pairs, protection=assessment,
                          rw_coin=coin, rw_probability=p, pool_transition=transition, counters=dict(counts)))
        window = []
    counts["tail_discarded"] = len(window)
    finish = dict(observed=n, accepted=counts["accepted"], rejected=n-counts["accepted"],
                  updates=counts["commits"], used=4*counts["commits"], tail=len(window))
    state = dict(schema="odr003-round2-protected-engine-v1", identity=dict(IDENTITY), mode=mode,
                 stream_records=stream, learner=dict(schema="odr003-round2-window-learner-v1",
                     batch=4, phase="finished", window_size=0, window_base=None,
                     finish_mode="original_adapter_finish" if mode == "disabled" else "rollback_partial_window",
                     learning=complete_learning(counts)), pools=pools.state_dict(),
                 provider_state={"manifest_sha256": IDENTITY["manifest_sha256"],
                                 "tensor_sha256": {row["sample_id"]: "e"*64 for row in stream}},
                 counters=counts, next_index=n, window_ids=[], transaction=None,
                 rw_rng_state=rng.getstate() if rng else None, rw_probability=p,
                 rw_seed=rw_control["rng_seed"] if mode == "RW" else None, rp_mask=mask,
                 scientific_trace=trace, finish_counts=finish, finished=True,
                 checkpoint_every=64, ledger_boundary={"sequence": 1, "byte_offset": 100})
    return state


class SavedArtifactVerifierContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good_p = synthetic_final()
        cls.good_disabled = synthetic_final("disabled")
        cls.controls = derive_controls("lightbox", 2021, 200, 75)
        cls.good_rw = synthetic_final("RW", 1024, rw_control=cls.controls["RW"])
        cls.good_rp = synthetic_final("RP", 1024, rp_control=cls.controls["RP"])

    def verify(self, state, *, rw=None, rp=None, **kwargs):
        return verify_final_state(state, expected_stream=kwargs.pop("expected_stream", state["stream_records"]),
                                  calibration=CALIBRATION, pool_seeds=POOL_SEEDS,
                                  expected_identity=IDENTITY, expected_mode=state["mode"],
                                  rw_control=rw, rp_control=rp, **kwargs)

    def test_P_recomputes_both_rules_pools_gates_counts_and_tail(self):
        proof = self.verify(self.good_p)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["observations"], 128)
        self.assertEqual(proof["counters"]["rollbacks"], 3)
        self.assertEqual(proof["counters"]["tail_discarded"], 1)
        self.assertEqual(proof["rule_triggered_window_counts"]["new_unavailable"], 2)
        self.assertEqual(proof["rule_triggered_window_counts"]["joint_geometry_worsening"], 1)
        self.assertTrue(proof["pool_replay_exact"])
        self.assertTrue(proof["no_new_model_forward"])

    def test_passing_flag_cannot_override_forged_gate_record(self):
        state = copy.deepcopy(self.good_p)
        state["independent_verified"] = True
        event = next(r for r in state["scientific_trace"] if r["kind"] == "observation")
        event["accepted"] = not event["accepted"]
        with self.assertRaisesRegex(VerificationError, "recomputed_G50"):
            self.verify(state)

    def test_gate_reason_and_scalar_score_are_checked(self):
        for field, value in (("gate_reason", "all"), ("gate_score", .777)):
            state = copy.deepcopy(self.good_p)
            state["scientific_trace"][0][field] = value
            with self.subTest(field=field), self.assertRaises(VerificationError):
                self.verify(state)

    def test_frozen_stream_asset_and_order_bound(self):
        expected = copy.deepcopy(self.good_p["stream_records"])
        expected[0]["image_sha256"] = "f"*64
        with self.assertRaisesRegex(VerificationError, "frozen_stream_records"):
            self.verify(self.good_p, expected_stream=expected)
        expected = copy.deepcopy(self.good_p["stream_records"])
        expected[0], expected[1] = expected[1], expected[0]
        with self.assertRaises(VerificationError):
            self.verify(self.good_p, expected_stream=expected)

    def test_window_cannot_reorder_or_substitute_accepted_inputs(self):
        state = copy.deepcopy(self.good_p)
        event = next(r for r in state["scientific_trace"] if r["kind"] == "window")
        event["sample_ids"].reverse()
        with self.assertRaisesRegex(VerificationError, "accepted_input_order"):
            self.verify(state)

    def test_unobserved_future_protection_image_is_rejected(self):
        state = copy.deepcopy(self.good_p)
        event = next(r for r in state["scientific_trace"] if r["kind"] == "window")
        event["protection_records"][0]["sample_id"] = "s0127"
        with self.assertRaisesRegex(VerificationError, "protection_records"):
            self.verify(state)

    def test_rule_self_report_not_trusted(self):
        state = copy.deepcopy(self.good_p)
        event = next(r for r in state["scientific_trace"] if r["kind"] == "window" and r["number"] == 2)
        event["protection"]["rollback"] = False
        event["decision"] = "commit"
        with self.assertRaisesRegex(VerificationError, "recomputed_protection_rules"):
            self.verify(state)

    def test_saved_pair_tamper_cannot_keep_previous_assessment(self):
        state = copy.deepcopy(self.good_p)
        event = next(r for r in state["scientific_trace"] if r["kind"] == "window" and r["number"] == 7)
        event["protection_pairs"][0]["post"]["reprojection"] = .01
        with self.assertRaisesRegex(VerificationError, "recomputed_protection_rules"):
            self.verify(state)

    def test_pool_replacement_pending_and_rng_state_replayed(self):
        for kind in ("rng", "pending", "candidate_count", "transition"):
            state = copy.deepcopy(self.good_p)
            if kind == "rng":
                state["pools"]["uniform"]["rng_state"] = list(random.Random(9).getstate())
            elif kind == "pending":
                state["pools"]["pending"] = []
            elif kind == "candidate_count":
                state["pools"]["uniform"]["candidate_count"] += 1
            else:
                event = next(r for r in state["scientific_trace"] if r["kind"] == "window")
                event["pool_transition"]["inserted_observation_ids"].reverse()
            with self.subTest(kind=kind), self.assertRaises(VerificationError):
                self.verify(state)

    def test_actual_adapter_counters_and_finish_counts_not_only_trace(self):
        for target in ("adapter", "finish", "tail", "window_counter"):
            state = copy.deepcopy(self.good_p)
            if target == "adapter":
                state["learner"]["learning"]["counters"]["updates"] += 1
            elif target == "finish":
                state["finish_counts"]["used"] += 4
            elif target == "tail":
                state["counters"]["tail_discarded"] = 0
            else:
                event = next(r for r in state["scientific_trace"] if r["kind"] == "window")
                event["counters"]["accepted"] -= 1
            with self.subTest(target=target), self.assertRaises(VerificationError):
                self.verify(state)

    def test_RW_replays_seed_probability_and_reports_actual_mismatch(self):
        proof = self.verify(self.good_rw, rw=self.controls["RW"])
        budget = proof["RW_budget"]
        self.assertEqual(budget["K_P"], 75)
        self.assertEqual(budget["K_RW"], self.good_rw["counters"]["commits"])
        self.assertEqual(budget["commit_difference"], budget["K_RW"]-75)
        self.assertEqual(budget["exact_commit_budget_match"], budget["K_RW"] == 75)
        self.assertTrue(budget["probability_match_does_not_guarantee_budget_match"])

    def test_RW_cannot_change_coin_final_rng_or_probability(self):
        for field in ("coin", "final_rng", "probability"):
            state = copy.deepcopy(self.good_rw)
            event = next(r for r in state["scientific_trace"] if r["kind"] == "window")
            if field == "coin":
                event["rw_coin"] = .314159
            elif field == "final_rng":
                state["rw_rng_state"] = random.Random(8).getstate()
            else:
                state["rw_probability"] = .5
            with self.subTest(field=field), self.assertRaises(VerificationError):
                self.verify(state, rw=self.controls["RW"])

    def test_RW_gate_is_recomputed_on_its_own_dynamic_arrivals(self):
        state = copy.deepcopy(self.good_rw)
        observation = next(r for r in state["scientific_trace"] if r["kind"] == "observation" and r["accepted"])
        observation["arrival"]["confidence"] = 0.
        with self.assertRaisesRegex(VerificationError, "recomputed_G50"):
            self.verify(state, rw=self.controls["RW"])

    def test_RP_verifies_actual_seed_permutation_mask_and_exact_K(self):
        proof = self.verify(self.good_rp, rp=self.controls["RP"])
        self.assertEqual(proof["counters"]["commits"], 75)
        self.assertEqual(proof["counters"]["accepted"], 300)
        self.assertEqual(proof["counters"]["tail_discarded"], 0)
        self.assertTrue(proof["random_position_mask_verified"])
        self.assertFalse(proof["gate_decisions_recomputed"])
        wrong = copy.deepcopy(self.controls["RP"])
        wrong["permutation"][0], wrong["permutation"][1] = wrong["permutation"][1], wrong["permutation"][0]
        with self.assertRaisesRegex(VerificationError, "frozen_permutation"):
            self.verify(self.good_rp, rp=wrong)

    def test_RP_cannot_lose_or_add_a_selected_input(self):
        state = copy.deepcopy(self.good_rp)
        state["rp_mask"][0] = not state["rp_mask"][0]
        with self.assertRaisesRegex(VerificationError, "actual_mask"):
            self.verify(state, rp=self.controls["RP"])

    def test_disabled_reference_compares_every_actual_learning_component(self):
        reference = copy.deepcopy(self.good_disabled["learner"]["learning"])
        obs = [r for r in self.good_disabled["scientific_trace"] if r["kind"] == "observation"]
        proof = verify_disabled_equivalence(self.good_disabled, reference, reference_observation_trace=obs)
        self.assertTrue(proof["full_learning_tree_equal"])
        self.assertFalse(proof["success_flags_used_as_evidence"])
        for field in ("model", "optimizer", "buffers", "rng", "bn_mean", "gradients", "modes"):
            bad = copy.deepcopy(reference)
            if field == "model":
                bad[field]["weight"][0] += 1
            elif field == "optimizer":
                bad[field]["state"][0]["step"] += 1
            elif field == "buffers":
                bad[field]["scratch"][0] += 1
            elif field == "rng":
                bad[field]["python"] = random.Random(1).getstate()
            elif field == "bn_mean":
                bad[field]["x"] = np.array([1.])
            elif field == "gradients":
                bad[field]["weight"] = np.array([0., 0.])
            else:
                bad[field][""] = True
            with self.subTest(field=field), self.assertRaises(VerificationError):
                verify_disabled_equivalence(self.good_disabled, bad)

    def test_incomplete_original_checkpoint_cannot_claim_full_equivalence(self):
        reference = {"model": self.good_disabled["learner"]["learning"]["model"], "strict": True}
        with self.assertRaisesRegex(VerificationError, "learning snapshot"):
            verify_disabled_equivalence(self.good_disabled, reference)

    def test_model_state_version_metadata_is_part_of_exact_equivalence(self):
        actual = copy.deepcopy(self.good_disabled)
        actual["learner"]["learning"]["model"] = OrderedDict(actual["learner"]["learning"]["model"])
        actual["learner"]["learning"]["model"]._metadata = {"": {"version": 1}}
        reference = copy.deepcopy(actual["learner"]["learning"])
        reference["model"]._metadata[""]["version"] = 2
        with self.assertRaisesRegex(VerificationError, "metadata"):
            verify_disabled_equivalence(actual, reference)

    def test_continuous_resumed_ignores_only_identity_ledger_and_retains_physical_costs(self):
        resumed = copy.deepcopy(self.good_p)
        resumed["identity"]["run_id"] = "synthetic-resume"
        resumed["ledger_boundary"] = {"sequence": 999, "byte_offset": 99999}
        proof = compare_continuous_resumed(self.good_p, resumed,
                    physical_work_continuous={"forwards_confirmed_completed": {"HE": 100}},
                    physical_work_resumed={"forwards_confirmed_completed": {"HE": 109}, "replay": 9})
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["physical_work_resumed"]["replay"], 9)
        self.assertEqual(proof["recorded_numeric_cost_resumed_minus_continuous"],
                         {"forwards_confirmed_completed": {"HE": 9}})
        self.assertFalse(proof["physical_cost_equality_required"])
        for field in ("provider_state", "pools", "learner", "scientific_trace", "counters"):
            bad = copy.deepcopy(resumed)
            if field == "provider_state":
                bad[field]["tensor_sha256"]["s0000"] = "f"*64
            elif field == "pools":
                bad[field]["pending"] = []
            elif field == "learner":
                bad[field]["learning"]["rng"]["python"] = random.Random(1).getstate()
            elif field == "scientific_trace":
                bad[field][0]["gate_reason"] = "changed"
            else:
                bad[field]["observed"] += 1
            with self.subTest(field=field), self.assertRaises(VerificationError):
                compare_continuous_resumed(self.good_p, bad)

    def test_nonfinite_learning_or_trace_and_incomplete_final_rejected(self):
        for kind in ("learning", "trace", "unfinished", "provider"):
            state = copy.deepcopy(self.good_p)
            if kind == "learning":
                state["learner"]["learning"]["model"]["weight"][0] = np.nan
            elif kind == "trace":
                state["scientific_trace"][0]["arrival"]["confidence"] = float("nan")
            elif kind == "unfinished":
                state["finished"] = False
            else:
                state["provider_state"]["tensor_sha256"]["future"] = "e"*64
            with self.subTest(kind=kind), self.assertRaises(VerificationError):
                self.verify(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
