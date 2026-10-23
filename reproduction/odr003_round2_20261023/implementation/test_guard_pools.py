"""Local synthetic contracts only. No model, real images, or shared-host work."""
import copy
import json
import math
import random
import unittest

import numpy as np

from guard_pools import (
    GuardPools, assess_protection_pair, assess_protection_window,
    decode_arrival_fragility, quaternion_distance_degrees,
    SOURCE_IMAGE_DIAGONAL,
)


def quaternion(degrees):
    angle = math.radians(degrees) / 2
    return [math.cos(angle), 0.0, 0.0, math.sin(angle)]


def prediction(degrees=0.0, disagreement=2.0, residual=0.01, legal=True):
    return dict(pose_legal=legal, h_available=True, e_available=True,
                heat_q=quaternion(degrees), original_h_rejected=not legal,
                rotation_disagreement=disagreement, reprojection=residual,
                used_indices=[0, 1, 2, 3, 4])


def record(index, *, accepted=True, fragile=False, missing=False):
    peaks = [0.9] * (5 if fragile else 11) + ([0.1] * 6 if fragile else [])
    return dict(sample_id=f"image/{index:04}.png", observation_index=index,
                image_sha256=f"{index:064x}", preprocess_sha256="f" * 64,
                g50_accepted=accepted,
                arrival=dict(heat_peaks=None if missing else peaks, pose_legal=True))


def add_window(pool, start, *, rejected_first=False):
    accepted = []
    for index in range(start, start + (5 if rejected_first else 4)):
        is_accepted = not (rejected_first and index == start)
        value = record(index, accepted=is_accepted, fragile=(index % 2 == 0))
        pool.observe(value)
        if is_accepted:
            accepted.append(value["sample_id"])
    return accepted


class FragileDecoderContracts(unittest.TestCase):
    def test_strict_threshold_and_lowered_effective_threshold(self):
        # Exactly 0.5 is not visible at the original 0.5 cutoff. The decoder
        # therefore lowers once; all five become visible at the real 0.4 pass.
        decoded = decode_arrival_fragility([0.5] * 5 + [0.2] * 6)
        self.assertEqual(decoded["used_indices"], list(range(5)))
        self.assertEqual(decoded["effective_threshold"], 0.4)
        self.assertEqual(decoded["used_count"], 5)
        self.assertTrue(decoded["enough_points"])
        self.assertIsNone(decoded["pnp_solved"])
        self.assertTrue(decoded["fragile"])

    def test_scalar_cast_happens_before_float32_comparison(self):
        # Python threshold reaches 0.30000000000000004. A float32 peak equal
        # to float32(0.3) must be equal, not greater. Selection happens at 0.2.
        decoded = decode_arrival_fragility([np.float32(0.3)] * 5 + [-1.0] * 6)
        self.assertAlmostEqual(decoded["effective_threshold"], 0.2)
        self.assertEqual(decoded["used_count"], 5)

    def test_last_positive_epsilon_does_not_accept_zero_peaks(self):
        decoded = decode_arrival_fragility([0.0] * 11)
        self.assertEqual(decoded["used_count"], 0)
        self.assertGreater(decoded["effective_threshold"], 0)
        self.assertLess(decoded["effective_threshold"], 1e-15)
        self.assertFalse(decoded["enough_points"])
        self.assertFalse(decoded["pnp_solved"])
        self.assertTrue(decoded["fragile"])
        self.assertIn("not_solvable_fewer_than_5_points", decoded["reasons"])

    def test_no_threshold_is_not_zero(self):
        decoded = decode_arrival_fragility([0.5] * 11, threshold=-0.1)
        self.assertIsNone(decoded["effective_threshold"])
        self.assertIsNone(decoded["used_count"])
        self.assertIsNone(decoded["fragile"])
        self.assertFalse(decoded["fragility_comparable"])

    def test_missing_and_nonfinite_are_distinct(self):
        self.assertIsNone(decode_arrival_fragility(None)["fragile"])
        with self.assertRaises(FloatingPointError):
            decode_arrival_fragility([0.9] * 10 + [float("nan")])

    def test_fragile_peak_without_low_point_count(self):
        decoded = decode_arrival_fragility([0.9] * 10 + [0.51])
        self.assertEqual(decoded["used_count"], 11)
        self.assertEqual(decoded["near_threshold_indices"], [10])
        self.assertTrue(decoded["fragile"])
        self.assertFalse(decode_arrival_fragility([0.9] * 11)["fragile"])


class ReservoirContracts(unittest.TestCase):
    def test_pending_arrivals_cannot_leak_into_history(self):
        pool = GuardPools(101, 202)
        window = add_window(pool, 0, rejected_first=True)
        protected = pool.protection_records(window)
        self.assertEqual([x["sample_id"] for x in protected], window)
        self.assertEqual(pool.uniform.slots, [])
        self.assertEqual(pool.fragile.slots, [])
        self.assertEqual(pool.diagnostics()["pending_count"], 5)
        self.assertTrue(all(x["protection_sources"][0]["pool"] == "current_window" for x in protected))

    def test_rejected_arrivals_are_eligible_for_both_pools(self):
        pool = GuardPools(101, 202)
        # Exactly four historical arrivals ensure all four enter uniform;
        # accepted status does not exclude the deliberately rejected first ID.
        for index in range(4):
            pool.observe(record(index, accepted=index != 0, fragile=index == 0))
        pool.finish_window()
        self.assertIn("image/0000.png", pool.uniform.slots)
        self.assertIn("image/0000.png", pool.fragile.slots)
        window = add_window(pool, 4)
        protected = pool.protection_records(window)
        ids = [x["sample_id"] for x in protected]
        self.assertEqual(ids[:4], window)
        self.assertEqual(ids[4:], [f"image/{i:04}.png" for i in range(4)])
        self.assertEqual(len(ids), len(set(ids)))
        rejected = next(x for x in protected if x["sample_id"] == "image/0000.png")
        self.assertEqual([s["pool"] for s in rejected["protection_sources"]], ["uniform", "fragile"])
        pool.record_check("window-2", protected)
        self.assertEqual(pool.diagnostics()["checked_unique_g50_rejected"], 1)

    def test_protection_selection_does_not_consume_rng(self):
        pool = GuardPools(101, 202)
        first = add_window(pool, 0)
        snapshot = pool.state_dict()
        self.assertEqual(pool.protection_records(first), pool.protection_records(first))
        self.assertEqual(snapshot, pool.state_dict())

    def test_full_checkpoint_roundtrip_replays_lists_and_replacements(self):
        pool = GuardPools(818, 919)
        for start in range(0, 20, 4):
            ids = add_window(pool, start)
            pool.record_check(f"window-{start}", pool.protection_records(ids))
            pool.finish_window()
        ids = add_window(pool, 20)
        frozen = json.loads(json.dumps(pool.state_dict(), allow_nan=False))
        restored = GuardPools(1, 2)
        restored.load_state_dict(frozen)
        self.assertEqual(pool.protection_records(ids), restored.protection_records(ids))
        self.assertEqual(pool.finish_window(), restored.finish_window())
        for start in range(24, 140, 4):
            left = add_window(pool, start)
            right = add_window(restored, start)
            a, b = pool.protection_records(left), restored.protection_records(right)
            self.assertEqual(a, b)
            pool.record_check(f"window-{start}", a)
            restored.record_check(f"window-{start}", b)
            self.assertEqual(pool.finish_window(), restored.finish_window())
        self.assertEqual(pool.state_dict(), restored.state_dict())

    def test_uniform_sampling_independent_of_fragile_eligibility(self):
        left, right = GuardPools(121, 242), GuardPools(121, 242)
        for index in range(100):
            left.observe(record(index, fragile=False))
            right.observe(record(index, fragile=True))
            if index % 4 == 3:
                left.finish_window()
                right.finish_window()
        self.assertEqual(left.uniform.state_dict(), right.uniform.state_dict())
        self.assertNotEqual(left.fragile.candidate_count, right.fragile.candidate_count)

    def test_pool_sampling_does_not_change_global_random_state(self):
        random.seed(77)
        before = random.getstate()
        pool = GuardPools(121, 242)
        for index in range(30):
            pool.observe(record(index, fragile=True))
        pool.finish_window()
        self.assertEqual(before, random.getstate())

    def test_duplicate_future_out_of_order_and_rejected_window_fail(self):
        pool = GuardPools(11, 22)
        window = add_window(pool, 0)
        with self.assertRaises(ValueError):
            pool.observe(record(0))
        with self.assertRaises(ValueError):
            pool.protection_records(window[:3] + ["image/9999.png"])
        with self.assertRaises(ValueError):
            pool.protection_records(list(reversed(window)))
        with self.assertRaises(ValueError):
            pool.protection_records(window[:3] + [window[0]])
        pool.observe(record(4, accepted=False))
        with self.assertRaises(ValueError):
            pool.protection_records(window[:3] + ["image/0004.png"])
        old_position = record(9)
        old_position["observation_index"] = 3
        with self.assertRaises(ValueError):
            pool.observe(old_position)

    def test_external_record_mutation_does_not_change_history(self):
        pool = GuardPools(11, 22)
        original = record(0)
        returned = pool.observe(original)
        original["arrival"]["heat_peaks"][0] = 12.0
        returned["sample_id"] = "corrupted"
        state = pool.state_dict()
        state["records"]["image/0000.png"]["arrival"]["heat_peaks"][0] = 13.0
        self.assertEqual(pool.records["image/0000.png"]["arrival"]["heat_peaks"][0], 0.9)

    def test_corrupt_checkpoint_rejected_without_partial_mutation(self):
        pool = GuardPools(11, 22)
        add_window(pool, 0)
        pool.finish_window()
        add_window(pool, 4)
        before = pool.state_dict()
        bad = copy.deepcopy(before)
        bad["uniform"]["slots"][0] = "image/0007.png"
        with self.assertRaises(ValueError):
            pool.load_state_dict(bad)
        self.assertEqual(pool.state_dict(), before)
        bad = copy.deepcopy(before)
        bad["pending"].reverse()
        with self.assertRaises(ValueError):
            pool.load_state_dict(bad)
        self.assertEqual(pool.state_dict(), before)

    def test_actual_check_event_is_idempotent_and_order_bound(self):
        pool = GuardPools(11, 22)
        ids = add_window(pool, 0)
        selected = pool.protection_records(ids)
        pool.record_check("decision-1", selected)
        pool.record_check("decision-1", selected)
        self.assertEqual(pool.diagnostics()["checked_images"], 4)
        with self.assertRaises(ValueError):
            pool.record_check("decision-1", list(reversed(selected)))

    def test_missing_fragility_still_enters_uniform(self):
        pool = GuardPools(11, 22)
        pool.observe(record(0, missing=True))
        pool.finish_window()
        self.assertEqual(pool.uniform.candidate_count, 1)
        self.assertEqual(pool.fragile.candidate_count, 0)
        self.assertEqual(pool.diagnostics()["fragility_unavailable"], 1)

    def test_nested_gt_is_rejected_before_record_insertion(self):
        pool = GuardPools(11, 22)
        value = record(0)
        value["arrival"]["ground_truth"] = {"rotation": [1, 0, 0, 0]}
        with self.assertRaises(ValueError):
            pool.observe(value)
        self.assertEqual(pool.records, {})

    def test_pending_accepted_arrival_cannot_be_silently_skipped(self):
        pool = GuardPools(11, 22)
        ids = add_window(pool, 0)
        pool.observe(record(4))
        with self.assertRaises(ValueError):
            pool.protection_records(ids)

    def test_same_empty_window_cannot_be_finalized_twice(self):
        pool = GuardPools(11, 22)
        add_window(pool, 0)
        pool.finish_window()
        with self.assertRaises(ValueError):
            pool.finish_window()


class ProtectionRuleContracts(unittest.TestCase):
    def test_no_single_or_two_conditions_can_veto(self):
        pre = prediction()
        cases = [prediction(degrees=40), prediction(disagreement=40), prediction(residual=0.1),
                 prediction(degrees=40, disagreement=40), prediction(degrees=40, residual=0.1),
                 prediction(disagreement=40, residual=0.1)]
        for post in cases:
            with self.subTest(post=post):
                self.assertFalse(assess_protection_pair("x", pre, post)["rollback"])

    def test_three_conditions_on_same_image_trigger(self):
        result = assess_protection_pair("x", prediction(),
                                       prediction(degrees=40, disagreement=40, residual=0.1))
        self.assertTrue(result["rollback"])
        self.assertTrue(result["rules"]["joint_geometry_worsening"]["triggered"])
        self.assertFalse(result["rules"]["new_unavailable"]["triggered"])

    def test_conditions_cannot_be_combined_across_images(self):
        pairs = [dict(sample_id="a", pre=prediction(), post=prediction(degrees=40)),
                 dict(sample_id="b", pre=prediction(), post=prediction(disagreement=40)),
                 dict(sample_id="c", pre=prediction(), post=prediction(residual=0.1))]
        result = assess_protection_window(pairs)
        self.assertFalse(result["rollback"])
        self.assertEqual(result["triggered_image_counts"]["h_rotation_change"], 1)
        self.assertEqual(result["triggered_image_counts"]["he_disagreement_increase"], 1)
        self.assertEqual(result["triggered_image_counts"]["reprojection_increase"], 1)

    def test_quaternion_sign_does_not_create_rotation_jump(self):
        pre = prediction(degrees=37)
        post = prediction(degrees=37, disagreement=40, residual=0.1)
        post["heat_q"] = [-x for x in post["heat_q"]]
        self.assertAlmostEqual(quaternion_distance_degrees(pre["heat_q"], post["heat_q"]), 0.0, places=5)
        self.assertFalse(assess_protection_pair("x", pre, post)["rollback"])

    def test_geodesic_uses_shortest_rotation(self):
        self.assertAlmostEqual(quaternion_distance_degrees(quaternion(350), quaternion(10)), 20.0, places=10)
        self.assertAlmostEqual(quaternion_distance_degrees([2, 0, 0, 0], quaternion(40)), 40.0, places=10)

    def test_point_set_change_alone_does_not_trigger(self):
        pre, post = prediction(), prediction()
        post["used_indices"] = [0, 1, 2, 3, 4, 9]
        result = assess_protection_pair("x", pre, post)
        self.assertTrue(result["point_set_changed"])
        self.assertFalse(result["rollback"])

    def test_prelegal_to_unavailable_or_physical_failure_triggers(self):
        for mode in ("pnp", "physical"):
            post = prediction(legal=False)
            if mode == "pnp":
                post.update(h_available=False, heat_q=None, reprojection=None,
                            rotation_disagreement=None)
            result = assess_protection_pair("x", prediction(), post)
            self.assertTrue(result["rollback"])
            self.assertTrue(result["rules"]["new_unavailable"]["triggered"])

    def test_invalid_to_valid_is_not_new_failure(self):
        self.assertFalse(assess_protection_pair("x", prediction(legal=False), prediction())["rollback"])

    def test_none_is_unavailable_denominator_not_zero(self):
        post = prediction(degrees=40, disagreement=40, residual=0.1)
        post["reprojection"] = None
        result = assess_protection_pair("x", prediction(), post)
        joint = result["rules"]["joint_geometry_worsening"]
        self.assertFalse(joint["comparable"])
        self.assertIsNone(joint["triggered"])
        self.assertEqual(result["denominators"]["joint_geometry_worsening"], 0)
        self.assertEqual(result["denominators"]["h_rotation_change"], 1)
        self.assertEqual(result["denominators"]["reprojection_increase"], 0)
        self.assertFalse(result["rollback"])

    def test_one_pixel_floor_prevents_tiny_residual_veto(self):
        pre = prediction(residual=0.0)
        post = prediction(degrees=40, disagreement=40, residual=0.5 / SOURCE_IMAGE_DIAGONAL)
        self.assertFalse(assess_protection_pair("x", pre, post)["rollback"])
        post["reprojection"] = 1.1 / SOURCE_IMAGE_DIAGONAL
        self.assertTrue(assess_protection_pair("x", pre, post)["rollback"])

    def test_thresholds_are_strict_not_greater_equal(self):
        pre = prediction(disagreement=2, residual=0)
        post = prediction(degrees=40, disagreement=12, residual=1.0 / SOURCE_IMAGE_DIAGONAL)
        result = assess_protection_pair("x", pre, post)
        conditions = result["rules"]["joint_geometry_worsening"]["subconditions"]
        self.assertFalse(conditions["he_disagreement_increase"]["triggered"])
        self.assertFalse(conditions["reprojection_increase"]["triggered"])
        self.assertFalse(result["rollback"])

    def test_exact_fifteen_degrees_and_relative_equality_do_not_trigger(self):
        pre = prediction()
        result = assess_protection_pair("x", pre, prediction(degrees=15, disagreement=30, residual=0.1))
        self.assertFalse(result["rollback"])
        result = assess_protection_pair("x", pre, prediction(degrees=40, disagreement=30, residual=0.0125))
        self.assertFalse(result["rollback"])
        self.assertTrue(assess_protection_pair("x", pre,
                        prediction(degrees=15.00001, disagreement=30, residual=0.01250001))["rollback"])

    def test_nonfinite_is_fatal_even_when_other_signal_is_missing(self):
        post = prediction()
        post["heat_q"] = None
        post["reprojection"] = float("inf")
        with self.assertRaises(FloatingPointError):
            assess_protection_pair("x", prediction(), post)
        with self.assertRaises(FloatingPointError):
            quaternion_distance_degrees(None, [1, 0, float("nan"), 0])

    def test_canonical_signal_adapter_and_no_double_window_count(self):
        pre = dict(legal=True, pnp_solved=True, physical_valid=True,
                   h_quaternion_wxyz=quaternion(0), he_rotation_deg=2,
                   reprojection_normalized=0.01)
        post = dict(legal=False, pnp_solved=False, physical_valid=False,
                    h_quaternion_wxyz=quaternion(40), he_rotation_deg=40,
                    reprojection_normalized=0.1)
        result = assess_protection_window([dict(sample_id="x", pre=pre, post=post),
                                            dict(sample_id="y", pre=pre, post=post)])
        self.assertEqual(result["rollback_window_count"], 1)
        self.assertEqual(result["rollback_image_count"], 2)
        self.assertEqual(result["triggered_image_counts"]["new_unavailable"], 2)
        self.assertEqual(result["triggered_image_counts"]["joint_geometry_worsening"], 2)

    def test_unknown_validity_not_assumed_failure(self):
        result = assess_protection_pair("x", {"legal": True}, {})
        self.assertIsNone(result["rules"]["new_unavailable"]["triggered"])
        self.assertEqual(result["denominators"]["new_unavailable"], 0)
        self.assertFalse(result["rollback"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
