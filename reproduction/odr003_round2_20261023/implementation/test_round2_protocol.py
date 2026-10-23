"""Local policy contracts only: no assets, torch, model calls, or shared host."""

import copy
import json
import unittest

from round2_protocol import (
    DOMAINS, ENGINEERING_CONTRACTS, SOURCE_SHA256, STREAM_SEEDS,
    assess_engineering, assess_exploration, build_complete_matrix,
    build_engineering_matrix, derive_controls, frozen_protocol,
)


def passed_contracts():
    return {name: True for name in ENGINEERING_CONTRACTS}


def screening_example():
    return {
        "lightbox": {
            "stream_seeds": [2021, 2022, 2023],
            "O": {"mean_error": 0.2},
            "A": {"mean_error": 0.1, "mean_D": 0.02, "tail_D": 0.2, "coverage": [512, 512, 512]},
            "G50": {"mean_D": 0.01, "tail_D": 0.1},
            "P": {"mean_error": 0.102, "mean_D": 0.011, "tail_D": 0.11,
                  "coverage": [512, 512, 512], "reference_missing": [0, 0, 0]},
        },
        "sunlamp": {
            "stream_seeds": [2021, 2022, 2023],
            "O": {"mean_error": 0.3},
            "A": {"mean_error": 0.2, "mean_D": 0.03, "tail_D": 0.2, "coverage": [511, 512, 512]},
            "G50": {"mean_D": 0.028, "tail_D": 0.22},
            "P": {"mean_error": 0.205, "mean_D": 0.029, "tail_D": 0.21,
                  "coverage": [512, 512, 512], "reference_missing": [0, 0, 0]},
        },
    }


class ProtocolMatrixContracts(unittest.TestCase):
    def test_frozen_document_is_json_and_callers_cannot_mutate_defaults(self):
        first = frozen_protocol()
        json.dumps(first, allow_nan=False)
        self.assertEqual(first["source_sha256"], SOURCE_SHA256)
        first["protection"]["h_rotation_change_degrees_gt"] = 999
        first["randomness"]["seeds"]["lightbox:2021"]["uniform_pool"] = 0
        second = frozen_protocol()
        self.assertEqual(second["protection"]["h_rotation_change_degrees_gt"], 15.0)
        self.assertEqual(second["protection"]["he_rotation_disagreement_increase_degrees_gt"], 10.0)
        self.assertEqual(second["protection"]["all_11_point_reprojection_increase_gt"]["relative_to_pre"], 0.25)
        self.assertNotEqual(second["randomness"]["seeds"]["lightbox:2021"]["uniform_pool"], 0)
        self.assertEqual(frozen_protocol()["protocol_sha256"], second["protocol_sha256"])

    def test_complete_matrix_has_exactly_six_pairs_per_group_and_no_extra_slots(self):
        rows = build_complete_matrix()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r["run_key"] for r in rows}), 18)
        self.assertEqual([r["run_limit_slot"] for r in rows], list(range(1, 19)))
        expected = {(d, s) for d in DOMAINS for s in STREAM_SEEDS}
        for group in ("P", "RW", "RP"):
            subset = [r for r in rows if r["group"] == group]
            self.assertEqual({(r["domain"], r["stream_seed"]) for r in subset}, expected)
            self.assertTrue(all(r["observations"] == 1024 for r in subset))
        self.assertFalse(any(r["counts_must_be_bound_before_start"] for r in rows[:6]))
        self.assertTrue(all(r["counts_must_be_bound_before_start"] for r in rows[6:]))
        self.assertEqual(rows[6]["paired_P_dependency"], "P:lightbox:2021")

    def test_engineering_has_eight_fixed_prefixes_and_resumes_do_not_expand_budget(self):
        rows = build_engineering_matrix()
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(r["logical_observation_budget"] for r in rows), 1024)
        self.assertTrue(all(r["observations"] == 128 and r["stream_seed"] == 2021 for r in rows))
        self.assertFalse(any(r["developer_accuracy_evaluation"] for r in rows))
        for domain in DOMAINS:
            paired = [r for r in rows if r["domain"] == domain]
            self.assertEqual(len(paired), 4)
            continuous = next(r for r in paired if r["trajectory"] == "P_continuous")
            resumed = next(r for r in paired if r["trajectory"] == "P_interrupted_resumed")
            self.assertEqual(continuous["rng_seeds"], resumed["rng_seeds"])
            self.assertEqual(len(resumed["interruptions"]), 2)

    def test_random_stream_roles_are_fixed_distinct_and_without_gt_inputs(self):
        seed_map = frozen_protocol()["randomness"]["seeds"]
        values = [v for pair in seed_map.values() for v in pair.values()]
        self.assertEqual(len(values), 24)
        self.assertEqual(len(set(values)), 24)
        self.assertEqual(seed_map["lightbox:2021"]["uniform_pool"], 2026102311)
        self.assertEqual(seed_map["sunlamp:2023"]["rw_coin"], 2026112521)
        self.assertFalse(frozen_protocol()["randomness"]["dependent_on_GT_or_scores"])

    def test_RP_is_prefix_of_one_fixed_permutation_then_sorted_by_input(self):
        small = derive_controls("sunlamp", 2022, 100, 31)
        large = derive_controls("sunlamp", 2022, 150, 72)
        self.assertEqual(small["RP"]["permutation"], large["RP"]["permutation"])
        positions = small["RP"]["selected_positions"]
        self.assertEqual(len(positions), 124)
        self.assertEqual(len(set(positions)), 124)
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(set(positions), set(small["RP"]["permutation"][:124]))
        self.assertTrue(set(positions).issubset(large["RP"]["selected_positions"]))
        self.assertEqual(small["RP"]["commits"], 31)
        self.assertEqual(small["RW"]["commit_probability"], 0.31)
        self.assertEqual(small["RW"]["budget_kind"], "commit_probability_matched_not_update_count_matched")
        self.assertFalse(small["RW"]["quota_filling"])

    def test_RP_never_uses_candidates_instead_of_successful_commits(self):
        paired = derive_controls("lightbox", 2021, 200, 7)
        self.assertEqual(paired["RP"]["used_images"], 28)
        self.assertEqual(paired["RP"]["commits"], 7)
        self.assertEqual(paired["RW"]["probability_denominator"], 200)
        self.assertEqual(paired["RW"]["probability_numerator"], 7)

    def test_Q0_probability_is_none_not_zero_and_endpoint_runs_are_not_auto_reused(self):
        q0 = derive_controls("lightbox", 2021, 0, 0)
        self.assertFalse(q0["RW"]["enabled"])
        self.assertIsNone(q0["RW"]["commit_probability"])
        self.assertIsNone(q0["RW"]["deterministic_endpoint"])
        self.assertEqual(q0["RP"]["selected_positions"], [])
        zero = derive_controls("lightbox", 2021, 10, 0)
        one = derive_controls("lightbox", 2021, 10, 10)
        self.assertTrue(zero["RW"]["enabled"])
        self.assertEqual(zero["RW"]["commit_probability"], 0)
        self.assertEqual(zero["RW"]["deterministic_endpoint"], "all_rollback")
        self.assertEqual(one["RW"]["commit_probability"], 1)
        self.assertEqual(one["RW"]["deterministic_endpoint"], "all_commit")
        self.assertIn("proof", one["RW"]["endpoint_reuse"])

    def test_unauthorized_domains_seeds_and_budgets_rejected(self):
        for args in (("other", 2021, 10, 5), ("lightbox", 2024, 10, 5),
                     ("lightbox", 2021, 257, 1), ("lightbox", 2021, 10, 11),
                     ("lightbox", 2021, -1, 0), ("lightbox", 2021, 10, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                derive_controls(*args)


class EngineeringAssessmentContracts(unittest.TestCase):
    def assess(self, q, k, longest=0, **kwargs):
        return assess_engineering(q, k, q-k, longest, contract_checks=passed_contracts(), **kwargs)

    def test_Q0_holds_and_rates_stay_none(self):
        result = self.assess(0, 0)
        self.assertTrue(result["hold_full_exploration"])
        self.assertIsNone(result["commit_fraction"])
        self.assertIsNone(result["trigger_fraction"])

    def test_no_real_trigger_does_not_hold_when_contracts_pass(self):
        result = self.assess(20, 20)
        self.assertFalse(result["hold_full_exploration"])
        self.assertTrue(any("no_real_trigger" in warning for warning in result["warnings"]))

    def test_near_frozen_uses_exact_predeclared_boundaries(self):
        self.assertTrue(self.assess(10, 1, 9)["hold_full_exploration"])
        self.assertTrue(self.assess(8, 0, 8)["hold_full_exploration"])
        self.assertFalse(self.assess(9, 1, 8)["hold_full_exploration"])
        few = self.assess(7, 0, 7)
        self.assertFalse(few["hold_full_exploration"])
        self.assertIn("zero_commits_report_separately", few["warnings"])

    def test_long_rollback_segment_is_warning_not_independent_hold(self):
        result = self.assess(20, 10, 8)
        self.assertFalse(result["hold_full_exploration"])
        self.assertIn("at_least_8_consecutive_rollbacks", result["warnings"])

    def test_missing_contracts_and_rollback_wiring_fail_closed(self):
        missing = assess_engineering(10, 10, 0, 0)
        self.assertTrue(missing["hold_full_exploration"])
        contracts = passed_contracts()
        contracts["pool_replay"] = False
        failed = assess_engineering(10, 8, 2, 2, contract_checks=contracts)
        self.assertTrue(failed["hold_full_exploration"])
        self.assertEqual(failed["failed_contracts"], ["pool_replay"])
        wrong_count = assess_engineering(10, 8, 1, 1, contract_checks=passed_contracts())
        self.assertTrue(wrong_count["hold_full_exploration"])
        self.assertIn("P_rollback_count_does_not_match_distinct_triggered_windows", wrong_count["hold_reasons"])

    def test_short_budget_and_impossible_counts_are_rejected(self):
        for counts in ((33, 30, 3, 2), (10, 11, 0, 0), (10, 5, 11, 1), (10, 9, 1, 8)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                assess_engineering(*counts, contract_checks=passed_contracts())


class ExplorationAssessmentContracts(unittest.TestCase):
    def test_partial_progress_can_pass_without_sunlamp_tail_beating_A(self):
        result = assess_exploration(screening_example())
        self.assertTrue(result["exploratory_screen_passed"])
        self.assertEqual(result["interpretation"], "partial_progress_sunlamp_tail_still_above_A")
        self.assertFalse(result["formal_efficacy_claim"])

    def test_stronger_tail_result_remains_exploratory(self):
        data = screening_example()
        data["sunlamp"]["P"]["tail_D"] = 0.19
        result = assess_exploration(data)
        self.assertTrue(result["exploratory_screen_passed"])
        self.assertEqual(result["interpretation"], "exploratory_candidate_sunlamp_tail_no_worse_than_A")
        self.assertFalse(result["formal_efficacy_claim"])

    def test_sunlamp_requires_direction_not_minimum_percentage(self):
        data = screening_example()
        data["sunlamp"]["P"]["tail_D"] = 0.219999
        self.assertTrue(assess_exploration(data)["exploratory_screen_passed"])
        data["sunlamp"]["P"]["tail_D"] = 0.22
        self.assertFalse(assess_exploration(data)["exploratory_screen_passed"])

    def test_lightbox_retains_80_percent_of_both_risk_benefits(self):
        data = screening_example()
        # A=.2 G=.1 -> fixed ceiling .12.  .121 loses too much prior benefit.
        data["lightbox"]["P"]["tail_D"] = 0.121
        result = assess_exploration(data)
        self.assertFalse(result["exploratory_screen_passed"])
        self.assertIn("retain_80_percent_G50_tail_D_benefit", result["domains"]["lightbox"]["failed_criteria"])
        data["lightbox"]["P"]["tail_D"] = 0.119
        data["lightbox"]["P"]["mean_D"] = 0.0121
        self.assertFalse(assess_exploration(data)["exploratory_screen_passed"])

    def test_mean_precision_5_percent_and_80_percent_A_gain_both_apply(self):
        data = screening_example()
        data["lightbox"]["P"]["mean_error"] = 0.106
        result = assess_exploration(data)
        self.assertIn("mean_error_at_most_105_percent_of_A", result["domains"]["lightbox"]["failed_criteria"])
        data["lightbox"]["O"]["mean_error"] = 0.102
        data["lightbox"]["P"]["mean_error"] = 0.101
        result = assess_exploration(data)
        self.assertIn("retain_80_percent_A_mean_error_benefit", result["domains"]["lightbox"]["failed_criteria"])

    def test_A_no_gain_compares_to_O_without_invalid_percentages(self):
        data = screening_example()
        data["sunlamp"]["O"]["mean_error"] = 0.19
        data["sunlamp"]["P"]["mean_error"] = 0.195
        result = assess_exploration(data)
        self.assertIn("A_no_gain_P_mean_error_no_worse_than_O", result["domains"]["sunlamp"]["failed_criteria"])
        data["sunlamp"]["P"]["mean_error"] = 0.19
        self.assertTrue(assess_exploration(data)["exploratory_screen_passed"])

    def test_new_reference_missing_cannot_be_offset_by_new_coverage(self):
        data = screening_example()
        data["sunlamp"]["P"]["reference_missing"] = [1, 0, 0]
        result = assess_exploration(data)
        self.assertFalse(result["exploratory_screen_passed"])
        coverage = next(c for c in result["domains"]["sunlamp"]["criteria"] if c["criterion"] == "coverage_and_fixed_reference")
        self.assertFalse(coverage["per_stream"][0]["passed"])

    def test_coverage_not_aggregated_to_hide_one_failing_stream(self):
        data = screening_example()
        data["sunlamp"]["A"]["coverage"] = [511, 512, 511]
        data["sunlamp"]["P"]["coverage"] = [512, 511, 512]
        self.assertFalse(assess_exploration(data)["exploratory_screen_passed"])

    def test_none_nan_and_infinity_are_unavailable_not_zero(self):
        for value in (None, float("nan"), float("inf"), -1, True):
            data = screening_example()
            data["sunlamp"]["P"]["tail_D"] = value
            with self.subTest(value=value):
                result = assess_exploration(data)
                self.assertFalse(result["exploratory_screen_passed"])
                criterion = next(c for c in result["domains"]["sunlamp"]["criteria"] if c["criterion"] == "tail_D_below_original_G50")
                self.assertFalse(criterion["available"])
                self.assertEqual(criterion["reason"], "missing_nonfinite_or_negative_metric")
                json.dumps(result, allow_nan=False)

    def test_no_selection_of_domains_or_streams_and_input_is_preserved(self):
        data = screening_example()
        original = copy.deepcopy(data)
        result = assess_exploration(data)
        self.assertEqual(data, original)
        self.assertEqual(result["domains"]["lightbox"]["input_metrics_retained"], data["lightbox"])
        del data["lightbox"]
        with self.assertRaises(ValueError):
            assess_exploration(data)
        data = screening_example()
        data["sunlamp"]["stream_seeds"] = [2021, 2023]
        self.assertFalse(assess_exploration(data)["exploratory_screen_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
