"""Synthetic aggregation contracts; never reads images, labels or models."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from round2_protocol import frozen_protocol, build_complete_matrix, derive_controls
from summarize_round2 import build_summary, validate_matrix, _verification_processes
from verify_complete import evaluate_fixed_reference
from test_verify_complete import rows


def fixture():
    proofs = []
    for definition in build_complete_matrix():
        d, seed, group = definition["domain"], definition["stream_seed"], definition["group"]
        o, a, g, new = rows(1.), rows(.8), rows(.7), rows(.72 if d == "lightbox" else .65)
        if d == "sunlamp":
            o[-1].update(valid=False, error=None); a[-1].update(valid=False, error=None)
        ids = [r["sample_id"] for r in o]
        risk = evaluate_fixed_reference(ids, o, a, {"G50": g, group: new}, expected_reference_count=512 if d == "lightbox" else 511)
        counts = dict(observed=1024, accepted=40, candidates=10, commits=8, rollbacks=2,
                      tail_discarded=0, consecutive_rollbacks=0, max_consecutive_rollbacks=1)
        if group == "RW": counts.update(candidates=11, commits=7, rollbacks=4, accepted=44)
        elif group == "RP": counts.update(candidates=8, commits=8, rollbacks=0, accepted=32)
        physical = {k: {"adaptation": 40, "developer_evaluation": 512} for k in (
            "forwards_confirmed_completed", "forwards_execution_lower_bound", "forwards_execution_upper_bound")}
        physical.update(forwards_inflight_unknown_upper_bound={}, replay_forwards_confirmed_completed={},
                        compute_seconds_confirmed=10., compute_time_incomplete=False)
        proofs.append(dict(schema="odr003-round2-complete-run-proof-v1", status="passed", definition=definition,
                           run_id=definition["run_key"], source_sha256=frozen_protocol()["source_sha256"],
                           protocol_sha256=frozen_protocol()["protocol_sha256"], counts=counts,
                           controls_derived_from_verified_counts=derive_controls(d, seed, 10, 8),
                           stream_binding=dict(path=f"{d}:{seed}", sha256="a"*64), risk=risk,
                           physical_work=physical, all_attempt_costs=dict(execution_attempt_elapsed_seconds_sum_recorded=12.),
                           independent_state=dict(rule_triggered_window_counts=dict(rule1=2)), queue_wait_seconds=1.))
    return proofs


class SummarizeRound2Tests(unittest.TestCase):
    def test_full_matrix_exactly_eighteen_and_RW_budget_honest(self):
        result = build_summary(fixture(), [], queue_wait_seconds=91.)
        self.assertEqual(result["complete_ODR_executed"], 18)
        self.assertEqual(result["actual_groups"], dict(P=6, RW=6, RP=6))
        budget = result["budgets"][0]
        self.assertEqual(budget["RW_commit_difference"], -1)
        self.assertEqual(budget["RW_used_image_difference"], -4)
        self.assertFalse(budget["RW_exact_actual_commit_budget_match"])
        self.assertTrue(budget["RP_exact_actual_commit_budget_match"])
        self.assertEqual(result["total_resources"]["queue_wait_seconds"], 91.)
        self.assertEqual(result["total_resources"]["compute_seconds_confirmed"], 180.)
        self.assertFalse(result["formal_efficacy_claim"])

    def test_actual_RW_equality_allowed_but_does_not_imply_equal_history(self):
        proofs = fixture()
        rw = next(p for p in proofs if p["definition"]["run_key"] == "RW:lightbox:2021")
        rw["counts"].update(commits=8, rollbacks=3)
        result = build_summary(proofs, [])
        self.assertTrue(result["budgets"][0]["RW_exact_actual_commit_budget_match"])
        self.assertIn("does not equalize history", result["control_interpretation"]["causal_limit"])

    def test_no_selective_dropping_stream_group_or_failed_direction(self):
        proofs = fixture()
        with self.assertRaisesRegex(ValueError, "missing complete"):
            build_summary(proofs[:-1], [])
        with self.assertRaises(ValueError): build_summary(proofs + [copy.deepcopy(proofs[0])], [])
        bad = copy.deepcopy(proofs)
        bad[0]["risk"]["methods"]["P"]["metrics"]["tail_D"] = 100.
        report = build_summary(bad, [])
        self.assertFalse(report["exploration_screen"]["exploratory_screen_passed"])
        self.assertFalse(report["automatic_parameter_changes_or_extra_runs"])
        self.assertEqual(report["complete_ODR_executed"], 18)

    def test_RP_must_really_match_K_not_claimed_label(self):
        bad = fixture()
        rp = next(p for p in bad if p["definition"]["group"] == "RP")
        rp["counts"]["commits"] -= 1
        with self.assertRaisesRegex(ValueError, "RP actual budget"):
            build_summary(bad, [])

    def test_controls_cannot_be_rederived_from_outcome_scores(self):
        bad = fixture()
        bad[6]["controls_derived_from_verified_counts"]["RW"]["commit_probability"] = .7
        with self.assertRaisesRegex(ValueError, "derived exclusively"):
            build_summary(bad, [])

    def test_Q_zero_skip_requires_actual_P_proof_and_no_silent_RP_skip(self):
        proofs = fixture()
        pair = [p for p in proofs if p["definition"]["domain"] == "lightbox" and p["definition"]["stream_seed"] == 2021]
        for p in pair:
            p["counts"].update(candidates=0, commits=0, accepted=0, rollbacks=0)
            p["controls_derived_from_verified_counts"] = derive_controls("lightbox", 2021, 0, 0)
        proofs = [p for p in proofs if p["definition"]["run_key"] != "RW:lightbox:2021"]
        skips = [dict(run_key="RW:lightbox:2021", reason="P_candidates_zero_probability_undefined")]
        result = build_summary(proofs, skips)
        self.assertEqual(result["complete_ODR_executed"], 17)
        self.assertEqual(result["domains"]["lightbox"]["method_summaries"]["RW"]["stream_count"], 2)
        self.assertIsNone(result["domains"]["lightbox"]["method_summaries"]["RW"]["metrics"]["mean_error"]["mean"])
        with self.assertRaises(ValueError): build_summary(proofs, [])
        proofs = [p for p in proofs if p["definition"]["run_key"] != "RP:lightbox:2021"]
        with self.assertRaises(ValueError): build_summary(proofs, skips)

    def test_false_Q_zero_skip_fails(self):
        proofs = fixture()
        proofs = [p for p in proofs if p["definition"]["run_key"] != "RW:lightbox:2021"]
        with self.assertRaisesRegex(ValueError, "zero P candidates"):
            build_summary(proofs, [dict(run_key="RW:lightbox:2021", reason="P_candidates_zero_probability_undefined")])

    def test_mean_is_stream_equal_weight_and_dispersion_retained(self):
        proofs = fixture()
        for p in proofs:
            if p["definition"]["group"] == "P" and p["definition"]["domain"] == "lightbox":
                p["risk"]["methods"]["P"]["metrics"]["mean_D"] = {2021:1.,2022:2.,2023:6.}[p["definition"]["stream_seed"]]
        result = build_summary(proofs, [])
        desc = result["domains"]["lightbox"]["method_summaries"]["P"]["metrics"]["mean_D"]
        self.assertEqual(desc["values"], [1.,2.,6.])
        self.assertEqual(desc["mean"], 3.)
        self.assertGreater(desc["sample_sd"], 0)

    def test_coverage_and_missing_not_averaged_away(self):
        proofs = fixture()
        p = proofs[0]
        p["risk"]["methods"]["P"].update(coverage=512, reference_missing=1)
        result = build_summary(proofs, [])
        self.assertFalse(result["exploration_screen"]["domains"]["lightbox"]["passed"])
        self.assertIn("coverage_and_fixed_reference", result["exploration_screen"]["domains"]["lightbox"]["failed_criteria"])

    def test_reused_baseline_or_reference_cannot_change_for_controls(self):
        for field in ("reference_ids", "methods"):
            bad = fixture()
            rw = bad[6]
            if field == "reference_ids": rw["risk"][field] = list(reversed(rw["risk"][field]))
            else: rw["risk"][field]["A"]["metrics"]["mean_error"] = .1
            with self.assertRaises(ValueError): build_summary(bad, [])

    def test_process_proof_alone_without_actual_exit_is_insufficient(self):
        with self.assertRaisesRegex(ValueError, "actual successful"):
            _verification_processes(dict(verification_attempts=[]), [dict(run_id="p")], {})

    def test_improvements_and_degradations_all_per_sample_kept(self):
        result = build_summary(fixture(), [])
        stream = result["domains"]["sunlamp"]["streams"]["2021"]
        self.assertEqual(len(stream["methods"]["P"]["per_sample"]), 512)
        self.assertEqual(stream["reference_count"], 511)
        self.assertFalse(stream["methods"]["P"]["per_sample"][-1]["in_reference"])


if __name__ == "__main__": unittest.main()
