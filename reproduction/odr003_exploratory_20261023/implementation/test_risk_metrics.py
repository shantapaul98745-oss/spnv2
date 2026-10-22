"""Synthetic contract tests only: no model, dataset or ground-truth access."""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

from risk_metrics import BASE_METHODS, DOMAINS, MetricContractError, evaluate_exploration


STREAMS = ("2021", "2022", "2023")


def fixture(n=40):
    ids = {domain: [str(i) for i in range(n)] for domain in DOMAINS}
    # A improves mean versus O, but harms two images. G reduces these harms
    # without sacrificing A's mean benefit. Other controls remain reported.
    values = {"O": [1.0] * n,
              "A": [2.0, 2.0] + [0.6] * (n - 2),
              "G50": [1.6, 1.6] + [0.6] * (n - 2),
              "G75": [1.6, 1.6] + [0.6] * (n - 2)}
    records = []
    for domain in DOMAINS:
        for stream in STREAMS:
            for method in BASE_METHODS:
                for sid, error in zip(ids[domain], values.get(method, values["A"])):
                    records.append({"domain": domain, "stream": stream, "method": method,
                                    "sample_id": sid, "error": error, "valid": True,
                                    "original_rejected": False, "rotation_deg": error * 10,
                                    "translation_m": error / 10})
    return records, ids


def edit(records, method, sid, value, *, domain=None, stream=None):
    for row in records:
        if row["method"] == method and row["sample_id"] == str(sid) and (domain is None or row["domain"] == domain) and (stream is None or row["stream"] == stream):
            row["error"] = value
            row["valid"] = value is not None


def summary(result, method="G50", domain="lightbox"):
    return result["domains"][domain]["method_summaries"][method]


class RiskMetricTests(unittest.TestCase):
    def test_reference_is_fixed_before_candidates_and_missing_cannot_be_rescued(self):
        records, ids = fixture()
        edit(records, "O", 39, None)
        edit(records, "A", 39, None)
        edit(records, "G50", 0, None)
        result = evaluate_exploration(records, ids)
        stream = result["domains"]["lightbox"]["streams"]["2021"]
        self.assertEqual(stream["reference_count"], 39)
        g = stream["methods"]["G50"]
        self.assertEqual(g["coverage"]["valid_count"], g["coverage"]["a_valid_count"])
        self.assertEqual(g["coverage"]["newly_missing_reference_ids"], ["0"])
        self.assertEqual(g["coverage"]["invalid_to_valid_vs_a_ids"], ["39"])
        self.assertFalse(g["coverage"]["pass"])
        self.assertTrue(all(value is None for value in g["metrics"].values()))
        self.assertIsNone(summary(result)["metrics"]["mean_error"]["mean"])
        self.assertEqual(stream["methods"]["G75"]["reference_count"], 39)

    def test_coverage_outside_reference_must_still_not_drop_below_a(self):
        records, ids = fixture()
        edit(records, "O", 39, None)
        edit(records, "G50", 39, None)
        result = evaluate_exploration(records, ids)
        g = result["domains"]["lightbox"]["streams"]["2021"]["methods"]["G50"]
        self.assertTrue(g["coverage"]["reference_complete"])
        self.assertFalse(g["coverage"]["pass"])
        self.assertIsNotNone(g["metrics"]["mean_error"])

    def test_tail_includes_zeros_and_uses_ceiling(self):
        records, ids = fixture(41)
        for method in BASE_METHODS:
            if method != "O":
                edit(records, method, 1, 0.6)
        result = evaluate_exploration(records, ids)
        a = result["domains"]["lightbox"]["streams"]["2021"]["methods"]["A"]
        self.assertEqual(a["tail_count"], 3)
        self.assertAlmostEqual(a["metrics"]["mean_positive_degradation"], 1 / 41)
        self.assertAlmostEqual(a["metrics"]["worst_5pct_degradation"], 1 / 3)
        self.assertAlmostEqual(a["metrics"]["negative_transfer_rate"], 1 / 41)

    def test_equal_weight_streams_not_pooled_images(self):
        records, ids = fixture()
        for row in records:
            if row["stream"] == "2022" and row["method"] == "A" and int(row["sample_id"]) >= 2:
                row.update(error=None, valid=False)
        result = evaluate_exploration(records, ids)
        values = summary(result, "A")["metrics"]["mean_error"]["values"]
        self.assertAlmostEqual(values[0], 0.67)
        self.assertAlmostEqual(values[1], 2.0)
        self.assertAlmostEqual(values[2], 0.67)
        self.assertAlmostEqual(summary(result, "A")["metrics"]["mean_error"]["mean"], (0.67 + 2 + 0.67) / 3)

    def test_selection_exact_tie_prefers_75(self):
        records, ids = fixture()
        result = evaluate_exploration(records, ids)
        self.assertEqual(result["selection"]["selected_method"], "G75")
        self.assertTrue(result["selection"]["run_random_controls"])
        json.dumps(result, allow_nan=False)

    def test_domain_tradeoff_cannot_cancel_other_domain(self):
        records, ids = fixture()
        for method in ("G50", "G75"):
            edit(records, method, 0, 2.01, domain="sunlamp")
            edit(records, method, 1, 2.01, domain="sunlamp")
        result = evaluate_exploration(records, ids)
        self.assertIsNone(result["selection"]["selected_method"])
        self.assertFalse(result["selection"]["candidates"]["G50"]["domains"]["sunlamp"]["risk"]["mean_positive_degradation"]["pass"])

    def test_five_percent_bound_rejects_despite_better_risk(self):
        records, ids = fixture()
        for row in records:
            if row["method"] in ("G50", "G75") and int(row["sample_id"]) > 1:
                row["error"] = 0.75
        result = evaluate_exploration(records, ids)
        acc = result["selection"]["candidates"]["G50"]["domains"]["lightbox"]["accuracy"]
        self.assertFalse(acc["pass"])
        self.assertGreater(acc["candidate_mean"], acc["five_percent_bound"])

    def test_eighty_percent_gain_bound_can_be_stronger(self):
        records, ids = fixture()
        # O=1, A mean=.95; G mean=.975 passes 1.05*.95 but loses half A benefit.
        for row in records:
            if row["method"] == "A":
                row["error"] = 1.1 if int(row["sample_id"]) < 2 else (38.0 - 2.2) / 38
            if row["method"] in ("G50", "G75"):
                row["error"] = 1.05 if int(row["sample_id"]) < 2 else (39.0 - 2.1) / 38
        result = evaluate_exploration(records, ids)
        acc = result["selection"]["candidates"]["G50"]["domains"]["lightbox"]["accuracy"]
        self.assertLess(acc["candidate_mean"], acc["five_percent_bound"])
        self.assertGreater(acc["candidate_mean"], acc["eighty_percent_bound"])
        self.assertFalse(acc["pass"])

    def test_no_a_benefit_uses_offline_without_gain_fraction(self):
        records, ids = fixture()
        for row in records:
            if row["method"] == "A":
                row["error"] = 1.2
            if row["method"] in ("G50", "G75"):
                row["error"] = 1.01
        result = evaluate_exploration(records, ids)
        acc = result["selection"]["candidates"]["G50"]["domains"]["lightbox"]["accuracy"]
        self.assertFalse(acc["a_improves_offline"])
        self.assertIsNone(acc["retained_gain_fraction"])
        self.assertEqual(acc["fallback_offline_bound"], 1.0)
        self.assertFalse(acc["pass"])
        for method in ("G50", "G75"):
            for sid in ids["lightbox"]:
                edit(records, method, sid, 1.0)
        self.assertEqual(evaluate_exploration(records, ids)["selection"]["selected_method"], "G75")

    def test_zero_risk_is_not_an_improvement_percentage_or_extra_run_trigger(self):
        records, ids = fixture()
        for row in records:
            if row["method"] != "O":
                row["error"] = 0.6
        result = evaluate_exploration(records, ids)
        self.assertIsNone(result["selection"]["selected_method"])
        risk = result["selection"]["candidates"]["G50"]["domains"]["lightbox"]["risk"]["worst_5pct_degradation"]
        self.assertTrue(risk["pass"])
        self.assertTrue(risk["baseline_zero"])
        self.assertIsNone(risk["relative_reduction"])

    def test_random_masks_are_not_averaged_predictions_or_independent_streams(self):
        records, ids = fixture()
        for domain in DOMAINS:
            for stream in STREAMS:
                for mask in ("0", "1", "2"):
                    for sid in ids[domain]:
                        # Separate masks harm different samples. Risk after
                        # averaging predictions would incorrectly remove harm.
                        error = 2.0 if sid == mask else 0.2
                        records.append(dict(domain=domain, stream=stream, method="R", mask=mask,
                                            sample_id=sid, error=error, valid=True))
        result = evaluate_exploration(records, ids)
        r = summary(result, "R")
        self.assertEqual(r["stream_count"], 3)
        self.assertEqual(r["independent_unit"], "observation_stream")
        self.assertAlmostEqual(r["metrics"]["mean_positive_degradation"]["mean"], 1 / 40)
        self.assertAlmostEqual(r["metrics"]["worst_5pct_degradation"]["mean"], 0.5)
        records[-1].update(error=None, valid=False)
        incomplete = evaluate_exploration(records, ids)
        self.assertIsNone(summary(incomplete, "R", "sunlamp")["metrics"]["mean_error"]["mean"])

    def test_zero_baseline_risk_in_one_domain_allows_no_increase(self):
        records, ids = fixture()
        for row in records:
            if row["domain"] == "sunlamp" and row["method"] != "O":
                row["error"] = 0.6
        result = evaluate_exploration(records, ids)
        self.assertEqual(result["selection"]["selected_method"], "G75")

    def test_strict_rate_and_tolerance_diagnostic_are_separate(self):
        records, ids = fixture()
        edit(records, "G50", 0, 1 + 1e-13)
        edit(records, "G50", 1, 0.6)
        result = evaluate_exploration(records, ids)
        metrics = summary(result)["metrics"]
        self.assertAlmostEqual(metrics["negative_transfer_rate"]["mean"], 1 / 40)
        self.assertEqual(metrics["negative_transfer_rate_above_tolerance"]["mean"], 0.0)

    def test_empty_reference_remains_none(self):
        records, ids = fixture()
        for row in records:
            if row["method"] == "O":
                row.update(error=None, valid=False)
        result = evaluate_exploration(records, ids)
        self.assertIsNone(summary(result)["metrics"]["mean_positive_degradation"]["mean"])
        self.assertIsNone(result["selection"]["selected_method"])
        json.dumps(result, allow_nan=False)

    def test_rejected_but_finite_extended_prediction_kept(self):
        records, ids = fixture()
        records[0]["original_rejected"] = True
        # Offline is immutable across streams, so repeat that annotation.
        for row in records:
            if row["domain"] == "lightbox" and row["method"] == "O" and row["sample_id"] == "0":
                row["original_rejected"] = True
        result = evaluate_exploration(records, ids)
        o = result["domains"]["lightbox"]["streams"]["2021"]["methods"]["O"]
        self.assertEqual(o["coverage"]["original_rejected_count"], 1)
        self.assertEqual(o["reference_count"], 40)

    def test_inputs_immutable_and_optional_raw_metrics_not_fabricated(self):
        records, ids = fixture()
        for row in records:
            row.pop("rotation_deg")
            row.pop("translation_m")
        before = copy.deepcopy(records)
        result = evaluate_exploration(records, ids)
        self.assertEqual(records, before)
        self.assertIsNone(summary(result)["metrics"]["mean_rotation_deg"]["mean"])
        self.assertIsNotNone(summary(result)["metrics"]["mean_error"]["mean"])

    def test_input_validation_blocks_nonfinite_negative_bool_and_mismatch(self):
        for value in (float("nan"), float("inf"), -1.0, True, "0.1"):
            records, ids = fixture()
            records[0]["error"] = value
            with self.subTest(value=value), self.assertRaises(MetricContractError):
                evaluate_exploration(records, ids)
        records, ids = fixture()
        records[0]["valid"] = False
        with self.assertRaises(MetricContractError):
            evaluate_exploration(records, ids)

    def test_derived_numeric_overflow_cannot_escape_as_nonstandard_json(self):
        records, ids = fixture()
        for row in records:
            row["error"] = 1.79e308 if row["method"] == "O" else 1.78e308
            row["rotation_deg"] = None
            row["translation_m"] = None
        with self.assertRaises(MetricContractError):
            evaluate_exploration(records, ids)

    def test_incomplete_duplicate_unknown_or_changed_offline_rows_block(self):
        cases = ("truncated", "duplicate", "outside", "offline_changed", "wrong_stream", "partial_R")
        for case in cases:
            records, ids = fixture()
            if case == "truncated":
                records.pop()
            elif case == "duplicate":
                records.append(copy.deepcopy(records[0]))
            elif case == "outside":
                records[0]["sample_id"] = "reserved_image"
            elif case == "offline_changed":
                records[0]["error"] = 1.1
            elif case == "wrong_stream":
                records[0]["stream"] = "2024"
            elif case == "partial_R":
                records.append(dict(records[-1], method="R", mask="0"))
            with self.subTest(case=case), self.assertRaises(MetricContractError):
                evaluate_exploration(records, ids)

    def test_max_domain_tail_precedes_better_average(self):
        records, ids = fixture()
        # G50: maximum domain tail .7, average .4. G75: .6, average .6.
        # Both qualify; predeclared minimax picks G75 despite G50 mean benefit.
        for domain, value in (("lightbox", 1.7), ("sunlamp", 1.1)):
            for sid in (0, 1):
                edit(records, "G50", sid, value, domain=domain)
        result = evaluate_exploration(records, ids)
        self.assertEqual(result["selection"]["selected_method"], "G75")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RiskMetricTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {"kind": "synthetic_cpu_contract_tests_not_scientific_results",
                "tests_run": result.testsRun, "failures": len(result.failures),
                "errors": len(result.errors), "passed": result.wasSuccessful(),
                "files": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("risk_metrics.py", "test_risk_metrics.py")}}
    print("ODR003_METRICS_TEST_EVIDENCE=" + json.dumps(evidence, allow_nan=False, sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
