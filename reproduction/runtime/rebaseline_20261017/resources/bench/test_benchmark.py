"""Small CPU regression contracts; fixtures never claim real GPU measurements."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from bench_contract import (BENCH, PLAN, HEADS, check_measurement, check_source_report,
                            inspect_parameters, latency_stats, sha)
from odr_resources import check_record, check_against_sources


def measurement():
    values = [float(x) for x in range(1, 31)]
    return {
        "input_shape": [1, 3, 512, 768], "input_kind": "fixed_zero_tensor_cuda_resident",
        "warmup_count": 10, "forward_count": 40, "dtype": "torch.float32",
        "eval_mode": True, "no_grad": True, "all_EHS_heads": list(HEADS), "autocast": False,
        "timing_method": "CUDA events with synchronize after every forward",
        "weights_unchanged": True, "strict_reload": True, "finite_output_verified": True,
        "model_digest_before": "same", "model_digest_after": "same",
        "latencies_ms": values, "statistics": latency_stats(values),
        "memory_bytes": {"baseline_allocated": 100, "baseline_reserved": 128,
                         "peak_allocated": 200, "peak_reserved": 256},
        "parameter_counts": {"whole_spnv2": 123, "odr_heads": 0},
    }


class Tests(unittest.TestCase):
    def test_statistics_independent_numpy_reference(self):
        values = [float(x) for x in range(1, 31)]
        stats = latency_stats(values)
        self.assertEqual(stats["sample_count"], 30)
        self.assertEqual(stats["mean_ms"], 15.5)
        self.assertEqual(stats["median_ms"], float(np.median(values)))
        for field, fraction in (("p05_ms", .05), ("p90_ms", .9), ("p95_ms", .95)):
            self.assertAlmostEqual(stats[field], float(np.quantile(values, fraction, method="linear")))
        self.assertEqual(stats["images_per_second"], 1000 / 15.5)
        self.assertAlmostEqual(stats["population_std_ms"], float(np.std(values)))

    def test_bad_sample_count_nonfinite_or_nonpositive_rejected(self):
        for values in ([1.] * 29, [1.] * 31, [1.] * 29 + [0.],
                       [1.] * 29 + [float("nan")], [1.] * 29 + [float("inf")],
                       [1.] * 29 + [True]):
            with self.subTest(last=values[-1], count=len(values)), self.assertRaises(ValueError):
                latency_stats(values)

    def test_complete_synthetic_measurement_contract(self):
        with patch("torch.cuda.init", side_effect=AssertionError("CUDA initialization forbidden")):
            self.assertEqual(check_measurement(measurement())["mean_ms"], 15.5)

    def test_bad_warmup_forward_dtype_shape_head_counts_rejected(self):
        cases = {
            "warmup_count": 9, "forward_count": 39, "dtype": "torch.float16",
            "input_shape": [4, 3, 512, 768], "all_EHS_heads": ["heatmap", "efficientpose"],
            "no_grad": False, "autocast": True, "strict_reload": False,
            "finite_output_verified": False, "model_digest_after": "different",
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                item = measurement(); item[key] = value
                with self.assertRaises(ValueError):
                    check_measurement(item)

    def test_modified_summary_memory_and_parameter_denominator_rejected(self):
        for field in ("statistics", "memory", "parameters"):
            item = measurement()
            if field == "statistics":
                item["statistics"]["mean_ms"] += 1
            elif field == "memory":
                item["memory_bytes"]["peak_allocated"] = 99
            else:
                item["parameter_counts"]["odr_heads"] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_measurement(item)

    def test_accepted_final_path_config_status_and_hash_required(self):
        with tempfile.TemporaryDirectory(prefix="source_contract_", dir=BENCH / "test_artifacts") as folder:
            final = Path(folder) / "model_final.pth.tar"
            cfg = Path(folder) / "config.yaml"
            final.write_bytes(b"synthetic source file; not a GPU model")
            cfg.write_text("fixture")
            report = {"status": "passed", "cfg": str(cfg), "final": str(final), "final_sha256": sha(final),
                      "strict_reload": True, "final_matches_checkpoint": True}
            check_source_report(report, cfg, final)
            for key, value in (("status", "running"), ("cfg", "/wrong"), ("final_sha256", "wrong"),
                               ("strict_reload", False), ("final_matches_checkpoint", False)):
                changed = copy.deepcopy(report); changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check_source_report(changed, cfg, final)
            final.write_bytes(b"changed synthetic file")
            with self.assertRaises(ValueError):
                check_source_report(report, cfg, final)

    def test_actual_cpu_parameter_inventory_denominators_and_corruption(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.efficientnet = torch.nn.Linear(2, 2)
                self.backbone.bifpn = torch.nn.Linear(2, 2)
                self.heads = torch.nn.ModuleList([torch.nn.Linear(2, 1) for _ in range(3)])
        model = Tiny()
        all_parameters = dict(model.named_parameters())
        encoder = sum(p.numel() for n, p in all_parameters.items() if n.startswith("backbone.efficientnet."))
        backbone = sum(p.numel() for n, p in all_parameters.items() if n.startswith("backbone."))
        inventory = {
            "full_parameter_inventory": [{"name": n, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype)}
                                         for n, p in all_parameters.items()],
            "odr_allowed_names": ["backbone.efficientnet.bias"],
            "parameter_counts": {"whole_spnv2": sum(p.numel() for p in all_parameters.values()),
                                 "efficientnet_encoder": encoder, "efficientdet_backbone": backbone,
                                 "odr_updated_backbone_norm_affine": 2},
        }
        with patch("torch.cuda.init", side_effect=AssertionError("CUDA initialization forbidden")):
            inspect_parameters(model, inventory)
        wrong = copy.deepcopy(inventory)
        wrong["parameter_counts"]["whole_spnv2"] += 1
        with self.assertRaises(ValueError):
            inspect_parameters(model, wrong)
        wrong = copy.deepcopy(inventory); wrong["full_parameter_inventory"][0]["shape"] = [999]
        with self.assertRaises(ValueError):
            inspect_parameters(model, wrong)

    def test_real_odr_resource_schema_and_bad_counts_or_hash_contract(self):
        task = {"id": "fixture_only", "N": 1024, "B": 4}
        good = {"task": task, "status": "adaptation_passed_evaluation_pending",
                "N_seen": 1024, "optimizer_steps": 256, "adaptation_seconds": 1.25,
                "peak_allocated_bytes": 100, "peak_reserved_bytes": 128, "trainable_parameters": 44}
        check_record(good, task)
        for key, value in (("N_seen", 1023), ("optimizer_steps", 255), ("adaptation_seconds", float("nan")),
                           ("peak_reserved_bytes", 99), ("trainable_parameters", 0)):
            bad = copy.deepcopy(good); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_record(bad, task)

    def test_odr_source_hash_and_parameter_denominator_crosscheck(self):
        entries = [{'name': 'phi3_bn_ehs', 'final_sha256': 'source-sha', 'parameter_counts': {'odr_updated_backbone_norm_affine': 44}}]
        actual = {'records': [{'phi': 3, 'source_sha256': 'source-sha', 'trainable_parameters': 44}]}
        self.assertTrue(check_against_sources(actual, entries))
        for key, wrong in [('source_sha256', 'wrong-sha'), ('trainable_parameters', 43)]:
            bad = copy.deepcopy(actual); bad['records'][0][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_against_sources(bad, entries)

    def test_pending_definition_not_an_execution_manifest(self):
        value = json.loads((BENCH / "pending_task.json").read_text())
        self.assertFalse(value["execution_manifest"])
        self.assertFalse(value["activation_allowed"])
        self.assertNotIn("tasks", value)
        task = value["pending_task_definition"]
        self.assertEqual(task["verify_gpu"], False)
        self.assertEqual(task["verify_env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(len(task["metadata"]["source_acceptance_task_ids"]), 16)
        self.assertEqual(len(task["metadata"]["fixed_odr_task_ids"]), 40)
        self.assertIn("eval_table4_phi3_gn_sunlamp", task["requires"])
        for path, expected in task["metadata"]["asset_sha256"].items():
            self.assertEqual(sha(path), expected)

    def test_main_cli_proves_gate_before_cuda_and_no_new_odr_command(self):
        source = (BENCH / "run_benchmark.py").read_text()
        self.assertLess(source.index("entries = sources()"), source.index("ready, reason = runner.gpu_available"))
        self.assertLess(source.index("ready, reason = runner.gpu_available"), source.index("import torch\n"))
        self.assertNotIn("run_task.py", source)
        self.assertNotIn("audited_entry.py", source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    (BENCH / "test_artifacts").mkdir(exist_ok=True)
    assert os.environ.get("STY") and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    value = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "scope": "Synthetic CPU statistics/contracts and tiny parameter inventory only; no real GPU measurements",
        "tests": result.testsRun, "failures": [(str(t), text) for t, text in result.failures + result.errors],
        "screen": os.environ["STY"], "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "code_sha256": {str(p): sha(p) for p in BENCH.glob("*.py")}, "command": sys.argv,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
