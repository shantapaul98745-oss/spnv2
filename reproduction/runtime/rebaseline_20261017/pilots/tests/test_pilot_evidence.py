"""CPU evidence acceptance/negative corruption tests; run only in a guarded screen.

Fixtures are tiny synthetic checkpoint evidence and never claim a GPU run happened.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

PILOTS = Path(__file__).resolve().parents[1]
PLAN = PILOTS.parent
sys.path.insert(0, str(PILOTS))
sys.path.insert(0, str(PLAN / "source/core"))
from config import cfg as defaults, update_config
from evidence import (ARTIFACT_NAMES, exact_tree, model_sha256, parameter_changes,
                      protocol, sha256_file, tree_sha256, write_json)
from verify_pilot import independent_order, verify


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cpu_fixture_", dir=PILOTS / "test_artifacts")
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        raw_cfg = yaml.safe_load((PILOTS / "configs/phi6_gn_ehs.yaml").read_text())
        raw_cfg["DATASET"]["ROOT"] = str(self.root / "data")
        self.cfg_path = self.root / "fixture.yaml"
        self.cfg_path.write_text(yaml.safe_dump(raw_cfg))
        self.cfg = defaults.clone()
        update_config(self.cfg, argparse.Namespace(cfg=str(self.cfg_path), opts=[]))
        self.config = yaml.safe_load(self.cfg.dump())
        self.rows = 128
        data_root = self.root / "data/speedplus"
        for name, count in (("train", 129), ("validation", 2)):
            csv_path = data_root / "synthetic/labels" / (name + ".csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                for index in range(count):
                    writer.writerow([name + str(index).zfill(4) + ".jpg", index])
        for field in ("CAMERA", "CADMODEL"):
            p = Path(self.config["DATASET"][field])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"tiny synthetic geometry")
        records, coverage = [], {}
        for split, csv_field in (("train", "TRAIN_CSV"), ("val", "VAL_CSV")):
            csv_path = data_root / self.config["TRAIN"][csv_field]
            rows = list(csv.reader(csv_path.open()))
            if split == "train":
                rows = rows[:128]
            folders = ["images_768x512_RGB"]
            if split == "train":
                folders += ["masks_192x128", "styles_768x512_RGB"]
            for row, values in enumerate(rows):
                for folder in folders:
                    asset = data_root / "synthetic" / folder / values[0]
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    asset.write_bytes(("fixture " + values[0]).encode())
                    stat = asset.stat()
                    records.append([split, row, str(asset), stat.st_size, stat.st_mtime_ns])
            coverage[split] = {"samples": len(rows), "folders": folders, "files": len(rows) * len(folders)}
        self.initial = {
            "backbone.weight": torch.tensor([.25, .5]),
            "heads.0.weight": torch.tensor([.1, .2]),
            "heads.1.weight": torch.tensor([.3, .4]),
            "heads.2.weight": torch.tensor([.5, .6]),
        }
        self.final = {k: v + .125 for k, v in self.initial.items()}
        code = {str(p.relative_to(PLAN / "source")): sha256_file(p)
                for folder in ("core", "tools") for p in sorted((PLAN / "source" / folder).rglob("*.py"))}
        geometry = {}
        for field in ("CAMERA", "KEYPOINTS", "CADMODEL"):
            p = Path(self.config["DATASET"][field])
            geometry[field] = {"path": str(p.absolute()), "exists": p.is_file(),
                               "sha256": sha256_file(p) if p.is_file() else None}
        csvs = {str(data_root / self.config["TRAIN"][key]): sha256_file(data_root / self.config["TRAIN"][key])
                for key in ("TRAIN_CSV", "VAL_CSV")}
        self.manifest = {
            "version": 1, "config": self.cfg.dump(),
            "config_sha256": hashlib.sha256(self.cfg.dump().encode()).hexdigest(),
            "code": code, "csv": csvs, "initialization": {},
            "auxiliary_geometry": geometry, "runtime": {"fixture": True},
            "sampler": "epoch-keyed-v1", "augmentation": "sample-keyed-v1",
            "initial_state_sha256": model_sha256(self.initial),
            "data_inventory": {
                "version": 1, "hash_kind": "sha256-of-path-size-mtime_ns-records",
                "raw_content_hashed": False, "records": records, "coverage": coverage,
                "sha256": hashlib.sha256(json.dumps(records, ensure_ascii=False,
                                                    separators=(",", ":")).encode()).hexdigest(),
            },
        }
        self.order = independent_order(128, 2021, True)
        self.ids = ["train" + str(i).zfill(4) + ".jpg" for i in self.order]
        group = {"params": list(range(4)), "lr": .001, "weight_decay": .0001,
                 "betas": (.9, .999), "eps": .0001}
        def checkpoint(attempts):
            successes = max(0, attempts - 4)
            rng = {
                "python": random.getstate(),
                "numpy": {"kind": "MT19937", "keys": list(range(624)), "pos": 2,
                          "has_gauss": 0, "cached_gaussian": 0.0},
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [torch.arange(8, dtype=torch.uint8)],
            }
            state = {
                "epoch": 0, "next_batch": attempts * 4, "update_attempts": attempts,
                "optimizer_updates": successes, "amp_skips": min(4, attempts),
                "consecutive_amp_skips": 0, "samples_seen": attempts * 4,
                "microbatches_seen": attempts * 4, "last_loss": 1.0 if attempts else None,
            }
            optimizer = {"param_groups": [copy.deepcopy(group)], "state": {}}
            if attempts:
                optimizer["state"] = {
                    index: {"step": torch.tensor(float(successes)), "exp_avg": torch.zeros(2),
                            "exp_avg_sq": torch.ones(2)} for index in range(4)}
            return {
                "format": "spnv2-boundary-v1", "state_dict": copy.deepcopy(self.final if attempts == 32 else self.initial),
                "optimizer": optimizer,
                "scaler": {"scale": 4096.0, "growth_factor": 2.0, "backoff_factor": .5,
                           "growth_interval": 2000, "_growth_tracker": successes},
                "training_state": state, "rng": rng,
                "sampler": {"version": 1, "size": 128, "batch_size": 1, "accum_steps": 4,
                            "seed": 2021, "shuffle": True, "epoch": 0, "next_batch": attempts * 4,
                            "total_batches": 128,
                            "order_sha256": hashlib.sha256(np.asarray(self.order, dtype="<i8").tobytes()).hexdigest()},
                "manifest": copy.deepcopy(self.manifest),
            }
        self.whole = checkpoint(32)
        files = {
            "model_initial.pth.tar": self.initial, "initial_checkpoint.pth.tar": checkpoint(0),
            "boundary_checkpoint.pth.tar": checkpoint(16), "model_final.pth.tar": self.final,
            "continuous_final_checkpoint.pth.tar": self.whole,
            "model_resumed_final.pth.tar": self.final,
            "resumed_final_checkpoint.pth.tar": copy.deepcopy(self.whole),
        }
        for name, value in files.items():
            torch.save(value, self.artifacts / name)
        write_json(self.artifacts / "manifest.json", self.manifest)
        self.report = {
            "schema": "spnv2-engineering-pilot-v1", "status": "passed", "exitcode": 0,
            "fixture_only_not_gpu_evidence": True, "source": str(PLAN / "source"),
            "cfg": str(self.cfg_path), "cfg_file_sha256": sha256_file(self.cfg_path),
            "effective_config": self.cfg.dump(), "effective_config_sha256": self.manifest["config_sha256"],
            "test_sha256": sha256_file(PILOTS / "gpu_smoke.py"),
            "evidence_helper_sha256": sha256_file(PILOTS / "evidence.py"),
            "attempts": 32, "checkpoint_attempt": 16, "physical_batch": 1, "accumulation": 4,
            "subset_rows": 128, "seed": 2021, "heads": ["heatmap", "efficientpose", "segmentation"],
            "protocol_checks": protocol(self.config, 32, 16),
            "fresh_initialization_equal": True, "pilot_weights_for_formal": False,
            "clip_norm": 1.0, "loss_normalization": "local_mean_divided_by_accumulation",
            "observed_training_shapes": [[1, 3, 512, 768]], "observed_forward_count": 192,
            "runtime": {"torch_home": str(PLAN.parents[1] / "env/cache/torch"), "screen": os.environ.get("STY")},
            "initialization_disclosure": "GN public code random initialization; author initialization unverified",
            "artifacts_directory": str(self.artifacts), "manifest": self.manifest, "code_sha256": code,
            "original_training_rows": 129, "expected_sample_ids": self.ids,
            "trainable_parameter_names": list(self.initial),
            "component_parameter_changes": parameter_changes(self.initial, self.final, list(self.initial),
                ["heatmap", "efficientpose", "segmentation"]),
            "continuous": {"state": copy.deepcopy(self.whole["training_state"]), "sample_ids": self.ids},
            "resume": {"state": copy.deepcopy(self.whole["training_state"]), "sample_ids": self.ids[64:],
                       "exact": True, "model_equal": True, "optimizer_equal": True,
                       "scaler_equal": True, "counters_equal": True, "rng_equal": True,
                       "sample_ids_equal": True, "checkpoint_attempt": 16},
        }
        self.refresh_artifacts()
        self.report_path = self.root / "synthetic_fixture_report.json"
        self.save_report()

    def tearDown(self):
        self.temp.cleanup()

    def refresh_artifacts(self):
        self.report["artifacts"] = {
            name: {"path": str(self.artifacts / name), "sha256": sha256_file(self.artifacts / name),
                   "size": (self.artifacts / name).stat().st_size} for name in ARTIFACT_NAMES}

    def save_report(self):
        write_json(self.report_path, self.report)

    def corrupt_checkpoint(self, name, change):
        path = self.artifacts / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        change(payload)
        torch.save(payload, path)
        self.refresh_artifacts()
        self.save_report()

    def test_cpu_fixture_all_evidence_passes(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
        self.assertEqual(os.environ.get("PYTHONNOUSERSITE"), "1")
        self.assertTrue(os.environ.get("STY"))
        with unittest.mock.patch("torch.cuda.init", side_effect=AssertionError("CUDA initialization forbidden")):
            result = verify(self.report_path)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["checkpoint_tree_sha256"]["continuous_final_checkpoint.pth.tar"],
                         result["checkpoint_tree_sha256"]["resumed_final_checkpoint.pth.tar"])

    def test_config_copies_only_change_owned_output_names(self):
        for name in ("phi6_gn_ehs", "table2_h", "table2_e", "table2_eh"):
            with self.subTest(name=name):
                old = yaml.safe_load((PLAN / "configs" / (name + ".yaml")).read_text())
                new = yaml.safe_load((PILOTS / "configs" / (name + ".yaml")).read_text())
                for key in ("EXP_NAME", "OUTPUT_DIR", "LOG_DIR"):
                    self.assertNotEqual(old.pop(key), new.pop(key))
                self.assertEqual(old, new)
                loaded = defaults.clone()
                update_config(loaded, argparse.Namespace(cfg=str(PILOTS / "configs" / (name + ".yaml")), opts=[]))
                self.assertTrue(all(protocol(yaml.safe_load(loaded.dump()), 32, 16).values()))

    def test_invalid_head_loss_and_physical_protocol_rejected(self):
        for field, value in (("LOSS_HEADS", ["heatmap"]), ("NAMES", ["heatmap", "heatmap"]),
                             ("LOSS_FACTORS", [1, .5, 1]), ("LOSS_NUMS", [1, 1, 1])):
            with self.subTest(field=field):
                cfg = copy.deepcopy(self.config)
                cfg["MODEL"]["HEAD"][field] = value
                with self.assertRaises(ValueError):
                    protocol(cfg, 32, 16)
        for field, value in (("IMAGES_PER_GPU", 2), ("ACCUM_STEPS", 1), ("LR", .004)):
            cfg = copy.deepcopy(self.config)
            cfg["TRAIN"][field] = value
            with self.assertRaises(ValueError):
                protocol(cfg, 32, 16)
        with self.assertRaises(ValueError):
            protocol(self.config, 31, 16)

    def test_unrehashable_artifact_corruption_rejected(self):
        with (self.artifacts / "model_final.pth.tar").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Artifact size/hash mismatch"):
            verify(self.report_path)

    def test_rehashed_model_corruption_rejected(self):
        self.corrupt_checkpoint("resumed_final_checkpoint.pth.tar",
                                lambda p: p["state_dict"]["heads.1.weight"].add_(1))
        with self.assertRaisesRegex(ValueError, "Full checkpoint state differs"):
            verify(self.report_path)

    def test_rehashed_optimizer_corruption_rejected(self):
        self.corrupt_checkpoint("resumed_final_checkpoint.pth.tar",
                                lambda p: p["optimizer"]["state"][0]["exp_avg"].add_(1))
        with self.assertRaisesRegex(ValueError, "Full checkpoint state differs"):
            verify(self.report_path)

    def test_rehashed_scaler_corruption_rejected(self):
        self.corrupt_checkpoint("resumed_final_checkpoint.pth.tar",
                                lambda p: p["scaler"].update(scale=2048.0))
        with self.assertRaisesRegex(ValueError, "Full checkpoint state differs"):
            verify(self.report_path)

    def test_rehashed_rng_corruption_rejected(self):
        self.corrupt_checkpoint("resumed_final_checkpoint.pth.tar",
                                lambda p: p["rng"]["torch_cuda"][0].add_(1))
        with self.assertRaisesRegex(ValueError, "Full checkpoint state differs"):
            verify(self.report_path)

    def test_rehashed_boundary_counter_corruption_rejected(self):
        self.corrupt_checkpoint("boundary_checkpoint.pth.tar",
                                lambda p: p["training_state"].update(next_batch=63))
        with self.assertRaisesRegex(ValueError, "complete update boundary"):
            verify(self.report_path)

    def test_rehashed_sampler_corruption_rejected(self):
        self.corrupt_checkpoint("resumed_final_checkpoint.pth.tar",
                                lambda p: p["sampler"].update(order_sha256="bad"))
        with self.assertRaisesRegex(AssertionError, "checkpoint.sampler"):
            verify(self.report_path)

    def test_final_model_snapshot_corruption_rejected(self):
        path = self.artifacts / "model_resumed_final.pth.tar"
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        snapshot["backbone.weight"].add_(1)
        torch.save(snapshot, path)
        self.refresh_artifacts()
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Final model snapshot differs"):
            verify(self.report_path)

    def test_actual_sample_id_and_shape_claim_corruption_rejected(self):
        original = copy.deepcopy(self.report)
        for field in ("ids", "shape", "forwards", "successful"):
            with self.subTest(field=field):
                self.report = copy.deepcopy(original)
                if field == "ids":
                    self.report["resume"]["sample_ids"][0] = "wrong.jpg"
                elif field == "shape":
                    self.report["observed_training_shapes"] = [[1, 3, 256, 384]]
                elif field == "forwards":
                    self.report["observed_forward_count"] = 190
                else:
                    self.report["continuous"]["state"]["optimizer_updates"] = 2
                self.save_report()
                with self.assertRaises((ValueError, AssertionError)):
                    verify(self.report_path)

    def test_missing_active_head_update_rejected(self):
        final = copy.deepcopy(self.final)
        final["heads.2.weight"] = self.initial["heads.2.weight"]
        with self.assertRaisesRegex(ValueError, "no changed parameter"):
            parameter_changes(self.initial, final, list(self.initial), self.report["heads"])

    def test_config_and_source_hash_claim_corruption_rejected(self):
        for field in ("cfg_file_sha256", "test_sha256", "effective_config_sha256"):
            with self.subTest(field=field):
                saved = self.report[field]
                self.report[field] = "bad"
                self.save_report()
                with self.assertRaises(ValueError):
                    verify(self.report_path)
                self.report[field] = saved
        self.report["code_sha256"] = {"core/engine/trainer.py": "bad"}
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Frozen source hash"):
            verify(self.report_path)

    def test_data_inventory_asset_change_rejected(self):
        asset = Path(self.manifest["data_inventory"]["records"][0][2])
        asset.write_bytes(b"asset changed")
        with self.assertRaisesRegex(ValueError, "Referenced input asset changed"):
            verify(self.report_path)

    def test_tree_digest_covers_dtype_shape_and_all_state_bytes(self):
        self.assertNotEqual(tree_sha256(torch.tensor([1.])), tree_sha256(torch.tensor([1.], dtype=torch.float64)))
        self.assertNotEqual(tree_sha256(torch.tensor([1.])), tree_sha256(torch.tensor([[1.]])))
        self.assertNotEqual(tree_sha256({"rng": [1, 2]}), tree_sha256({"rng": [1, 3]}))
        with self.assertRaises(ValueError):
            tree_sha256(torch.tensor([float("nan")]))


if __name__ == "__main__":
    from unittest import mock
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    (PILOTS / "test_artifacts").mkdir(exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PilotTests))
    evidence = {
        "scope": "synthetic CPU evidence/negative corruption tests only; GPU pilots not run",
        "tests_run": result.testsRun, "successful": result.wasSuccessful(),
        "failures": [(str(t), message) for t, message in result.failures],
        "errors": [(str(t), message) for t, message in result.errors],
        "screen": os.environ.get("STY"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "code_sha256": {str(p): sha256_file(p) for p in
            (Path(__file__).resolve(), PILOTS / "gpu_smoke.py", PILOTS / "verify_pilot.py", PILOTS / "evidence.py")},
        "command": sys.argv,
    }
    write_json(args.summary, evidence)
    raise SystemExit(0 if result.wasSuccessful() else 1)
