"""Local mock-only checks of developer-evaluation recovery; no model forwards."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASELINE = Path(__file__).resolve().parents[2] / "odr003_exploratory_20261023" / "implementation"
sys.path.insert(0, str(BASELINE))
from durable_state import DurableRunStore
from frozen_backend import sha256
import run_round2


class PlaceholderImage:
    def unsqueeze(self, _axis):
        return self


class FakeDeveloperDataset:
    def __init__(self):
        self.manifest = {"items": [{"sample_id": f"synthetic/{i:04d}"} for i in range(512)]}
        self.requested = []

    def __len__(self):
        return 512

    def __getitem__(self, index):
        self.requested.append(index)
        return {"sample_id": self.manifest["items"][index]["sample_id"], "image": PlaceholderImage()}


class FakePrediction:
    def __init__(self, *, fail_call=None):
        self.calls = 0
        self.fail_call = fail_call

    def extract(self, *args, **kwargs):
        self.calls += 1
        if self.fail_call is not None and self.calls == self.fail_call:
            raise RuntimeError("synthetic interruption; no model was invoked")
        return {"synthetic_prediction": 7, "h_available": True, "e_available": True,
                "original_h_rejected": False, "heat_q": [1, 0, 0, 0], "effi_t": [0, 0, 10],
                "timing_seconds": {"forward": 0.0}}

    def developer_metrics(self, record, label, cfg):
        return {"valid": True, "original_valid": True, "error": 0.0, "original_error": 0.0,
                "rotation_deg": 0.0, "translation_m": 0.0, "synthetic_metric": 0.0}


class EvaluationRecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = FakeDeveloperDataset()
        labels = [{"sample_id": row["sample_id"], "quaternion_wxyz": [1, 0, 0, 0], "translation_m": [0, 0, 10]}
                  for row in self.dataset.manifest["items"]]
        self.label_path = self.root / "synthetic_dev_labels.json"
        self.write_json(self.label_path, {"items": labels, "image_manifest_sha256": "d" * 64})
        self.task = {"baseline_task": {"bindings": {
            "developer_labels": {"path": str(self.label_path), "sha256": sha256(self.label_path)},
            "developer_manifest": {"path": "unused-mock-manifest", "sha256": "d" * 64}}}}
        self.paths = {"developer_labels": self.label_path, "developer_manifest": "unused-mock-manifest"}
        identity = dict(run_id="mock-evaluation", source_sha256="a" * 64,
                        config_sha256="b" * 64, manifest_sha256="c" * 64)
        self.store = DurableRunStore(self.root / "store", identity, state_validator=lambda state: None)
        self.state = dict(identity=identity,
                          counters=dict(observed=1024, candidates=256, commits=128, rollbacks=128),
                          next_index=1024, learner={"learning": {"model": {"fixed": 7}}})
        self.cursor = dict(observed=1024, candidates=256, commits=128, rollbacks=128, next_index=1024)
        self.store.save(self.state, self.cursor)
        self.model = SimpleNamespace(state_dict=lambda: {"fixed": 7})
        self.module = SimpleNamespace(ImageManifestDataset=lambda *a, **k: self.dataset)
        self.progress = []

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_json(path, value):
        import json
        Path(path).write_text(json.dumps(value), encoding="utf8")

    def evaluate(self, prediction, conflict=lambda: None):
        with patch.object(run_round2, "atomic_json", side_effect=self.write_json), \
             patch.object(run_round2, "_progress", side_effect=lambda *a, **k: self.progress.append(k)):
            return run_round2.evaluate_developer(
                self.task, self.paths, None, None, None, prediction,
                self.module, None, self.model, None, SimpleNamespace(phase="adaptation"),
                self.store, self.store.recover().state, self.root, conflict, lambda: None)

    def test_interruption_resumes_after_complete_64_rows_and_charges_replay(self):
        failed = FakePrediction(fail_call=71)
        with self.assertRaises(RuntimeError):
            self.evaluate(failed)
        saved = self.store.recover().state
        self.assertEqual(saved["evaluation"]["next_index"], 64)
        self.assertEqual(len(saved["evaluation"]["rows"]), 64)
        self.dataset.requested.clear()
        resumed = FakePrediction()
        report = self.evaluate(resumed)
        self.assertEqual(self.dataset.requested[0], 64)
        self.assertEqual(resumed.calls, 448)
        self.assertEqual(report["count"], 512)
        final = self.store.recover().state["evaluation"]
        self.assertEqual(len(final["rows"]), 512)
        self.assertEqual(len({row["sample_id"] for row in final["rows"]}), 512)
        work = self.store.work_summary()
        self.assertEqual(work["forwards_confirmed_completed"]["developer_evaluation"], 518)
        self.assertEqual(work["replay_forwards_confirmed_completed"]["developer_evaluation"], 7)
        self.assertEqual(work["forwards_inflight_unknown_upper_bound"], {"developer_evaluation": 1})
        publications = [event["payload"] for event in self.store.read_journal()[0]
                        if event["kind"] == "checkpoint_published"]
        self.assertTrue(publications)
        self.assertTrue(all(event["checkpoint_seconds"] >= 0.0 for event in publications))
        self.assertNotIn("checkpoint_seconds", str(final["rows"]))

    def test_final_evaluation_resume_performs_zero_extra_model_calls(self):
        self.evaluate(FakePrediction())
        repeated = FakePrediction()
        report = self.evaluate(repeated)
        self.assertEqual(repeated.calls, 0)
        self.assertEqual(report["count"], 512)

    def test_resource_wait_prevents_next_evaluation_batch(self):
        count = {"checks": 0}
        def conflict():
            count["checks"] += 1
            return "foreign_compute" if count["checks"] == 2 else None
        fake = FakePrediction()
        report = self.evaluate(fake, conflict)
        self.assertTrue(report["waiting_resources"])
        self.assertEqual(fake.calls, 64)
        self.assertEqual(self.store.recover().state["evaluation"]["next_index"], 64)
        self.assertEqual(self.progress[-1]["status"], "waiting_resources")
        self.assertEqual(self.progress[-1]["evaluated"], 64)

    def test_scientific_count_adapter_supports_reference_without_engine_counters(self):
        reference = SimpleNamespace(adapter=SimpleNamespace(observed=128, accepted=75, updates=18, tail=3))
        counts = run_round2.scientific_counts(reference)
        self.assertEqual(counts, dict(observed=128, accepted=75, candidates=18, commits=18,
                                     rollbacks=0, tail_discarded=3, consecutive_rollbacks=0,
                                     max_consecutive_rollbacks=0))
        protected = SimpleNamespace(counters=dict(observed=128, commits=7, rollbacks=10))
        self.assertEqual(run_round2.scientific_counts(protected), protected.counters)

    def test_malformed_complete_evaluation_must_not_claim_512_results(self):
        bad = copy.deepcopy(self.state)
        bad["evaluation"] = {"next_index": 512, "rows": []}
        self.store.save(bad, self.cursor)
        fake = FakePrediction()
        with self.assertRaises(ValueError):
            self.evaluate(fake)
        self.assertEqual(fake.calls, 0)

    def test_reordered_saved_evaluation_ids_are_rejected_before_forward(self):
        bad = copy.deepcopy(self.state)
        rows = [{"sample_id": f"synthetic/{i:04d}", "prediction": {},
                 "metrics": {"valid": True, "original_valid": True}} for i in range(64)]
        rows.reverse()
        bad["evaluation"] = {"next_index": 64, "rows": rows}
        self.store.save(bad, self.cursor)
        fake = FakePrediction()
        with self.assertRaises(ValueError):
            self.evaluate(fake)
        self.assertEqual(fake.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
