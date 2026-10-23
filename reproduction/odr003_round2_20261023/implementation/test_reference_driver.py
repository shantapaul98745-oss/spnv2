"""Independent direct-A reference tests on local synthetic CPU data."""
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

BASELINE = Path(__file__).resolve().parents[2] / "odr003_exploratory_20261023" / "implementation"
sys.path.insert(0, str(BASELINE))
from durable_state import DurableRunStore
from guard_pools import GuardPools
from protected_engine import ProtectedEngine, validate_engine_state
from protected_learning import WindowLearner
from reference_driver import ReferenceDriver, validate_reference_state
from state_adapter import OriginalAAdapter, equal_tree
from test_protected_engine import Provider, legal_signal
from test_protected_learning import TinyEHS


class ReferenceContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.adapters = []

    def tearDown(self):
        for adapter in self.adapters:
            adapter.close()
        self.temp.cleanup()

    def make(self, directory, *, count=13, disabled=False, rejects=("s0001", "s0004"), production=False):
        torch.manual_seed(17)
        np.random.seed(18)
        random.seed(19)
        adapter = OriginalAAdapter.configure(TinyEHS(burn_random=True, pollute=True), torch.device("cpu"))
        self.adapters.append(adapter)
        provider = Provider(count)
        records = [dict(sample_id=f"s{index:04d}", image_sha256=f"{index:064x}", preprocess_sha256="f" * 64)
                   for index in range(count)]
        identity = dict(run_id=directory, source_sha256="a" * 64, config_sha256="b" * 64,
                        manifest_sha256="c" * 64)
        def scorer(image):
            row = legal_signal()
            sid = next(key for key, value in provider.images.items() if value is image)
            row["synthetic_accept"] = sid not in rejects
            return row
        gate = lambda row: (row["synthetic_accept"], "synthetic_gate", .4)
        store = DurableRunStore(Path(self.temp.name) / directory, identity,
                                state_validator=validate_engine_state if disabled else validate_reference_state)
        if disabled:
            return ProtectedEngine(mode="disabled", stream_records=records,
                                   learner=WindowLearner(adapter), pools=GuardPools(91, 92),
                                   store=store, image_provider=provider, arrival_scorer=scorer,
                                   protection_scorer=scorer, gate=gate)
        return ReferenceDriver(stream_records=records, adapter=adapter, store=store,
                               image_provider=provider, scorer=scorer, gate=gate,
                               allow_stub_prefix=not production)

    def test_direct_original_reference_equals_disabled_engine_including_tail(self):
        reference = self.make("direct")
        result = reference.run()
        disabled = self.make("disabled", disabled=True)
        actual = disabled.run()
        self.assertTrue(equal_tree(result["learning"], disabled.learner.snapshot()["learning"]))
        self.assertEqual(result["observation_trace"],
                         [row for row in actual["scientific_trace"] if row["kind"] == "observation"])
        self.assertEqual(result["finish_counts"], actual["finish_counts"])
        self.assertEqual(result["finish_counts"]["tail"], 3)
        self.assertEqual(result["physical_work"]["forwards_confirmed_completed"],
                         {"arrival_scoring": 13, "adaptation": 11})

    def test_reference_never_uses_transactional_update_or_public_snapshot_methods(self):
        reference = self.make("independent")
        names = ("begin_window", "accumulate", "candidate", "commit", "rollback", "finish_tail", "snapshot", "load")
        patches = [patch.object(WindowLearner, name, side_effect=AssertionError("Reference used protected path"))
                   for name in names]
        for item in patches:
            item.start()
        try:
            result = reference.run()
        finally:
            for item in reversed(patches):
                item.stop()
        self.assertEqual(result["finish_counts"]["updates"], 2)

    def test_partial_checkpoint_resume_replays_physical_work_but_not_scientific_counts(self):
        reference = self.make("continuous", count=12, rejects=())
        expected = reference.run()
        interrupted = self.make("resume", count=12, rejects=())
        for _ in range(6):
            interrupted.step()
        interrupted.checkpoint("synthetic_partial")
        for _ in range(3):
            interrupted.step()
        resumed = self.make("resume", count=12, rejects=())
        actual = resumed.run()
        self.assertTrue(equal_tree(expected["learning"], actual["learning"]))
        self.assertEqual(expected["observation_trace"], actual["observation_trace"])
        self.assertEqual(actual["finish_counts"], dict(observed=12, accepted=12, rejected=0, updates=3, used=12, tail=0))
        self.assertEqual(actual["physical_work"]["forwards_confirmed_completed"]["adaptation"], 15)
        self.assertEqual(actual["physical_work"]["replay_forwards_confirmed_completed"]["adaptation"], 3)

    def test_complete_final_state_restoration_runs_no_additional_forward(self):
        reference = self.make("finished")
        first = reference.run()
        resumed = self.make("finished")
        second = resumed.run()
        self.assertTrue(equal_tree(first["learning"], second["learning"]))
        self.assertEqual(first["physical_work"], second["physical_work"])

    def test_production_prefix_and_regular_checkpoint_boundaries(self):
        with self.assertRaises(ValueError):
            self.make("short_production", count=12, production=True)
        reference = self.make("full_prefix", count=128, production=True)
        result = reference.run()
        generations = list(reference.store.generations.glob("generation-*"))
        self.assertEqual(len(generations), 4)  # initial, 64, 128, normal finish.
        self.assertEqual(result["finish_counts"]["observed"], 128)
        journal, _ = reference.store.read_journal()
        cp = [event["payload"] for event in journal if event["kind"] == "checkpoint_published"]
        self.assertTrue(all(event["checkpoint_seconds"] >= 0 for event in cp))
        self.assertNotIn("checkpoint_seconds", str(result["observation_trace"]))

    def test_corrupt_trace_rejected_without_blessing_journal(self):
        reference = self.make("corrupt")
        reference.step()
        state = reference.snapshot()
        state["observation_trace"][0]["sample_id"] = "future"
        with self.assertRaises(ValueError):
            validate_reference_state(state)
        reference.store.append_event("commit", {"updates": 999})
        self.assertEqual(reference.store.recover().scientific_cursors["commits"], 0)

    def test_impure_score_raises_and_keeps_last_complete_cursor(self):
        reference = self.make("impure")
        original = reference.scorer
        def impurity(image):
            row = original(image)
            random.random()
            return row
        reference.scorer = impurity
        with self.assertRaises(RuntimeError):
            reference.run()
        self.assertEqual(reference.store.recover().scientific_cursors["observed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
