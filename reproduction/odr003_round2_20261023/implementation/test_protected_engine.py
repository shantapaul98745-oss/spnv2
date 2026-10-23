"""Local synthetic CPU engine integration, no real dataset/model/host access."""
from __future__ import annotations

import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

BASELINE = Path(__file__).resolve().parents[2] / "odr003_exploratory_20261023" / "implementation"
sys.path.insert(0, str(BASELINE))
from durable_state import DurableRunStore
from guard_pools import GuardPools
from protected_engine import ProtectedEngine, ControlledInterruption, validate_engine_state
from protected_learning import WindowLearner
from state_adapter import OriginalAAdapter, equal_tree
from test_protected_learning import TinyEHS


class Provider:
    def __init__(self, count):
        generator = torch.Generator().manual_seed(821)
        self.images = {f"s{index:04d}": torch.rand((1, 3, 8, 8), generator=generator)
                       for index in range(count)}
        self.requests = []

    def __call__(self, sample_id):
        self.requests.append(sample_id)
        return self.images[sample_id]

    def state_dict(self):
        return {"requests": list(self.requests)}

    def load_state_dict(self, state):
        self.requests = list(state["requests"])


def legal_signal():
    return dict(pose_legal=True, h_available=True, e_available=True,
                original_h_rejected=False, heat_q=[1., 0., 0., 0.],
                rotation_disagreement=2., reprojection=.01, heat_peaks=[.9] * 11,
                used_indices=list(range(11)), used_count=11, effective_heat_threshold=.5,
                timing_seconds={"forward": random.Random(9).random()})


class EngineContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.adapters = []

    def tearDown(self):
        for adapter in self.adapters:
            adapter.close()
        self.temp.cleanup()

    def engine(self, directory, *, count=12, mode="P", reject_windows=(), faults=(),
               p=None, mask=None, reject_ids=()):
        torch.manual_seed(17)
        np.random.seed(18)
        random.seed(19)
        model = TinyEHS(burn_random=True, pollute=True)
        adapter = OriginalAAdapter.configure(model, torch.device("cpu"))
        self.adapters.append(adapter)
        learner = WindowLearner(adapter)
        provider = Provider(count)
        records = [dict(sample_id=f"s{index:04d}", image_sha256=f"{index:064x}",
                        preprocess_sha256="f" * 64) for index in range(count)]
        identity = dict(run_id=directory, source_sha256="a" * 64, config_sha256="b" * 64,
                        manifest_sha256="c" * 64)
        store = DurableRunStore(Path(self.temp.name) / directory, identity,
                                state_validator=validate_engine_state)
        holder = {}

        def arrival(image):
            row = legal_signal()
            # Synthetic, no model forward: contract checks only. Real runner
            # supplies the actual isolated EHS/HE scorer and truthful counters.
            sid = next(key for key, value in provider.images.items() if value is image)
            row["synthetic_accept"] = sid not in reject_ids
            return row

        def protection(image):
            row = legal_signal()
            current = holder["engine"]
            if learner.phase == "candidate" and current.counters["candidates"] in reject_windows:
                row.update(pose_legal=False, h_available=False, original_h_rejected=True,
                           heat_q=None, rotation_disagreement=None, reprojection=None)
            return row

        engine = ProtectedEngine(mode=mode, stream_records=records, learner=learner,
                                 pools=GuardPools(91, 92), store=store, image_provider=provider,
                                 arrival_scorer=arrival, protection_scorer=protection,
                                 gate=lambda row: (row["synthetic_accept"], "synthetic_gate", .4),
                                 rw_probability=p, rw_seed=77 if mode == "RW" else None,
                                 rp_mask=mask, fault_points=faults)
        holder["engine"] = engine
        return engine

    def science(self, engine):
        return dict(counters=copy.deepcopy(engine.counters), trace=copy.deepcopy(engine.scientific_trace),
                    learning=engine.learner.snapshot(), pools=engine.pools.state_dict(),
                    provider=engine.image_provider.state_dict(), finished=engine.finished,
                    finish_counts=copy.deepcopy(engine.finish_counts),
                    rng=engine.rw_rng.getstate() if engine.rw_rng else None)

    def test_normal_rollback_preserves_observation_history_and_no_failed_replay(self):
        engine = self.engine("rollback", reject_windows=(1, 3))
        result = engine.run()
        self.assertEqual(result["counters"]["observed"], 12)
        self.assertEqual(result["counters"]["accepted"], 12)
        self.assertEqual(result["counters"]["candidates"], 3)
        self.assertEqual(result["counters"]["commits"], 1)
        self.assertEqual(result["counters"]["rollbacks"], 2)
        self.assertEqual(engine.pools.uniform.candidate_count, 12)
        self.assertEqual(result["physical_work"]["forwards_confirmed_completed"]["adaptation"], 12)
        self.assertEqual(result["physical_work"]["replay_forwards_confirmed_completed"], {})
        self.assertEqual([r["decision"] for r in result["scientific_trace"] if r["kind"] == "window"],
                         ["rollback", "commit", "rollback"])
        self.assertNotIn("timing_seconds", str(result["scientific_trace"]))
        for window in (row for row in result["scientific_trace"] if row["kind"] == "window"):
            self.assertEqual(len(window["protection_pairs"]), len(window["protection_records"]))
            for pair in window["protection_pairs"]:
                for when in ("pre", "post"):
                    self.assertEqual(len(pair[when]["heat_peaks"]), 11)
                    self.assertIn("used_indices", pair[when])
                    self.assertIn("effective_heat_threshold", pair[when])

    def test_prepared_has_four_ids_but_only_three_gradients_and_resumes_exactly(self):
        reference = self.engine("prepared_reference", reject_windows=(1,))
        reference.run()
        expected = self.science(reference)
        interrupted = self.engine("prepared", reject_windows=(1,))
        for _ in range(3):
            interrupted.step()
        interrupted.step(stop_after_prepare=True)
        info = interrupted.checkpoint("test_explicit_prepared")
        state = interrupted.store.recover().state
        self.assertEqual(len(state["window_ids"]), 4)
        self.assertEqual(state["learner"]["window_size"], 3)
        self.assertEqual(state["counters"]["accepted"], 4)
        self.assertEqual(state["learner"]["learning"]["counters"]["accepted"], 3)
        resumed = self.engine("prepared", reject_windows=(1,))
        self.assertEqual(resumed.resume().generation, info.generation)
        resumed.run()
        self.assertTrue(equal_tree(expected, self.science(resumed)))
        self.assertEqual(resumed.store.work_summary()["forwards_confirmed_completed"]["adaptation"], 12)

    def test_partial_checkpoint_replays_unpublished_decision_without_double_science(self):
        reference = self.engine("partial_reference", mode="disabled")
        reference.run()
        expected = self.science(reference)
        interrupted = self.engine("partial", mode="disabled")
        for _ in range(6):
            interrupted.step()
        interrupted.checkpoint("test_partial")
        self.assertEqual(interrupted.learner.window_size, 2)
        for _ in range(3):
            interrupted.step()
        journal, _ = interrupted.store.read_journal()
        self.assertTrue(any(r["kind"] == "window_decision_audit" and r["payload"]["id"] == "candidate:0002"
                            for r in journal))
        self.assertEqual(interrupted.store.recover().scientific_cursors["commits"], 1)
        resumed = self.engine("partial", mode="disabled")
        resumed.run()
        self.assertTrue(equal_tree(expected, self.science(resumed)))
        work = resumed.store.work_summary()
        self.assertEqual(work["forwards_confirmed_completed"]["adaptation"], 15)
        self.assertEqual(work["replay_forwards_confirmed_completed"]["adaptation"], 3)
        self.assertEqual(resumed.counters["commits"], 3)
        self.assertEqual(len(resumed.scientific_trace), 15)

    def test_both_fixed_faults_fire_once_and_recovery_matches_continuous(self):
        reference = self.engine("fault_reference", count=36, reject_windows=(2, 5))
        reference.run()
        expected = self.science(reference)
        engine = self.engine("fault", count=36, reject_windows=(2, 5),
                             faults=("candidate_2_tentative", "observation_31"))
        observed = []
        for _ in range(3):
            try:
                engine.run()
                break
            except ControlledInterruption as error:
                observed.append(str(error))
                engine = self.engine("fault", count=36, reject_windows=(2, 5),
                                     faults=("candidate_2_tentative", "observation_31"))
        self.assertEqual(observed, ["candidate_2_tentative", "observation_31"])
        self.assertTrue(engine.finished)
        self.assertTrue(equal_tree(expected, self.science(engine)))
        work = engine.store.work_summary()
        self.assertEqual(work["forwards_confirmed_completed"]["adaptation"], 37)
        self.assertEqual(work["replay_forwards_confirmed_completed"]["adaptation"], 1)
        self.assertEqual(len({x["id"] for x in engine.scientific_trace}), len(engine.scientific_trace))

    def test_rw_zero_all_rollbacks_and_one_all_commits(self):
        zero = self.engine("rw_zero", mode="RW", p=0.0)
        initial_model = copy.deepcopy(zero.learner.adapter.model.state_dict())
        zero.run()
        self.assertEqual(zero.counters["commits"], 0)
        self.assertEqual(zero.counters["rollbacks"], 3)
        self.assertTrue(equal_tree(initial_model, zero.learner.adapter.model.state_dict()))
        coins = [x["rw_coin"] for x in zero.scientific_trace if x["kind"] == "window"]
        rng = random.Random(77)
        self.assertEqual(coins, [rng.random() for _ in range(3)])
        one = self.engine("rw_one", mode="RW", p=1.0)
        one.run()
        one_model = copy.deepcopy(one.learner.adapter.model.state_dict())
        disabled = self.engine("rw_disabled", mode="disabled")
        disabled.run()
        self.assertEqual(one.counters["commits"], 3)
        self.assertEqual(one.counters["rollbacks"], 0)
        self.assertTrue(equal_tree(one_model, disabled.learner.adapter.model.state_dict()))

    def test_rw_mid_probability_resume_does_not_redraw_a_committed_coin(self):
        reference = self.engine("rw_reference", mode="RW", p=.5)
        reference.run()
        expected = self.science(reference)
        interrupted = self.engine("rw_resume", mode="RW", p=.5)
        for _ in range(4):
            interrupted.step()
        interrupted.checkpoint("committed_window")
        for _ in range(5):
            interrupted.step()
        resumed = self.engine("rw_resume", mode="RW", p=.5)
        resumed.run()
        self.assertTrue(equal_tree(expected, self.science(resumed)))

    def test_rp_exact_mask_budget_and_no_geometry_scoring(self):
        mask = [True, False, True, True, False, True, True, False, True, True, False, True]
        engine = self.engine("rp", mode="RP", mask=mask)
        result = engine.run()
        self.assertEqual(result["counters"]["observed"], 12)
        self.assertEqual(result["counters"]["accepted"], 8)
        self.assertEqual(result["counters"]["commits"], 2)
        self.assertEqual(result["physical_work"]["forwards_confirmed_completed"], {"adaptation": 8})
        self.assertEqual(engine.pools.records, {})
        self.assertEqual([r["observation_index"] for r in result["scientific_trace"]
                          if r["kind"] == "observation" and r["accepted"]],
                         [index for index, value in enumerate(mask) if value])

    def test_rejected_ids_enter_history_without_becoming_learning_inputs(self):
        engine = self.engine("rejected", count=13, reject_ids=("s0000",))
        engine.run()
        self.assertFalse(engine.pools.records["s0000"]["g50_accepted"])
        windows = [r for r in engine.scientific_trace if r["kind"] == "window"]
        self.assertTrue(all("s0000" not in r["sample_ids"] for r in windows))
        self.assertEqual(engine.pools.uniform.candidate_count, 13)

    def test_bad_provider_rng_and_impure_scorer_stop_before_scientific_progress(self):
        engine = self.engine("bad_provider")
        original = engine.image_provider
        def impure_provider(sample_id):
            torch.rand(1)
            return original(sample_id)
        engine.image_provider = impure_provider
        with self.assertRaises(RuntimeError):
            engine.run()
        self.assertEqual(engine.store.recover().scientific_cursors["observed"], 0)
        engine = self.engine("bad_scorer")
        def impure_scorer(image):
            engine.learner.adapter.model.scratch.add_(1)
            return legal_signal()
        engine.arrival_scorer = impure_scorer
        with self.assertRaises(RuntimeError):
            engine.run()
        self.assertEqual(engine.store.recover().scientific_cursors["observed"], 0)

    def test_final_recovery_does_not_repeat_a_single_forward(self):
        engine = self.engine("done")
        engine.run()
        before = engine.store.work_summary()["forwards_confirmed_completed"]
        fresh = self.engine("done")
        result = fresh.run()
        self.assertEqual(before, result["physical_work"]["forwards_confirmed_completed"])

    def test_regular_checkpoint_only_at_64_and_exit_without_per_window_generation(self):
        engine = self.engine("interval", count=68, mode="disabled")
        engine.run()
        generations = list(engine.store.generations.glob("generation-*"))
        self.assertEqual(len(generations), 3)  # Initial, observation 64, normal exit.
        self.assertEqual(engine.counters["commits"], 17)
        journal, _ = engine.store.read_journal()
        publications = [event["payload"] for event in journal if event["kind"] == "checkpoint_published"]
        self.assertEqual(len(publications), 3)
        self.assertTrue(all(event["checkpoint_seconds"] >= 0.0 for event in publications))
        self.assertNotIn("checkpoint_seconds", str(engine.scientific_trace))
        self.assertNotIn("checkpoint_seconds", engine.store.recover().state)

    def test_protocol_change_is_rejected_on_recovery(self):
        engine = self.engine("protocol", mode="RW", p=.5)
        engine.step()
        changed = self.engine("protocol", mode="RW", p=.7)
        with self.assertRaises(ValueError):
            changed.resume()

    def test_tentative_candidate_cannot_be_saved_as_complete_authority(self):
        engine = self.engine("tentative")
        for _ in range(3):
            engine.step()
        engine.step(stop_after_prepare=True)
        engine.checkpoint("prepared")
        before = engine.store.recover().info.generation
        engine.learner.candidate(engine.image_provider("s0003"))
        with self.assertRaises(ValueError):
            engine.checkpoint("forbidden_tentative")
        self.assertEqual(engine.store.recover().info.generation, before)

    def test_forged_observation_or_window_identity_is_rejected(self):
        engine = self.engine("trace")
        for _ in range(6):
            engine.step()
        good = engine.snapshot()
        bad = copy.deepcopy(good)
        bad["scientific_trace"][0]["sample_id"] = "future"
        with self.assertRaises(ValueError):
            validate_engine_state(bad)
        bad = copy.deepcopy(good)
        bad["window_ids"].reverse()
        with self.assertRaises(ValueError):
            validate_engine_state(bad)

    def test_unresolved_dispatch_remains_unknown_and_replay_is_charged(self):
        engine = self.engine("dispatch", mode="disabled")
        original = engine.arrival_scorer
        def broken(image):
            original(image)
            raise RuntimeError("synthetic interruption after possible dispatched work")
        engine.arrival_scorer = broken
        with self.assertRaises(RuntimeError):
            engine.run()
        self.assertEqual(engine.store.work_summary()["forwards_inflight_unknown_upper_bound"],
                         {"arrival_scoring": 1})
        resumed = self.engine("dispatch", mode="disabled")
        result = resumed.run()
        self.assertEqual(result["counters"]["observed"], 12)
        self.assertEqual(result["physical_work"]["replay_forwards_confirmed_completed"]["arrival_scoring"], 1)
        self.assertEqual(result["physical_work"]["forwards_execution_upper_bound"]["arrival_scoring"], 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
