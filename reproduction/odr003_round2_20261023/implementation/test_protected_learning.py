"""Synthetic local-CPU contracts for whole-window learning transactions."""
from __future__ import annotations

import copy
import io
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

# The parent runner supplies its reviewed baseline path at runtime. Local tests
# use the existing exact baseline copy; no remote source/model access occurs.
BASELINE = Path(__file__).resolve().parents[2] / "odr003_exploratory_20261023" / "implementation"
sys.path.insert(0, str(BASELINE))
from state_adapter import OriginalAAdapter, clone_tree, equal_tree, rng_state, restore_rng
from protected_learning import WindowLearner


class TinyEHS(torch.nn.Module):
    def __init__(self, *, burn_random=False, pollute=False):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))
        self.seg = torch.nn.Conv2d(3, 1, 1)
        self.register_buffer("scratch", torch.tensor(0.0))
        self.register_buffer("nonpersistent_scratch", torch.tensor(0.0), persistent=False)
        self.burn_random, self.pollute = burn_random, pollute

    def forward(self, image, is_train=False, gpu=None):
        if self.burn_random:
            # Deliberate synthetic use of all model-compute RNGs exercises a
            # stronger restoration contract than this deterministic eval net.
            perturbation = torch.rand((), device=image.device) + random.random() + float(np.random.rand())
            image = image + perturbation * 0.01
        features = self.backbone(image)
        if self.pollute:
            self.scratch.add_(1.0)
            self.nonpersistent_scratch.add_(2.0)
        logits = self.seg(features)
        loss = -(torch.sigmoid(logits) * torch.nn.functional.logsigmoid(logits)).mean()
        return loss, {"ent": loss.detach()}


def tensor_devices(value):
    if isinstance(value, torch.Tensor):
        return [value.device.type]
    if isinstance(value, dict):
        return [device for item in value.values() for device in tensor_devices(item)]
    if isinstance(value, (list, tuple)):
        return [device for item in value for device in tensor_devices(item)]
    return []


class LearningContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        np.random.seed(18)
        random.seed(19)
        self.images = [torch.rand(1, 3, 8, 8) for _ in range(24)]
        self.initial = clone_tree(TinyEHS().state_dict())
        self.adapters = []

    def tearDown(self):
        for adapter in self.adapters:
            adapter.close()

    def new(self, *, burn_random=False, pollute=False):
        model = TinyEHS(burn_random=burn_random, pollute=pollute)
        model.load_state_dict(self.initial)
        adapter = OriginalAAdapter.configure(model, torch.device("cpu"))
        self.adapters.append(adapter)
        return WindowLearner(adapter)

    def accept(self, learner, image):
        learner.adapter.observed += 1
        if learner.phase == "idle":
            learner.begin_window()
        if learner.window_size < 3:
            return learner.accumulate(image)
        return learner.candidate(image)

    def full_window(self, learner, images, decision="commit"):
        losses = [self.accept(learner, image) for image in images]
        result = getattr(learner, decision)()
        return losses, result

    def assert_learning_equal(self, first, second, *, ignore_counters=False):
        a, b = first.snapshot()["learning"], second.snapshot()["learning"]
        if ignore_counters:
            a.pop("counters")
            b.pop("counters")
        self.assertTrue(equal_tree(a, b))

    def test_protection_disabled_matches_original_A_exactly(self):
        reference = self.new(burn_random=True, pollute=True)
        wrapper = self.new(burn_random=True, pollute=True)
        start = clone_tree(rng_state())
        expected_losses = []
        for image in self.images[:12]:
            reference.adapter.observed += 1
            expected_losses.append(reference.adapter.accept(image))
        expected = reference.snapshot()["learning"]
        restore_rng(start)
        actual_losses = []
        for start_index in range(0, 12, 4):
            losses, _ = self.full_window(wrapper, self.images[start_index:start_index + 4])
            actual_losses.extend(losses)
        self.assertEqual(expected_losses, actual_losses)
        self.assertTrue(equal_tree(expected, wrapper.snapshot()["learning"]))
        self.assertEqual(wrapper.adapter.updates, 3)
        self.assertEqual(wrapper.adapter.accepted, 12)

    def test_first_three_images_cannot_step_optimizer(self):
        learner = self.new()
        before = learner.snapshot()["learning"]
        for image in self.images[:3]:
            self.accept(learner, image)
        self.assertEqual(learner.adapter.updates, 0)
        self.assertTrue(equal_tree(before["model"], learner.adapter.model.state_dict()))
        self.assertTrue(equal_tree(before["optimizer"], learner.adapter.optimizer.state_dict()))
        self.assertTrue(any(parameter.grad is not None for parameter in learner.adapter.model.parameters()))
        self.accept(learner, self.images[3])
        self.assertEqual(learner.phase, "candidate")
        self.assertEqual(learner.adapter.updates, 1)
        self.assertTrue(all(parameter.grad is None for parameter in learner.adapter.model.parameters()))

    def test_rollback_restores_full_learning_but_keeps_input_counts(self):
        learner = self.new(burn_random=True, pollute=True)
        self.full_window(learner, self.images[:4])
        base = learner.snapshot()["learning"]
        self.full_window(learner, self.images[4:8], "rollback")
        after = learner.snapshot()["learning"]
        self.assertEqual(after["counters"], dict(accepted=8, observed=8, updates=1, tail=0))
        base.pop("counters")
        after.pop("counters")
        self.assertTrue(equal_tree(base, after))
        self.assertTrue(all(parameter.grad is None for parameter in learner.adapter.model.parameters()))
        self.assertEqual(learner.phase, "idle")

    def test_repeated_rollbacks_do_not_accumulate_adam_steps(self):
        learner = self.new(burn_random=True)
        self.full_window(learner, self.images[:4])
        baseline = clone_tree(learner.adapter.optimizer.state_dict())
        for start in (4, 8, 12):
            self.full_window(learner, self.images[start:start + 4], "rollback")
            self.assertTrue(equal_tree(baseline, learner.adapter.optimizer.state_dict()))
            self.assertEqual(learner.adapter.updates, 1)
        self.assertEqual(learner.adapter.accepted, 16)
        self.full_window(learner, self.images[16:20])
        self.assertEqual(learner.adapter.updates, 2)
        for state in learner.adapter.optimizer.state.values():
            self.assertEqual(float(state["step"]), 2.0)
            self.assertEqual(state["step"].device.type, "cpu")

    def test_rollback_then_commit_matches_skipping_failed_window_learning(self):
        rejected = self.new(burn_random=True, pollute=True)
        direct = self.new(burn_random=True, pollute=True)
        shared_rng = clone_tree(rng_state())
        self.full_window(rejected, self.images[:4], "rollback")
        self.full_window(rejected, self.images[4:8])
        actual = rejected.snapshot()["learning"]
        restore_rng(shared_rng)
        self.full_window(direct, self.images[4:8])
        expected = direct.snapshot()["learning"]
        actual.pop("counters")
        expected.pop("counters")
        self.assertTrue(equal_tree(actual, expected))
        self.assertEqual(rejected.adapter.accepted, 8)
        self.assertEqual(direct.adapter.accepted, 4)

    def test_midwindow_cpu_checkpoint_restores_gradients_and_trajectory(self):
        continuous = self.new(burn_random=True, pollute=True)
        self.full_window(continuous, self.images[:4])
        self.accept(continuous, self.images[4])
        self.accept(continuous, self.images[5])
        checkpoint = continuous.snapshot()
        self.assertEqual(set(tensor_devices(checkpoint)), {"cpu"})
        self.assertTrue(any(value is not None for value in checkpoint["learning"]["gradients"].values()))
        # Round-trip through the actual torch checkpoint format on local memory.
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        losses = [self.accept(continuous, image) for image in self.images[6:8]]
        continuous.rollback()
        more, _ = self.full_window(continuous, self.images[8:12])
        expected = continuous.snapshot()
        resumed = self.new(burn_random=True, pollute=True)
        resumed.load(checkpoint)
        self.assertEqual(resumed.phase, "collecting")
        self.assertEqual(resumed.window_size, 2)
        self.assertEqual(losses, [self.accept(resumed, image) for image in self.images[6:8]])
        resumed.rollback()
        replayed, _ = self.full_window(resumed, self.images[8:12])
        self.assertEqual(more, replayed)
        self.assertTrue(equal_tree(expected, resumed.snapshot()))

    def test_candidate_checkpoint_can_restore_the_same_rollback(self):
        learner = self.new(burn_random=True, pollute=True)
        self.full_window(learner, self.images[:4])
        for image in self.images[4:8]:
            self.accept(learner, image)
        checkpoint = learner.snapshot()
        learner.rollback()
        expected = learner.snapshot()
        resumed = self.new(burn_random=True, pollute=True)
        resumed.load(checkpoint)
        resumed.rollback()
        self.assertTrue(equal_tree(expected, resumed.snapshot()))

    def test_default_tail_restores_complete_base_including_rng_and_cache(self):
        learner = self.new(burn_random=True, pollute=True)
        self.full_window(learner, self.images[:4])
        before = learner.snapshot()["learning"]
        for image in self.images[4:7]:
            self.accept(learner, image)
        learner.adapter.observed += 2  # Observed rejects do not enter gradients.
        result = learner.finish_tail()
        self.assertEqual(result, dict(observed=9, accepted=7, rejected=2, updates=1, used=4, tail=3))
        after = learner.snapshot()["learning"]
        before.pop("counters")
        after.pop("counters")
        self.assertTrue(equal_tree(before, after))
        self.assertEqual(learner.finish_mode, "rollback_partial_window")
        with self.assertRaises(RuntimeError):
            learner.begin_window()

    def test_legacy_tail_exactly_matches_original_finish(self):
        original = self.new(burn_random=True, pollute=True)
        disabled = self.new(burn_random=True, pollute=True)
        shared_rng = clone_tree(rng_state())
        for image in self.images[:7]:
            original.adapter.observed += 1
            original.adapter.accept(image)
        reference_counts = original.adapter.finish()
        expected = original.snapshot()["learning"]
        restore_rng(shared_rng)
        self.full_window(disabled, self.images[:4])
        for image in self.images[4:7]:
            self.accept(disabled, image)
        counts = disabled.finish_tail(legacy_reference=True)
        self.assertEqual(reference_counts, counts)
        self.assertTrue(equal_tree(expected, disabled.snapshot()["learning"]))
        self.assertEqual(disabled.finish_mode, "original_adapter_finish")

    def test_tail_policies_same_scientific_model_optimizer_different_auxiliary_state(self):
        protected = self.new(burn_random=True, pollute=True)
        legacy = self.new(burn_random=True, pollute=True)
        shared_rng = clone_tree(rng_state())
        self.full_window(protected, self.images[:4])
        for image in self.images[4:7]:
            self.accept(protected, image)
        protected.finish_tail()
        p = protected.snapshot()["learning"]
        restore_rng(shared_rng)
        self.full_window(legacy, self.images[:4])
        for image in self.images[4:7]:
            self.accept(legacy, image)
        legacy.finish_tail(legacy_reference=True)
        r = legacy.snapshot()["learning"]
        for name in ("model", "optimizer", "buffers", "gradients", "counters"):
            self.assertTrue(equal_tree(p[name], r[name]), name)
        self.assertNotEqual(bool(p["bn_features"]), bool(r["bn_features"]))
        self.assertFalse(equal_tree(p["rng"], r["rng"]))

    def test_rollback_does_not_rewind_independent_pool_or_decision_rng(self):
        pool_rng, decision_rng = random.Random(100), random.Random(200)
        learner = self.new(burn_random=True)
        learner.adapter.observed += 1
        learner.begin_window()
        pool_rng.random()
        decision_rng.random()
        pool_after, decision_after = pool_rng.getstate(), decision_rng.getstate()
        learner.accumulate(self.images[0])
        for image in self.images[1:4]:
            self.accept(learner, image)
        learner.rollback()
        self.assertEqual(pool_after, pool_rng.getstate())
        self.assertEqual(decision_after, decision_rng.getstate())

    def test_snapshots_are_cpu_independent_and_not_aliases(self):
        learner = self.new()
        self.accept(learner, self.images[0])
        checkpoint = learner.snapshot()
        saved_model = clone_tree(checkpoint["learning"]["model"])
        for image in self.images[1:4]:
            self.accept(learner, image)
        learner.commit()
        self.assertTrue(equal_tree(saved_model, checkpoint["learning"]["model"]))
        gradient = next(value for value in checkpoint["learning"]["gradients"].values() if value is not None)
        gradient.add_(1000.0)
        self.assertTrue(all(parameter.grad is None for parameter in learner.adapter.model.parameters()))

    def test_named_module_modes_and_nonpersistent_buffers_restore(self):
        learner = self.new(pollute=True)
        # Keep root train flag different from children; no recursive .train()
        # should erase this configuration during restore.
        learner.adapter.model.training = True
        learner.adapter.model.seg.training = True
        modes = {name: module.training for name, module in learner.adapter.model.named_modules()}
        self.full_window(learner, self.images[:4], "rollback")
        self.assertEqual(modes, {name: module.training for name, module in learner.adapter.model.named_modules()})
        self.assertEqual(float(learner.adapter.model.scratch), 0.0)
        self.assertEqual(float(learner.adapter.model.nonpersistent_scratch), 0.0)

    def test_invalid_phase_or_duplicate_decision_fails(self):
        learner = self.new()
        with self.assertRaises(RuntimeError):
            learner.accumulate(self.images[0])
        learner.begin_window()
        with self.assertRaises(RuntimeError):
            learner.begin_window()
        with self.assertRaises(RuntimeError):
            learner.candidate(self.images[0])
        for image in self.images[:4]:
            self.accept(learner, image)
        with self.assertRaises(RuntimeError):
            learner.finish_tail()
        learner.commit()
        with self.assertRaises(RuntimeError):
            learner.commit()
        learner.finish_tail()
        with self.assertRaises(RuntimeError):
            learner.finish_tail()

    def test_external_counter_mutation_is_detected(self):
        learner = self.new()
        self.accept(learner, self.images[0])
        learner.adapter.accepted += 1
        with self.assertRaises(RuntimeError):
            learner.snapshot()

    def test_checkpoint_corruption_detected_before_restore(self):
        learner = self.new()
        self.accept(learner, self.images[0])
        good = learner.snapshot()
        bad = copy.deepcopy(good)
        bad["learning"]["counters"]["updates"] = 1
        with self.assertRaises(ValueError):
            learner.load(bad)
        self.assertTrue(equal_tree(good, learner.snapshot()))
        bad = copy.deepcopy(good)
        bad["window_base"]["model"].pop("scratch")
        with self.assertRaises(ValueError):
            learner.load(bad)
        self.assertTrue(equal_tree(good, learner.snapshot()))

    def test_empty_stream_and_empty_started_window_finish(self):
        empty = self.new()
        self.assertEqual(empty.finish_tail(), dict(observed=0, accepted=0, rejected=0, updates=0, used=0, tail=0))
        started = self.new()
        started.adapter.observed = 3
        started.begin_window()
        self.assertEqual(started.finish_tail(), dict(observed=3, accepted=0, rejected=3, updates=0, used=0, tail=0))

    def test_finished_checkpoint_is_fully_restorable(self):
        learner = self.new(burn_random=True)
        self.full_window(learner, self.images[:4])
        self.accept(learner, self.images[4])
        learner.finish_tail()
        checkpoint = learner.snapshot()
        restored = self.new(burn_random=True)
        restored.load(checkpoint)
        self.assertTrue(equal_tree(checkpoint, restored.snapshot()))
        self.assertEqual(restored.phase, "finished")


if __name__ == "__main__":
    unittest.main(verbosity=2)
