"""Exact production-trainer resume at AMP, validation and epoch boundaries.

Validation here deliberately consumes all three CPU RNGs and returns no accepted
poses. Actual inference summary/null behavior is verified independently.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
source = Path(args.source).resolve()
sys.path.insert(0, str(source/'core'))
from config import cfg as defaults
from engine.trainer import do_train
from utils.training_state import (EpochBatchSampler, SeededDataset, TrainingState,
    capture_rng_state, save_training_checkpoint, load_training_checkpoint, sha256_file)

torch.set_num_threads(2)
warnings.filterwarnings('ignore', message='.*CUDA is not available.*')
warnings.filterwarnings('ignore', category=FutureWarning)
MANIFEST = dict(version=1, code={str(p.relative_to(source)): sha256_file(p)
    for folder in ('core', 'tools') for p in sorted((source/folder).rglob('*.py'))})


def seed(value=2021):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def config():
    cfg = defaults.clone()
    cfg.defrost()
    cfg.TRAIN.ACCUM_STEPS = 4
    cfg.TRAIN.END_EPOCH = 2
    cfg.TRAIN.AMP_SKIP_WARMUP = 64
    cfg.TRAIN.AMP_MAX_CONSECUTIVE_SKIPS = 16
    cfg.MODEL.HEAD.LOSS_HEADS = ['heatmap']
    cfg.VERBOSE = False
    cfg.freeze()
    return cfg


class Data(Dataset):
    def __len__(self):
        return 25

    def __getitem__(self, index):
        noise = (random.random()+np.random.rand()+torch.rand(()).item())*.01
        x = torch.tensor([index/31, (index % 5)/5, 1.], dtype=torch.float64)+noise
        return x, dict(y=torch.tensor([index/41], dtype=torch.float64),
                       imgpath=f'sample-{index:03d}')


class InfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient*float('inf')


class Toy(nn.Module):
    def __init__(self, inject_overflow=False):
        super().__init__()
        self.linear = nn.Linear(3, 1, dtype=torch.float64)
        self.dropout = nn.Dropout(.2)
        self.register_buffer('forwards', torch.tensor(0))
        self.inject_overflow = inject_overflow
        self.seen = []

    def forward(self, x, is_train=True, gpu=None, y=None, imgpath=None, **targets):
        self.seen.extend(imgpath)
        self.forwards += 1
        x = self.dropout(x)*(1+.1*random.random()+.1*np.random.rand())
        loss = ((self.linear(x)-y)**2).mean()
        parts = dict(hmap=loss.detach())
        if self.inject_overflow and int(self.forwards) <= 4:
            loss = InfiniteBackward.apply(loss)
        return loss, parts


def make(inject_overflow=False, amp=False):
    model = Toy(inject_overflow)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    sampler = EpochBatchSampler(25, 2, 4, 2021, True)
    loader = DataLoader(SeededDataset(Data(), 2021), batch_sampler=sampler,
                        num_workers=0, generator=torch.Generator().manual_seed(852))
    scaler = torch.amp.GradScaler('cpu', init_scale=128) if amp else None
    return model, optimizer, sampler, loader, scaler


def validation(model):
    model.eval()
    # Model and loader validation helpers may consume RNG; checkpoints after
    # validation must preserve the state after these draws, not replay them.
    draws = (random.random(), float(np.random.rand()), torch.rand(5))
    return dict(score=None, n_samples=7, heatmap_rejections=7,
                metrics={'final_pose': dict(mean=None, count=0, unit='-')}), draws


def next_epoch(state, sampler):
    state.epoch += 1
    state.next_batch = 0
    sampler.set_epoch(state.epoch, 0)


def tree_equal(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right), (left, right))
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            tree_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            tree_equal(case, a, b)
    else:
        case.assertEqual(left, right)


class Tests(unittest.TestCase):
    def compare(self, reference, resumed, whole, state, rng):
        rm, ro, rs, rd, rsc = reference
        cm, co, cs, cd, csc = resumed
        tree_equal(self, rm.state_dict(), cm.state_dict())
        tree_equal(self, ro.state_dict(), co.state_dict())
        if rsc is not None:
            tree_equal(self, rsc.state_dict(), csc.state_dict())
        tree_equal(self, whole.state_dict(), state.state_dict())
        tree_equal(self, rng, capture_rng_state())

    def test_resume_immediately_after_actual_cpu_amp_skip(self):
        seed()
        reference = make(True, True)
        rm, ro, rs, rd, rsc = reference
        whole = do_train(0, config(), rm, rd, ro, scaler=rsc)
        rng = capture_rng_state()
        self.assertEqual((whole.update_attempts, whole.optimizer_updates, whole.amp_skips), (3, 2, 1))
        seed()
        partial = make(True, True)
        pm, po, ps, pd, psc = partial
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'after_amp_skip.pth.tar'
            def boundary(state):
                if state.next_batch == 4:
                    self.assertEqual((state.update_attempts, state.optimizer_updates, state.amp_skips), (1, 0, 1))
                    save_training_checkpoint(path, pm, po, psc, state, ps, MANIFEST)
                    raise InterruptedError('controlled stop after AMP skip')
            with self.assertRaises(InterruptedError):
                do_train(0, config(), pm, pd, po, scaler=psc, on_boundary=boundary)
            prefix = pm.seen.copy()
            seed(79)
            resumed = make(True, True)
            cm, co, cs, cd, csc = resumed
            state, _ = load_training_checkpoint(path, cm, co, csc, cs, MANIFEST)
            do_train(0, config(), cm, cd, co, scaler=csc, state=state)
            self.assertEqual(rm.seen, prefix+cm.seen)
            self.compare(reference, resumed, whole, state, rng)

    def test_resume_last_window_before_validation(self):
        seed()
        reference = make()
        rm, ro, rs, rd, _ = reference
        whole = do_train(0, config(), rm, rd, ro)
        expected_summary, expected_draws = validation(rm)
        rng = capture_rng_state()
        seed()
        pm, po, ps, pd, _ = make()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'last_window_pre_validation.pth.tar'
            def boundary(state):
                if state.next_batch == ps.total_batches:
                    save_training_checkpoint(path, pm, po, None, state, ps, MANIFEST)
                    raise InterruptedError('controlled stop before validation')
            with self.assertRaises(InterruptedError):
                do_train(0, config(), pm, pd, po, on_boundary=boundary)
            seed(85)
            resumed = make()
            cm, co, cs, cd, _ = resumed
            state, _ = load_training_checkpoint(path, cm, co, None, cs, MANIFEST)
            self.assertEqual(state.next_batch, cs.total_batches)
            do_train(0, config(), cm, cd, co, state=state)
            self.assertEqual(cm.seen, [])  # No repeated last-window forwards.
            actual_summary, actual_draws = validation(cm)
            tree_equal(self, expected_summary, actual_summary)
            tree_equal(self, expected_draws, actual_draws)
            self.compare(reference, resumed, whole, state, rng)
            self.assertIsNone(actual_summary['score'])
            self.assertEqual(actual_summary['metrics']['final_pose']['count'], 0)

    def test_resume_after_validation_and_epoch_advance_rng_exact(self):
        seed()
        reference = make()
        rm, ro, rs, rd, _ = reference
        whole = do_train(0, config(), rm, rd, ro)
        validation(rm)
        next_epoch(whole, rs)
        expected_prefix = len(rm.seen)
        do_train(1, config(), rm, rd, ro, state=whole)
        rng = capture_rng_state()
        seed()
        pm, po, ps, pd, _ = make()
        state = do_train(0, config(), pm, pd, po)
        validation(pm)
        next_epoch(state, ps)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'post_validation_epoch_boundary.pth.tar'
            save_training_checkpoint(path, pm, po, None, state, ps, MANIFEST)
            seed(76)
            resumed = make()
            cm, co, cs, cd, _ = resumed
            state, _ = load_training_checkpoint(path, cm, co, None, cs, MANIFEST)
            self.assertEqual((state.epoch, state.next_batch), (1, 0))
            do_train(1, config(), cm, cd, co, state=state)
            self.assertEqual(rm.seen[expected_prefix:], cm.seen)
            self.compare(reference, resumed, whole, state, rng)


suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = dict(status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
              failures=[dict(test=str(test), trace=trace) for test, trace in result.failures+result.errors],
              source=str(source), code_sha256=MANIFEST['code'],
              test_sha256=sha256_file(__file__), device='cpu', actual_cpu_grad_scaler=True)
Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
sys.exit(0 if result.wasSuccessful() else 1)
