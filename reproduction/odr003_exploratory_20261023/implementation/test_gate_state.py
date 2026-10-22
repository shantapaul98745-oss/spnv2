"""Engineering tests use synthetic inputs only; no accuracy conclusions."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch

FROZEN = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017/odr/source_A/core')
sys.path.insert(0,str(FROZEN))
from engine.adapter import do_adapt
from gate_core import rank, fit_calibration, decide, random_budget_mask
from state_adapter import OriginalAAdapter, isolated_scoring, equal_tree, clone_tree, rng_state


class TinyEHS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.BatchNorm2d(3))
        self.seg=torch.nn.Conv2d(3,1,1)
        self.register_buffer('scratch',torch.tensor(0.))
        self.pollute=False
    def forward(self,x,is_train=False,gpu=None):
        x=self.backbone(x)
        if self.pollute:
            self.scratch.add_(1)
        logit=self.seg(x)
        loss=-(torch.sigmoid(logit)*torch.nn.functional.logsigmoid(logit)).mean()
        return loss,{'ent':loss.detach()}


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17); np.random.seed(17)
        self.images=[torch.rand(1,3,8,8) for _ in range(12)]
        self.initial=TinyEHS().state_dict()
    def new(self):
        m=TinyEHS();m.load_state_dict(self.initial)
        return OriginalAAdapter.configure(m,torch.device('cpu'))
    def test_all_accepted_matches_frozen_A(self):
        ref=self.new(); got=self.new()
        cfg=NS(ODR=NS(NUM_TRAIN_SAMPLES=12,IMAGES_PER_BATCH=4),TRAIN=NS(IMAGES_PER_GPU=1),MODEL=NS(USE_GROUPNORM_BACKBONE=False),VERBOSE=False)
        do_adapt(0,cfg,ref.model,ref.bn_features,self.images,ref.optimizer,device=torch.device('cpu'))
        for image in self.images:
            before=clone_tree(got.model.state_dict())
            with isolated_scoring(got.model,got.bn_features):
                got.model(image)
                torch.rand(2);np.random.rand(2)
            self.assertTrue(equal_tree(before,got.model.state_dict()))
            got.observed+=1;got.accept(image)
        got.finish()
        self.assertTrue(equal_tree(ref.model.state_dict(),got.model.state_dict()))
        self.assertTrue(equal_tree(ref.optimizer.state_dict(),got.optimizer.state_dict()))
        self.assertEqual(got.updates,3)
        ref.close();got.close()
    def test_scoring_preserves_partial_gradients_rng_and_cache(self):
        a=self.new();a.accept(self.images[0]);a.model.pollute=True
        grads={n:None if p.grad is None else p.grad.clone() for n,p in a.model.named_parameters()}
        params=clone_tree(a.model.state_dict()); caches=clone_tree(a.bn_features); optim=clone_tree(a.optimizer.state_dict()); rng=clone_tree(rng_state())
        with isolated_scoring(a.model,a.bn_features):
            a.model(self.images[1]);torch.rand(3);np.random.rand(3)
        self.assertTrue(equal_tree(params,a.model.state_dict()))
        self.assertTrue(equal_tree(caches,a.bn_features))
        self.assertTrue(equal_tree(optim,a.optimizer.state_dict()))
        self.assertTrue(equal_tree(rng,rng_state()))
        self.assertTrue(equal_tree(grads,{n:p.grad for n,p in a.model.named_parameters()}));a.close()
    def test_reject_does_not_change_model_and_tail_restores_buffers(self):
        a=self.new();a.model.pollute=True
        for image in self.images[:4]:a.accept(image)
        boundary=clone_tree(a.model.state_dict());optimizer=clone_tree(a.optimizer.state_dict())
        for image in self.images[4:7]:a.accept(image)
        a.observed=12
        counts=a.finish()
        self.assertEqual(counts,dict(observed=12,accepted=7,rejected=5,updates=1,used=4,tail=3))
        self.assertTrue(equal_tree(boundary,a.model.state_dict()))
        self.assertTrue(equal_tree(optimizer,a.optimizer.state_dict()))
        self.assertTrue(all(p.grad is None for p in a.model.parameters()));a.close()
    def test_complete_checkpoint_resume(self):
        a=self.new()
        for image in self.images[:4]:a.observed+=1;a.accept(image)
        ck=a.checkpoint()
        for image in self.images[4:]:a.observed+=1;a.accept(image)
        b=self.new();b.restore(ck)
        for image in self.images[4:]:b.observed+=1;b.accept(image)
        self.assertTrue(equal_tree(a.checkpoint(),b.checkpoint()));a.close();b.close()
    def test_incomplete_checkpoint_rejected(self):
        a=self.new();a.accept(self.images[0])
        with self.assertRaises(ValueError):a.checkpoint()
        a.close()
    def test_combined_calibration_ties_and_frozen_threshold(self):
        records=[]
        for d in ('lightbox','sunlamp'):
            for i in range(100):
                records.append(dict(domain=d,sample_id=str(i),geometry_scoreable=True,confidence=float(i),reprojection=float(99-i),rotation_disagreement=float(99-i),translation_disagreement=float(99-i),entropy=float(99-i)))
        c=fit_calibration(records)
        for m,want in [('G50',100),('G75',150),('E50',100),('E75',150)]:
            self.assertEqual(sum(decide(m,r,c)[0] for r in records),want)
        self.assertEqual(float(rank(1,[1,1,2,3])),.25)
        illegal=dict(records[0],geometry_scoreable=False,geometry_reason='pnp_rejected')
        self.assertFalse(decide('G75',illegal,c)[0])
    def test_random_mask_budget_exact_reproducible(self):
        x=random_budget_mask(1024,683,9301)
        self.assertEqual(sum(x),680)
        self.assertEqual(x,random_budget_mask(1024,683,9301))
        self.assertNotEqual(x,random_budget_mask(1024,683,9302))


if __name__=='__main__':
    assert os.environ.get('STY'),'Use unique guarded screen for engineering tests'
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
