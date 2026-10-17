"""Acceptance for actual production trainer and boundary checkpoint helpers."""
import argparse
import copy
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

parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True)
parser.add_argument('--report',required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(args.source)/'core'))
from config import cfg as defaults
import engine.trainer as trainer
from utils.training_state import (TrainingState,EpochBatchSampler,SeededDataset,
    save_training_checkpoint,load_training_checkpoint,capture_rng_state,sha256_file)

torch.set_num_threads(2)
warnings.filterwarnings('ignore',message='.*CUDA is not available.*')
warnings.filterwarnings('ignore',category=FutureWarning)

def seed(value=2021):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)

def config(k):
    cfg=defaults.clone();cfg.defrost()
    cfg.TRAIN.ACCUM_STEPS=k
    cfg.TRAIN.END_EPOCH=1
    cfg.TRAIN.AMP_SKIP_WARMUP=0
    cfg.TRAIN.AMP_MAX_CONSECUTIVE_SKIPS=16
    cfg.MODEL.HEAD.LOSS_HEADS=['heatmap']
    cfg.VERBOSE=False
    cfg.freeze();return cfg

class Data(Dataset):
    def __init__(self,n=37,augment=False):self.n=n;self.augment=augment;self.loaded=[]
    def __len__(self):return self.n
    def __getitem__(self,i):
        self.loaded.append(i)
        x=torch.tensor([i/31,(i%5)/5,1.],dtype=torch.float64)
        if self.augment:x=x+(random.random()+np.random.rand()+torch.rand(()).item())*.01
        return x,dict(y=torch.tensor([i/41],dtype=torch.float64),imgpath=f'sample-{i:03d}')

class Toy(nn.Module):
    def __init__(self,stochastic=False):
        super().__init__();self.linear=nn.Linear(3,1,dtype=torch.float64)
        self.stochastic=stochastic;self.dropout=nn.Dropout(.2);self.seen=[]
    def forward(self,x,is_train=True,gpu=None,y=None,imgpath=None,**kw):
        self.seen.extend(list(imgpath or []))
        if self.stochastic:x=self.dropout(x)*(1+.1*random.random()+.1*np.random.rand())
        loss=((self.linear(x)-y)**2).mean()
        return loss,dict(hmap=loss.detach())

def loader(k=4,n=37,b=2,workers=0,augment=False):
    raw=Data(n,augment);s=EpochBatchSampler(n,b,k,2021,True)
    d=DataLoader(SeededDataset(raw,2021),batch_sampler=s,num_workers=workers,
                 generator=torch.Generator().manual_seed(812))
    return d,s,raw

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)

def equal_tree(case,a,b):
    if isinstance(a,torch.Tensor):case.assertTrue(torch.equal(a,b),(a,b))
    elif isinstance(a,dict):
        case.assertEqual(set(a),set(b))
        for k in a:equal_tree(case,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        case.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal_tree(case,x,y)
    else:case.assertEqual(a,b)

class Tests(unittest.TestCase):
    def test_accum1_preserves_old_math(self):
        seed();new=Toy();old=copy.deepcopy(new)
        a,b=optimizer(new),optimizer(old)
        d,_,_=loader(k=1,n=10,b=2)
        trainer.do_train(0,config(1),new,d,a)
        old.train()
        for x,t in d:
            t.pop('_sample_index');b.zero_grad()
            loss,_=old(x,**t);loss.backward();torch.nn.utils.clip_grad_norm_(old.parameters(),1.);b.step()
        equal_tree(self,new.state_dict(),old.state_dict());equal_tree(self,a.state_dict(),b.state_dict())

    def test_accum4_equals_mean_of_local_losses(self):
        seed();new=Toy();ref=copy.deepcopy(new);a,b=optimizer(new),optimizer(ref)
        d,_,_=loader(n=18)
        state=trainer.do_train(0,config(4),new,d,a)
        b.zero_grad();losses=[]
        for x,t in d:
            t.pop('_sample_index');loss,_=ref(x,**t);losses.append(loss)
            if len(losses)==4:
                (sum(losses)/4).backward();torch.nn.utils.clip_grad_norm_(ref.parameters(),1.);b.step();b.zero_grad();losses=[]
        for key,value in new.state_dict().items():torch.testing.assert_close(value,ref.state_dict()[key],rtol=1e-14,atol=1e-14)
        self.assertEqual(state.update_attempts,2);self.assertEqual(state.samples_seen,16)

    def test_tail_and_formal_counts(self):
        for physical,samples,windows in [(4,47952,2997),(1,47964,11991)]:
            s=EpochBatchSampler(47966,physical,4)
            self.assertEqual(s.total_batches,windows*4);self.assertEqual(len(s.indices()),samples)
        seed();m=Toy();d,s,raw=loader(n=19,b=2)
        state=trainer.do_train(0,config(4),m,d,optimizer(m))
        self.assertEqual(len(raw.loaded),16);self.assertEqual(len(m.seen),16)
        self.assertEqual(state.next_batch,8);self.assertEqual(state.optimizer_updates,2)
        with self.assertRaises(ValueError):s.set_epoch(0,3)

    def test_sample_rng_independent_of_workers_and_resume_position(self):
        seed();a,_,_=loader(n=37,augment=True,workers=0)
        b,s,_=loader(n=37,augment=True,workers=2)
        first=list(a);second=list(b)
        for (x,t),(y,u) in zip(first,second):
            self.assertTrue(torch.equal(x,y));self.assertEqual(t['imgpath'],u['imgpath'])
        s.set_epoch(0,4)
        for (x,t),(y,u) in zip(first[4:],b):
            self.assertTrue(torch.equal(x,y));self.assertEqual(t['imgpath'],u['imgpath'])

    def test_exact_resume_rng_samples_weights_optimizer(self):
        manifest=dict(version=1,code={'trainer':sha256_file(Path(args.source)/'core/engine/trainer.py')})
        seed();a=Toy(True);ao=optimizer(a);ad,_,_=loader(augment=True,workers=2)
        whole=trainer.do_train(0,config(4),a,ad,ao);whole_rng=capture_rng_state()
        seed();b=Toy(True);bo=optimizer(b);bd,bs,_=loader(augment=True,workers=2)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'boundary.pth.tar'
            def interrupt(state):
                if state.next_batch==8:
                    save_training_checkpoint(path,b,bo,None,state,bs,manifest)
                    raise InterruptedError('controlled boundary interruption')
            with self.assertRaises(InterruptedError):trainer.do_train(0,config(4),b,bd,bo,on_boundary=interrupt)
            prefix=b.seen.copy();seed(97);c=Toy(True);co=optimizer(c);cd,cs,_=loader(augment=True,workers=2)
            resumed,_=load_training_checkpoint(path,c,co,None,cs,manifest)
            trainer.do_train(0,config(4),c,cd,co,state=resumed)
            self.assertEqual(a.seen,prefix+c.seen)
            equal_tree(self,a.state_dict(),c.state_dict());equal_tree(self,ao.state_dict(),co.state_dict())
            equal_tree(self,whole.state_dict(),resumed.state_dict())
            equal_tree(self,whole_rng,capture_rng_state())
            with self.assertRaises(RuntimeError):load_training_checkpoint(path,c,co,None,cs,dict(version=99))
            with self.assertRaises(RuntimeError):load_training_checkpoint(path,nn.Linear(3,2),co,None,cs,manifest)
            resumed.next_batch=3
            with self.assertRaises(ValueError):save_training_checkpoint(path,c,co,None,resumed,cs,manifest)

    def test_amp_window_boundary_skip_and_clipping(self):
        class InfiniteBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x):return x.clone()
            @staticmethod
            def backward(ctx,g):return g*float('inf')
        class AmpToy(Toy):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,*a,**kw):
                loss,parts=super().forward(*a,**kw);self.calls+=1
                if self.calls<=4:loss=InfiniteBackward.apply(loss)
                return loss,parts
        class Spy:
            def __init__(self):self.inner=torch.amp.GradScaler('cpu',init_scale=128);self.calls=[]
            def scale(self,l):self.calls.append('scale');return self.inner.scale(l)
            def unscale_(self,o):self.calls.append('unscale');return self.inner.unscale_(o)
            def step(self,o):self.calls.append('step');return self.inner.step(o)
            def update(self):self.calls.append('update');return self.inner.update()
            def get_scale(self):return self.inner.get_scale()
        seed();m=AmpToy();d,_,_=loader(n=16);o=optimizer(m);scaler=Spy();clips=[]
        original=trainer.clip_grad_norm_
        def clip(*a,**kw):clips.append(m.calls);return original(*a,**kw)
        trainer.clip_grad_norm_=clip
        try:state=trainer.do_train(0,config(4),m,d,o,scaler=scaler)
        finally:trainer.clip_grad_norm_=original
        self.assertEqual(scaler.calls,['scale']*4+['unscale','step','update']+['scale']*4+['unscale','step','update'])
        self.assertEqual(clips,[8]);self.assertEqual(state.update_attempts,2)
        self.assertEqual(state.optimizer_updates,1);self.assertEqual(state.amp_skips,1)

    def test_nonfinite_loss_stops_before_update(self):
        class Bad(Toy):
            def forward(self,*a,**kw):
                loss,parts=super().forward(*a,**kw);return loss*float('nan'),parts
        seed();m=Bad();d,_,_=loader(n=8);state=TrainingState()
        with self.assertRaises(FloatingPointError):trainer.do_train(0,config(4),m,d,optimizer(m),state=state)
        self.assertEqual(state.update_attempts,0)

suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
            failures=[dict(test=str(t),trace=trace) for t,trace in result.failures+result.errors],
            source=str(Path(args.source).resolve()),code_sha256={str(p.relative_to(args.source)):sha256_file(p)
                for folder in ('core','tools') for p in sorted((Path(args.source)/folder).rglob('*.py'))})
Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
sys.exit(0 if result.wasSuccessful() else 1)
