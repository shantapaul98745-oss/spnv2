"""Observation logging and scope checks shared by real execution and CPU tests."""
import torch
from torch.utils.data import Dataset
class IndexedDataset(Dataset):
 def __init__(self,base):self.base=base
 def __len__(self):return len(self.base)
 def __getitem__(self,index):return self.base[index],index
class LoggedLoader:
 def __init__(self,loader,split,limit,domain=None,ids=None):
  self.base=loader;self.dataset=loader.dataset;self.split=split;self.limit=limit;self.ids=[];self.domain=domain;self.names=ids
 def __len__(self):return len(self.base)
 def __iter__(self):
  for item in self.base:
   if self.split=='train':
    if len(self.ids)>=self.limit:return
    images,indices=item;assert images.shape[0]==len(indices)==1,'One actual observation per physical forward'
    for index in indices.tolist():
     base=self.dataset.base;name=self.names[index] if self.names is not None else str(base.csv.iloc[index,0])
     self.ids.append(f'{self.domain or base.domain}/{name}')
    yield images
   else:yield item

def norm_scope(model):
 allowed=set();running=set();momenta={}
 for name,module in model.backbone.named_modules():
  if isinstance(module,(torch.nn.BatchNorm2d,torch.nn.GroupNorm)):
   allowed.update('backbone.'+name+'.'+p for p,_ in module.named_parameters(recurse=False))
  if isinstance(module,torch.nn.BatchNorm2d):
   momenta[name]=module.momentum
   running.update('backbone.'+name+'.'+s for s in ['running_mean','running_var'])
 return allowed,running,momenta

def check_scope(model,before,line):
 after={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 assert set(after)==set(before)
 assert all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in after.values())
 allowed,running,momenta=norm_scope(model)
 trainable={n for n,p in model.named_parameters() if p.requires_grad}
 assert trainable==allowed and trainable
 changed={k for k in after if not torch.equal(after[k],before[k])}
 buffers_unchanged=all(torch.equal(after[k],before[k]) for k,_ in model.named_buffers())
 assert changed and changed<=allowed|(running if line!='A' else set()),'Unexpected update scope'
 assert all(torch.equal(after[k],before[k]) for k in after if not k.startswith('backbone.')),'Prediction heads changed'
 if line=='A':assert buffers_unchanged
 elif running:
  assert changed&running==running,'Every repaired BN running tensor must update'
  assert all(bool((after[k]>=0).all()) for k in running if k.endswith('running_var'))
 return after,dict(trainable_names=sorted(trainable),changed_names=sorted(changed),expected_running_keys=sorted(running),changed_running_buffers=sorted(changed&running),bn_momenta=momenta,heads_unchanged=True,buffers_unchanged=buffers_unchanged)
