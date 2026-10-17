"""Engineering only: real frozen adapter math with tiny deterministic mock networks.
No source weights, real images, GPU, target scores, or scientific results are used.
"""
import os,sys,json,ast,copy,types,importlib.util,time,traceback
from pathlib import Path
from types import SimpleNamespace as NS
from common import *
assert os.environ.get('STY'),'Use unique guarded screen'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU tests must hide CUDA'
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from audit_core import *
from generate import generate
results=[]
def check(name,fn):
 started=time.time();value=fn();results.append(dict(name=name,status='passed',seconds=time.time()-started,detail=value));print('PASS',name,json.dumps(value),flush=True)
class Meter:
 def __init__(self,*a,**k):pass
 def update(self,*a,**k):pass
class Progress:
 def __init__(self,*a,**k):pass
 def display(self,*a,**k):pass
 def display_summary(self,*a,**k):pass
# Import the real adapter bytes; replace display-only dependencies, never math.
u=types.ModuleType('utils.utils');u.AverageMeter=Meter;u.ProgressMeter=Progress
v=types.ModuleType('utils.visualize');v.torch=torch
sys.modules['utils.utils']=u;sys.modules['utils.visualize']=v

def adapter(line):
 p=O/('source_'+line)/'core/engine/adapter.py';spec=importlib.util.spec_from_file_location('real_adapter_'+line,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.do_adapt
class Tiny(nn.Module):
 def __init__(self,gn=False,momentum=.01):
  super().__init__();self.backbone=nn.Sequential();self.backbone.add_module('conv',nn.Conv2d(2,4,1,bias=False,dtype=torch.float64));self.backbone.add_module('norm',nn.GroupNorm(2,4,dtype=torch.float64) if gn else nn.BatchNorm2d(4,momentum=momentum,dtype=torch.float64));self.heads=nn.Conv2d(4,1,1,dtype=torch.float64)
 def forward(self,images,is_train=True,gpu=None,**kwargs):
  y=self.heads(self.backbone(images));loss=(y-.3).square().mean();return loss,{'ent':loss.detach()}

def configure(model):
 model.eval();model.requires_grad_(False)
 for m in model.backbone.modules():
  if isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):m.requires_grad_(True)
 return model

def math_case(line,gn,b,n):
 torch.manual_seed(209);model=configure(Tiny(gn,.1 if line=='B2' else .01));reference=copy.deepcopy(model);before=copy.deepcopy(model.state_dict());features={}
 hook=None
 if not gn:hook=model.backbone.norm.register_forward_hook(lambda m,i,o:features.update(norm=i[0].detach()))
 data=[torch.randn(1,2,3,3,dtype=torch.float64)+j*.03 for j in range(n+3)]
 class Stream:
  def __init__(self):self.forwards=0
  def __len__(self):return len(data)
  def __iter__(self):
   for image in data:self.forwards+=1;yield image
 stream=Stream();opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.025);refopt=torch.optim.SGD([p for p in reference.parameters() if p.requires_grad],lr=.025)
 steps=[];handle=opt.register_step_post_hook(lambda *a:steps.append(stream.forwards))
 cfg=NS(ODR=NS(NUM_TRAIN_SAMPLES=n,IMAGES_PER_BATCH=b),TRAIN=NS(IMAGES_PER_GPU=1),MODEL=NS(USE_GROUPNORM_BACKBONE=gn),VERBOSE=False)
 adapter(line)(0,cfg,model,features,stream,opt,device=torch.device('cpu'),scaler=None)
 for start in range(0,n,b):
  window_features=[]
  for image in data[start:start+b]:
   if not gn:window_features.append(reference.backbone.conv(image).detach())
   loss,_=reference(image);(loss/b).backward()
  refopt.step();refopt.zero_grad(set_to_none=True)
  if not gn and line!='A':
   # Independent pooled moment formula; biased variance matches original adapter.
   merged=torch.cat(window_features,dim=0);mean=merged.mean((0,2,3));variance=merged.square().mean((0,2,3))-mean.square();norm=reference.backbone.norm
   with torch.no_grad():norm.running_mean.mul_(1-norm.momentum).add_(mean,alpha=norm.momentum);norm.running_var.mul_(1-norm.momentum).add_(variance,alpha=norm.momentum)
 for k,value in model.state_dict().items():assert torch.allclose(value,reference.state_dict()[k],rtol=1e-12,atol=1e-13),k
 _,scope=check_scope(model,before,line)
 assert stream.forwards==n and steps==list(range(b,n+1,b))
 if hook:hook.remove()
 handle.remove()
 return dict(line=line,norm='GN' if gn else 'BN',N=n,B=b,forwards=stream.forwards,steps=len(steps),heads_unchanged=scope['heads_unchanged'],running_changes=len(scope['changed_running_buffers']),max_abs_error=max(float((v-reference.state_dict()[k]).abs().max()) for k,v in model.state_dict().items()))

class SampleDataset(Dataset):
 domain='mock'
 def __init__(self,size):self.size=size
 def __len__(self):return self.size
 def __getitem__(self,index):return torch.tensor([float(index)])

def observations(seed,size,n):
 torch.manual_seed(seed);base=SampleDataset(size);loader=DataLoader(base,batch_size=1,shuffle=True,num_workers=0,drop_last=True)
 object.__setattr__(loader,'dataset',IndexedDataset(base));logged=LoggedLoader(loader,'train',n,domain='mock',ids=[f'image_{i}' for i in range(size)])
 iterator=iter(logged)
 while len(logged.ids)<n:
  try:image=next(iterator)
  except StopIteration:iterator=iter(logged);image=next(iterator)
  assert image.shape[0]==1
 assert len(logged.ids)==n and len(set(logged.ids))==min(n,size)
 for begin in range(0,n,size):assert len(set(logged.ids[begin:begin+size]))==len(logged.ids[begin:begin+size])
 return logged.ids

def stream_test():
 orders={}
 for seed in range(2021,2026):
  for size,n in [(7,4),(7,7),(7,19),(2791,8192)]:
   a=observations(seed,size,n);b=observations(seed,size,n);assert a==b;orders[(seed,size,n)]=a
 assert len({tuple(orders[(s,7,19)]) for s in range(2021,2026)})==5
 return dict(seeds=list(range(2021,2026)),cases=20,max_N=8192,cycles_and_truncation=True,repeatable_same_seed=True,different_streams_between_seeds=True)

def manifest_test():
 m=generate();assert m['counts']=={'fixed':40,'Bscan':120,'Nscan':240};assert m['unique_executions']==370 and m['reuse_requests']==30
 assert len({t['id'] for t in m['tasks']})==400
 keys=['phi','line','domain','seed','N','B']
 sys.path.insert(0,str(O/'source_A/core'))
 from config import cfg as defaults,update_config
 for t in m['tasks']:
  loaded=defaults.clone();update_config(loaded,NS(cfg=t['config'],opts=[]));assert loaded.TRAIN.ACCUM_STEPS==1 and loaded.TRAIN.REPRODUCIBLE is False
  assert t['N']%t['B']==0 and t['seed'] in range(2021,2026)
  if t.get('reuse_of'):
   ref=next(q for q in m['tasks'] if q['id']==t['reuse_of']);assert all(t[k]==ref[k] for k in keys)
  import yaml
  c=yaml.safe_load(Path(t['config']).read_text());assert c['TRAIN']['REPRODUCIBLE'] is False and c['TRAIN']['ACCUM_STEPS']==c['TRAIN']['IMAGES_PER_GPU']==1 and c['FP16'] is False
  assert c['ODR']['IMAGES_PER_BATCH']==t['B'] and c['ODR']['NUM_TRAIN_SAMPLES']==t['N']
  assert c['MODEL']['PRETRAIN_FILE']==str(source_path(t['phi'])) and str(PLAN/'outputs') in c['MODEL']['PRETRAIN_FILE']
  assert c['MODEL']['HEAD']['LOSS_HEADS']==[] and c['ODR']['MIN_ENTROPY']
  assert c['ROOT']==str(O/('source_'+t['line']))
 for line in ['A','B1','B2']:verify_code(line)
 a=tree(O/'source_A');b1=tree(O/'source_B1');b2=tree(O/'source_B2')
 assert {k for k in a if a[k]!=b1[k]}=={'core/engine/adapter.py'}
 assert {k for k in a if a[k]!=b2[k]}=={'core/engine/adapter.py','tools/odr.py'}
 for f in O.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 verify_implementation()
 return {k:m[k] for k in ['requested','unique_executions','reuse_requests','counts']}

def negative_scope():
 torch.manual_seed(2);m=configure(Tiny());before=copy.deepcopy(m.state_dict())
 with torch.no_grad():m.heads.bias.add_(1)
 try:check_scope(m,before,'A')
 except AssertionError:pass
 else:raise AssertionError('Head mutation accepted')
 return dict(head_mutation_rejected=True)

try:
 check('manifest_configs_snapshots',manifest_test)
 for line in ['A','B1','B2']:
  for b in [1,4,32]:check(f'actual_adapter_{line}_BN_B{b}',lambda line=line,b=b:math_case(line,False,b,64))
 for b in [1,4,32]:check(f'actual_adapter_A_GN_B{b}',lambda b=b:math_case('A',True,b,64))
 check('observation_stream',stream_test);check('scope_rejects_head_mutation',negative_scope)
 report=dict(status='passed',scope='CPU engineering acceptance only; no real image/weights/GPU accuracy experiment',tests=results,passed=len(results),gpu_runtime_validated=False,scientific_results_created=False)
 atomic(O/'reports/math_cpu_acceptance_v2.json',report);print(json.dumps(report),flush=True)
except BaseException as error:
 atomic(O/'reports/math_cpu_acceptance_v2.json',dict(status='failed',tests=results,error=repr(error),traceback=traceback.format_exc(),scientific_results_created=False));raise
