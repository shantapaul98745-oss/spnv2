"""Reusable input, event, optimizer and RNG validation without GPU side effects."""
from pathlib import Path
import csv,datetime,hashlib,json,math,os,random

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit_event(name,**fields):
 value=dict(event=name,phase='odr',timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat());value.update(fields)
 print('SPNV2_EVENT '+json.dumps(value,sort_keys=True,allow_nan=False),flush=True);return value

def input_evidence(csv_path,image_root,camera,keypoints,domain):
 csv_path=Path(csv_path).resolve();image_root=Path(image_root).resolve()
 with csv_path.open() as f:ids=[row[0] for row in csv.reader(f)]
 assert ids and len(ids)==len(set(ids)),'Target CSV sample IDs must be nonempty and unique'
 records=[]
 for index,name in enumerate(ids):
  path=(image_root/name).resolve();assert path.is_relative_to(image_root),'Target image escapes actual dataset folder'
  stat=path.stat();assert path.is_file()
  records.append([index,name,str(path),stat.st_size,stat.st_mtime_ns])
 geometry={name:dict(path=str(Path(path).resolve()),sha256=digest(path)) for name,path in [('camera',camera),('keypoints',keypoints)]}
 return dict(schema='odr-input-evidence-v1',domain=domain,csv=dict(path=str(csv_path),sha256=digest(csv_path),sample_count=len(ids)),geometry=geometry,image_root=str(image_root),raw_image_content_hashed=False,image_metadata_fields=['row','sample_id','absolute_path','size_bytes','mtime_ns'],image_records=records,image_metadata_sha256=hashlib.sha256(json.dumps(records,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())

def verify_input_evidence(saved):
 assert saved['schema']=='odr-input-evidence-v1' and saved['raw_image_content_hashed'] is False
 actual=input_evidence(saved['csv']['path'],saved['image_root'],saved['geometry']['camera']['path'],saved['geometry']['keypoints']['path'],saved['domain'])
 assert actual==saved,'Target CSV, geometry or image metadata changed since adaptation start'
 return True

def finite_tree(value):
 import torch
 if torch.is_tensor(value):return not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all())
 if isinstance(value,float):return math.isfinite(value)
 if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
 if isinstance(value,(tuple,list)):return all(finite_tree(v) for v in value)
 return True

def validate_rng(states,cuda_count=1):
 import numpy as np,torch
 assert isinstance(states,dict) and set(states)=={'python','numpy','torch_cpu','torch_cuda'} and finite_tree(states),'Nonfinite/incomplete RNG state'
 py=states['python'];assert isinstance(py,tuple) and len(py)==3 and py[0]==3 and isinstance(py[1],tuple) and len(py[1])==625
 assert all(isinstance(x,int) and 0<=x<2**32 for x in py[1][:-1]) and 0<=py[1][-1]<=624
 assert py[2] is None or isinstance(py[2],float) and math.isfinite(py[2])
 random.Random().setstate(py)
 n=states['numpy'];assert set(n)=={'kind','keys','pos','has_gauss','cached_gaussian'}
 assert n['kind']=='MT19937' and len(n['keys'])==624 and all(isinstance(x,int) and 0<=x<2**32 for x in n['keys'])
 assert isinstance(n['pos'],int) and 0<=n['pos']<=624 and n['has_gauss'] in [0,1] and math.isfinite(n['cached_gaussian'])
 np.random.RandomState().set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
 cpu=states['torch_cpu'];assert torch.is_tensor(cpu) and cpu.dtype==torch.uint8 and cpu.ndim==1 and cpu.numel()>0
 torch.Generator(device='cpu').set_state(cpu.cpu())
 cuda=states['torch_cuda'];assert isinstance(cuda,list) and len(cuda)==cuda_count
 for value in cuda:assert torch.is_tensor(value) and value.dtype==torch.uint8 and value.ndim==1 and value.numel()>0
 return True

def validate_optimizer(payload,model,expected_steps,cfg=None):
 import torch
 assert isinstance(payload,dict) and set(payload)=={'state','param_groups'} and finite_tree(payload),'Nonfinite/incomplete optimizer state'
 parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 groups=payload['param_groups'];assert len(groups)==1 and len(groups[0]['params'])==len(parameters)
 ids=groups[0]['params'];assert len(set(ids))==len(ids) and set(payload['state'])<=set(ids) and payload['state']
 names={pid:name for pid,(name,param) in zip(ids,parameters)};params={pid:param for pid,(name,param) in zip(ids,parameters)}
 if cfg is not None:
  assert groups[0]['lr']==cfg.TRAIN.LR and groups[0]['weight_decay']==cfg.TRAIN.WD and tuple(groups[0]['betas'])==(cfg.TRAIN.GAMMA1,cfg.TRAIN.GAMMA2)
 for pid,state in payload['state'].items():
  assert set(state)==({'step','exp_avg','exp_avg_sq','max_exp_avg_sq'} if groups[0].get('amsgrad',False) else {'step','exp_avg','exp_avg_sq'})
  step=state['step'];assert (torch.is_tensor(step) and step.numel()==1 or isinstance(step,(int,float)))
  assert float(step)==expected_steps and expected_steps>0
  for name,value in state.items():
   if name!='step':assert torch.is_tensor(value) and value.shape==params[pid].shape,'Optimizer moment shape mismatch'
  assert bool((state['exp_avg_sq']>=0).all()),'Negative AdamW second moment'
 return sorted(names[pid] for pid in payload['state'])

def equal_tree(a,b):
 import torch
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
 return a==b


def validate_weights(state,reference):
 import torch
 assert state.keys()==reference.keys() and finite_tree(state),'Nonfinite/missing model state'
 for name,value in state.items():
  expected=reference[name];assert torch.is_tensor(value) and value.dtype==expected.dtype and value.shape==expected.shape,'Model state tensor schema mismatch: '+name
 return True


def anchor_input_evidence(saved,csv_path,image_root,camera,keypoints,domain):
 assert saved['domain']==domain and Path(saved['csv']['path'])==Path(csv_path).resolve()
 assert Path(saved['image_root'])==Path(image_root).resolve(),'Image inventory does not cover the configured target image folder'
 assert saved['geometry']['camera']['path']==str(Path(camera).resolve()) and saved['geometry']['keypoints']['path']==str(Path(keypoints).resolve()),'Geometry evidence paths differ from actual configuration'
 return True
