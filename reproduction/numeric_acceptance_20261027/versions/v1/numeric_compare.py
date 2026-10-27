"""Explicit cross-device numeric acceptance; internal restoration stays exact."""
import json,hashlib,math,sys
from pathlib import Path
ROOT=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
PLAN=ROOT/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
POLICY=None
STATS={'comparisons':0,'numeric_values':0,'max_absolute_difference':0.,'max_tolerance_fraction':0.,'failures':0}
LAST_FAILURE=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_bound(x):
 if sha(x['path'])!=x['sha256']:raise ValueError('Numeric acceptance binding changed')
 return json.loads(Path(x['path']).read_text())
def configure(policy):
 global POLICY
 if policy.get('schema')!='spnv2-cross-gpu-numeric-acceptance-v1' or policy.get('authorization')!='user_2026-10-27_authorized_scientific_acceptance_revision':raise ValueError('Numeric acceptance authorization differs')
 if policy['absolute_tolerance']!=1e-6 or policy['relative_tolerance']!=1e-5 or not policy['finite_only'] or policy['same_run_rollback_and_branch_isolation']!='bitwise_exact':raise ValueError('Fixed scientific tolerance/strict isolation differs')
 if POLICY is not None and POLICY!=policy:raise ValueError('Acceptance policy cannot change during a run')
 POLICY=policy
 return policy

def current():
 if POLICY is None:
  idx=json.loads((PLAN/'index.json').read_text());rev=read_bound(idx['numeric_acceptance']);configure(read_bound(rev['policy']))
 return POLICY

def bindings():
 idx=json.loads((PLAN/'index.json').read_text());rev=read_bound(idx['numeric_acceptance'])
 return {'revision':idx['numeric_acceptance'],'policy':rev['policy']}

def _exact_path(path):
 return any(('rng' in part.lower() or part.lower() in ('random_state','param_groups','identity','threshold','thresholds','calibration')) for part in path)

def equal(actual,expected):
 global LAST_FAILURE
 p=current();STATS['comparisons']+=1;LAST_FAILURE=None
 def fail(path,reason):
  global LAST_FAILURE
  STATS['failures']+=1;LAST_FAILURE={'path':'.'.join(map(str,path)),'reason':reason};return False
 def visit(x,y,path=(),exact=False):
  exact=exact or _exact_path(path)
  if type(x) is not type(y):return fail(path,'type_changed')
  if isinstance(x,dict):
   if set(x)!=set(y):return fail(path,'keys_changed')
   return all(visit(x[k],y[k],path+(str(k),),exact) for k in x)
  if isinstance(x,(list,tuple)):
   if len(x)!=len(y):return fail(path,'length_changed')
   return all(visit(a,b,path+(str(i),),exact) for i,(a,b) in enumerate(zip(x,y)))
  if isinstance(x,float):
   if not math.isfinite(x) or not math.isfinite(y):return fail(path,'nonfinite')
   delta=abs(x-y);bound=0. if exact else p['absolute_tolerance']+p['relative_tolerance']*abs(y)
   STATS['numeric_values']+=1;STATS['max_absolute_difference']=max(STATS['max_absolute_difference'],delta)
   if bound:STATS['max_tolerance_fraction']=max(STATS['max_tolerance_fraction'],delta/bound)
   return delta<=bound or fail(path,'numeric_tolerance_exceeded' if not exact else 'exact_state_changed')
  if type(x).__module__.startswith('numpy'):
   import numpy as np
   if isinstance(x,np.ndarray):
    if x.shape!=y.shape or x.dtype!=y.dtype:return fail(path,'shape_or_dtype_changed')
    if x.dtype.kind in 'fc':
     if not np.isfinite(x).all() or not np.isfinite(y).all():return fail(path,'nonfinite')
     if exact:return bool(np.array_equal(x,y)) or fail(path,'exact_state_changed')
     diff=np.abs(x-y);bounds=p['absolute_tolerance']+p['relative_tolerance']*np.abs(y)
     STATS['numeric_values']+=int(x.size)
     if x.size:STATS['max_absolute_difference']=max(STATS['max_absolute_difference'],float(np.max(diff)));STATS['max_tolerance_fraction']=max(STATS['max_tolerance_fraction'],float(np.max(diff/bounds)))
     return bool(np.all(diff<=bounds)) or fail(path,'numeric_tolerance_exceeded')
    return bool(np.array_equal(x,y)) or fail(path,'discrete_array_changed')
   if isinstance(x,np.floating):return visit(float(x),float(y),path,exact)
  if type(x).__module__.startswith('torch') and hasattr(x,'detach'):
   import torch
   if x.shape!=y.shape or x.dtype!=y.dtype:return fail(path,'shape_or_dtype_changed')
   if x.is_floating_point() or x.is_complex():
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):return fail(path,'nonfinite')
    if exact:return bool(torch.equal(x,y)) or fail(path,'exact_state_changed')
    return visit(x.detach().cpu().numpy(),y.detach().cpu().numpy(),path,False)
   return bool(torch.equal(x,y)) or fail(path,'discrete_tensor_changed')
  try:return bool(x==y) or fail(path,'discrete_value_changed')
  except (ValueError,TypeError):return fail(path,'unsupported_value')
 return visit(actual,expected)

def receipt():return {'standard':'cross_gpu_numeric_v1','bindings':bindings(),'statistics':dict(STATS),'internal_branch_and_rollback':'bitwise_exact','discrete_decisions_and_rng':'exact'}
