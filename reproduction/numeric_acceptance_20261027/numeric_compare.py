"""Explicit cross-device numeric acceptance; internal restoration stays exact."""
import json,hashlib,math,sys,copy
from pathlib import Path
ROOT=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
PLAN=ROOT/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
POLICY=None
STATS={'comparisons':0,'numeric_values':0,'max_absolute_difference':0.,'max_tolerance_fraction':0.,'failures':0}
LAST_FAILURE=None
GATE_RUNTIME=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_bound(x):
 if sha(x['path'])!=x['sha256']:raise ValueError('Numeric acceptance binding changed')
 return json.loads(Path(x['path']).read_text())
def configure(policy):
 global POLICY
 if policy.get('schema')!='spnv2-cross-gpu-numeric-acceptance-v1' or policy.get('authorization')!='user_2026-10-27_authorized_scientific_acceptance_revision':raise ValueError('Numeric acceptance authorization differs')
 if policy['absolute_tolerance']!=1e-6 or policy['relative_tolerance']!=1e-5 or not policy['finite_only'] or policy['same_run_rollback_and_branch_isolation']!='bitwise_exact' or policy.get('rotation_disagreement_comparison')!='fixed_absolute_angle_degrees' or policy.get('rotation_angular_tolerance_degrees')!=0.01 or policy.get('derived_HE_delta_acceptance')!='recompute_own_arithmetic_and_keep_discrete_conditions_exact' or policy.get('bn_feature_cache_acceptance')!='per_channel_mean_and_mean_square' or policy.get('bn_channel_mean_absolute_tolerance')!=2e-5 or policy.get('bn_window_mean_absolute_tolerance')!=1e-5 or policy.get('bn_two_means_authorization')!='explicit_user_reply_2026-10-28_only_window_mean_1e5_cache_mean_2e5' or policy.get('geometric_prediction_comparison')!='per_object_norm' or policy.get('geometry_norm_authorization')!='explicit_user_reply_2026-10-28_geometry_object_norm' or policy.get('derived_gate_score_acceptance')!='exact_own_input_frozen_calibration_recomputation' or policy.get('heat_peaks_comparison')!='fixed11_vector_L2_norm':raise ValueError('Fixed scientific tolerance/strict isolation differs')
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


def _he_record(row):
 if not isinstance(row,dict) or row.get('schema')!='odr003-round2-protection-rules-v1' or not {'pre','post','rules'}<=set(row):return row
 r=copy.deepcopy(row)
 cond=r['rules']['joint_geometry_worsening']['subconditions']['he_disagreement_increase']
 if not cond['comparable']:return r
 pre,post=r['pre']['he_rotation_deg'],r['post']['he_rotation_deg'];threshold=cond['threshold']
 values=[pre,post,threshold,cond['value'],cond['compared_value'],cond['compared_threshold']]
 if not all(type(x) in (int,float) and math.isfinite(x) for x in values):raise ValueError('Nonfinite derived protection record')
 if cond['operator']!='post > pre + threshold' or any(not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in [(cond['value'],post-pre),(cond['compared_value'],post),(cond['compared_threshold'],pre+threshold)]) or cond['triggered'] is not (post>pre+threshold):raise ValueError('Derived protection arithmetic is not self-consistent')
 # Values are independently checked against their own accepted inputs above.
 # Normalize only this redundant representation for cross-device comparison;
 # never mutate the persisted records, predicates or actual computed values.
 for k in ('value','compared_value','compared_threshold'):cond[k]=0.
 STATS['derived_HE_records_recomputed']=STATS.get('derived_HE_records_recomputed',0)+1
 return r


def _gate_record(row):
 """Audit the original gate computation; never rewrite persisted score data."""
 global GATE_RUNTIME
 if not isinstance(row,dict) or row.get('kind')!='observation' or not {'arrival','accepted','gate_reason','gate_score'}<=set(row):return row
 p=current()
 if GATE_RUNTIME is None:
  import importlib.util
  item=p['derived_gate_code']
  if sha(item['path'])!=item['sha256']:raise ValueError('Frozen gate code changed')
  spec=importlib.util.spec_from_file_location('numeric_bound_original_gate',item['path']);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
  bundle=read_bound(p['derived_gate_calibration'])
  if bundle.get('status')!='passed' or bundle['calibration']['schema']!='odr003-frozen-shared-calibration-v1':raise ValueError('Frozen gate calibration invalid')
  GATE_RUNTIME=(gate,bundle['calibration'])
 gate,calibration=GATE_RUNTIME
 decision,reason,score=gate.decide('G50',row['arrival'],calibration)
 if type(row['accepted']) is not bool or row['accepted'] is not decision or row['gate_reason']!=reason:raise ValueError('Actual gate threshold decision is not self-consistent')
 stored=row['gate_score']
 if score is None:
  if stored is not None:raise ValueError('Unavailable gate score was replaced')
 elif type(stored) is not float or not math.isfinite(stored) or stored!=score:raise ValueError('Derived gate score does not exactly recompute from own frozen inputs')
 # Its original inputs and discrete outcome still undergo their unchanged
 # cross-device checks. Only the redundant derived score is normalized in a
 # comparison copy, after exact independent recomputation on each side.
 result=dict(row);result['gate_score']=0.
 STATS['derived_gate_records_recomputed']=STATS.get('derived_gate_records_recomputed',0)+1
 return result

def absolute_tolerance(policy,path):
 if len(path)>=3 and path[0]=='learner' and path[1] in ('learning','window_base'):
  if path[2]=='bn_mean':return policy['bn_window_mean_absolute_tolerance']
  if path[2]=='bn_features' and path[-1]=='channel_mean':return policy['bn_channel_mean_absolute_tolerance']
 return policy['absolute_tolerance']

# Only full pose-prediction records are eligible. Model/optimizer/gradient
# tensors with coincidentally similar key names retain pointwise comparison.
GEOMETRY_SHAPES={'effi_R':(3,3),'effi_t':(3,),'heat_q':(4,),'heat_t':(3,),'bbox':(4,),'heat_peaks':(11,)}
def geometry_prediction_record(row):
 return isinstance(row,dict) and {'e_available','h_available','effi_R','effi_t','heat_q','heat_t','bbox'}<=set(row)

def geometry_compare(x,y,key,policy):
 import numpy as np
 if x is None or y is None:return (x is y),{'reason':'geometry_availability_changed'}
 if type(x) is not type(y):return False,{'reason':'geometry_type_changed'}
 if hasattr(x,'detach'):
  if x.dtype!=y.dtype:return False,{'reason':'geometry_dtype_changed'}
  x,y=x.detach().cpu().numpy(),y.detach().cpu().numpy()
 a,b=np.asarray(x),np.asarray(y)
 if a.shape!=b.shape or a.shape!=GEOMETRY_SHAPES[key] or a.dtype!=b.dtype or a.dtype.kind!='f':return False,{'reason':'geometry_shape_or_dtype_changed'}
 if not np.isfinite(a).all() or not np.isfinite(b).all():return False,{'reason':'geometry_nonfinite'}
 a,b=a.astype(np.float64),b.astype(np.float64)
 if key=='effi_R':
  identity=np.eye(3)
  if any(not np.allclose(z.T@z,identity,atol=policy['absolute_tolerance'],rtol=policy['relative_tolerance']) or not math.isclose(float(np.linalg.det(z)),1.,abs_tol=policy['absolute_tolerance'],rel_tol=policy['relative_tolerance']) for z in (a,b)):return False,{'reason':'invalid_rotation_matrix'}
  delta=float(np.linalg.norm(a-b));bound=2.*math.sqrt(2.)*math.sin(math.radians(policy['rotation_angular_tolerance_degrees'])/2.)
  return delta<=bound,dict(reason='rotation_angular_tolerance_exceeded',difference_norm=delta,bound=bound,angular_tolerance_degrees=policy['rotation_angular_tolerance_degrees'],comparison='raw_matrix_Frobenius_chord_no_projection')
 if key=='heat_q':
  na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b))
  if not(math.isfinite(na) and math.isfinite(nb)) or na==0. or nb==0.:return False,{'reason':'geometry_zero_quaternion'}
  a,b=a/na,b/nb
  if float(np.dot(a,b))<0.:a=-a
  delta=float(np.linalg.norm(a-b));bound=2.*math.sin(math.radians(policy['rotation_angular_tolerance_degrees'])/4.)
  return delta<=bound,dict(reason='quaternion_angular_tolerance_exceeded',difference_norm=delta,bound=bound,angular_tolerance_degrees=policy['rotation_angular_tolerance_degrees'],comparison='normalized_sign_equivalent_quaternion_chord')
 delta=float(np.linalg.norm((a-b).ravel()));scale=float(np.linalg.norm(b.ravel()));bound=policy['absolute_tolerance']+policy['relative_tolerance']*scale
 return delta<=bound,dict(reason='geometry_object_norm_exceeded',difference_norm=delta,expected_norm=scale,bound=bound,shape=list(a.shape))

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
   if path and path[-1]=='bn_features' and not exact:
    if set(x)!=set(y):return fail(path,'BN_cache_keys_changed')
    import numpy as np
    for key in x:
     xx,yy=x[key],y[key]
     if type(xx) is not type(yy):return fail(path+(str(key),),'BN_cache_type_changed')
     if hasattr(xx,'detach'):xx,yy=xx.detach().cpu().numpy(),yy.detach().cpu().numpy()
     if not isinstance(xx,np.ndarray) or xx.shape!=yy.shape or xx.dtype!=yy.dtype or xx.ndim!=4:return fail(path+(str(key),),'BN_cache_shape_or_dtype_changed')
     if not np.isfinite(xx).all() or not np.isfinite(yy).all():return fail(path+(str(key),),'BN_cache_nonfinite')
     xx,yy=xx.astype(np.float64),yy.astype(np.float64)
     for label,aa,bb in [('channel_mean',xx.mean(axis=(0,2,3)),yy.mean(axis=(0,2,3))),('channel_mean_square',np.square(xx).mean(axis=(0,2,3)),np.square(yy).mean(axis=(0,2,3)))]:
      if not visit(aa,bb,path+(str(key),label),False):return False
     STATS['bn_cache_tensors_compared']=STATS.get('bn_cache_tensors_compared',0)+1
    return True
   if not exact:
    try:x,y=_he_record(x),_he_record(y)
    except (ValueError,KeyError,TypeError):return fail(path,'inconsistent_derived_protection_record')
    try:x,y=_gate_record(x),_gate_record(y)
    except (ValueError,KeyError,TypeError):return fail(path,'inconsistent_derived_gate_record')
   if set(x)!=set(y):return fail(path,'keys_changed')
   if not exact and not any(part.lower() in ('model','optimizer','gradients','gradient','parameters','param_groups','bn_features') for part in path) and geometry_prediction_record(x) and geometry_prediction_record(y):
    for key in x:
     if key in GEOMETRY_SHAPES:
      ok,detail=geometry_compare(x[key],y[key],key,p)
      STATS['geometry_objects_compared']=STATS.get('geometry_objects_compared',0)+1
      if 'bound' in detail:
       STATS['max_geometry_tolerance_fraction']=max(STATS.get('max_geometry_tolerance_fraction',0.),detail['difference_norm']/detail['bound'])
      if not ok:
       fail(path+(str(key),),detail['reason']);LAST_FAILURE.update(detail);return False
     elif not visit(x[key],y[key],path+(str(key),),exact):return False
    return True
   return all(visit(x[k],y[k],path+(str(k),),exact) for k in x)
  if isinstance(x,(list,tuple)):
   if len(x)!=len(y):return fail(path,'length_changed')
   return all(visit(a,b,path+(str(i),),exact) for i,(a,b) in enumerate(zip(x,y)))
  if isinstance(x,float):
   if path and path[-1] in ('rotation_disagreement','he_rotation_deg') and not exact:
    if not (math.isfinite(x) and math.isfinite(y) and 0.<=x<=180. and 0.<=y<=180.):return fail(path,'invalid_derived_angle')
    delta=abs(x-y);bound=p['rotation_angular_tolerance_degrees']
    STATS['numeric_values']+=1;STATS['max_angular_difference_degrees']=max(STATS.get('max_angular_difference_degrees',0.),delta)
    return delta<=bound or fail(path,'derived_angle_tolerance_exceeded')
   if not math.isfinite(x) or not math.isfinite(y):return fail(path,'nonfinite')
   delta=abs(x-y);bound=0. if exact else absolute_tolerance(p,path)+p['relative_tolerance']*abs(y)
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
     diff=np.abs(x-y);bounds=absolute_tolerance(p,path)+p['relative_tolerance']*np.abs(y)
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

def receipt():return {'standard':'cross_gpu_numeric_v10_heat_peaks_vector_norm','bindings':bindings(),'statistics':dict(STATS),'internal_branch_and_rollback':'bitwise_exact','discrete_decisions_and_rng':'exact'}
