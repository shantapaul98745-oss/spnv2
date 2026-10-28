"""Admitted zero-forward comparison of the saved w54 complete matching checkpoint."""
import pathlib,json,os,sys,time,hashlib,uuid,importlib.util,copy
from contextlib import ExitStack
P=pathlib.Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
sys.path.insert(0,str(P/'implementation'))
from host_entry import host_modules,project_lock,launch_record,conflict,read,sha,PROJECT
from budget import GlobalBudget,atomic_json

def resolve(value,path):
 obj=value;remaining=path
 while remaining:
  if isinstance(obj,dict):
   keys=[k for k in obj if remaining==str(k) or remaining.startswith(str(k)+'.')]
   if not keys:raise KeyError(remaining)
   k=max(keys,key=lambda x:len(str(x)));remaining=remaining[len(str(k)):].lstrip('.');obj=obj[k]
  elif isinstance(obj,(list,tuple)):
   k,_,remaining=remaining.partition('.');obj=obj[int(k)]
  else:
   if remaining in ('channel_mean','channel_mean_square') and hasattr(obj,'shape'):
    import numpy as np
    arr=obj.detach().cpu().numpy() if hasattr(obj,'detach') else obj
    if arr.ndim!=4:raise ValueError('BN statistic requires original NCHW cache')
    arr=arr.astype(np.float64);obj=arr.mean(axis=(0,2,3)) if remaining=='channel_mean' else np.square(arr).mean(axis=(0,2,3));remaining=''
   else:raise KeyError(remaining)
 return obj

def main():
 idx=read(P/'index.json');nonce=os.environ['SPNV2_RUN_NONCE'];root=P/'revisions/numeric_acceptance_20261027/saved_state_diagnosis';root.mkdir(exist_ok=True);attempt=P/'launches'/nonce;attempt.mkdir(exist_ok=False);output=attempt/'result.json';budget=None
 with ExitStack() as stack:
  for name in ['rebaseline_20261017','odr003_exploratory_20261023','odr003_round2_20261023','g50p_risk_diagnostic_20261025']:stack.enter_context(project_lock(P.parent/name/'pipeline.lock'))
  stack.enter_context(project_lock(P/'manager.lock'))
  gate,collector=host_modules(idx);registry=P/'launches/registry.json'
  before=collector.collect_snapshot(registry,project_root=P,project_id=PROJECT);after=collector.collect_snapshot(registry,project_root=P,project_id=PROJECT);decision=gate.assess_launch_boundary(before,after,now=time.time(),project_lock_held=True,task_kind='heavy_cpu');from dataclasses import asdict
  atomic_json(attempt/'admission.json',dict(before=before,after=after,decision=asdict(decision)))
  if not decision.allow_start:atomic_json(output,dict(status='waiting_resources',model_forwards=0,reasons=decision.reasons));return 75
  record=launch_record({'run_id':'g50p_w54_full_state_difference_20261028'},str(root/(nonce+'.log')),gate);record['diagnostic_code_sha256']=sha(__file__);atomic_json(attempt/'launch.json',record)
  doc=read(registry);doc['records'].append(dict(record_path=str(attempt/'launch.json'),record_sha256=gate.launch_record_sha256(record),expected_record=record));atomic_json(registry,doc)
  if conflict(collector.collect_snapshot(registry,project_root=P,project_id=PROJECT),record,gate):raise ValueError('Own diagnostic registration unavailable')
  global_budget=GlobalBudget(P/'budget',dict(project_id=PROJECT,frozen_inputs_sha256=idx['frozen_inputs']['sha256']),initial_cost_receipt=idx['initial_cost_receipt'],budget_policy=idx['budget_policy']);budget=global_budget.begin_attempt(record,allow_model_forwards=False)
  try:
   import pair_runner
   rev=read(idx['numeric_acceptance']['path']);spec=rev['comparator'];assert sha(spec['path'])==spec['sha256'];module_spec=importlib.util.spec_from_file_location('saved_numeric_comparator',spec['path']);numeric=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(numeric)
   row=next(x for x in idx['tasks'] if x['run_id']=='g50p_diag_v1_sunlamp_s2021_w0054');task=read(row['task_path']);store=P/'pairs'/row['run_id']/'isolated_durable';current=read(store/'CURRENT');directory=store/'checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=read(directory/'metadata.json');actual_spec=dict(directory=str(directory),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256'])
   actual,_=pair_runner.read_generation(actual_spec);expected,_=pair_runner.read_generation(task['selection']['match_checkpoint']);a=pair_runner.science_state(actual);e=pair_runner.science_state(expected)
   if meta['scientific_cursors']!=task['selection']['match_checkpoint']['cursors']:raise ValueError('Saved current checkpoint is not the exact selected complete boundary')
   ok=numeric.equal(a,e);failure=copy.deepcopy(numeric.LAST_FAILURE);detail=None
   atomic_json(attempt/'first_difference.json',dict(run_id=row['run_id'],first_difference=failure,accepted_under_current_policy=ok,model_forwards=0))
   if failure:
    x,y=resolve(a,failure['path']),resolve(e,failure['path']);detail={'actual_type':type(x).__name__,'expected_type':type(y).__name__}
    import numpy as np
    if hasattr(x,'shape'):
     if hasattr(x,'detach'):x=x.detach().cpu().numpy();y=y.detach().cpu().numpy()
     detail.update(shape=list(x.shape),dtype=str(x.dtype),different_elements=int(np.count_nonzero(x!=y)))
     if x.dtype.kind in 'fc':
      xx,yy=x.astype(np.float64),y.astype(np.float64);diff=np.abs(xx-yy);atol=numeric.absolute_tolerance(numeric.current(),tuple(failure['path'].split('.')));bounds=atol+numeric.current()['relative_tolerance']*np.abs(yy);ind=np.unravel_index(int(np.argmax(diff/bounds)),x.shape) if x.shape else ();detail.update(max_abs=float(np.max(diff)),max_fraction=float(np.max(diff/bounds)),max_index=[int(i) for i in ind],actual_value=float(x[ind]),expected_value=float(y[ind]))
     else:
      ix=np.argwhere(x!=y)[:12];detail['first_indices']=ix.tolist();detail['first_actual']=[int(x[tuple(i)]) for i in ix];detail['first_expected']=[int(y[tuple(i)]) for i in ix]
    elif isinstance(x,(str,int,float,bool,type(None))):detail.update(actual=x,expected=y)
    else:detail.update(actual_repr=repr(x)[:500],expected_repr=repr(y)[:500])
   excluded=[]
   def diagnostic_view(value,path=()):
    if isinstance(value,dict):
     out={}
     for k,z in value.items():
      if k=='bn_features':excluded.append('.'.join(path+(str(k),)))
      else:out[k]=diagnostic_view(z,path+(str(k),))
     return out
    if isinstance(value,list):return [diagnostic_view(z,path+(str(i),)) for i,z in enumerate(value)]
    if isinstance(value,tuple):return tuple(diagnostic_view(z,path+(str(i),)) for i,z in enumerate(value))
    return value
   av,ev=diagnostic_view(a),diagnostic_view(e)
   remaining_ok=numeric.equal(av,ev);remaining_failure=numeric.LAST_FAILURE
   diagnostic_only=dict(excluded_fields=sorted(set(excluded)),all_remaining_fields_pass_current_policy=remaining_ok,first_remaining_difference=remaining_failure,not_an_acceptance_waiver=True)
   result=dict(schema='spnv2-saved-full-state-difference-v1',run_id=row['run_id'],status='compared',accepted_under_current_policy=ok,first_difference=failure,detail=detail,diagnostic_excluding_bn_features=diagnostic_only,actual_checkpoint=actual_spec,expected_checkpoint=task['selection']['match_checkpoint'],actual_cursors=meta['scientific_cursors'],expected_cursors=task['selection']['match_checkpoint']['cursors'],numeric_statistics=numeric.STATS,model_forwards=0,no_images_or_GT_read=True,registered_record_sha256=gate.launch_record_sha256(record),budget_before_close=budget.snapshot());atomic_json(output,result);closed=budget.close('saved_state_comparison_complete');atomic_json(attempt/'budget_closed.json',closed);atomic_json(attempt/'outcome.json',dict(status='compared',exit_code=0,result=str(output),result_sha256=sha(output),model_forwards=0));print(json.dumps(dict(status='compared',result=str(output),accepted=ok,first_difference=failure,detail=detail,diagnostic_excluding_bn_features=diagnostic_only)));return 0
  except Exception as exc:
   if budget is not None and not budget.closed:budget.close('saved_state_comparison_failed')
   atomic_json(attempt/'outcome.json',dict(status='failed',exit_code=1,error_type=type(exc).__name__,error=str(exc),model_forwards=0));raise
if __name__=='__main__':raise SystemExit(main())
