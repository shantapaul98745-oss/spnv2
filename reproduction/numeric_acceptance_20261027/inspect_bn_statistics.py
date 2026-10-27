"""Admitted zero-forward comparison of the actual persisted matching checkpoint."""
import pathlib,json,os,sys,time,hashlib,uuid,importlib.util
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
  record=launch_record({'run_id':'g50p_saved_state_difference_20261027'},str(root/(nonce+'.log')),gate);record['diagnostic_code_sha256']=sha(__file__);atomic_json(attempt/'launch.json',record)
  doc=read(registry);doc['records'].append(dict(record_path=str(attempt/'launch.json'),record_sha256=gate.launch_record_sha256(record),expected_record=record));atomic_json(registry,doc)
  if conflict(collector.collect_snapshot(registry,project_root=P,project_id=PROJECT),record,gate):raise ValueError('Own diagnostic registration unavailable')
  global_budget=GlobalBudget(P/'budget',dict(project_id=PROJECT,frozen_inputs_sha256=idx['frozen_inputs']['sha256']),initial_cost_receipt=idx['initial_cost_receipt'],budget_policy=idx['budget_policy']);budget=global_budget.begin_attempt(record,allow_model_forwards=False)
  try:
   import pair_runner
   rev=read(idx['numeric_acceptance']['path']);spec=rev['comparator'];assert sha(spec['path'])==spec['sha256'];module_spec=importlib.util.spec_from_file_location('saved_numeric_comparator',spec['path']);numeric=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(numeric)
   row=next(x for x in idx['tasks'] if x['run_id']=='g50p_diag_v1_sunlamp_s2021_w0023');task=read(row['task_path']);store=P/'pairs'/row['run_id']/'isolated_durable';current=read(store/'CURRENT');directory=store/'checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=read(directory/'metadata.json');actual_spec=dict(directory=str(directory),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256'])
   actual,_=pair_runner.read_generation(actual_spec);expected,_=pair_runner.read_generation(task['selection']['match_checkpoint'])
   import numpy as np
   def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else x
   sections=[('learning',actual['learner']['learning']['bn_features'],expected['learner']['learning']['bn_features']),('window_base',actual['learner']['window_base']['bn_features'],expected['learner']['window_base']['bn_features'])]
   worst={};summary={'channel_mean':{'max_abs':0.,'max_current_ratio':0.},'channel_mean_square':{'max_abs':0.,'max_current_ratio':0.}};layers=0;channels=0
   for section,aa,bb in sections:
    if set(aa)!=set(bb):raise ValueError('BN cache keys changed')
    for key in aa:
     x,y=array(aa[key]),array(bb[key])
     if x.shape!=y.shape or x.dtype!=y.dtype or x.ndim!=4 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('BN cache shape/dtype/finiteness differs')
     x,y=x.astype(np.float64),y.astype(np.float64);layers+=1;channels+=x.shape[1]
     for name,a,b in [('channel_mean',x.mean(axis=(0,2,3)),y.mean(axis=(0,2,3))),('channel_mean_square',np.square(x).mean(axis=(0,2,3)),np.square(y).mean(axis=(0,2,3)))]:
      diff=np.abs(a-b);bound=1e-6+1e-5*np.abs(b);ratio=diff/bound;i=int(np.argmax(ratio));stats=summary[name];stats['max_abs']=max(stats['max_abs'],float(np.max(diff)))
      if float(ratio[i])>stats['max_current_ratio']:
       stats['max_current_ratio']=float(ratio[i]);worst[name]={'cache':section,'layer':str(key),'channel':i,'actual':float(a[i]),'expected':float(b[i]),'abs_diff':float(diff[i]),'fixed_bound':float(bound[i]),'excess_ratio':float(ratio[i])}
   result=dict(schema='spnv2-all-BN-cache-statistical-difference-v1',status='diagnosed_not_accepted',layers=layers,channels=channels,summary=summary,worst=worst,actual_checkpoint=actual_spec,expected_checkpoint=task['selection']['match_checkpoint'],model_forwards=0,acceptance_policy_unchanged=True,no_images_or_GT_read=True,proposed_tolerances_not_applied=True)
   atomic_json(output,result);closed=budget.close('BN_statistical_difference_diagnosed');atomic_json(attempt/'budget_closed.json',closed);atomic_json(attempt/'outcome.json',dict(status='diagnosed',exit_code=0,result=str(output),result_sha256=sha(output),model_forwards=0));print(json.dumps(result));return 0
  except Exception as exc:
   if budget is not None and not budget.closed:budget.close('saved_state_comparison_failed')
   atomic_json(attempt/'outcome.json',dict(status='failed',exit_code=1,error_type=type(exc).__name__,error=str(exc),model_forwards=0));raise
if __name__=='__main__':raise SystemExit(main())
