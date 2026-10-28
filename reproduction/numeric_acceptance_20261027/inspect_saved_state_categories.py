"""Admitted read-only complete saved-state classification; never model inference."""
import pathlib,json,os,sys,time,hashlib,importlib.util,copy,math
from contextlib import ExitStack
P=pathlib.Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
sys.path.insert(0,str(P/'implementation'))
from host_entry import host_modules,project_lock,launch_record,conflict,read,sha,PROJECT
from budget import GlobalBudget,atomic_json
if len(sys.argv)!=3 or sys.argv[1]!='--run-id':raise SystemExit('Explicit --run-id required')
RID=sys.argv[2]
LEARNING={'model','buffers','optimizer','gradients','bn_features','bn_mean','bn_mean_sq','boundary_buffers','rng'}
MAX_DETAILS=50

def wrapped(value,path):
 for part in reversed(path):value={part:value}
 return value

def resolve(value,path):
 obj=value;remaining=path
 while remaining:
  if isinstance(obj,dict):
   keys=[k for k in obj if remaining==str(k) or remaining.startswith(str(k)+'.')]
   if not keys:raise KeyError(remaining)
   key=max(keys,key=lambda k:len(str(k)));remaining=remaining[len(str(key)):].lstrip('.');obj=obj[key]
  elif isinstance(obj,(list,tuple)):
   key,_,remaining=remaining.partition('.');obj=obj[int(key)]
  elif remaining in ('channel_mean','channel_mean_square') and hasattr(obj,'shape'):
   import numpy as np
   arr=obj.detach().cpu().numpy() if hasattr(obj,'detach') else np.asarray(obj)
   if arr.ndim!=4:raise ValueError('BN cache statistic requires NCHW')
   arr=arr.astype(np.float64);obj=arr.mean(axis=(0,2,3)) if remaining=='channel_mean' else np.square(arr).mean(axis=(0,2,3));remaining=''
  else:raise KeyError(remaining)
 return obj

def learning_roots(a,e,path=()):
 found=[]
 if isinstance(a,dict) and isinstance(e,dict):
  if LEARNING<=set(a) and LEARNING<=set(e):return [(path,a,e)]
  for key in a.keys()&e.keys():found+=learning_roots(a[key],e[key],path+(key,))
 elif isinstance(a,(list,tuple)) and type(a) is type(e):
  for i,(x,y) in enumerate(zip(a,e)):found+=learning_roots(x,y,path+(i,))
 return found

def tensor_detail(x,y,path,numeric,*,atol_override=None,bn_affine_paths=()):
 import numpy as np
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 if hasattr(y,'detach'):y=y.detach().cpu().numpy()
 x,y=np.asarray(x),np.asarray(y);out=dict(path='.'.join(map(str,path)),shape=list(x.shape),dtype=str(x.dtype))
 if x.shape!=y.shape or x.dtype!=y.dtype:return dict(out,structure_mismatch=True)
 if x.dtype.kind not in 'fc':return dict(out,different_elements=int(np.count_nonzero(x!=y)))
 xx,yy=x.astype(np.complex128 if x.dtype.kind=='c' else np.float64),y.astype(np.complex128 if y.dtype.kind=='c' else np.float64)
 if not np.isfinite(xx).all() or not np.isfinite(yy).all():return dict(out,nonfinite=True)
 atol=(numeric.absolute_tolerance(numeric.current(),tuple(map(str,path)),bn_affine_paths=bn_affine_paths) if hasattr(numeric,'trainable_bn_affine_paths') else numeric.absolute_tolerance(numeric.current(),tuple(map(str,path)))) if atol_override is None else atol_override
 exact=numeric._exact_path(tuple(map(str,path)))
 bounds=np.zeros_like(np.abs(yy)) if exact else atol+numeric.current()['relative_tolerance']*np.abs(yy);delta=np.abs(xx-yy)
 ratio=np.divide(delta,bounds,out=np.zeros_like(delta),where=bounds!=0);bad=(bounds==0)&(delta>0)
 if np.any(bad):ratio[bad]=np.inf
 flat=int(np.argmax(ratio)) if ratio.size else 0;index=np.unravel_index(flat,x.shape) if x.shape and x.size else ()
 fraction=float(np.max(ratio)) if ratio.size else 0.;maximum=float(np.max(delta)) if delta.size else 0.
 out.update(atol=0. if exact else atol,rtol=0. if exact else numeric.current()['relative_tolerance'],max_abs=maximum,max_fraction=fraction if math.isfinite(fraction) else 'infinite_exact_difference',violating_elements=int(np.count_nonzero(delta>bounds)),different_elements=int(np.count_nonzero(xx!=yy)),max_index=[int(i) for i in index])
 if x.size and x.dtype.kind!='c':out.update(actual_value=float(xx[index]),expected_value=float(yy[index]))
 return out

def main():
 idx=read(P/'index.json');index_sha=sha(P/'index.json');nonce=os.environ.get('SPNV2_RUN_NONCE','')
 if len(nonce)!=32 or any(x not in '0123456789abcdef' for x in nonce) or not os.environ.get('STY'):raise ValueError('Unique owned screen/nonce required')
 if RID not in {row['run_id'] for row in idx['tasks']}:raise ValueError('Diagnostic must name one frozen window')
 state=read(P/'status.json')
 if state['status']!='blocked' or (state.get('active') or {}).get('run_id')!=RID or RID in state['accepted']:raise ValueError('Only current failed unaccepted window may be diagnosed')
 attempt=P/'launches'/nonce;attempt.mkdir(exist_ok=False);output=attempt/'result.json';budget=None
 with ExitStack() as stack:
  for name in ('rebaseline_20261017','odr003_exploratory_20261023','odr003_round2_20261023','g50p_risk_diagnostic_20261025'):stack.enter_context(project_lock(P.parent/name/'pipeline.lock'))
  stack.enter_context(project_lock(P/'manager.lock'))
  if sha(P/'index.json')!=index_sha:raise ValueError('Diagnostic index changed before locks')
  gate,collector=host_modules(idx);registry=P/'launches/registry.json';before=collector.collect_snapshot(registry,project_root=P,project_id=PROJECT);after=collector.collect_snapshot(registry,project_root=P,project_id=PROJECT);decision=gate.assess_launch_boundary(before,after,now=time.time(),project_lock_held=True,task_kind='heavy_cpu');from dataclasses import asdict
  atomic_json(attempt/'admission.json',dict(before=before,after=after,decision=asdict(decision)))
  if not decision.allow_start:atomic_json(output,dict(status='waiting_resources',model_forwards=0,reasons=decision.reasons));return 75
  record=launch_record({'run_id':RID+'_saved_state_categories'},str(attempt/'diagnostic.log'),gate);record['diagnostic_code_sha256']=sha(__file__);atomic_json(attempt/'launch.json',record);doc=read(registry);doc['records'].append(dict(record_path=str(attempt/'launch.json'),record_sha256=gate.launch_record_sha256(record),expected_record=record));atomic_json(registry,doc)
  if conflict(collector.collect_snapshot(registry,project_root=P,project_id=PROJECT),record,gate):raise ValueError('Own registered diagnostic identity unavailable')
  global_budget=GlobalBudget(P/'budget',dict(project_id=PROJECT,frozen_inputs_sha256=idx['frozen_inputs']['sha256']),initial_cost_receipt=idx['initial_cost_receipt'],budget_policy=idx['budget_policy']);budget=global_budget.begin_attempt(record,allow_model_forwards=False)
  try:
   import pair_runner
   rev=read(idx['numeric_acceptance']['path']);spec=rev['comparator']
   if sha(spec['path'])!=spec['sha256']:raise ValueError('Current comparator binding changed')
   loader=importlib.util.spec_from_file_location('current_category_comparator',spec['path']);numeric=importlib.util.module_from_spec(loader);loader.loader.exec_module(numeric)
   row=next(x for x in idx['tasks'] if x['run_id']==RID);task=read(row['task_path']);store=P/'pairs'/RID/'isolated_durable';current=read(store/'CURRENT');directory=store/'checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=read(directory/'metadata.json');actual_spec=dict(directory=str(directory),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256'])
   if meta['scientific_cursors']!=task['selection']['match_checkpoint']['cursors']:raise ValueError('Expected the saved complete matching boundary')
   actual,_=pair_runner.read_generation(actual_spec);expected,_=pair_runner.read_generation(task['selection']['match_checkpoint']);a,e=pair_runner.science_state(actual),pair_runner.science_state(expected)
   full_ok=numeric.equal(a,e);full_failure=copy.deepcopy(numeric.LAST_FAILURE);atomic_json(attempt/'first_difference.json',dict(accepted_under_current_policy=full_ok,first_difference=full_failure,model_forwards=0))
   affine_paths=numeric.trainable_bn_affine_paths(a,e) if hasattr(numeric,'trainable_bn_affine_paths') else set()
   def current_tensor_detail(x,y,path,_numeric,*,atol_override=None):return tensor_detail(x,y,path,_numeric,atol_override=atol_override,bn_affine_paths=affine_paths)
   roots=learning_roots(a,e);groups=[];detail_count=0
   if not roots:raise ValueError('No complete learning snapshot identified')
   def compare(x,y,path):
    ok=numeric.equal(wrapped(x,path),wrapped(y,path),learning_context=(a,e)) if hasattr(numeric,'trainable_bn_affine_paths') else numeric.equal(wrapped(x,path),wrapped(y,path));return ok,copy.deepcopy(numeric.LAST_FAILURE)
   def detail(failure):
    path=failure['path'];out=dict(path=path,reason=failure['reason'])
    try:
     xx,yy=resolve(a,path),resolve(e,path)
     if hasattr(xx,'shape') or isinstance(xx,float):out.update(current_tensor_detail(xx,yy,tuple(path.split('.')),numeric))
     elif isinstance(xx,(str,int,bool,type(None))):out.update(actual=xx,expected=yy)
     else:out.update(actual_type=type(xx).__name__,expected_type=type(yy).__name__)
    except (KeyError,ValueError,TypeError,IndexError) as exc:out['optional_detail_unavailable']=repr(exc)
    return out
   def leaves(x,y,path):
    if type(x) is not type(y):yield x,y,path;return
    if isinstance(x,dict):
     if set(x)!=set(y):yield x,y,path
     for key in x.keys()&y.keys():yield from leaves(x[key],y[key],path+(key,))
    elif isinstance(x,(list,tuple)):
     if len(x)!=len(y):yield x,y,path
     for i,(xx,yy) in enumerate(zip(x,y)):yield from leaves(xx,yy,path+(i,))
    else:yield x,y,path
   for prefix,x,y in roots:
    for category in sorted(set(x)|set(y)):
     budget.check_time('read-only category '+str(category));path=prefix+(category,);label='.'.join(map(str,path))
     if category not in x or category not in y:groups.append(dict(path=label,category=str(category),passes_current_policy=False,failed_items=1,structure_changed=True));continue
     ok,failure=compare(x[category],y[category],path);group=dict(path=label,category=str(category),passes_current_policy=ok,first_difference=failure,failed_items=0,failed_tensor_items=0,failure_details=[],details_truncated=False,worst_numeric_detail=None)
     if not ok:
      for xx,yy,leaf_path in leaves(x[category],y[category],path):
       leaf_ok,leaf_failure=compare(xx,yy,leaf_path)
       if leaf_ok:continue
       group['failed_items']+=1;group['failed_tensor_items']+=int(hasattr(xx,'shape'));item=detail(leaf_failure)
       if detail_count<MAX_DETAILS:group['failure_details'].append(item);detail_count+=1
       else:group['details_truncated']=True
       ratio=item.get('max_fraction');old=group['worst_numeric_detail'];score=float('inf') if isinstance(ratio,str) else ratio
       if score is not None and (old is None or score>(float('inf') if isinstance(old.get('max_fraction'),str) else old.get('max_fraction',-1))):group['worst_numeric_detail']=item
     if category=='bn_features':
      import numpy as np
      moments={name:dict(tensors=0,channels=0,max_abs=0.,max_fraction=0.,violating_elements=0,min_required_atol_with_fixed_rtol=0.,worst=None) for name in ('channel_mean','channel_mean_square')}
      for key in x[category].keys()&y[category].keys():
       xx,yy=x[category][key],y[category][key]
       xx=xx.detach().cpu().numpy() if hasattr(xx,'detach') else np.asarray(xx);yy=yy.detach().cpu().numpy() if hasattr(yy,'detach') else np.asarray(yy)
       if xx.shape!=yy.shape or xx.dtype!=yy.dtype or xx.ndim!=4 or not np.isfinite(xx).all() or not np.isfinite(yy).all():raise ValueError('BN cache structure/finiteness differs')
       xx,yy=xx.astype(np.float64),yy.astype(np.float64)
       for moment,aa,bb in [('channel_mean',xx.mean(axis=(0,2,3)),yy.mean(axis=(0,2,3))),('channel_mean_square',np.square(xx).mean(axis=(0,2,3)),np.square(yy).mean(axis=(0,2,3)))]:
        stat=current_tensor_detail(aa,bb,path+(key,moment),numeric);item=moments[moment];item['tensors']+=1;item['channels']+=int(aa.size);item['max_abs']=max(item['max_abs'],stat['max_abs']);item['violating_elements']+=stat['violating_elements'];item['min_required_atol_with_fixed_rtol']=max(item['min_required_atol_with_fixed_rtol'],float(np.max(np.maximum(np.abs(aa-bb)-numeric.current()['relative_tolerance']*np.abs(bb),0.))))
        if stat['max_fraction']>=item['max_fraction']:item['max_fraction']=stat['max_fraction'];item['worst']=stat
      group['both_cache_moments_exhaustively_checked']=moments
     if category in ('bn_mean','bn_mean_sq'):
      summary=dict(tensors=0,max_abs=0.,max_fraction=0.,violating_elements=0,min_required_atol_with_fixed_rtol=0.,proposed_atol_1e5_passes_diagnostic_only=True,proposed_atol=1e-5,proposed_rtol=1e-5,proposal_not_applied=True,worst_current=None,worst_proposed=None)
      for xx,yy,leaf_path in leaves(x[category],y[category],path):
       if not hasattr(xx,'shape'):
        if not compare(xx,yy,leaf_path)[0]:summary['proposed_atol_1e5_passes_diagnostic_only']=False
        continue
       cur=current_tensor_detail(xx,yy,leaf_path,numeric);pro=current_tensor_detail(xx,yy,leaf_path,numeric,atol_override=1e-5);summary['tensors']+=1
       import numpy as np
       xa=xx.detach().cpu().numpy() if hasattr(xx,'detach') else np.asarray(xx);ya=yy.detach().cpu().numpy() if hasattr(yy,'detach') else np.asarray(yy)
       if xa.shape==ya.shape and np.isfinite(xa).all() and np.isfinite(ya).all():summary['min_required_atol_with_fixed_rtol']=max(summary['min_required_atol_with_fixed_rtol'],float(np.max(np.maximum(np.abs(xa.astype(np.float64)-ya.astype(np.float64))-numeric.current()['relative_tolerance']*np.abs(ya.astype(np.float64)),0.)))) if xa.size else summary['min_required_atol_with_fixed_rtol']
       if cur.get('nonfinite') or cur.get('structure_mismatch') or pro.get('nonfinite') or pro.get('structure_mismatch'):summary['proposed_atol_1e5_passes_diagnostic_only']=False;continue
       summary['max_abs']=max(summary['max_abs'],cur.get('max_abs',0.));summary['violating_elements']+=cur.get('violating_elements',0);ratio=cur.get('max_fraction',0.)
       if isinstance(ratio,str) or ratio>=summary['max_fraction']:summary['max_fraction']=ratio;summary['worst_current']=cur
       summary['proposed_atol_1e5_passes_diagnostic_only'] &= pro.get('violating_elements',1)==0
       prev=summary['worst_proposed']
       if prev is None or pro.get('max_fraction',0.)>prev.get('max_fraction',0.):summary['worst_proposed']=pro
      group['all_accumulator_tensors']=summary
     groups.append(group)
   root_paths={path for path,_,_ in roots}
   def remaining(value,path=()):
    if path in root_paths:return {'independently_classified_learning_path':'.'.join(map(str,path))}
    if isinstance(value,dict):return {k:remaining(v,path+(k,)) for k,v in value.items()}
    if isinstance(value,list):return [remaining(v,path+(i,)) for i,v in enumerate(value)]
    if isinstance(value,tuple):return tuple(remaining(v,path+(i,)) for i,v in enumerate(value))
    return value
   remaining_ok=numeric.equal(remaining(a),remaining(e));remaining_failure=copy.deepcopy(numeric.LAST_FAILURE)
   other_ok=all(g['passes_current_policy'] for g in groups if g['category'] not in ('bn_mean','bn_mean_sq')) and remaining_ok
   result=dict(schema='spnv2-saved-state-categories-v2',status='classified',run_id=RID,actual_checkpoint=actual_spec,expected_checkpoint=task['selection']['match_checkpoint'],cursors=meta['scientific_cursors'],accepted_under_current_policy=full_ok,first_difference=full_failure,learning_snapshots=['.'.join(map(str,p)) for p,_,_ in roots],groups=groups,remaining_scientific_state=dict(passes_current_policy=remaining_ok,first_difference=remaining_failure,only_already_classified_learning_subtrees_partitioned=True,not_an_acceptance_waiver=True),all_non_accumulator_groups_and_remaining_pass_current_policy=other_ok,detail_limit=MAX_DETAILS,details_emitted=detail_count,details_truncated=any(g.get('details_truncated') for g in groups),numeric_acceptance=numeric.bindings(),numeric_statistics=numeric.STATS,BN_note='bn_mean/bn_mean_sq are pending four-input mean/raw-second-moment accumulators; true running buffers are model/buffers/boundary_buffers. Cache-statistics authorization does not cover these accumulators.',proposal_not_applied=True,model_forwards=0,no_images_or_GT_read=True,no_tensor_or_scientific_state_writes=True,registered_record_sha256=gate.launch_record_sha256(record),budget_before_close=budget.snapshot());atomic_json(output,result);closed=budget.close('saved_state_category_scan_complete');atomic_json(attempt/'budget_closed.json',closed);atomic_json(attempt/'outcome.json',dict(status='compared',exit_code=0,result=str(output),result_sha256=sha(output),model_forwards=0));print(json.dumps(dict(status='classified',result=str(output),accepted_under_current_policy=full_ok,failing_categories=[g['path'] for g in groups if not g['passes_current_policy']],remaining_passes=remaining_ok,model_forwards=0)));return 0
  except Exception as exc:
   if budget is not None and not budget.closed:budget.close('saved_state_category_scan_failed')
   atomic_json(attempt/'outcome.json',dict(status='failed',exit_code=1,error_type=type(exc).__name__,error=str(exc),model_forwards=0));raise
if __name__=='__main__':raise SystemExit(main())
