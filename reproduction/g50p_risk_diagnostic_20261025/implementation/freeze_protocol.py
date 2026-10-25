"""Stdlib-only immutable preparation; never opens labels or model weights."""
import json, hashlib, pathlib, random, os, datetime

BASE = pathlib.Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
OLD = BASE/'spnv2_runs/plans/odr003_round2_20261023'
PLAN = BASE/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
WINDOWS = {
 'lightbox:2021': ([1,22,44,66,88,110,132,154], []),
 'lightbox:2022': ([1,22,44,65,87,108,130,152], []),
 'lightbox:2023': ([1,22,44,65,87,108,130,152], []),
 'sunlamp:2021': ([1,54,84,114], [9,16,23,31]),
 'sunlamp:2022': ([1,5,9,14], [15,46,77,108]),
 'sunlamp:2023': ([1,17,33,49,65,81,97,113], []),
}
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(pathlib.Path(p).read_text())
def bind(p): return dict(path=str(p), sha256=sha(p))
def checked(b):
 assert sha(b['path']) == b['sha256'], b['path']
 return read(b['path'])
def immutable(p, v):
 p.parent.mkdir(parents=True, exist_ok=True)
 b=(json.dumps(v, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
 if p.exists():
  assert p.read_bytes()==b, 'immutable mismatch: '+str(p)
 else:
  fd=os.open(p, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
  with os.fdopen(fd,'wb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
 return bind(p)
def main():
 import fcntl
 PLAN.mkdir(exist_ok=True)
 with (PLAN/'preparation.lock').open('a+b') as lock:
  fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=read(OLD/'status.json'); assert status['status']=='complete' and status['active'] is None
  old_index=read(OLD/'engineering_index.json')
  assert sha(OLD/'engineering_index.json')=='887e230c74c5ae7c51ba69b61e4d69f2ff93effb443cd0b7a4beeb602db9253e'
  assert all(sha(OLD/'implementation'/k)==v for k,v in old_index['implementation_sha256'].items())
  producer=dict(commit='3ef5cbd32f16bcfcb650f4b5d7a04a69e718aa47', implementation_dir=str(OLD/'implementation'), implementation_sha256=old_index['implementation_sha256'])
  signals=[dict(name=k,field=k,risk_direction=d) for k,d in [('entropy','higher'),('confidence','lower'),('reprojection','higher'),('rotation_disagreement','higher'),('translation_disagreement','higher')]]
  static=dict(signals=signals,targets=[dict(name=g+'_final_minus_O_positive',method=g,kind='greater_than',field='positive_degradation',value=0) for g in ['A','G50','P','RW','RP']], initial_error_quantiles=[0,.25,.5,.75,1], strata_method='within each original Iref, stable sort by (finite Offline error, sample_id), four equal-count rank strata; missing initial error separate')
  protocol=dict(schema='g50p-risk-diagnostic-protocol-v1',diagnostic_id=PLAN.name,version='1.0',innovations=['004','009'],purpose_for_future='005 evidence only',producer=producer,max_new_complete_ODR=0,experiment_A_model_forwards=0,max_windows=48,max_physical_forwards=40000,max_active_seconds=14400,resource_wait_separate=True,first_two=['lightbox:2021:1','sunlamp:2021:9'],panel_size=128,panel_tail_count=7,panel_sampling='random.Random(aux_seed).sample(sorted eligible IDs,128), evaluate sorted chosen IDs',aux_seed_formula='2026102502 + domain_index*10000 + (stream_seed-2021)*100',static_diagnostic=static,branch_labels=dict(mean_error='accept.mean_error - rollback.mean_error > 0',tail_D='accept.tail7_D - rollback.tail7_D > 0',coverage='new missing relative to rollback; separately tabulated'),continuous_window_signals=[dict(name='max_H_rotation_change',direction='higher'),dict(name='max_HE_rotation_disagreement_increase',direction='higher'),dict(name='max_all11_residual_increase',direction='higher')],frozen_rules='all original science, B4, FP32, optimizer LR/WD, seeds, rules, pools, backend/software versions unchanged',whole_window_base_required=True,strict_equal=True,allow_approximation=False,GT_only='A saved errors; B branch offline evaluation only; reserved GT sealed',selection_changes_after_GT=False,stop_at_any_budget=True,scope_exclusions=['C','new ODR','new loss','regularization','pseudo labels','dynamic stopping','formal confirmation','full-domain','B/N','150 source epochs'])
  protocol_binding=immutable(PLAN/'protocol.json',protocol)
  selected=[]; runs={}; panels={}; offline=[]; all_candidates=rollbacks=0
  for key,(yes,no) in WINDOWS.items():
   domain,seed_s=key.split(':');seed=int(seed_s);rid=f'g50p2_det_20261024_p_{domain}_s{seed}_N1024_B4';entry=status['complete_accepted'][rid]
   t=checked(entry['run_task']);r=checked(entry['report']);proof=checked(entry['proof'])
   assert proof['status']=='passed' and proof['report_sha256']==entry['report']['sha256'] and t['implementation_sha256']==producer['implementation_sha256']
   all_candidates+=r['counts']['candidates'];rollbacks+=r['counts']['rollbacks']
   stream=checked(t['baseline_task']['bindings']['stream_manifest']);dev=checked(t['baseline_task']['bindings']['developer_manifest'])
   assert not stream['contains_gt'] and not dev['contains_gt'] and len(stream['items'])==1024 and len(dev['items'])==512
   refb=old_index['fixed_reference_bindings'][key];ref=checked(refb)
   adapt={x['sample_id'] for x in stream['items']};devmap={x['sample_id']:x for x in dev['items']}
   eligible=sorted(set(ref['reference_ids']) & set(devmap) - adapt)
   assert len(eligible)>=128, 'insufficient eligible panel: '+key
   aux=2026102502+(0 if domain=='lightbox' else 1)*10000+(seed-2021)*100
   ids=sorted(random.Random(aux).sample(eligible,128));assert not set(ids)&adapt
   panel={**{k:v for k,v in dev.items() if k!='items'},'schema':'g50p-diagnostic-panel-v1','run_key':key,'auxiliary_seed':aux,'items':[devmap[i] for i in ids],'eligible_count':len(eligible),'reference_binding':refb,'stream_binding':t['baseline_task']['bindings']['stream_manifest'],'developer_binding':t['baseline_task']['bindings']['developer_manifest'],'selection_depends_on_outcomes':False}
   pb=immutable(PLAN/'panels'/(domain+'_'+seed_s+'.json'),panel)
   oid=f'g50p2_det_20261024_o_{domain}_offline_dev512';oe=status['complete_accepted'][oid];orp=checked(oe['report']);op=orp['evaluation']
   ob=dict(path=op['prediction_path'],sha256=op['prediction_sha256'])
   panels[key]=dict(manifest=pb,labels=t['baseline_task']['bindings']['developer_labels'],offline_predictions=ob)
   if seed==2021:offline.append(dict(domain=domain,**ob,report=oe['report'],proof=oe['proof']))
   od=pathlib.Path(t['output_dir']);audit={}
   for jp in sorted((od/'durable/journal').glob('segment-*.jsonl')):
    for line in jp.open():
     event=json.loads(line)
     if event['kind']=='window_decision_audit':
      n=int(event['payload']['id'].split(':')[1]);assert n not in audit; audit[n]=event['payload']
   assert len(audit)==r['counts']['candidates'] and sum(x['decision']=='rollback' for x in audit.values())==r['counts']['rollbacks']
   generations=[]
   for gp in sorted((od/'durable/checkpoints').glob('generation-*')):
    mp=gp/'metadata.json';m=read(mp);assert sha(mp)==(gp/'metadata.sha256').read_text().strip()
    assert m['identity']['config_sha256']==entry['run_task']['sha256']
    generations.append(dict(directory=str(gp),metadata_sha256=sha(mp),state_sha256=m['state_sha256'],generation=m['generation'],cursors=m['scientific_cursors']))
   final=next(x for x in generations if x['generation']==r['final_generation']);assert final['state_sha256']==r['final_state_sha256']
   runs[key]=dict(run_key='P:'+key,run_id=rid,original_task=entry['run_task'],original_report=entry['report'],original_proof=entry['proof'],original_final_checkpoint=final,source_sha256=r['source_sha256'],protocol_sha256=r['protocol_sha256'])
   for n in sorted(yes+no):
    decision='commit' if n in yes else 'rollback';assert audit[n]['decision']==decision,(key,n)
    observed=audit[n]['counters']['observed']
    parents=[x for x in generations if x['cursors']['candidates']<n and x['cursors']['observed']<observed];assert parents
    parent=max(parents,key=lambda x:x['generation'])
    match=min((x for x in generations if x['cursors']['candidates']>=n and x['cursors']['observed']>=observed),key=lambda x:x['generation'])
    selected.append(dict(window_id=key+':'+str(n),domain=domain,stream_seed=seed,candidate_number=n,original_decision=decision,observed_at_decision=observed,original_task=entry['run_task'],parent_checkpoint=parent,match_checkpoint=match,original_final_checkpoint=final))
  assert all_candidates==793 and rollbacks==117 and len(selected)==48
  order=protocol['first_two']+[x['window_id'] for x in selected if x['window_id'] not in protocol['first_two']]
  frozen=dict(schema='g50p-risk-diagnostic-inputs-v1',diagnostic_id=PLAN.name,protocol=protocol_binding,producer=producer,windows=selected,execution_order=order,panels=panels,runs=runs,offline_predictions=offline,old_summary=bind(OLD/'summaries/round2/summary.json'),old_summary_proof=bind(OLD/'summaries/round2/proof.json'),candidate_population=793,rollback_population=117,labels_opened=False,new_model_forwards=0)
  fb=immutable(PLAN/'frozen_inputs.json',frozen)
  print(json.dumps(dict(status='frozen',binding=fb,panels={k:read(v['manifest']['path'])['eligible_count'] for k,v in panels.items()},windows=48,forward_count=0)))
def validate_index(index):
 frozen=checked(index['frozen_inputs'])
 assert frozen['schema']=='g50p-risk-diagnostic-inputs-v1'
 assert len(frozen['windows'])==48 and frozen['candidate_population']==793 and frozen['rollback_population']==117
 expected={(key,int(n),'commit' if n in yes else 'rollback') for key,(yes,no) in WINDOWS.items() for n in yes+no}
 actual={(x['domain']+':'+str(x['stream_seed']),x['candidate_number'],x['original_decision']) for x in frozen['windows']}
 assert actual==expected and len(actual)==48
 assert frozen['execution_order'][:2]==['lightbox:2021:1','sunlamp:2021:9']
 assert index['complete_ODR_runs']==0
 for i,row in enumerate(index['tasks']):
  task=checked(dict(path=row['task_path'],sha256=row['task_sha256']))
  sel=task['selection'];wid=sel['domain']+':'+str(sel['stream_seed'])+':'+str(sel['candidate_number'])
  assert wid==frozen['execution_order'][i]
  assert sel==next(x for x in frozen['windows'] if x['window_id']==wid)
  key=sel['domain']+':'+str(sel['stream_seed'])
  assert task['panel']==frozen['panels'][key] and task['producer']==frozen['producer']
  assert task['index_binding']==index['frozen_inputs'] and row['pilot']==(i<2)
 return True
if __name__=='__main__':main()
