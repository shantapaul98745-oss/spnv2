"""Independently inspect the candidate via existing pure append contracts."""
from pathlib import Path
import copy,hashlib,json,os,sys
from datetime import datetime,timezone,timedelta
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'pipeline'))
from runner import validate_manifest,initialize_state,asset_hash_error
from revision import extend_state,canonical_sha256
def read(f): return json.loads(f.read_text())
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    with (D/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert os.environ.get('STY'),'Independent screen required'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only verification required'
assert D==P/'queue_proposals/table2_pilots_20261022'
receipt,diff=read(D/'build_receipt.json'),read(D/'diff_audit.json')
assert os.environ['STY']!=receipt['screen']
for name,expected in receipt['files_sha256'].items():assert sha(D/name)==expected,name
assert sha(P/'queue_proposals/pending_source_tasks.json')==diff['source_pending_file_sha256']
for name,expected in diff['pipeline_sha256'].items():assert sha(Path(name))==expected,name
old=read(Path(diff['old_manifest']));new=read(D/'proposal.json')
snapshot=read(D/'baseline_status_snapshot.json');live=read(P/'status.json')
pending={t['id']:t for t in read(P/'queue_proposals/pending_source_tasks.json')['pending_task_definitions']}
assert len(old['tasks'])==49 and len(new['tasks'])==52 and old['tasks']==new['tasks'][:49]
assert canonical_sha256(old)==diff['old_manifest_canonical_sha256']==live['manifest_sha256']
assert canonical_sha256(new)==diff['candidate_canonical_sha256']==receipt['candidate_canonical_sha256']
assert live['status']=='phase_complete' and live.get('active') is None
assert len(live['completed'])==49 and all(t['status']=='accepted' for t in live['tasks'])
assert snapshot['tasks']==live['tasks'] and snapshot['completed']==live['completed']
assert snapshot['manager']==live['manager']
m=live['manager'];proc=Path('/proc')/str(m['pid']);st=(proc/'stat').read_text().rsplit(')',1)[1].split()
assert m['pid']==1851843 and st[19]==str(m['starttime'])=='526573814'
assert st[0] not in ['T','Z','X']
assert str(diff['old_manifest']).encode() in (proc/'cmdline').read_bytes()
validate_manifest(old);validate_manifest(new)
assets={};per_task=[];previous=old['tasks'][-1]['id']
for ident,t in zip(['pilot_table2_h','pilot_table2_e','pilot_table2_eh'],new['tasks'][49:]):
    assert t['id']==ident and t['requires']==[previous]
    reference=copy.deepcopy(pending[ident]);reference['requires']=[previous]
    assert t==reference,ident
    assert t['metadata']['pilot_weights_for_formal'] is False and t['metadata']['attempts']==32 and t['metadata']['checkpoint_attempt']==16
    assert t['metadata']['physical_batch']==4 and t['metadata']['accum_steps']==4 and t['metadata']['engineering_subset_rows']==512
    assert asset_hash_error(t) is None,ident
    absent=[t['log'],t['log']+'.exitcode',t['log']+'.verify',t['log']+'.verify.exitcode',t['metadata']['report'],t['metadata']['artifacts']]+t['expected_outputs']
    assert all(not Path(x).exists() for x in absent),ident
    for name,expected in t['metadata']['asset_sha256'].items():
        actual=sha(Path(name));assert actual==expected,name
        assets[name]={'expected':expected,'actual':actual}
    per_task.append({'id':ident,'assets':len(t['metadata']['asset_sha256']),'absent_outputs':absent,'requires':t['requires']})
    previous=ident
before=canonical_sha256(live)
amended,audit=extend_state(live,old,new,caller_context={'operation':'independent_cpu_candidate_acceptance_only','screen':os.environ['STY']})
assert canonical_sha256(live)==before
revert=copy.deepcopy(amended);revert['manifest_sha256']=live['manifest_sha256'];assert revert==live
simulated=initialize_state(amended,new)
assert simulated['completed']==live['completed']
for before_task,after_task in zip(live['tasks'],simulated['tasks'][:49]):
    for key in ['id','status','runtime','history','error','accepted_at','started_at']:
        assert before_task.get(key)==after_task.get(key),(before_task['id'],key)
assert all(t['status']=='queued' and not t.get('runtime') for t in simulated['tasks'][49:])
save('independent_append_contract_audit.json',audit)
save('new_pilot_asset_verification.json',{'status':'passed','unique_assets':len(assets),'assets':assets,'per_task':per_task})
save('verified.json',{'status':'passed','at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),'screen':os.environ['STY'],
'candidate':str(D/'proposal.json'),'candidate_canonical_sha256':canonical_sha256(new),
'candidate_file_sha256':sha(D/'proposal.json'),'old_manifest_canonical_sha256':canonical_sha256(old),
'old_tasks_preserved':49,'new_queued_pilots':3,'unique_new_assets_checked':len(assets),
'checks':['candidate_bytes_match_build_receipt','original49_exact_prefix','new3_only_dependency_changes','existing_manifest_validator','existing_pure_append_contract','simulation_preserves49_execution_evidence','only3_new_pilots_queued_without_runtime','current_original_manager_identity_and_phase_complete','new3_assets_actual_sha256','new3_outputs_still_absent'],
'files_sha256':{x:sha(D/x) for x in ['build_candidate.py','verify_candidate.py','build_receipt.json','diff_audit.json','new_pilot_asset_verification.json','independent_append_contract_audit.json']},
'activation_allowed':False,'new_manager_only_authorization_required':True,'production_write_performed':False,'gpu_experiment_started':False})
print(json.dumps({'status':'passed','candidate_canonical_sha256':canonical_sha256(new),'new_assets':len(assets),'preserved':49,'new_queued':3,'activation_allowed':False}))
