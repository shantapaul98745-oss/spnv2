"""Build only the three-pilot append candidate; never publish production state."""
from pathlib import Path
import copy,hashlib,json,os,sys
from datetime import datetime,timezone,timedelta
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'pipeline'))
from runner import validate_manifest
from revision import extend_state,canonical_sha256
IDS=['pilot_table2_h','pilot_table2_e','pilot_table2_eh']
OLD=P/'queue_observed_stage_20261018.json'
PENDING=P/'queue_proposals/pending_source_tasks.json'
EXPECTED='a6f82b1efc62ebb6223e1cfec4919e44de35424b53709230f39b0655ea3e27a5'
def read(f): return json.loads(f.read_text())
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
def save(name,value):
    with (D/name).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def proc(pid):
    f=Path('/proc')/str(pid); stat=(f/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'starttime':stat[19],'state':stat[0],'command':(f/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
assert os.environ.get('STY'),'Independent screen required'
assert D==P/'queue_proposals/table2_pilots_20261022'
old,previous,pending=read(OLD),read(P/'status.json'),read(PENDING)
assert canonical_sha256(old)==EXPECTED and len(old['tasks'])==49
assert previous['status']=='phase_complete' and previous.get('active') is None
assert len(previous['completed'])==49 and all(t['status']=='accepted' for t in previous['tasks'])
manager=proc(previous['manager']['pid'])
assert manager['pid']==1851843 and manager['starttime']=='526573814'
assert str(OLD) in manager['command']
candidate=copy.deepcopy(old)
candidate['scope']='observed_stage49_plus_table2_three_pilots_only'
source_tasks={t['id']:t for t in pending['pending_task_definitions']}
dependency=old['tasks'][-1]['id']
assert dependency=='fixed_phi3_B2_sunlamp_s2025_N1024_B4'
changes=[]
for ident in IDS:
    task=copy.deepcopy(source_tasks[ident])
    old_requires=task['requires']
    task['requires']=[dependency]
    comparison=copy.deepcopy(task);comparison['requires']=old_requires
    assert comparison==source_tasks[ident]
    changes.append({'id':ident,'field':'requires','before':old_requires,'after':task['requires']})
    candidate['tasks'].append(task);dependency=ident
validate_manifest(old);validate_manifest(candidate)
amended,audit=extend_state(previous,old,candidate,caller_context={'operation':'offline_candidate_preparation_only','production_mutation':False,'screen':os.environ['STY'],'builder':str(Path(__file__).resolve())})
check=copy.deepcopy(amended);check['manifest_sha256']=previous['manifest_sha256']
assert check==previous and candidate['tasks'][:49]==old['tasks']
save('proposal.json',candidate)
save('baseline_status_snapshot.json',previous)
save('append_contract_audit.json',audit)
save('diff_audit.json',{'status':'prepared_pending_independent_cpu_verification','activation_allowed':False,
'old_manifest':str(OLD),'old_manifest_canonical_sha256':EXPECTED,'candidate':str(D/'proposal.json'),
'candidate_canonical_sha256':canonical_sha256(candidate),'old_tasks':49,'new_tasks':52,
'original49_exact_prefix':True,'existing_manifest_fields_changed':['scope','tasks'],
'backlog_preserved_exactly':candidate.get('backlog')==old.get('backlog'),
'added_ids':IDS,'pending_definition_changes':changes,'manager_identity':manager,
'source_pending_file_sha256':sha(PENDING),'pipeline_sha256':{str(P/'pipeline'/x):sha(P/'pipeline'/x) for x in ['runner.py','revision.py','progress.py','task_entry.py']},
'not_activated':True,'formal_table2_training_added':False})
save('build_receipt.json',{'status':'passed','at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
'screen':os.environ['STY'],'candidate_canonical_sha256':canonical_sha256(candidate),
'files_sha256':{x:sha(D/x) for x in ['proposal.json','baseline_status_snapshot.json','append_contract_audit.json','diff_audit.json']},
'production_write_performed':False,'gpu_experiment_started':False})
print(json.dumps({'status':'passed','candidate':str(D/'proposal.json'),'canonical_sha256':canonical_sha256(candidate),'tasks':52,'added':IDS}))
