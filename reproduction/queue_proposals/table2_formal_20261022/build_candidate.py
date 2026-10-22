# Build an inactive minimal Table2 formal append candidate, with no process control.
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy, hashlib, json, os, sys
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D=Path(__file__).resolve().parent
OLD=P/'queue_proposals/table2_pilots_20261022/proposal.json'
PENDING=P/'queue_proposals/pending_source_tasks.json'
EXPECTED='976cf1bf4caf473d47b861e01b69a216399da0e492b6bcb9fe282134365e5adc'
sys.path[:0]=[str(P/'pipeline'),str(P/'evaluation_entrypoints')]
from runner import validate_manifest
from revision import extend_state,canonical_sha256
from entry_contract import verify_protocol,source_paths
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(name,x):
 with (D/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert D==P/'queue_proposals/table2_formal_20261022'
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
old=read(OLD);previous=read(P/'status.json');pending=read(PENDING)
assert canonical_sha256(old)==EXPECTED==previous['manifest_sha256']
assert len(old['tasks'])==52 and previous['status']=='phase_complete' and previous.get('active') is None
assert len(previous['completed'])==52 and all(t['status']=='accepted' for t in previous['tasks'])
manager=previous['manager'];proc=Path('/proc')/str(manager['pid'])
stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
assert manager['pid']==3305820 and str(manager['starttime'])==stat[19]=='564320727'
assert stat[0] not in ('Z','X','T') and str(OLD).encode() in (proc/'cmdline').read_bytes()
protocol=verify_protocol()
E=P/'evaluation_entrypoints';X=P/'extended_evaluation'
observer={}
for mapping,root in [(protocol['wrapper_code_sha256'],E),(protocol['observer_code_sha256'],X),(protocol['base_entries_sha256'],P)]:
 for rel,digest in mapping.items():
  path=root/rel;assert sha(path)==digest,str(path);observer[str(path)]=digest
for path in [E/'implementation_manifest.json',E/'protocol_manifest.json',X/'preparation_manifest.json',P/'source_manifest.json']:
 observer[str(path)]=sha(path)
templates={t['id']:t for t in pending['pending_task_definitions']}
candidate=copy.deepcopy(old);candidate['scope']='observed_stage52_plus_table2_formal_H10_E20_EH20_three_domains'
changes=[];added=[]
pilots=['pilot_table2_h','pilot_table2_e','pilot_table2_eh']
for label in ['h','e','eh']:
 ids=['train_table2_'+label]+['eval_table2_'+label+'_'+d for d in ['synthetic','lightbox','sunlamp']]
 for ident in ids:
  t=copy.deepcopy(templates[ident]);before=copy.deepcopy(t)
  if ident=='train_table2_h':t['requires']=pilots
  elif ident=='train_table2_e':t['requires']=['pilot_table2_e','eval_table2_h_sunlamp']
  elif ident=='train_table2_eh':t['requires']=['pilot_table2_eh','eval_table2_e_sunlamp']
  if t['kind']=='full_source_evaluation':
   for field,mode in [('command','evaluate'),('verify_command','verify_eval')]:
    cmd=t[field];i=cmd.index(str(P/'tasks/source_artifacts.py'));assert cmd[i+1]==mode
    t[field]=cmd[:i]+[str(E/'capture_source.py'),'--base-entry']+cmd[i:]
   assets=t['metadata']['asset_sha256']
   for path,digest in observer.items():
    assert path not in assets or assets[path]==digest,path
    assets[path]=digest
   t['metadata']['pending_observer_entry']=str(E/'capture_source.py')
  delta={k:{'before':before.get(k),'after':t.get(k)} for k in set(before)|set(t) if before.get(k)!=t.get(k)}
  changes.append({'id':ident,'changes':delta})
  candidate['tasks'].append(t);added.append(ident)
assert len(candidate['tasks'])==64 and candidate['tasks'][:52]==old['tasks']
assert candidate['backlog']==old['backlog']
validate_manifest(old);validate_manifest(candidate)
amended,audit=extend_state(previous,old,candidate,caller_context={'operation':'offline_candidate_preparation_only','production_mutation':False,'screen':os.environ['STY'],'builder':str(Path(__file__).resolve())})
revert=copy.deepcopy(amended);revert['manifest_sha256']=previous['manifest_sha256'];assert revert==previous
save('proposal.json',candidate)
save('baseline_status_snapshot.json',previous)
save('append_contract_audit.json',audit)
save('diff_audit.json',{'status':'prepared_pending_independent_cpu_verification','activation_allowed':False,
 'old_manifest':str(OLD),'old_manifest_canonical_sha256':EXPECTED,'candidate':str(D/'proposal.json'),
 'candidate_canonical_sha256':canonical_sha256(candidate),'old_tasks':52,'new_tasks':64,
 'original52_exact_prefix':True,'existing_manifest_fields_changed':['scope','tasks'],
 'backlog_preserved_exactly':True,'added_ids':added,'template_changes':changes,
 'source_pending_file_sha256':sha(PENDING),'manager_identity':manager,
 'pipeline_sha256':{str(P/'pipeline'/x):sha(P/'pipeline'/x) for x in ['runner.py','revision.py','progress.py','task_entry.py']},
 'observer_assets':observer,'not_activated':True,'formal_epochs_added':50,'evaluations_added':9,
 'single_head_contract':{'table2_h':{'heads':['heatmap'],'extended_primary_mode':'H_rotation_H_translation','absent_heads':['efficientpose'],'no_entropy_odr':True},
 'table2_e':{'heads':['efficientpose'],'extended_primary_mode':'E_only_original','extension_applicability':'not_applicable_no_heatmap','absent_heads':['heatmap'],'no_entropy_odr':True},
 'table2_eh':{'heads':['heatmap','efficientpose'],'extended_primary_mode':'H_rotation_E_translation','no_entropy_odr':True}},
 'denominator_policy':'Paper Table2 primary uses original accepted head-specific evidence; extension separate, unavailable protocols metrics=None, no missing-H zero-fill or E substitution, all_samples only on actual N/N.'})
save('build_receipt.json',{'status':'passed','at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
 'screen':os.environ['STY'],'candidate_canonical_sha256':canonical_sha256(candidate),
 'files_sha256':{x:sha(D/x) for x in ['proposal.json','baseline_status_snapshot.json','append_contract_audit.json','diff_audit.json']},
 'production_write_performed':False,'gpu_experiment_started':False})
print(json.dumps({'status':'passed','candidate':str(D/'proposal.json'),'canonical_sha256':canonical_sha256(candidate),'tasks':64,'added':added}))
