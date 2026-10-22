# Independent bounded CPU candidate acceptance; no GPU inference or state publication.
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast,copy,csv,hashlib,json,os,sys
import yaml
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'pipeline'),str(P/'evaluation_entrypoints')]
from runner import validate_manifest,initialize_state
from revision import extend_state,canonical_sha256
from entry_contract import verify_protocol,source_paths,source_spec,extension_api
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(name,x):
 with (D/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cfg(p):return yaml.safe_load(p.read_text())
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
assert D==P/'queue_proposals/table2_formal_20261022'
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
receipt,diff=read(D/'build_receipt.json'),read(D/'diff_audit.json')
assert os.environ['STY']!=receipt['screen']
assert (D/'build.log.exitcode').read_text().strip()=='0'
for path,want in receipt['files_sha256'].items():assert sha(D/path)==want,path
for path,want in diff['pipeline_sha256'].items():assert sha(Path(path))==want,path
assert sha(P/'queue_proposals/pending_source_tasks.json')==diff['source_pending_file_sha256']
old=read(Path(diff['old_manifest']));new=read(D/'proposal.json')
live=read(P/'status.json');baseline=read(D/'baseline_status_snapshot.json')
assert len(old['tasks'])==52 and len(new['tasks'])==64
assert new['tasks'][:52]==old['tasks']
assert canonical_sha256(old)==live['manifest_sha256']==diff['old_manifest_canonical_sha256']
assert canonical_sha256(new)==receipt['candidate_canonical_sha256']==diff['candidate_canonical_sha256']
assert {k:v for k,v in old.items() if k not in ('scope','tasks')}=={k:v for k,v in new.items() if k not in ('scope','tasks')}
assert live['status']=='phase_complete' and live.get('active') is None
assert len(live['completed'])==52 and all(t['status']=='accepted' for t in live['tasks'])
assert baseline['tasks']==live['tasks'] and baseline['completed']==live['completed']
assert baseline['manager']==live['manager']
m=live['manager'];pr=Path('/proc')/str(m['pid']);st=(pr/'stat').read_text().rsplit(')',1)[1].split()
assert m['pid']==3305820 and str(m['starttime'])==st[19]=='564320727'
assert st[0] not in ('Z','X','T') and diff['old_manifest'].encode() in (pr/'cmdline').read_bytes()
validate_manifest(old);validate_manifest(new)
protocol=verify_protocol();E=P/'evaluation_entrypoints';X=P/'extended_evaluation'
observer={}
for mapping,root in [(protocol['wrapper_code_sha256'],E),(protocol['observer_code_sha256'],X),(protocol['base_entries_sha256'],P)]:
 for rel,digest in mapping.items():observer[str(root/rel)]=digest
for path in [E/'implementation_manifest.json',E/'protocol_manifest.json',X/'preparation_manifest.json',P/'source_manifest.json']:
 observer[str(path)]=sha(path)
assert observer==diff['observer_assets']
templates={t['id']:t for t in read(P/'queue_proposals/pending_source_tasks.json')['pending_task_definitions']}
reference_heads={'h':['heatmap'],'e':['efficientpose'],'eh':['heatmap','efficientpose']}
reference_modes={'h':'H_rotation_H_translation','e':'E_only_original','eh':'H_rotation_E_translation'}
added_ids=[];per_task=[];assets={};branch_contracts=[];csv_counts={}
pilot_ids=['pilot_table2_h','pilot_table2_e','pilot_table2_eh']
expected_dep={'train_table2_h':pilot_ids,'train_table2_e':['pilot_table2_e','eval_table2_h_sunlamp'],'train_table2_eh':['pilot_table2_eh','eval_table2_e_sunlamp']}
for label,epochs in [('h',10),('e',20),('eh',20)]:
 train_cfg=cfg(P/'configs'/('table2_'+label+'.yaml'));tr=train_cfg['TRAIN']
 assert tr['END_EPOCH']==epochs and tr['BEGIN_EPOCH']==0 and tr['IMAGES_PER_GPU']==4 and tr['ACCUM_STEPS']==4
 assert tr['OPTIMIZER']=='AdamW' and tr['LR']==0.001 and tr['WD']==0.0001 and tr['LR_STEP']==[15,18]
 assert train_cfg['SEED']==2021 and train_cfg['FP16'] is True and train_cfg['AUTO_RESUME'] is False and tr['RESUME_FILE']==''
 assert train_cfg['DATASET']['INPUT_SIZE']==[768,512]
 assert train_cfg['MODEL']['EFFICIENTDET_PHI']==3 and train_cfg['MODEL']['USE_GROUPNORM_BACKBONE'] is False
 assert train_cfg['MODEL']['HEAD']['NAMES']==reference_heads[label] and 'segmentation' not in train_cfg['MODEL']['HEAD']['NAMES']
 assert train_cfg['AUGMENT']==cfg(P/'configs/phi3_bn_ehs.yaml')['AUGMENT']
 out=Path(train_cfg['OUTPUT_DIR'])/train_cfg['MODEL']['BACKBONE']['NAME']/train_cfg['EXP_NAME']
 trainlog=Path(train_cfg['LOG_DIR'])/train_cfg['MODEL']['BACKBONE']['NAME']/train_cfg['EXP_NAME']
 assert not out.exists() and not trainlog.exists(),(out,trainlog)
 ids=['train_table2_'+label]+['eval_table2_'+label+'_'+d for d in ['synthetic','lightbox','sunlamp']]
 for ident in ids:
  index=52+len(added_ids);t=new['tasks'][index];ref=copy.deepcopy(templates[ident]);assert t['id']==ident
  added_ids.append(ident)
  if ident in expected_dep:ref['requires']=expected_dep[ident]
  absent=[t['log'],t['log']+'.exitcode',t['log']+'.verify',t['log']+'.verify.exitcode']+t['expected_outputs']
  if ident.startswith('train_'):
   assert t['metadata']['independently_initialized'] is True and t['metadata']['pilot_weights_for_formal'] is False
   assert t['metadata']['epochs']==epochs and t['metadata']['physical_batch']==4 and t['metadata']['accum_steps']==4
   assert '--resume' not in t['command'] and 'gpu_smoke.py' not in ' '.join(t['command'])
   absent.extend([str(out),str(trainlog)])
  else:
   for field,mode in [('command','evaluate'),('verify_command','verify_eval')]:
    cmd=ref[field];i=cmd.index(str(P/'tasks/source_artifacts.py'));assert cmd[i+1]==mode
    ref[field]=cmd[:i]+[str(E/'capture_source.py'),'--base-entry']+cmd[i:]
    parsed=source_spec(cmd[i+1:]);assert parsed.mode==mode
   ref['metadata']['asset_sha256'].update(observer)
   ref['metadata']['pending_observer_entry']=str(E/'capture_source.py')
   cmd=templates[ident]['command'];i=cmd.index(str(P/'tasks/source_artifacts.py'));a=source_spec(cmd[i+1:])
   ec=cfg(Path(a.cfg));domain=t['metadata']['domain'];n={'synthetic':11994,'lightbox':6740,'sunlamp':2791}[domain]
   assert a.expected_samples==n==t['metadata']['samples'] and t['metadata']['final_only'] is True
   assert ec['TEST']['HEAD']==reference_heads[label]==ec['MODEL']['HEAD']['NAMES']
   assert ec['TEST']['MODEL_FILE']==str(out/'model_final.pth.tar')
   assert a.source_report==str(P/'reports'/('source_table2_'+label+'_verified.json'))
   assert ec['TEST']['IMAGES_PER_GPU']==1
   expected_csv='synthetic/labels/validation.csv' if domain=='synthetic' else domain+'/labels/test.csv'
   assert ec['TEST']['TEST_CSV']==expected_csv
   csv_path=Path(ec['DATASET']['ROOT'])/ec['DATASET']['DATANAME']/ec['TEST']['TEST_CSV']
   if str(csv_path) not in csv_counts:
    with csv_path.open(newline='') as f:count=sum(1 for row in csv.reader(f) if row)
    csv_counts[str(csv_path)]=count
   assert csv_counts[str(csv_path)]==n,(csv_path,csv_counts[str(csv_path)],n)
   paths=source_paths(a.report)
   absent.extend([a.output,a.report,str(paths['root']),str(paths['wrapper_proof'])])
   branch_contracts.append({'id':ident,'heads':reference_heads[label],'source_final':ec['TEST']['MODEL_FILE'],'expected_samples':n,
    'extension_applicability':'applicable' if label!='e' else 'not_applicable_no_heatmap',
    'extended_primary_mode':reference_modes[label],'original_paper_table2_head_specific_accepted_evidence_preserved':True,
    'missing_head_metrics':None,'actual_numeric_coverage_required_for_all_samples':True,'no_extra_forward':True,'no_entropy_odr':True})
  assert t==ref,ident
  for path,want in t['metadata']['asset_sha256'].items():
   if path in assets:assert assets[path]['expected']==want
   else:
    actual=sha(Path(path));assert actual==want,path;assets[path]={'expected':want,'actual':actual}
  assert all(not Path(x).exists() for x in absent),[x for x in absent if Path(x).exists()]
  assert t['gpu']==templates[ident]['gpu'] and t['verify_gpu'] is False
  per_task.append({'id':ident,'requires':t['requires'],'assets':len(t['metadata']['asset_sha256']),'absent_outputs':absent})
assert added_ids==diff['added_ids'] and len(added_ids)==12
# Check branch selection expressions from the unchanged accepted producer and independent verifier.
for rel,function in [('extended_evaluation/report.py','calculate'),('extended_evaluation/verify.py','recompute')]:
 tree=ast.parse((P/rel).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
 assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mode' for t in n.targets))
 for label in reference_heads:
  scope={'h':'heatmap' in reference_heads[label],'e':'efficientpose' in reference_heads[label]}
  assert eval(compile(ast.Expression(assignment.value),rel,'eval'),{'__builtins__':{}},scope)==reference_modes[label]
 text=(P/rel).read_text()
 assert 'metrics=None' in text or '"metrics": None' in text
 assert 'count else None' in text
_,report,_=extension_api()
assert report.statistics([None,None],2,mask_name='candidate_only')['mean'] is None
assert report.statistics([1.0,None],2,mask_name='candidate_only')['denominator_scope']=='numeric_pose_available_subset'
assert report.statistics([1.0,2.0],2,mask_name='candidate_only')['denominator_scope']=='all_samples'
assert report.statistics([1.0,2.0],2,mask_name='candidate_only',original=True)['denominator_scope']=='original_accepted_subset'
before=canonical_sha256(live)
amended,audit=extend_state(live,old,new,caller_context={'operation':'independent_cpu_candidate_acceptance_only','screen':os.environ['STY']})
assert canonical_sha256(live)==before
revert=copy.deepcopy(amended);revert['manifest_sha256']=live['manifest_sha256'];assert revert==live
sim=initialize_state(amended,new);assert sim['completed']==live['completed']
for a,b in zip(live['tasks'],sim['tasks'][:52]):
 for key in ['id','status','runtime','history','error','accepted_at','started_at']:
  assert a.get(key)==b.get(key),(a['id'],key)
assert all(t['status']=='queued' and not t.get('runtime') for t in sim['tasks'][52:])
save('independent_append_contract_audit.json',audit)
save('new_asset_verification.json',{'status':'passed','unique_assets':len(assets),'assets':assets,'per_task':per_task,'csv_rows':csv_counts})
save('evaluation_branch_contract.json',{'status':'passed','scope':'Static candidate configuration and accepted-code contract, not scientific accuracy or new inference','tasks':branch_contracts,
 'checks':['nine_original_argument_vectors_preserved','same_forward_wrapper_protocol_pins','single_head_mode_expressions_match_producer_and_independent_verifier','missing_protocol_metrics_None','empty_mean_None','actual_finite_count_controls_all_samples','original_accepted_subset_preserved']})
save('activation_review.json',{'status':'reviewable_not_authorized_to_activate','candidate':str(D/'proposal.json'),'candidate_canonical_sha256':canonical_sha256(new),
 'old_manifest':str(Path(diff['old_manifest'])),'old_manifest_canonical_sha256':canonical_sha256(old),
 'manager':m,'first_new_task':'train_table2_h','new_task_ids':added_ids,
 'required_actions_before_activation':['Obtain a new explicit one-time manager-only switch authorization; previous pilot switch authorization is consumed.',
 'Recheck current 52-task manifest, all 52 accepted task records and no active worker; exact manager PID/starttime/cmd and exclusive pipeline.lock ownership.',
 'Recheck this candidate/acceptance and all new assets and output absence; preserve source configs, frozen source, existing 52 definitions and execution evidence.',
 'Use the existing guarded manager-only graceful cutover under the same lock, durable revision and GPU resource gate. No direct state or active manifest mutation.',
 'Start replacement manager with --manifest candidate plus the one-time --extend-from current52 only for this authorized cutover; verify lock/new identity and only train_table2_h newly starts.',
 'For later ordinary fault recovery use only --manifest candidate, never repeat extension flags or the one-time cutover controller.'],
 'production_write_performed':False,'gpu_experiment_started':False,'activation_allowed':False})
save('verified.json',{'status':'passed','at':now(),'screen':os.environ['STY'],
 'candidate':str(D/'proposal.json'),'candidate_canonical_sha256':canonical_sha256(new),'candidate_file_sha256':sha(D/'proposal.json'),
 'old_manifest_canonical_sha256':canonical_sha256(old),'old_tasks_preserved':52,'new_queued_tasks':12,'formal_source_epochs_added':50,'new_domain_evaluations':9,'unique_new_assets_checked':len(assets),
 'checks':['build_receipt_bytes','original52_exact_prefix','all52_execution_evidence_preserved_in_pure_append_simulation','three_train_configs_frozen_4x4_independent_final_H10_E20_EH20','all_three_pilots_precede_H','E_EH_wait_previous_full_source_acceptance','new12_only_allowed_template_changes','new_assets_actual_sha256','new_outputs_absent','evaluation_branch_contract','no_gpu_or_production_mutation'],
 'files_sha256':{x:sha(D/x) for x in ['build_candidate.py','verify_candidate.py','build_receipt.json','diff_audit.json','new_asset_verification.json','independent_append_contract_audit.json','evaluation_branch_contract.json','activation_review.json']},
 'activation_allowed':False,'new_manager_only_authorization_required':True,'production_write_performed':False,'gpu_experiment_started':False})
print(json.dumps({'status':'passed','candidate_canonical_sha256':canonical_sha256(new),'new_assets':len(assets),'preserved':52,'new_queued':12,'formal_epochs':50,'evals':9,'activation_allowed':False}))
