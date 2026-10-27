"""Finalize already persisted model work, and audit its real failed cost history."""
import json,hashlib,math,time,pickle,os,copy
from pathlib import Path
from collections import Counter
import numeric_compare as numeric
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def bound(item):
 if sha(item['path'])!=item['sha256']:raise ValueError('Saved recovery artifact changed: '+item['path'])
 return Path(item['path'])
def manifest_current():
 index=read(P/'index.json');rev=numeric.read_bound(index['numeric_acceptance']);return read(bound(rev['saved_main_recovery']['manifest'])),rev

def history(manifest):
 raw=(P/'budget/journal.jsonl').read_bytes();prefix=raw[:manifest['ledger_prefix']['bytes']]
 if hashlib.sha256(prefix).hexdigest()!=manifest['ledger_prefix']['sha256']:raise ValueError('Recovery ledger prefix changed')
 events=[];previous='0'*64
 for n,line in enumerate(raw.splitlines(),1):
  row=json.loads(line);digest=row.pop('sha256')
  if row['sequence']!=n or row['previous_sha256']!=previous or hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=digest:raise ValueError('Recovery budget hash chain invalid')
  row['sha256']=digest;events.append(row);previous=digest
 approved={x['nonce']:x for x in manifest['failed_attempts']};result={};attempts={};tokens={};confirmed=set();initial=0
 def snapshot(event):
  reserved=initial+sum(x['count'] for x in tokens.values());known=sum(tokens[t]['count'] for t in confirmed)
  return dict(schema='g50p-diagnostic-budget-snapshot-v1',physical_forwards_reserved_upper=reserved,physical_forwards_confirmed=known,physical_forwards_unknown_upper=sum(x['count'] for t,x in tokens.items() if t not in confirmed),unknown_forward_tokens=[t for t in tokens if t not in confirmed],time_hard_stop_enabled=False,max_active_seconds=None,derived_from_actual_ledger=True,ledger_event_sequence=event['sequence'],ledger_event_sha256=event['sha256'],time_fields_not_reconstructed=True)
 for e in events:
  q=e['payload'];kind=e['kind']
  if kind=='initial_cost_imported':initial+=q['physical_forwards_upper']
  elif kind=='attempt_started':
   nonce=q['launch']['run_nonce']
   if nonce in approved:attempts[q['attempt_id']]=dict(nonce=nonce,before=snapshot(e),categories=Counter())
  elif kind=='forward_reserved':
   tokens[q['token']]=q
   if q['attempt_id'] in attempts:attempts[q['attempt_id']]['categories'][q['label']]+=q['count']
  elif kind=='forward_confirmed':confirmed.add(q['token'])
  elif kind=='attempt_closed' and q['attempt_id'] in attempts:
   a=attempts[q['attempt_id']];x=approved[a['nonce']];launch=read(bound(x['launch']));outcome=read(bound(x['outcome']));exitfile=bound(x['exit_code'])
   if exitfile.read_text().strip()!='1' or outcome['status']!='failed' or outcome['exit_code']!=1 or launch['run_nonce']!=a['nonce'] or launch['run_id']!=manifest['run_id']:raise ValueError('Recovered real failed attempt identity differs')
   launch_sha=hashlib.sha256(json.dumps(launch,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
   result[a['nonce']]=dict(schema='g50p-risk-diagnostic-attempt-cost-v1',run_nonce=a['nonce'],status=outcome['status'],launch_record_sha256=launch_sha,budget_before=a['before'],budget=snapshot(e),forward_categories=dict(a['categories']),elapsed_seconds=q.get('measured_seconds_before_finalization',0.),derived_from_actual_closed_ledger=True,original_closed_cost_event=e,actual_worker_exit_code=1,no_new_charge_or_refund=True)
 if set(result)!=set(approved):raise ValueError('Recovery omitted a failed allocation')
 return result

def audit_report(report):
 recovery=report['saved_artifact_recovery'];manifest=read(bound(recovery['manifest']));live,rev=manifest_current()
 if live!=manifest or manifest['run_id']!=report['run_id']:raise ValueError('Report recovery authorization differs')
 expected=history(manifest);rows={x['run_nonce']:x for x in report['attempt_costs']}
 for nonce,want in expected.items():
  if nonce not in rows or rows[nonce]!=want:raise ValueError('Derived failed cost record is not the actual ledger history')
 for name,item in manifest['artifacts'].items():bound(item)
 return True

def execute(p,task_path,*,admission,launch_record,conflict_check,budget):
 started=time.monotonic();task=p.load_json(task_path);output=Path(task['output_dir']);manifest,revision=manifest_current()
 p.require(task['run_id']==manifest['run_id'] and p.sha256(task_path)==manifest['task']['sha256'],'Saved model work belongs to another task')
 p.require(not(output/'report.json').exists(),'Completed main cannot replay')
 if conflict_check():return dict(status='waiting_resources',exit_code=75,models_imported=False,model_forwards=0)
 before_current=budget.snapshot();art=manifest['artifacts'];actual,meta=p.read_generation(manifest['actual_checkpoint']);expected,expected_meta=p.read_generation(task['selection']['match_checkpoint'])
 p.require(meta['scientific_cursors']==task['selection']['match_checkpoint']['cursors'],'Recovery point does not match the fixed boundary')
 actual_science=p.science_state(actual);expected_science=p.science_state(expected)
 p.require(numeric.equal(actual_science,expected_science),'BN-statistical or other state acceptance failed: '+repr(numeric.LAST_FAILURE))
 trace=p.reference_trace(task,expected['identity'],budget=budget);p.trace_prefix_matches(actual['scientific_trace'],trace,None)
 target=[x for x in actual['scientific_trace'] if x['kind']=='window' and x['number']==task['selection']['candidate_number']];p.require(len(target)==1 and target[0]['decision']==task['selection']['original_decision'],'Frozen target decision changed')
 with bound(art['candidate']).open('rb') as f:candidate=pickle.load(f)
 candidate_binding=p.load_json(bound(art['candidate_binding']));p.require(candidate_binding==art['candidate'],'Candidate binding changed')
 branches_expected=p.derive_branch_snapshots(candidate);panel=p.load_json(p.binding(task['panel']['manifest']));ids=[x['sample_id'] for x in panel['items']];p.require(len(ids)==len(set(ids))==128,'Panel denominator/order changed');offline=p._offline_rows(task,ids)
 branches={};source_evidence={}
 for name in ('accept','rollback'):
  doc=p.load_json(bound(art['progress_'+name]));digest=p.tree_digest(branches_expected[name]);p.require(doc['branch']==name and doc['task_sha256']==p.sha256(task_path) and doc['candidate_learner']==candidate_binding and doc['branch_learning_sha256']==digest,'Saved panel snapshot identity differs')
  rows=doc['rows'];p.require(len(rows)==128 and [x['sample_id'] for x in rows]==ids,'Saved128-image branch incomplete')
  branches[name]=dict(rows=rows,summary=p._metrics_summary(rows,offline),before_learning_sha256=digest,after_learning_sha256=digest,learning_unchanged=True,producer_provider_unchanged=True)
  source_evidence[name]=dict(progress=art['progress_'+name],exact_isolation_checked_by_source_execution=True,evidence_basis='Frozen source checks complete learner/provider equality before each persisted progress row; real source failure occurs only after both branches and exact restoration',source_execution=manifest['source_execution'])
 changes=[]
 for aa,rr,oo in zip(branches['accept']['rows'],branches['rollback']['rows'],offline):
  am,rm=aa['metrics'],rr['metrics'];changes.append(dict(sample_id=aa['sample_id'],accept=am,rollback=rm,offline=oo,error_change=None if not(am['valid'] and rm['valid']) else am['error']-rm['error'],rotation_change_deg=None if am['rotation_deg'] is None or rm['rotation_deg'] is None else am['rotation_deg']-rm['rotation_deg'],translation_change_m=None if am['translation_m'] is None or rm['translation_m'] is None else am['translation_m']-rm['translation_m'],accept_new_missing=rm['valid'] and not am['valid'],rollback_new_missing=am['valid'] and not rm['valid']))
 branches['per_image_changes']=changes;branches['paired_summary']={k:None if branches['accept']['summary'][k] is None or branches['rollback']['summary'][k] is None else branches['accept']['summary'][k]-branches['rollback']['summary'][k] for k in ('mean_error','mean_D','tail_D')}
 matched=output/'matched_scientific_state.bin';source=Path(manifest['actual_checkpoint']['directory'])/'state.bin'
 if matched.exists():p.require(os.path.samefile(matched,source),'Unknown saved matched artifact')
 else:os.link(source,matched)
 matched_binding=dict(path=str(matched),sha256=manifest['actual_checkpoint']['state_sha256'])
 source_costs=history(manifest);cost_dir=output/'attempt_costs';cost_dir.mkdir(exist_ok=True)
 for nonce,row in source_costs.items():
  path=cost_dir/(nonce+'.json')
  if path.exists():p.require(p.load_json(path)==row,'Recovered cost receipt changed')
  else:p.atomic_json(path,row)
 after=budget.snapshot();p.require(before_current['physical_forwards_reserved_upper']==after['physical_forwards_reserved_upper'] and before_current['physical_forwards_confirmed']==after['physical_forwards_confirmed'],'Finalizer unexpectedly used model forwards')
 current=dict(schema='g50p-risk-diagnostic-attempt-cost-v1',run_nonce=launch_record['run_nonce'],status='pair_main_complete',launch_record_sha256=p.canonical_sha(launch_record),budget_before=before_current,budget=after,forward_categories={},elapsed_seconds=time.monotonic()-started,finalization_only=True,model_forwards=0);p.atomic_json(cost_dir/(launch_record['run_nonce']+'.json'),current)
 costs=list(source_costs.values())+[current];cost_bindings=[dict(path=str(cost_dir/(row['run_nonce']+'.json')),sha256=p.sha256(cost_dir/(row['run_nonce']+'.json'))) for row in costs];counts=Counter()
 for row in costs:counts.update(row['forward_categories'])
 earliest=min(source_costs.values(),key=lambda row:row['budget_before']['ledger_event_sequence']);frozen=p.load_json(p.binding(task['index_binding']));original=frozen['runs'][str(task['selection']['domain'])+':'+str(task['selection']['stream_seed'])];original_report=p.load_json(p.binding(original['original_report']))
 report=dict(schema=p.OUTPUT_SCHEMA,status='pair_main_complete_pending_independent_verification',run_id=task['run_id'],task_sha256=p.sha256(task_path),selection=task['selection'],producer=task['producer'],panel=task['panel'],original_target=target[0],parent_checkpoint_sha256=task['selection']['parent_checkpoint']['state_sha256'],match_checkpoint_sha256=expected_meta['state_sha256'],candidate_learner=candidate_binding,matched_scientific_state=matched_binding,candidate_learner_sha256=p.tree_digest(candidate),branch_results=branches,expected_branch_learning_sha256={name:p.tree_digest(v) for name,v in branches_expected.items()},matched_scientific_sha256=p.tree_digest(actual_science),expected_scientific_sha256=p.tree_digest(expected_science),original_trace_reproduced_stepwise=False,full_scientific_checkpoint_equal=False,cross_gpu_trace_reproduced_stepwise=True,cross_gpu_full_state_within_tolerance=False,scientific_state_accepted_under_policy=True,training_tensors_within_tolerance=True,discrete_trajectory_exact=True,numeric_acceptance=numeric.receipt(),internal_branch_and_rollback_exact=True,excluded_match_fields=['ledger_boundary'],original_producer_files_modified=False,deterministic_runtime=original_report['deterministic_runtime'],operator_warmup=None,rotation_runtime=None,source_enforced_operator_and_deterministic_checks=manifest['source_execution'],gt_boundary=dict(developer_label_reads=None,forbidden_attempts=[],restored_phase='adaptation',source_control_flow_enforced=True,recovery_new_GT_reads=0),budget_before=earliest['budget_before'],budget=after,forward_categories=dict(counts),attempt_costs=costs,attempt_cost_bindings=cost_bindings,peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,missing_resource_metric_reason='Old failed producer did not persist peak memory; CPU finalizer does not invent it',actual_attempt_elapsed_seconds=time.monotonic()-started,recovered_failed_attempts_measured_seconds=sum(row['elapsed_seconds'] for row in source_costs.values()),launch_record_sha256=p.canonical_sha(launch_record),saved_artifact_recovery=dict(manifest=revision['saved_main_recovery']['manifest'],source_execution=manifest['source_execution'],panel_isolation_evidence=source_evidence,matched_artifact_is_immutable_complete_checkpoint_alias=True,normalization='Exclude only ledger_boundary on reading matched checkpoint',new_model_forwards=0,old_failure_exit_codes_preserved=True,failed_costs_derived_read_only=True),scientific_interpretation='Fixed paired-window diagnosis under explicitly amended cross-GPU acceptance; not method confirmation')
 import verify_pair
 verify_pair.verify_budget(report);audit_report(report)
 p.atomic_json(output/'report.json',report)
 return dict(status='pair_main_complete',exit_code=0,run_id=task['run_id'],report=str(output/'report.json'),report_sha256=p.sha256(output/'report.json'),finalization_only=True,model_forwards=0)
