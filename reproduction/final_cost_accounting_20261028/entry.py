"""Candidate zero-forward settlement/publisher. No action without --execute.

Run inside the existing SYSU namespace with its current index/config and an
owned screen, SPNV2_RUN_NONCE and --log-path matching the real screen wrapper.
Request JSON binds this entry, core.py, current index, and action inputs.
"""
from pathlib import Path
from contextlib import contextmanager,ExitStack
import argparse,copy,hashlib,importlib.util,json,os,sys,time
ROOT=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
P=ROOT/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
M=ROOT/'spnv2_runs/migrations/sysu_20261027'
SOURCE_SHA='f1dcd1b56790aa9bc8856d539c4dd90e03a6f877546f7e1849aff6be05ce323a'
IDENTITY_SHA='05e91d7414e9f753d4b700a782f09f6ed6675f8e0b1b7cc11bb93f1133194857'
core=None
EXECUTION=None

def sha(path,limit=None):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  left=limit
  while left is None or left:
   block=f.read((1<<20) if left is None else min(left,1<<20))
   if not block:break
   h.update(block)
   if left is not None:left-=len(block)
 if left is not None and left:raise ValueError('Requested journal prefix is unavailable')
 return h.hexdigest()
def allowed(path):
 p=Path(path).resolve(strict=True)
 if not(p.is_relative_to(P) or p.is_relative_to(M)) or any(v in ('transfer_keys','.ssh') for v in p.parts):raise ValueError('Metadata evidence escapes this own plan/migration or enters key storage')
 if not p.is_file() or p.stat().st_size>8<<20 or p.suffix not in ('.json','.exitcode','.log','.events','.txt','.md'):raise ValueError('Only bounded metadata/exit evidence is allowed, never a model/key')
 return p
def checked(spec,*,raw=False):
 if not core.is_binding(spec):raise ValueError('Exact immutable metadata binding required')
 p=allowed(spec['path'])
 if sha(p)!=spec['sha256']:raise ValueError('Metadata evidence hash changed: '+str(p))
 data=p.read_text();return data if raw else json.loads(data)
def bind(path):return dict(path=str(path),sha256=sha(path))
def immutable(path,value):
 from budget import atomic_json
 path=Path(path)
 if path.exists():
  if json.loads(path.read_text())!=value:raise ValueError('Refusing to rewrite existing accounting evidence: '+str(path))
 else:atomic_json(path,value)
 return bind(path)
def load_core(spec=None):
 global core
 path=Path(__file__).with_name('core.py').resolve()
 if spec is not None and (Path(spec['path']).resolve()!=path or sha(path)!=spec['sha256']):raise ValueError('Accounting core code binding changed')
 loader=importlib.util.spec_from_file_location('g50p_final_cost_core',path);core=importlib.util.module_from_spec(loader);loader.loader.exec_module(core);return core

def original_prefix():
 source=M/'source_handoff_snapshot_20261027/budget/journal.jsonl';length=source.stat().st_size
 if sha(source)!=SOURCE_SHA or sha(P/'budget/journal.jsonl',length)!=SOURCE_SHA or sha(P/'budget/identity.json')!=IDENTITY_SHA:raise ValueError('Original source prefix or ledger identity changed')
 return dict(bytes=length,sha256=SOURCE_SHA,identity_sha256=IDENTITY_SHA)

def transfer_closed(manifest):
 closure=manifest['transfer_closure'];source=checked(closure['source_sender_status']);target=checked(closure['target_receiver_status']);exit_code=int(checked(closure['source_sender_exit'],raw=True).strip())
 for item in (source,target):
  core.require(item.get('phase')=='complete' and item.get('verified_files')==item.get('expected_files')==276 and item.get('verified_raw_bytes')==item.get('expected_raw_bytes')==98724602862 and item.get('model_forwards')==0,'Source/target transfer is not actually fully verified and closed')
 core.require(exit_code==0 and target.get('error') is None,'Real final source/receiver session exit was not successful')
 for item in manifest['items']:
  identities=[]
  for evidence in item['evidence']:
   text=checked(evidence,raw=True)
   if Path(evidence['path']).suffix=='.json':
    doc=json.loads(text);candidate=doc.get('identity',doc) if isinstance(doc,dict) else None
    if isinstance(candidate,dict) and all(k in candidate for k in ('pid','starttime','boot_id','cmdline_sha256')):
     normalized={k:candidate[k] for k in ('pid','starttime','boot_id','cmdline_sha256')};normalized['run_nonce']=candidate.get('run_nonce',candidate.get('nonce'));identities.append(normalized)
  if item['kind']=='process':
   core.require(any(all(item['identity'].get(k)==observed.get(k) for k in ('pid','starttime','boot_id','cmdline_sha256','run_nonce')) for observed in identities),'Historical external process identity has no matching immutable real status/launch evidence')
   core.require(int(checked(item['terminal_exit'],raw=True).strip())==item['actual_exit_code'],'External process actual exit differs')

def paid_identity_conflicts(rows,info):
 for item in info['new_items']:
  key=core.process_key(item)
  if key is None:continue
  for row in rows:
   if row['kind']!='attempt_started':continue
   launch=row['payload']['launch']
   if (launch['boot_id'],launch['pid'],launch['starttime'],launch['cmdline_sha256'],launch['run_nonce'])==key[1:]:raise ValueError('A newly charged external process already has an original-ledger attempt; split/prove nonoverlap explicitly instead of charging twice')

def _reconcile_failure(ctx):
 """While original locks are held, close only this returned own attempt."""
 budget=ctx.get('budget');ctx.setdefault('accounting',{})
 if budget is None:return
 try:
  budget._reload();nonce=ctx['nonce'];starts=[r for r in budget.rows if r['kind']=='attempt_started' and r['payload']['launch']['run_nonce']==nonce]
  if len(starts)>1:raise ValueError('Own nonce has multiple ledger allocations')
  if not starts:return
  aid=starts[0]['payload']['attempt_id'];ctx['accounting'].update(attempt_id=aid,started_monotonic=starts[0]['payload']['started_monotonic'])
  closes=[r for r in budget.rows if r['kind']=='attempt_closed' and r['payload']['attempt_id']==aid]
  attempt=ctx.get('attempt')
  if not closes and attempt is not None and not attempt.closed:
   attempt.close('final_cost_metadata_failed');budget._reload();closes=[r for r in budget.rows if r['kind']=='attempt_closed' and r['payload']['attempt_id']==aid]
  if len(closes)>1:raise ValueError('Own actual attempt was closed more than once')
  if closes:
   close=closes[0];observations=[r['payload']['charged_seconds_upper'] for r in budget.rows if r['kind']=='time_finalization_observed' and r['payload']['attempt_id']==aid]
   ctx['accounting'].update(closed=True,close_event_sequence=close['sequence'],close_event_sha256=close['sha256'],charged_seconds_upper=max([close['payload']['charged_seconds_upper']]+observations),external_debit_seconds_upper=close['payload'].get('retrospective_external_debit_seconds_upper',0.),external_cost_settlement_id=close['payload'].get('external_cost_settlement_id'))
   if attempt is not None:attempt.closed=True;budget._attempt=None
  else:ctx['accounting'].update(closed=False,allocation_unresolved=True)
 except Exception as error:ctx['accounting']['reconciliation_error']=repr(error)

@contextmanager
def admitted(request,run_id,log_path,started):
 global EXECUTION
 if not P.is_dir():raise ValueError('Use the approved SYSU namespace, not a new plan root')
 nonce=os.environ.get('SPNV2_RUN_NONCE','');core.require(len(nonce)==32 and all(x in '0123456789abcdef' for x in nonce) and bool(os.environ.get('STY')),'Unique owned screen and nonce required')
 log_path=Path(log_path);core.require(log_path.is_absolute() and log_path.resolve().is_relative_to(P) and log_path.suffix=='.log','Actual wrapper log must be a named project log')
 directory=P/'launches'/nonce;directory.mkdir(exist_ok=False);ctx=dict(directory=directory,directory_created_here=True,attempt=None,budget=None,started=started,nonce=nonce,log_path=log_path,stage='index_and_code_validation',accounting={});EXECUTION=ctx
 index=checked(request['index']);core.require(request['index']['path']==str(P/'index.json'),'Wrong current index')
 for name in ('host_entry.py','budget.py'):core.require(sha(P/'implementation'/name)==index['implementation_sha256'][name],'Canonical accounting/admission bytes changed')
 sys.path.insert(0,str(P/'implementation'))
 from host_entry import host_modules,project_lock,launch_record,conflict,read,PROJECT
 from budget import GlobalBudget,atomic_json
 with ExitStack() as stack:
  try:
   ctx['stage']='original_locks'
   for name in ('rebaseline_20261017','odr003_exploratory_20261023','odr003_round2_20261023','g50p_risk_diagnostic_20261025'):stack.enter_context(project_lock(P.parent/name/'pipeline.lock'))
   stack.enter_context(project_lock(P/'manager.lock'));core.require(sha(P/'index.json')==request['index']['sha256'],'Index changed before original locks')
   ctx['stage']='scientific_quiescence';state=read(P/'status.json');core.require(state.get('active') is None and state.get('status') in ('scientific_complete_costs_pending','partial_delivery_required'),'Scientific dispatch is not quiescent at its real delivery boundary')
   ctx['stage']='resource_admission';gate,collector=host_modules(index);registry=P/'launches/registry.json';before=collector.collect_snapshot(registry,project_root=P,project_id=PROJECT);after=collector.collect_snapshot(registry,project_root=P,project_id=PROJECT);decision=gate.assess_launch_boundary(before,after,now=time.time(),project_lock_held=True,task_kind='heavy_cpu');from dataclasses import asdict
   atomic_json(directory/'admission.json',dict(before=before,after=after,decision=asdict(decision)))
   if not decision.allow_start:ctx['resource_wait_reasons']=list(decision.reasons);raise SystemExit(75)
   ctx['stage']='registry_readback';record=launch_record(dict(run_id=run_id),str(log_path),gate);record['diagnostic_code_sha256']=sha(__file__);atomic_json(directory/'launch.json',record);ctx['record']=record;reg=read(registry);reg['records'].append(dict(record_path=str(directory/'launch.json'),record_sha256=gate.launch_record_sha256(record),expected_record=record));atomic_json(registry,reg)
   core.require(not conflict(collector.collect_snapshot(registry,project_root=P,project_id=PROJECT),record,gate),'Own precise registration was not read back')
   ctx['stage']='source_prefix_validation';prefix=original_prefix()
   ctx['stage']='budget_constructor';budget=GlobalBudget(P/'budget',dict(project_id=PROJECT,frozen_inputs_sha256=index['frozen_inputs']['sha256']),initial_cost_receipt=index['initial_cost_receipt'],budget_policy=index['budget_policy']);ctx['budget']=budget;core.safe_snapshot(budget.snapshot());ctx.update(index=index,state=state,prefix=prefix,journal_before=dict(bytes=(P/'budget/journal.jsonl').stat().st_size,sha256=sha(P/'budget/journal.jsonl')),stage='bound_action_inputs')
   yield ctx
  except BaseException:
   _reconcile_failure(ctx);raise

def begin(ctx):
 ctx['stage']='attempt_start_requested';ctx['attempt_start_requested']=True;ctx['pre_budget_wall']=time.monotonic()-ctx['started'];ctx['attempt']=ctx['budget'].begin_attempt(ctx['record'],allow_model_forwards=False);ctx['accounting']=dict(attempt_id=ctx['attempt'].attempt_id,closed=False,started_monotonic=ctx['attempt'].record['started_monotonic']);ctx['before']=ctx['budget'].snapshot();ctx['stage']='metadata_action_running'

def close(ctx,*,external_upper=0.,info=None,manifest_binding=None):
 budget=ctx['budget'];attempt=ctx['attempt'];core.require(not attempt.closed and not any(r['kind']=='attempt_closed' and r['payload']['attempt_id']==attempt.attempt_id for r in budget.rows),'Actual metadata attempt was already closed; never append a duplicate close');core.require(not attempt.pending and attempt.deadline_monotonic is None,'No forwards or time deadline are permitted for accounting')
 elapsed=budget.clock.monotonic()-attempt.record['started_monotonic'];payload=core.close_payload(attempt.attempt_id,elapsed,ctx['pre_budget_wall'],external_upper,info=info,manifest_binding=manifest_binding)
 event=budget._append('attempt_closed',payload);attempt.closed=True;budget._attempt=None;ctx['tail_started']=time.monotonic();ctx['close_event']=event;ctx['stage']='metadata_action_closed';ctx['accounting'].update(closed=True,close_event_sequence=event['sequence'],close_event_sha256=event['sha256'],charged_seconds_upper=payload['charged_seconds_upper'],external_debit_seconds_upper=payload['retrospective_external_debit_seconds_upper'],external_cost_settlement_id=payload.get('external_cost_settlement_id'))
 snapshot=budget.snapshot();core.safe_snapshot(snapshot);core.require(snapshot['physical_forwards_reserved_upper']==ctx['before']['physical_forwards_reserved_upper'] and snapshot['physical_forwards_confirmed']==ctx['before']['physical_forwards_confirmed'],'Metadata action unexpectedly changed physical forwards')
 before=ctx['journal_before'];core.require(sha(P/'budget/journal.jsonl',before['bytes'])==before['sha256'],'Existing journal bytes were rewritten');return event,snapshot

def enforce_tail(ctx):
 if time.monotonic()-ctx['tail_started']>2.:
  event=ctx['close_event'];budget=ctx['budget'];observed=budget.clock.monotonic()-ctx['attempt'].record['started_monotonic'];budget._append('time_finalization_observed',dict(attempt_id=ctx['attempt'].attempt_id,measured_total_seconds=observed+ctx['pre_budget_wall'],charged_seconds_upper=observed+ctx['pre_budget_wall']+event['payload'].get('retrospective_external_debit_seconds_upper',0.)+2.,time_hard_stop_enabled=False,exact_total_time_claim=False));raise ValueError('Actual publication exceeded its reserved2s tail; preserve evidence, do not accept any candidate final manifest')

def finish(ctx,result):
 from budget import atomic_json
 atomic_json(ctx['directory']/'result.json',result);enforce_tail(ctx)
 atomic_json(ctx['directory']/'outcome.json',dict(status=result['status'],exit_code=0,result=str(ctx['directory']/'result.json'),result_sha256=sha(ctx['directory']/'result.json'),model_forwards=0,measured_finalization_seconds_before_last_outcome_write=time.monotonic()-ctx['tail_started'],actual_wrapper_exit_still_required=True));enforce_tail(ctx)
 print(json.dumps(result))

def settlement_receipt(info,event,start,manifest_binding):
 return dict(schema='spnv2-final-external-cost-settlement-v1',status='settled',migration_id='sysu_20261027',plan_path=str(P),budget_journal_path=str(P/'budget/journal.jsonl'),costs_appended_to_original_ledger=True,all_external_work_closed=True,pending_cost_items=[],settlement_id=info['settlement_id'],external_manifest=manifest_binding,import_attempt_id=event['payload']['attempt_id'],import_event_sequence=event['sequence'],import_event_sha256=event['sha256'],import_launch_nonce=start['payload']['launch']['run_nonce'],external_debit_seconds_upper=event['payload']['retrospective_external_debit_seconds_upper'],measurement_note=info['measurement_note'],source_prefix_anchor_sha256=SOURCE_SHA,exact_total_time_claim=False)

def settle(request,log_path,started):
 manifest=checked(request['inventory']);pre=core.inventory(manifest,[item.get('existing_attempt_id') for item in manifest['items'] if item.get('accounting')=='already_in_original_ledger'])
 with admitted(request,'g50p_ext_cost_'+pre['settlement_id'],log_path,started) as ctx:
  budget=ctx['budget'];closed_ids={r['payload']['attempt_id'] for r in budget.rows if r['kind']=='attempt_closed'};info=core.inventory(manifest,closed_ids);transfer_closed(manifest);paid_identity_conflicts(budget.rows,info);existing=core.find_import(budget.rows,info);ctx['settlement_id']=info['settlement_id'];ctx['existing_import_event']=None if existing is None else dict(sequence=existing['sequence'],sha256=existing['sha256']);begin(ctx)
  if existing:
   event=existing;start=next(r for r in budget.rows if r['kind']=='attempt_started' and r['payload']['attempt_id']==event['payload']['attempt_id']);event_manifest=event['payload']['external_cost_manifest'];receipt=settlement_receipt(info,event,start,event_manifest);current_event,snapshot=close(ctx);imported_now=False
  else:
   event,snapshot=close(ctx,external_upper=info['external_debit_seconds_upper'],info=info,manifest_binding=request['inventory']);start=next(r for r in budget.rows if r['kind']=='attempt_started' and r['payload']['attempt_id']==event['payload']['attempt_id']);receipt=settlement_receipt(info,event,start,request['inventory']);imported_now=True
  rb=immutable(P/'revisions/final_cost_accounting_20261028/settlements'/(info['settlement_id']+'.json'),receipt);bb=immutable(ctx['directory']/'budget_closed.json',snapshot);finish(ctx,dict(status='external_cost_settlement_verified',settlement=rb,settlement_id=info['settlement_id'],external_debit_imported_now=imported_now,new_external_debit_seconds_upper=info['external_debit_seconds_upper'] if imported_now else 0.,duplicate_external_debit=False,current_metadata_execution_budget=bb,unknown_measurements_preserved=True,model_forwards=0))

def successful_attempt(nonce,expected_status):
 core.require(isinstance(nonce,str) and len(nonce)==32 and all(x in '0123456789abcdef' for x in nonce),'Exact successful accounting nonce required');directory=P/'launches'/nonce;launch=json.loads(allowed(directory/'launch.json').read_text());outcome=json.loads(allowed(directory/'outcome.json').read_text());log=Path(launch['log_path'])
 core.require(int(allowed(Path(str(log)+'.exitcode')).read_text().strip())==outcome.get('exit_code')==0 and outcome.get('status')==expected_status,'Real prior metadata process has no matching successful wrapper/outcome');result=json.loads(allowed(outcome['result']).read_text());core.require(sha(outcome['result'])==outcome['result_sha256'] and launch['run_nonce']==nonce,'Actual metadata result/identity hash differs');return result,bind(directory/'launch.json'),bind(directory/'outcome.json')

def publish(request,log_path,started):
 provisional=checked(request['provisional']);receipt=checked(request['settlement']);pubid=core.digest(dict(summary=provisional['summary'],proof=provisional['proof'],settlement_id=receipt['settlement_id']))
 with admitted(request,'g50p_final_cost_publish_'+pubid,log_path,started) as ctx:
  core.require(provisional.get('schema')=='g50p-scientific-summary-costs-pending-v1' and provisional.get('status')=='scientific_complete_costs_pending','Only the real guard provisional may be finalized')
  summary=checked(provisional['summary']);proof=checked(provisional['proof']);core.require(proof['summary_sha256']==provisional['summary']['sha256'],'Saved summary/proof hash differs');n=core.summary_scope(summary,proof,provisional,len(ctx['state']['accepted']))
  item=provisional['actual_summary_attempt'];directory=Path(item['launch_directory']);outcome=json.loads(allowed(directory/'outcome.json').read_text());launch=json.loads(allowed(directory/'launch.json').read_text());exit_code=int(allowed(Path(item['log_path']+'.exitcode')).read_text().strip());core.require(exit_code==outcome.get('exit_code')==item.get('exit_code')==0 and outcome.get('status')=='diagnostic_delivered' and outcome.get('summary')==provisional['summary']['path'] and outcome.get('proof')==provisional['proof']['path'] and outcome.get('proof_sha256')==provisional['proof']['sha256'] and launch.get('run_nonce')==item['nonce'],'Actual scientific summary exit/outcome/identity differs')
  core.require(sha(allowed(item['task_path']))==item['task_sha256'] and launch.get('project_id')==ctx['record']['project_id'],'Scientific summary actual task/project binding differs')
  argv=launch.get('argv',[]);core.require('--task-json' in argv and argv[argv.index('--task-json')+1]==item['task_path'],'Actual scientific summary command used another task')
  old_budget=checked(provisional['summary_worker_closed_budget']);core.safe_snapshot(old_budget)
  manifest=checked(receipt['external_manifest']);closed_ids={r['payload']['attempt_id'] for r in ctx['budget'].rows if r['kind']=='attempt_closed'};info=core.inventory(manifest,closed_ids);transfer_closed(manifest);core.require(receipt.get('schema')=='spnv2-final-external-cost-settlement-v1' and receipt.get('status')=='settled' and receipt.get('pending_cost_items')==[] and receipt.get('all_external_work_closed') is True and receipt.get('costs_appended_to_original_ledger') is True and receipt['settlement_id']==info['settlement_id'],'External fees are not genuinely settled')
  imported=core.find_import(ctx['budget'].rows,info);core.require(imported is not None and imported['sequence']==receipt['import_event_sequence'] and imported['sha256']==receipt['import_event_sha256'] and imported['payload']['attempt_id']==receipt['import_attempt_id'],'External import is not the exact recognized original-ledger close event')
  previous,previous_launch,previous_outcome=successful_attempt(request['settlement_success_nonce'],'external_cost_settlement_verified');core.require(previous['settlement']==request['settlement'],'Successful settlement result refers to another receipt');core.safe_snapshot(checked(previous['current_metadata_execution_budget']))
  prior_starts=[r for r in ctx['budget'].rows if r['kind']=='attempt_started' and r['payload']['launch']['run_nonce']==request['settlement_success_nonce']];core.require(len(prior_starts)==1,'Settlement actual process has no unique original-ledger allocation');prior_id=prior_starts[0]['payload']['attempt_id'];prior_closes=[r for r in ctx['budget'].rows if r['kind']=='attempt_closed' and r['payload']['attempt_id']==prior_id];core.require(len(prior_closes)==1 and not prior_starts[0]['payload']['model_forwards_allowed'] and not prior_closes[0]['payload']['unresolved_forward_tokens'],'Settlement metadata execution did not actually close without model work')
  previous_record=checked(previous_launch);core.require(all(previous_record[k]==prior_starts[0]['payload']['launch'][k] for k in ('pid','starttime','boot_id','cmdline_sha256','run_nonce','run_id')),'Settlement exact registered identity differs from actual ledger allocation')
  final_path=P/'delivery/final_manifest.json';core.require(not final_path.exists(),'A final manifest already exists; preserve it and inspect its actual publisher exit, never overwrite/reprice')
  begin(ctx);event,snapshot=close(ctx);bb=immutable(ctx['directory']/'budget_closed.json',snapshot)
  proof_doc=dict(schema='g50p-final-cost-publication-proof-v1',status='metadata_checks_passed_pending_actual_wrapper_exit',summary=provisional['summary'],scientific_summary_proof=provisional['proof'],external_settlement=request['settlement'],external_import_event=dict(sequence=imported['sequence'],sha256=imported['sha256']),fresh_closed_budget=bb,own_publication_close_event=dict(sequence=event['sequence'],sha256=event['sha256']),no_summary_or_model_reexecution=True,model_forwards=0)
  pb=immutable(ctx['directory']/'publication_proof.json',proof_doc)
  final=dict(schema='g50p-diagnostic-delivery-manifest-v1',status='complete' if n==48 else 'partial',summary=provisional['summary'],proof=provisional['proof'],provisional=request['provisional'],final_settled_budget=bb,external_cost_settlement=request['settlement'],final_cost_publication_proof=pb,settlement_actual_successful_outcome=previous_outcome,active_seconds_charged_upper=snapshot['active_seconds_charged_upper'],time_hard_stop_enabled=False,budget_policy=snapshot['budget_policy'],initial_cost_receipts=snapshot['initial_cost_receipts'],physical_forwards_reserved_upper=snapshot['physical_forwards_reserved_upper'],queue_wait_seconds=snapshot['queue_wait_seconds'],complete_ODR_runs=0,independently_accepted_windows=n,scientific_scope_complete=n==48,source_prefix=ctx['prefix'],ledger_terminal_close_event=dict(sequence=event['sequence'],sha256=event['sha256']),publication_launch=bind(ctx['directory']/'launch.json'),publication_outcome_path=str(ctx['directory']/'outcome.json'),publication_actual_exit_code_path=str(log_path)+'.exitcode',valid_only_with_actual_publisher_wrapper_exit_0=True,exact_total_time_claim=False,publication_measurement_tail_accounted_as_upper_seconds=2.,note='Original scientific summary preserved. All imported external scopes and this real publication attempt are in the continuous original ledger. CPU-unavailable/raw-wall/conservative-debit details remain in bound inventory; no exact global elapsed-time claim.')
  store_candidate_and_finish(ctx,final,dict(publication_proof=pb,fresh_budget=bb,publication_id=pubid,model_forwards=0))

def store_candidate_and_finish(ctx,fields,result,*,store=None,complete=None):
 store=immutable if store is None else store;complete=finish if complete is None else complete
 candidate=dict(fields);candidate.update(schema='g50p-final-cost-publication-candidate-v1',status='candidate_pending_actual_exit_and_paid_promotion',scientific_delivery_status=fields['status'],candidate_only=True,final_manifest_not_published=True,promotion_costs_pending=True,explicit_paid_atomic_promotion_required=True)
 candidate['candidate_stage_closed_budget']=candidate.pop('final_settled_budget');candidate['candidate_cutoff_active_seconds_charged_upper']=candidate.pop('active_seconds_charged_upper')
 candidate['candidate_stage_proof']=candidate.pop('final_cost_publication_proof');candidate['proposed_final_target_path']=str(P/'delivery/final_manifest.json')
 candidate['note']='Not a final manifest or final total-cost receipt. After real publisher wrapper0, verify source/own closure; promotion must itself be explicitly paid/measured and bind a fresh final ledger snapshot.'
 cb=store(ctx['directory']/'candidate_final_manifest.json',candidate)
 output=dict(result,status='final_cost_candidate_ready_pending_wrapper_exit',candidate_final_manifest=cb,final_manifest_not_published=True,promotion_costs_pending=True,root_must_verify_actual_wrapper_exit_before_promotion=True)
 return complete(ctx,output)

def failure_payload(error,observation,ctx,now_monotonic,now_wall):
 """Pure failure accounting record, including before-yield/before-begin errors."""
 ctx=ctx or {};account=dict(ctx.get('accounting') or {});elapsed=now_monotonic-observation['started_monotonic'];elapsed=elapsed if elapsed>=0 else None
 known_id=account.get('attempt_id');closed=account.get('closed') is True;charge=account.get('charged_seconds_upper');external=account.get('external_debit_seconds_upper')
 self_upper=charge-external if type(charge) in (int,float) and type(external) in (int,float) else None
 covered=closed and elapsed is not None and self_upper is not None and self_upper>=elapsed and not account.get('reconciliation_error')
 code=error.code if isinstance(error,SystemExit) and type(error.code) is int else 130 if isinstance(error,KeyboardInterrupt) else 1
 start=account.get('started_monotonic');unadmitted=(start-observation['started_monotonic']) if type(start) in (int,float) and start>=observation['started_monotonic'] else elapsed if known_id is None else None
 return dict(schema='spnv2-final-cost-entry-failure-v1',status='waiting_resources' if code==75 else 'failed',exit_code=code,nonce=observation.get('nonce'),request_path=observation.get('request_path'),request_sha256_claimed=observation.get('request_sha256'),request_binding_verified=observation.get('request_verified',False),log_path=observation.get('log_path'),argv=observation.get('argv'),process_identity=observation.get('process_identity'),code_sha256=observation.get('code_sha256'),started_wall_time=observation['started_wall_time'],started_monotonic=observation['started_monotonic'],observed_failure_wall_time=now_wall,measured_total_main_wall_seconds_before_failure_receipt=elapsed,measured_unadmitted_wall_seconds=unadmitted,cpu_seconds=None,cpu_measurement_status='unavailable',stage=ctx.get('stage',observation.get('stage','request_or_code_validation')),attempt_id=known_id,ledger_attempt_closed=closed,existing_ledger_attempt_must_not_be_reimported=known_id is not None,ledger_accounting=account,accounting_pending=not covered,failed_native_wall_is_not_assumed_zero=True,uncovered_wall_lower_bound=None if elapsed is None or self_upper is None else max(0.,elapsed-self_upper),attempt_start_requested=ctx.get('attempt_start_requested',False),existing_external_import=ctx.get('existing_import_event'),error_type=type(error).__name__,error=str(error),model_forwards=0,actual_wrapper_exit_still_required=True,no_final_success_claim=True,measurement_scope='Observed before failure-receipt/exit IO; existing ledger charge is preserved, never refunded or imported twice')

def _atomic_failure(path,value):
 path=Path(path);temp=path.with_name(path.name+'.'+str(os.getpid())+'.'+str(time.time_ns())+'.tmp');data=(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
 with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def record_failure(error,observation):
 ctx=EXECUTION;payload=failure_payload(error,observation,ctx,time.monotonic(),time.time());directory=None
 try:
  if ctx is not None and ctx.get('directory_created_here'):directory=ctx['directory']
  elif P.is_dir():
   # Never overwrite a previous nonce directory when this invocation was rejected.
   directory=P/'launches'/('final_cost_unadmitted_'+str(os.getpid())+'_'+str(time.time_ns()));directory.mkdir(exist_ok=False)
  if directory is not None:
   _atomic_failure(directory/'entry_failure_receipt.json',payload)
   _atomic_failure(directory/'outcome.json',dict(status=payload['status'],exit_code=payload['exit_code'],entry_failure_receipt=str(directory/'entry_failure_receipt.json'),entry_failure_receipt_sha256=sha(directory/'entry_failure_receipt.json'),attempt_id=payload['attempt_id'],ledger_attempt_closed=payload['ledger_attempt_closed'],accounting_pending=payload['accounting_pending'],model_forwards=0,no_final_success_claim=True))
   return
 except BaseException as write_error:payload['failure_receipt_write_error']=repr(write_error)
 print(json.dumps(payload,ensure_ascii=True,default=repr),file=sys.stderr)

def guarded_action(operation,observation,on_failure=None):
 try:return operation()
 except BaseException as error:
  (record_failure if on_failure is None else on_failure)(error,observation);raise

def main():
 global EXECUTION
 started=time.monotonic();started_wall=time.time();EXECUTION=None
 def pure_tests():
  load_core();loader=importlib.util.spec_from_file_location('g50p_final_cost_selftest',Path(__file__).with_name('selftest.py'));module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);return module.run(core,sys.modules[__name__])
 if sys.argv[1:]==['--self-test']:return pure_tests()
 observation=dict(started_monotonic=started,started_wall_time=started_wall,nonce=os.environ.get('SPNV2_RUN_NONCE'),argv=list(sys.argv),request_path=None,request_sha256=None,request_verified=False,log_path=None,stage='argument_or_request_validation')
 try:
  proc=Path('/proc')/str(os.getpid());stat=(proc/'stat').read_text().rsplit(')',1)[1].split();observation['process_identity']=dict(pid=os.getpid(),starttime=int(stat[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest());observation['code_sha256']=sha(__file__)
 except (OSError,ValueError):observation['process_identity']=dict(pid=os.getpid(),identity_metadata_unavailable=True)
 def execute():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--action',choices=('settle','publish'));parser.add_argument('--request');parser.add_argument('--request-sha256');parser.add_argument('--log-path');parser.add_argument('--execute',action='store_true');args=parser.parse_args();observation.update(request_path=args.request,request_sha256=args.request_sha256,log_path=args.log_path)
  if args.self_test:parser.error('Pure test entry must be exactly --self-test')
  if not(args.execute and args.action and args.request and args.request_sha256 and args.log_path):parser.error('Explicit --execute/action and bound request/log are required')
  request_path=Path(args.request).resolve(strict=True)
  if not(request_path.is_relative_to(P) or request_path.is_relative_to(M)) or request_path.stat().st_size>8<<20 or sha(request_path)!=args.request_sha256:raise ValueError('Accounting request binding/path changed')
  observation['request_verified']=True;request=json.loads(request_path.read_text());entry=request.get('entry',{})
  if Path(entry.get('path','')).resolve()!=Path(__file__).resolve() or entry.get('sha256')!=sha(__file__):raise ValueError('Actual accounting entry code is not explicitly bound')
  load_core(request['core']);core.require(request.get('schema')=='spnv2-final-cost-accounting-request-v1' and request.get('action')==args.action,'Wrong explicit accounting action request')
  if args.action=='settle':settle(request,args.log_path,started)
  else:publish(request,args.log_path,started)
  return 0
 return guarded_action(execute,observation)
if __name__=='__main__':raise SystemExit(main())
