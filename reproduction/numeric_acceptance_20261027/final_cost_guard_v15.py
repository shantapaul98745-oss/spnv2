"""Bound metadata-only manager driver: never publish final cost claims.

Arguments match the canonical manager. Revision binding:
 final_cost_guard={code:{path,sha256},original_manager:{path,sha256},
                   external_settlement_receipt?:{path,sha256}}
If the optional receipt is absent, M/status.final_external_cost_settlement is
consulted. A hash-verified receipt may only mark an explicit final-cost review
ready; this driver never estimates fees or writes delivery/final_manifest.json.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys,time
ROOT=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
PLAN=ROOT/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
MIGRATION=ROOT/'spnv2_runs/migrations/sysu_20261027'
MANAGER=PLAN/'implementation/manager.py'
MANAGER_SHA='4f372e4117744ace342f2a9a25984d8fa9cf5d03cf2542e8f0dbd13f15efa52c'
PENDING_STATUS='scientific_complete_costs_pending'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def checked(item):
 if not isinstance(item,dict) or set(item)!={'path','sha256'}:raise ValueError('Exact metadata binding required')
 path=Path(item['path']).resolve(strict=True)
 if sha(path)!=item['sha256']:raise ValueError('Final-cost guard metadata binding changed: '+str(path))
 return read(path)

def settlement_readiness(migration_status,receipt_binding=None,receipt=None,receipt_hash_verified=False):
 """Pure readiness declaration; no claim that a final ledger was audited here."""
 reasons=[]
 if migration_status.get('migration_control_costs_pending_settlement') is not False:reasons.append('migration_control_costs_pending_settlement')
 if not receipt_hash_verified or not receipt_binding or not isinstance(receipt,dict):reasons.append('no_hash_verified_final_external_cost_settlement_receipt')
 else:
  if receipt.get('schema')!='spnv2-final-external-cost-settlement-v1' or receipt.get('status')!='settled':reasons.append('external_settlement_receipt_not_settled')
  if receipt.get('migration_id')!='sysu_20261027' or receipt.get('plan_path')!=str(PLAN):reasons.append('external_settlement_receipt_identity_differs')
  if receipt.get('budget_journal_path')!=str(PLAN/'budget/journal.jsonl') or receipt.get('costs_appended_to_original_ledger') is not True:reasons.append('original_continuous_ledger_append_not_attested')
  if receipt.get('all_external_work_closed') is not True or receipt.get('pending_cost_items')!=[]:reasons.append('external_work_or_cost_items_unclosed')
 return dict(ready_for_explicit_metadata_only_final_cost_review=not reasons,
             pending_reasons=reasons,external_settlement_receipt=receipt_binding,
             receipt_hash_verified=receipt_hash_verified,final_budget_verified_here=False,
             final_cost_publication_required=True,no_cost_estimation_performed=True)

def external_metadata(guard):
 try:status=read(MIGRATION/'status.json')
 except (OSError,ValueError) as error:return dict(settlement_readiness({}),metadata_error=repr(error))
 item=guard.get('external_settlement_receipt') or status.get('final_external_cost_settlement')
 receipt=None;verified=False;error=None
 if item is not None:
  try:
   path=Path(item['path']).resolve(strict=True)
   if not(path.is_relative_to(PLAN) or path.is_relative_to(MIGRATION)):raise ValueError('External cost receipt escapes this project plan/migration')
   receipt=checked(item);verified=True
  except (OSError,ValueError,KeyError,TypeError) as exception:error=repr(exception)
 result=settlement_readiness(status,item,receipt,verified)
 result['migration_pending_flag']=status.get('migration_control_costs_pending_settlement')
 if error:result['metadata_error']=error
 return result

def make_provisional(state,source,readiness,guard_binding):
 """Pure metadata plan. The source worker snapshot is explicitly not final."""
 complete=source['proof'].get('complete_scope') is True and source['proof'].get('completed_windows')==48 and len(state['accepted'])==48
 receipt=dict(schema='g50p-scientific-summary-costs-pending-v1',status=PENDING_STATUS,
              scientific_scope_complete=complete,scientific_scope_status='complete' if complete else 'partial',
              independently_accepted_windows=len(state['accepted']),maximum_windows=48,
              summary=source['summary_binding'],proof=source['proof_binding'],
              actual_summary_worker_exit_code=0,actual_summary_attempt=copy.deepcopy(source['item']),
              summary_worker_closed_budget=source['worker_budget_binding'],
              summary_worker_budget_is_not_final_external_cost_accounting=True,
              original_index=copy.deepcopy(state['index']),final_cost_guard=copy.deepcopy(guard_binding),
              external_cost_readiness=readiness,final_settled_budget_not_published=True,
              final_manifest_not_published=True,model_forwards_added_by_guard=0,
              preserved_summary_must_not_be_reexecuted=True,
              next_stage='explicit_metadata_only_external_cost_settlement_and_final_publication',
              note='Scientific outputs are preserved. This is a provisional manifest, never a settled total-cost receipt.')
 return receipt

def install_finish_guard(manager,guard):
 original=manager.finish
 def finish(state):
  item=state.get('active')
  if not isinstance(item,dict):return original(state)
  exit_path=Path(item['log_path']+'.exitcode');directory=Path(item['launch_directory']);outcome_path=directory/'outcome.json'
  if not(exit_path.is_file() and outcome_path.is_file()):return original(state)
  code=int(exit_path.read_text().strip());outcome=manager.read(outcome_path)
  if code!=0 or outcome.get('exit_code')!=code:return original(state)
  task=manager.read(item['task_path'])
  if task.get('action')!='summarize_diagnostic' or outcome.get('status')!='diagnostic_delivered':return original(state)
  # Exact original terminal handoff and successful summary-branch checks.
  handoff=item.get('terminal_evidence_handoff')
  if handoff is not None:
   handoff.update(resolved_at=time.time(),resolution='existing_terminal_evidence_received')
   if state.get('status')=='awaiting_terminal_evidence':state.update(status='running',reason='Terminal evidence arrived; continue normal strict outcome validation')
  item.update(exit_code=code,outcome=outcome)
  proof_path=Path(outcome['proof']);proof=manager.read(proof_path);summary_path=Path(outcome['summary'])
  if proof.get('status')!='passed' or proof.get('summary_sha256')!=manager.sha(summary_path):raise ValueError('Diagnostic delivery lacks the actual summary/proof binding')
  worker_budget_path=directory/'budget_closed.json';worker_budget=manager.read(worker_budget_path)
  if (worker_budget.get('schema')!='g50p-diagnostic-budget-snapshot-v1' or worker_budget.get('unresolved_attempt_ids') or (worker_budget.get('time_hard_stop_enabled',True) and worker_budget.get('active_time_bound_violation_attempt_ids'))):raise ValueError('Final delivery requires the actual settled budget receipt after process exit')
  source=dict(item=item,proof=proof,summary_binding=manager.binding(summary_path),proof_binding=manager.binding(proof_path),worker_budget_binding=manager.binding(worker_budget_path))
  readiness=external_metadata(guard);provisional=make_provisional(state,source,readiness,guard)
  nonce=item.get('nonce','')
  if len(nonce)<16 or not all(c.isalnum() or c in '_-' for c in nonce):raise ValueError('Actual summary launch nonce invalid')
  path=PLAN/'delivery'/('provisional_costs_pending_'+nonce+'_'+guard['code']['sha256'][:12]+'.json')
  manager.immutable(path,provisional)
  state.update(status=PENDING_STATUS,active=None,provisional_delivery=manager.binding(path),
               scientific_scope_complete=provisional['scientific_scope_complete'],
               final_cost_publication_required=True,
               ready_for_explicit_metadata_only_final_cost_review=readiness['ready_for_explicit_metadata_only_final_cost_review'],
               reason=('Scientific summary preserved; explicit metadata-only final-cost publication remains required' if readiness['ready_for_explicit_metadata_only_final_cost_review'] else 'Scientific summary preserved; migration/control costs await genuine settlement'))
  manager.write_state(state)
  print(json.dumps(dict(status=PENDING_STATUS,provisional_manifest=str(path),scientific_scope_complete=provisional['scientific_scope_complete'],external_cost_readiness=readiness,final_manifest_written=False)))
  raise SystemExit(0)
 manager.finish=finish
 return original

def install_audit_handoff_guard(manager):
 """Persist the same accepted cohort before the unchanged audit dispatcher reads it.

 The canonical loop may complete its last verifier less than sixty seconds
 after its previous status write. Its audit receipt then contains the new
 acceptance while compose_single_task still reads the old on-disk cohort.
 Only the existing atomic state writer is called here; no receipt comparison,
 task, scientific evidence or resource/budget entry is changed.
 """
 original=manager.launch
 def launch(state,task_path,run_id,index_path):
  if manager.read(task_path).get('action') in ('verify_pilots','summarize_diagnostic'):
   manager.write_state(state)
  return original(state,task_path,run_id,index_path)
 manager.launch=launch
 return original

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--index',required=True);parser.add_argument('--deployment-manifest',required=True);parser.add_argument('--deployment-sha256',required=True);args=parser.parse_args()
 index_path=Path(args.index).resolve(strict=True)
 if index_path!=PLAN/'index.json':raise ValueError('Guard may only wrap this original diagnostic plan')
 index=read(index_path);revision=checked(index['numeric_acceptance']);guard=revision.get('final_cost_guard')
 if not isinstance(guard,dict):raise ValueError('Explicit current index.numeric_acceptance.final_cost_guard binding required')
 self_spec=guard.get('code',{});manager_spec=guard.get('original_manager',{})
 if Path(self_spec.get('path','')).resolve()!=Path(__file__).resolve() or self_spec.get('sha256')!=sha(__file__):raise ValueError('Final-cost guard code is not the explicitly bound file')
 if Path(manager_spec.get('path','')).resolve()!=MANAGER or manager_spec.get('sha256')!=MANAGER_SHA or index['implementation_sha256'].get('manager.py')!=MANAGER_SHA or sha(MANAGER)!=MANAGER_SHA:raise ValueError('Canonical manager changed; refuse an unreviewed finish wrapper')
 sys.path.insert(0,str(PLAN/'implementation'));loader=importlib.util.spec_from_file_location('bound_canonical_diagnostic_manager',MANAGER);manager=importlib.util.module_from_spec(loader);sys.modules[loader.name]=manager;loader.loader.exec_module(manager)
 install_finish_guard(manager,guard)
 install_audit_handoff_guard(manager)
 # A repeated manager invocation must not rerun the preserved successful summary.
 if (PLAN/'status.json').exists():
  state=manager.read(PLAN/'status.json')
  if state.get('status')==PENDING_STATUS:
   if not state.get('provisional_delivery'):raise ValueError('Deferred state lacks its immutable provisional evidence')
   manager.checked(state['provisional_delivery'])
   print(json.dumps(dict(status=PENDING_STATUS,final_cost_publication_required=True,scientific_summary_reexecuted=False,final_manifest_written=False)));return 0
 return manager.main(index_path,args.deployment_manifest,args.deployment_sha256)

def pure_metadata_self_test():
 """In-memory branch fixtures only: no files, ledger, processes or models."""
 from types import SimpleNamespace
 import contextlib,io
 global Path,external_metadata
 saved_path,saved_external=Path,external_metadata
 class FixturePath:
  def __init__(self,value):self.value=str(value)
  def __truediv__(self,value):return FixturePath(self.value+'/'+str(value))
  def __str__(self):return self.value
  def is_file(self):return True
  def read_text(self):return '0'
 base=dict(accepted={str(i):{} for i in range(48)},index=dict(path='index',sha256='d'*64),status='running',active=dict(nonce='1'*32,log_path='summary.log',launch_directory='launch',task_path='task.json'))
 guard=dict(code=dict(path='guard.py',sha256='0'*64))
 receipt=dict(schema='spnv2-final-external-cost-settlement-v1',status='settled',migration_id='sysu_20261027',plan_path=str(PLAN),budget_journal_path=str(PLAN/'budget/journal.jsonl'),costs_appended_to_original_ledger=True,all_external_work_closed=True,pending_cost_items=[])
 pending=settlement_readiness(dict(migration_control_costs_pending_settlement=True))
 missing=settlement_readiness(dict(migration_control_costs_pending_settlement=False))
 ready=settlement_readiness(dict(migration_control_costs_pending_settlement=False),dict(path='receipt',sha256='e'*64),receipt,True)
 assert not pending['ready_for_explicit_metadata_only_final_cost_review'] and not missing['ready_for_explicit_metadata_only_final_cost_review'] and ready['ready_for_explicit_metadata_only_final_cost_review']
 count=0
 try:
  Path=FixturePath
  for action,readiness,bad_proof in [('run_pair',pending,False),('verify_pair',pending,False),('summarize_diagnostic',pending,False),('summarize_diagnostic',missing,False),('summarize_diagnostic',ready,False),('summarize_diagnostic',pending,True)]:
   writes=[];delegated=[];state=copy.deepcopy(base)
   outcome=dict(exit_code=0,status='diagnostic_delivered' if action=='summarize_diagnostic' else 'pair_main_complete',proof='proof.json',summary='summary.json')
   proof=dict(status='failed' if bad_proof else 'passed',summary_sha256='a'*64,complete_scope=True,completed_windows=48)
   budget=dict(schema='g50p-diagnostic-budget-snapshot-v1',unresolved_attempt_ids=[],time_hard_stop_enabled=False)
   def fixture_read(path):
    value=str(path)
    if value.endswith('outcome.json'):return outcome
    if value.endswith('budget_closed.json'):return budget
    if value.endswith('proof.json'):return proof
    if value.endswith('task.json'):return dict(action=action)
    raise AssertionError('Unexpected fixture read: '+value)
   manager=SimpleNamespace(finish=lambda st:delegated.append(st) or 'original_finish',read=fixture_read,sha=lambda path:'a'*64,binding=lambda path:dict(path=str(path),sha256='a'*64),immutable=lambda path,obj:writes.append((str(path),copy.deepcopy(obj))),write_state=lambda st:writes.append(('status',copy.deepcopy(st))))
   external_metadata=lambda unused:readiness
   install_finish_guard(manager,guard)
   try:
    with contextlib.redirect_stdout(io.StringIO()):result=manager.finish(state)
   except SystemExit as exit:
    assert exit.code==0 and action=='summarize_diagnostic' and not bad_proof
    assert state['status']==PENDING_STATUS and state['active'] is None and not delegated
    assert len(writes)==2 and all(not path.endswith('/final_manifest.json') for path,_ in writes)
    assert writes[0][1]['final_settled_budget_not_published'] is True and 'final_settled_budget' not in writes[0][1]
   except ValueError:
    assert bad_proof and not writes and not delegated
   else:
    assert action in ('run_pair','verify_pair') and result=='original_finish' and len(delegated)==1 and not writes
   count+=1
 finally:Path,external_metadata=saved_path,saved_external
 print(json.dumps(dict(status='pure_metadata_tests_passed',cases=count,model_forwards=0,no_runtime_file_or_ledger_reads=True,no_runtime_writes=True)))
 return 0

if __name__=='__main__':
 if sys.argv[1:]==['--self-test']:raise SystemExit(pure_metadata_self_test())
 raise SystemExit(main())
