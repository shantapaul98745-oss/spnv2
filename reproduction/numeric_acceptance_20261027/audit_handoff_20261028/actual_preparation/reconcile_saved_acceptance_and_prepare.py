import sys,copy,json,uuid,hashlib
from pathlib import Path
from contextlib import ExitStack
R=Path('/home/ps/Downloads/UsrSourceCode/LuXinghai/spnv2/reproduction/numeric_acceptance_20261027');sys.path.insert(0,str(R))
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M;FIX=D/'revisions/summary_handoff_20261028'
with ExitStack() as stack:
 for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
 stack.enter_context(b.lock(D/'manager.lock'))
 oldindex=b.read(FIX/'before/index.json');state=b.read(D/'status.json');a=state['active'];index=copy.deepcopy(oldindex);index['numeric_acceptance']=b.bind(FIX/'revision.json')
 task_path=b.mapped(oldindex['audit_task_paths']['summary']);task=b.read(task_path);receipts_path=b.mapped(task['execution_receipts']['path']);rc=b.read(receipts_path)
 assert b.sha(receipts_path)==task['execution_receipts']['sha256'];planned=[x['run_id'] for x in oldindex['tasks']];assert rc['planned_run_ids']==planned and rc['missing_main_run_ids']==[] and rc['missing_independent_acceptance_run_ids']==[]
 assert len(state['accepted'])==47 and len(state['main'])==48 and len(rc['accepted'])==48 and {x['run_id']:x for x in rc['runs']}==state['main'];assert all(rc['accepted'][k]==v for k,v in state['accepted'].items())
 last='g50p_diag_v1_sunlamp_s2023_w0113';v=rc['accepted'][last]['verification_attempt'];assert a['nonce']==v['nonce']=='f9222d5f12954aad97d96a1d8b3e376c' and a['task_sha256']==v['task_sha256'] and a['run_id']==v['run_id']
 checks=[]
 for run in planned[42:]:
  ac=rc['accepted'][run];vv=ac['verification_attempt'];main=state['main'][run];proof=b.read(b.mapped(ac['proof']['path']));report=b.read(b.mapped(ac['report']['path']))
  assert b.sha(b.mapped(ac['proof']['path']))==ac['proof']['sha256'] and b.sha(b.mapped(ac['report']['path']))==ac['report']['sha256']
  assert proof['status']=='passed' and proof['actual_main_exit_verified'] is True and proof['report_sha256']==ac['report']['sha256'] and proof['independent_result']['complete_state_accepted_under_policy'] is True
  assert report['scientific_state_accepted_under_policy'] is True and report['discrete_trajectory_exact'] is True and report['internal_branch_and_rollback_exact'] is True
  assert vv['exit_code']==0 and vv['wrapper_returncode']==0 and Path(str(b.mapped(vv['log_path']))+'.exitcode').read_text().strip()=='0'
  vo=b.read(b.mapped(vv['launch_directory'])/'outcome.json');vl=b.read(b.mapped(vv['launch_directory'])/'launch.json');assert vo==vv['outcome'] and vo['exit_code']==0 and vo['proof_sha256']==ac['proof']['sha256'] and vl['run_nonce']==vv['nonce'] and vl['task_sha256']==vv['task_sha256']
  assert Path(b.mapped(main['exit_code_path'])).read_text().strip()=='0' and b.read(b.mapped(main['outcome_path']))['exit_code']==0
  checks.append(dict(run_id=run,proof=ac['proof'],report=ac['report'],forward_upper=report['budget']['physical_forwards_reserved_upper'],unknown_upper=report['budget']['physical_forwards_unknown_upper']))
 # Restore only the actual terminalized final attempt from the already durable
 # receipt. Earlier attempts must be exactly the same and preserve all costs.
 ra=rc['all_attempts'];assert len(ra)==len(state['attempts'])
 for old,new in zip(state['attempts'],ra):
  if old['nonce']==a['nonce']:
   assert new==v
   assert all(new.get(k)==value for k,value in old.items() if k not in ['terminal_evidence_handoff','exit_code','outcome'])
  else:assert old==new
 journal=(D/'budget/journal.jsonl').read_bytes();prefix=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes();assert journal.startswith(prefix)
 events=[json.loads(x) for x in journal.splitlines()];assert not ({x['payload']['attempt_id'] for x in events if x['kind']=='attempt_started'}-{x['payload']['attempt_id'] for x in events if x['kind']=='attempt_closed'})
 b.immutable_bytes(FIX/'failed_preparation_index.json',(D/'index.json').read_bytes());b.immutable_bytes(FIX/'pre_reconciliation_status.json',(D/'status.json').read_bytes())
 b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json');state.update(index=ib,accepted=copy.deepcopy(rc['accepted']),attempts=copy.deepcopy(ra),active=None,manager=None,status='between_tasks',next_resource_check=0.,reason='Recovered real48 accepted cohort from unchanged durable final receipt; original strict audit follows')
 b.atomic(D/'status.json',b.raw(state));b.immutable(FIX/'reconciled_state.json',state)
 control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v14.json');handoff['diagnostic_index']=ib;hp=b.immutable(control/'budget_handoff_summary_handoff2.json',handoff)
 template=b.read(control/'template_numeric_v14.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_summary_handoff2.json',template)
 dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_summary_handoff2.json',dep)
 # Compose the actual existing final-summary task against the faithfully
 # restored original receipt; all original strict receipt checks still run.
 config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proofout=control/'config_versions'/('summary_handoff2_validation_'+nonce+'.json');validator=R/'prepare_runtime.py'
 req=b.immutable(FIX/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proofout)))
 python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
 ma=ns.build_manager_bwrap(config,b.bind(dp),version);ma[ma.index(b.logical(D/'implementation/manager.py'))]=b.logical(R/'final_cost_guard_v15.py')
 assert (D/'budget/journal.jsonl').read_bytes()==journal
 result=dict(schema='spnv2-final-audit-handoff-preparation-v1',published_code_commit='b37e5309e80dc721c6ab7c481417dd6adf7a8260',index=ib,revision=index['numeric_acceptance'],deployment=b.bind(dp),config=version,validation_nonce=nonce,validation_output=b.logical(proofout),validation_argv=va,manager_argv=ma,screen_socket_dir=config['screen_socket_dir'],accepted_reused=48,main_reused=48,successful_last_verifier_reused=True,model_forwards=0,original_final_receipts=task['execution_receipts'],newly_verified_pairs=checks,ledger_prefix=dict(bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()))
 b.immutable(FIX/'preparation_receipt.json',result);b.immutable(FIX/'preparation_failures.json',dict(real_failed_preparations=[dict(actual_exit_code=1,error='Missing routing pilot metadata',code=b.bind(R/'resume_summary_handoff_20261028.py')),dict(actual_exit_code=1,error='Strict acceptance namespace rejected audit-path/template change',code=b.bind(FIX/'finish_preparation_recovery.py'))],recovery_code=b.bind(Path(__file__)),original_immutable_task_and_receipts_unchanged=True,model_forwards=0,ledger_unchanged=True))
 print(json.dumps({k:v for k,v in result.items() if k not in ['validation_argv','manager_argv','newly_verified_pairs']}))
