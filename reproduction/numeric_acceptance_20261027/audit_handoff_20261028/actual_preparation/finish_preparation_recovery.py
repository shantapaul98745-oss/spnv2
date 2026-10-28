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
 index=b.read(D/'index.json');ib=b.bind(D/'index.json');state=b.read(D/'status.json');dp=M/'runtime_control/deployment_summary_handoff1.json';dep=b.read(dp)
 if dep['index']!=ib or b.sha(D/'index.json')!=b.sha(FIX/'index_candidate.json') or len(state['accepted'])!=47 or len(state['main'])!=48 or state['active']['nonce']!='f9222d5f12954aad97d96a1d8b3e376c':raise ValueError('Partial preparation boundary changed')
 journal=(D/'budget/journal.jsonl').read_bytes();events=[json.loads(x) for x in journal.splitlines()]
 if {x['payload']['attempt_id'] for x in events if x['kind']=='attempt_started'}-{x['payload']['attempt_id'] for x in events if x['kind']=='attempt_closed'}:raise ValueError('Active allocation')
 # No old pilot task existed at this technical routing path. Create only the
 # canonical fixed-pilot metadata used by the namespace validator, never run it.
 selected=index['tasks'][:2];receipts=dict(schema='g50p-diagnostic-execution-receipts-v1',runs=[state['main'][x['run_id']] for x in selected],accepted={x['run_id']:state['accepted'][x['run_id']] for x in selected},planned_run_ids=[x['run_id'] for x in selected],missing_main_run_ids=[],missing_independent_acceptance_run_ids=[],all_attempts=state['attempts'],queue_wait_seconds=state['queue_wait_seconds'],preprocessing=state['preprocessing'])
 rp=b.immutable(D/'pilot_receipts_summary_handoff_validation.json',receipts);task=copy.deepcopy(index['pilot_admission_template']);task.update(execution_receipts=b.bind(rp),index_path=str(J/ns.DIAGNOSTIC/'index.json'));task_path=b.immutable(b.mapped(index['audit_task_paths']['pilot_admission']),task)
 config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proof=M/'runtime_control/config_versions'/('summary_handoff1_validation_'+nonce+'.json');validator=R/'prepare_runtime.py'
 req=b.immutable(FIX/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proof)))
 python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
 ma=ns.build_manager_bwrap(config,b.bind(dp),version);ma[ma.index(b.logical(D/'implementation/manager.py'))]=b.logical(R/'final_cost_guard_v15.py')
 if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Preparation changed ledger')
 result=dict(schema='spnv2-final-audit-handoff-preparation-v1',published_code_commit='b37e5309e80dc721c6ab7c481417dd6adf7a8260',index=ib,revision=index['numeric_acceptance'],deployment=b.bind(dp),config=version,validation_nonce=nonce,validation_output=b.logical(proof),validation_argv=va,manager_argv=ma,screen_socket_dir=config['screen_socket_dir'],accepted_reused=47,main_reused=48,successful_last_verifier_reused=True,model_forwards=0,preparation_recovery='Missing current-routing pilot metadata materialized for validator only; no pilot replay',ledger_prefix=dict(bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()))
 b.immutable(FIX/'preparation_receipt.json',result);b.immutable(FIX/'preparation_failure.json',dict(actual_exit_code=1,error='Missing tasks/shared_gpu_20261027/verify_pilots.json at validation-only compose',original_preparation_code=b.bind(R/'resume_summary_handoff_20261028.py'),recovery_code=b.bind(Path(__file__)),scientific_workers_launched=0,model_forwards=0,ledger_unchanged=True))
 print(json.dumps({k:v for k,v in result.items() if k not in ['validation_argv','manager_argv']}))
