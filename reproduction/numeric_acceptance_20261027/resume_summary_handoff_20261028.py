"""Versioned metadata-only last-verifier to final-audit handoff repair.
Never replays an accepted main, verifier, model or saved panel.
"""
import sys,copy,json,hashlib,uuid,os
from pathlib import Path
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
R=U/'spnv2/reproduction/numeric_acceptance_20261027'
FIX=D/'revisions/summary_handoff_20261028'
BASE='dea906ca7dac4bc431e67c208d7441a3cf50c67a3d55442e2090ee3a39ce546b'
LAST='g50p_diag_v1_sunlamp_s2023_w0113'
def main(commit):
 if len(commit)!=40:raise ValueError('Published commit required')
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!=BASE:raise ValueError('Exact V14 index changed')
  before=b.read(D/'index.json');state=b.read(D/'status.json');active=state.get('active') or {}
  if len(state['main'])!=48 or len(state['accepted'])!=47 or active.get('nonce')!='f9222d5f12954aad97d96a1d8b3e376c' or LAST in state['accepted']:raise ValueError('Not the saved final-verifier handoff')
  oldlog=D/'revisions/numeric_acceptance_20261027/v14_fixed_heat_vector_bound/manager.log'
  if Path(str(oldlog)+'.exitcode').read_text().strip()!='1':raise ValueError('Old manager actual exit missing')
  outcome=b.read(b.mapped(active['launch_directory'])/'outcome.json');proof_path=b.mapped(outcome['proof']);proof=b.read(proof_path)
  if Path(str(b.mapped(active['log_path']))+'.exitcode').read_text().strip()!='0' or outcome.get('exit_code')!=0 or outcome.get('status')!='pair_verified' or outcome['run_id']!=LAST:raise ValueError('Final independent worker actual success missing')
  if proof['status']!='passed' or proof['report_sha256']!=state['main'][LAST]['report']['sha256'] or outcome['proof_sha256']!=b.sha(proof_path):raise ValueError('Final proof binding changed')
  journal=(D/'budget/journal.jsonl').read_bytes();prefix=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes()
  if not journal.startswith(prefix):raise ValueError('Original budget prefix changed')
  events=[json.loads(line) for line in journal.splitlines()];starts={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if starts-closed:raise ValueError('Unclosed allocation blocks metadata cutover')
  for path in [D/'index.json',D/'status.json',oldlog,Path(str(oldlog)+'.exitcode'),b.mapped(before['audit_task_paths']['summary'])]:
   b.immutable_bytes(FIX/'before'/path.relative_to(D),path.read_bytes())
  oldtask=b.read(b.mapped(before['audit_task_paths']['summary']));receipt=b.mapped(oldtask['execution_receipts']['path']);b.immutable_bytes(FIX/'before'/receipt.relative_to(D),receipt.read_bytes())
  revision=copy.deepcopy(b.read(b.mapped(before['numeric_acceptance']['path'])))
  oldguard=copy.deepcopy(revision['final_cost_guard']);revision.update(published_code_commit=commit,final_cost_guard=dict(code=b.bind(R/'final_cost_guard_v15.py'),original_manager=oldguard['original_manager']),supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(FIX/'before/index.json'),metadata_handoff_fix=dict(authorization='existing_technical_repair_and_safe_continuation',failure_analysis=b.bind(R/'audit_handoff_20261028/failure_analysis.json'),tests=b.bind(R/'audit_handoff_20261028/targeted_tests.json'),reason='Flush actual in-memory accepted evidence before unchanged strict audit composer reads status',scientific_policy_unchanged=True))
  rp=b.immutable(FIX/'revision.json',revision);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(rp)
  index['audit_task_paths']['summary']=str(J/ns.DIAGNOSTIC/'tasks'/index['sysu_task_revision']/'summarize_diagnostic_handoff1.json')
  index['summary_template']['run_id']+='__handoff1'
  b.immutable(FIX/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json')
  state.update(index=ib,manager=None,status='running',reason='Preserve actual final verifier; canonical finish then flush before strict final audit',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='summary_handoff_flush_20261028',revision=index['numeric_acceptance'],accepted_reused=47,main_reused=48,pending_successful_verifier_reused=True,model_forwards=0))
  b.atomic(D/'status.json',b.raw(state))
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v14.json');handoff['diagnostic_index']=ib
  hp=b.immutable(control/'budget_handoff_summary_handoff1.json',handoff)
  template=b.read(control/'template_numeric_v14.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance'])
  tp=b.immutable(control/'template_summary_handoff1.json',template)
  dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_summary_handoff1.json',dep)
  # This existing pilot task is used only to validate the namespace/entry metadata.
  # No pilot/scientific worker is executed or re-admitted by this preparation.
  task_path=b.mapped(index['audit_task_paths']['pilot_admission']);config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped)
  nonce=uuid.uuid4().hex;proofout=control/'config_versions'/('summary_handoff1_validation_'+nonce+'.json');validator=R/'prepare_runtime.py'
  req=b.immutable(FIX/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proofout)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True)
  va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
  ma=ns.build_manager_bwrap(config,b.bind(dp),version);ma[ma.index(b.logical(D/'implementation/manager.py'))]=b.logical(R/'final_cost_guard_v15.py')
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Metadata cutover changed ledger')
  if before['tasks']!=index['tasks'] or before['implementation_sha256']!=index['implementation_sha256'] or before['protocol']!=index['protocol'] or before['budget_policy']!=index['budget_policy']:raise ValueError('Scientific or budget binding changed')
  result=dict(schema='spnv2-final-audit-handoff-preparation-v1',published_code_commit=commit,index=ib,revision=index['numeric_acceptance'],deployment=b.bind(dp),config=version,validation_nonce=nonce,validation_output=b.logical(proofout),validation_argv=va,manager_argv=ma,screen_socket_dir=config['screen_socket_dir'],accepted_reused=47,main_reused=48,successful_last_verifier_reused=True,scientific_policy=revision['policy'],model_forwards=0,ledger_prefix=dict(bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()))
  b.immutable(FIX/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['validation_argv','manager_argv']}))
if __name__=='__main__':main(sys.argv[1])
