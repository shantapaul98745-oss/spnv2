"""Apply only the explicitly authorized BN channel-mean atol1e-5."""
import sys,json,copy,hashlib,time,uuid,pathlib
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
R=U/'spnv2/reproduction/numeric_acceptance_20261027';REV=D/'revisions/numeric_acceptance_20261027';V5=REV/'v5_BN_mean_atol'
def main(commit):
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!='43537b1d3aa49f6e7a61a159425323590d3ea36bfc27668f6b333d0a405d80a5':raise ValueError('Exact BN-statistics baseline changed')
  state=b.read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0023'
  if state['status']!='blocked' or active['nonce']!='c29aa1c640cd4aaa89b48fda4afd9214' or active.get('exit_code')!=1 or 'channel_mean' not in active['outcome'].get('error','') or len(state['accepted'])!=27 or len(state['main'])!=27:raise ValueError('Only the approved BN-mean stop can resume')
  if (REV/'v4_bn_statistics/manager.log.exitcode').read_text().strip()!='1':raise ValueError('Old manager exit unresolved')
  before=b.read(D/'index.json');oldrev=b.read(b.mapped(before['numeric_acceptance']['path']));oldmanifest=b.read(b.mapped(oldrev['saved_main_recovery']['manifest']['path']));manifest=copy.deepcopy(oldmanifest)
  stats_path=D/'launches/d0aa0247305f4abab989e383583506d1/result.json';stats=b.read(stats_path)
  if stats['model_forwards']!=0 or stats['actual_checkpoint']!=manifest['actual_checkpoint'] or stats['summary']['channel_mean']['max_abs']>=1e-5 or stats['summary']['channel_mean_square']['max_current_ratio']>1.:raise ValueError('Saved all-cache statistical evidence does not justify the exact approved bound')
  for path in [D/'index.json',D/'status.json']:b.immutable_bytes(V5/'before'/path.relative_to(D),path.read_bytes())
  past=[x for x in state['attempts'] if x['run_id']==rid];known={x['nonce'] for x in manifest['failed_attempts']}|{active['nonce']}
  if {x['nonce'] for x in past}!=known:raise ValueError('Unknown historical failure prevents recovery')
  failure=[]
  for row in past:
   directory=b.mapped(row['launch_directory']);outcome=b.read(directory/'outcome.json');launch=b.read(directory/'launch.json');exitfile=pathlib.Path(str(b.mapped(row['log_path']))+'.exitcode')
   if row['exit_code']!=1 or outcome!=row['outcome'] or exitfile.read_text().strip()!='1' or launch['run_nonce']!=row['nonce'] or row['task_sha256']!=manifest['task']['sha256']:raise ValueError('Old actual failure evidence changed')
   failure.append(dict(nonce=row['nonce'],task_sha256=row['task_sha256'],launch_directory=row['launch_directory'],launch=b.bind(directory/'launch.json'),outcome=b.bind(directory/'outcome.json'),exit_code=b.bind(exitfile)))
  journal=(D/'budget/journal.jsonl').read_bytes();source=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes()
  if not journal.startswith(source):raise ValueError('Source ledger prefix changed')
  events=[json.loads(x) for x in journal.splitlines()];started={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if started-closed:raise ValueError('Outstanding allocation prevents continuation')
  manifest.update(failed_attempts=failure,ledger_prefix=dict(path=b.logical(D/'budget/journal.jsonl'),bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()),additional_zero_forward_finalizer_failure_preserved=active['nonce'])
  mp=b.immutable(V5/'recovery_manifest.json',manifest);rev=copy.deepcopy(oldrev);rev.update(published_code_commit=commit,policy=b.bind(REV/'policy_v5.json'),comparator=b.bind(R/'numeric_compare.py'),supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(V5/'before/index.json'),saved_main_recovery=dict(run_id=rid,manifest=b.bind(mp)),BN_mean_only_authorization=b.bind(V5/'authorization.json'))
  rp=b.immutable(REV/'revision_v5.json',rev);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(rp);b.immutable(V5/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json')
  state.update(index=ib,status='between_tasks',active=None,manager=None,reason='explicit_BN_mean_atol_1e-5_resume_saved_work',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='BN_mean_only_atol_20261028',revision=index['numeric_acceptance'],accepted_reused=27,all_previous_failures_and_costs_preserved=True,no_model_replay=True));b.atomic(D/'status.json',b.raw(state))
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v4.json');handoff['diagnostic_index']=ib;hp=b.immutable(control/'budget_handoff_numeric_v5.json',handoff);template=b.read(control/'template_numeric_v4.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_numeric_v5.json',template);dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_numeric_v5.json',dep)
  task=b.mapped(manifest['task']['path']);config=dc.compose_single_task(b.bind(dp),b.bind(task),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_v5_validation_'+nonce+'.json');validator=R/'prepare_runtime.py';req=b.immutable(V5/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task),output=b.logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Budget changed during metadata revision')
  result=dict(index=ib,revision=index['numeric_acceptance'],policy=rev['policy'],config=version,deployment=b.bind(dp),validation_nonce=nonce,validation_output=b.logical(proof),validation_argv=va,manager_argv=ns.build_manager_bwrap(config,b.bind(dp),version),screen_socket_dir=config['screen_socket_dir'],accepted_reused=27,model_forwards=0,saved_main_recovery=b.bind(mp),published_code_commit=commit);b.immutable(V5/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))
if __name__=='__main__':
 if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise SystemExit('Explicit published commit required')
 main(sys.argv[1])
