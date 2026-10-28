"""Apply fixed0.01degree rotation acceptance; resume only failed w54."""
import sys,json,copy,hashlib,uuid,pathlib,os
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
R=U/'spnv2/reproduction/numeric_acceptance_20261027';REV=D/'revisions/numeric_acceptance_20261027';V8=REV/'v8_rotation_0_01_degree'
def main(commit):
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!='2b7357bb2998e7df296e4c78ae8966c9ad6686299ab786f786a0e5b79ea32161':raise ValueError('Exact V7 baseline changed')
  state=b.read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0054'
  if state['status']!='blocked' or active['nonce']!='968ebcce7bc3495fb80241059a92d9c7' or active.get('exit_code')!=1 or active['outcome']['error_type']!='TraceMismatch' or len(state['accepted'])!=29 or len(state['main'])!=29:raise ValueError('Not the specifically authorized geometry stop')
  if (REV/'v7_gate_score_self_consistency/manager.log.exitcode').read_text().strip()!='1':raise ValueError('Manager actual exit missing')
  before=b.read(D/'index.json');row=next(x for x in before['tasks'] if x['run_id']==rid);task_path=b.mapped(row['task_path']);task=b.read(task_path);output=D/'pairs'/rid
  if rid in state['main'] or (output/'report.json').exists():raise ValueError('Completed main cannot replay')
  import numeric_compare as numeric
  native_policy=copy.deepcopy(b.read(REV/'policy_v8.json'))
  for key in ('derived_gate_code','derived_gate_calibration'):native_policy[key]['path']=str(b.mapped(native_policy[key]['path']))
  numeric.configure(native_policy);failure=b.read(output/'failure.json');diff=failure['trace_difference']
  if not numeric.equal(diff['actual'],diff['expected']):raise ValueError('Saved event fails authorized policy: '+repr(numeric.LAST_FAILURE))
  current=b.read(output/'isolated_durable/CURRENT');source=output/'isolated_durable/checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=b.read(source/'metadata.json')
  cursors=task['selection']['parent_checkpoint']['cursors']
  if cursors['next_index']!=448:raise ValueError('Frozen parent boundary changed')
  if meta['scientific_cursors']!=cursors or b.sha(source/'metadata.json')!=current['metadata_sha256']:raise ValueError('Saved complete448 checkpoint changed')
  # Immutable hardlinks protect the complete saved recovery point from rolling
  # pruning without copying or modifying its state, metadata or checksum.
  preserved=V8/'saved_complete_checkpoint'
  preserved.mkdir(exist_ok=True)
  for name in ('state.bin','metadata.json','metadata.sha256'):
   dest=preserved/name
   if dest.exists():
    if not os.path.samefile(dest,source/name):raise ValueError('Unknown preserved recovery point')
   else:os.link(source/name,dest)
  artifacts={}
  if any((output/name).exists() for name in ('candidate_learner.bin','candidate_binding.json','progress_accept.json','progress_rollback.json')):raise ValueError('Unexpected persisted candidate/panel; do not replay without an explicit artifact binding')
  past=[x for x in state['attempts'] if x['run_id']==rid]
  if len(past)!=1 or past[0]['nonce']!=active['nonce']:raise ValueError('Unaccounted prior w54 attempt')
  failed=[]
  for x in past:
   directory=b.mapped(x['launch_directory']);outcome=b.read(directory/'outcome.json');launch=b.read(directory/'launch.json');exitfile=pathlib.Path(str(b.mapped(x['log_path']))+'.exitcode')
   if x['exit_code']!=1 or outcome!=x['outcome'] or exitfile.read_text().strip()!='1' or launch['run_nonce']!=x['nonce'] or x['task_sha256']!=row['task_sha256']:raise ValueError('Actual failure lineage changed')
   failed.append(dict(nonce=x['nonce'],task_sha256=x['task_sha256'],launch_directory=x['launch_directory'],launch=b.bind(directory/'launch.json'),outcome=b.bind(directory/'outcome.json'),exit_code=b.bind(exitfile)))
  journal=(D/'budget/journal.jsonl').read_bytes();source_journal=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes()
  if not journal.startswith(source_journal):raise ValueError('Original budget prefix changed')
  events=[json.loads(x) for x in journal.splitlines()];starts={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if starts-closed:raise ValueError('Unclosed attempt blocks continuation')
  for path in [D/'index.json',D/'status.json',output/'failure.json',output/'isolated_durable/CURRENT']:
   b.immutable_bytes(V8/'before'/path.relative_to(D),path.read_bytes())
  manifest=dict(schema='spnv2-partial-saved-main-recovery-v1',run_id=rid,task=dict(path=row['task_path'],sha256=row['task_sha256']),actual_checkpoint=dict(directory=b.logical(preserved),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256']),saved_cursors=cursors,original_checkpoint_directory=b.logical(source),artifacts=artifacts,failed_attempts=failed,ledger_prefix=dict(path=b.logical(D/'budget/journal.jsonl'),bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()),source_execution=dict(nonce=active['nonce'],actual_exit_code=1,no_old_exit_rewritten=True),recompute_only_unpersisted_tail=True,no_preexisting_panels=True,panel_evaluation_pending=True)
  mp=b.immutable(V8/'partial_recovery_manifest.json',manifest);oldrev=b.read(b.mapped(before['numeric_acceptance']['path']));revision=copy.deepcopy(oldrev)
  revision.update(published_code_commit=commit,policy=b.bind(REV/'policy_v8.json'),comparator=b.bind(R/'numeric_compare.py'),overrides={name:b.bind(R/(name+'.py')) for name in ('pair_runner','verify_pair')},recovery_code=b.bind(R/'recover_saved_main.py'),partial_main_recovery=dict(run_id=rid,manifest=b.bind(mp)),supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(V8/'before/index.json'),rotation_angular_authorization='explicit_user_reply_2026-10-28_rotation_related_fixed_0_01_degree',existing_accepted_pairs_reused=29)
  rp=b.immutable(REV/'revision_v8.json',revision);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(rp);b.immutable(V8/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json')
  b.immutable_bytes(V8/'archived_w54_failure.json',(output/'failure.json').read_bytes());(output/'failure.json').unlink()
  state.update(index=ib,status='between_tasks',active=None,manager=None,reason='authorized_rotation_fixed_0_01_degree_resume_w54',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='rotation_fixed_0_01_degree_20261028',revision=index['numeric_acceptance'],accepted_reused=29,all_failures_and_costs_preserved=True,saved_panels_reused=False));b.atomic(D/'status.json',b.raw(state))
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v7.json');handoff['diagnostic_index']=ib;hp=b.immutable(control/'budget_handoff_numeric_v8.json',handoff);template=b.read(control/'template_numeric_v7_pinned_receipt.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_numeric_v8.json',template);dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_numeric_v8.json',dep)
  config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_v8_validation_'+nonce+'.json');validator=R/'prepare_runtime.py';req=b.immutable(V8/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Metadata revision altered original budget')
  result=dict(index=ib,revision=index['numeric_acceptance'],policy=revision['policy'],config=version,deployment=b.bind(dp),validation_nonce=nonce,validation_output=b.logical(proof),validation_argv=va,manager_argv=ns.build_manager_bwrap(config,b.bind(dp),version),screen_socket_dir=config['screen_socket_dir'],accepted_reused=29,model_forwards=0,partial_main_recovery=b.bind(mp),published_code_commit=commit);b.immutable(V8/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))
if __name__=='__main__':
 if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise SystemExit('Published commit required')
 main(sys.argv[1])
