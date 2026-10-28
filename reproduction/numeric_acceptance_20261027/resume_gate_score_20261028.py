"""Apply only explicitly authorized derived G50 gate-score self-consistency."""
import sys,json,copy,hashlib,uuid,pathlib,os
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
R=U/'spnv2/reproduction/numeric_acceptance_20261027';REV=D/'revisions/numeric_acceptance_20261027';V7=REV/'v7_gate_score_self_consistency'
def main(commit):
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!='2ed51c643e5c826c2c5a72048773d28e8f535df5a2c608a851b697d890c7d513':raise ValueError('Exact V5 baseline changed')
  state=b.read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0031'
  if state['status']!='blocked' or active['nonce']!='13aba9dafd9345039ee78876179dcad2' or active.get('exit_code')!=1 or active['outcome']['error_type']!='TraceMismatch' or len(state['accepted'])!=28 or len(state['main'])!=28:raise ValueError('Not the specifically authorized geometry stop')
  if (REV/'v6_geometry_norm/manager.log.exitcode').read_text().strip()!='1':raise ValueError('Manager actual exit missing')
  before=b.read(D/'index.json');row=next(x for x in before['tasks'] if x['run_id']==rid);task_path=b.mapped(row['task_path']);task=b.read(task_path);output=D/'pairs'/rid
  if rid in state['main'] or (output/'report.json').exists():raise ValueError('Completed main cannot replay')
  import numeric_compare as numeric
  native_policy=copy.deepcopy(b.read(REV/'policy_v7.json'))
  for key in ('derived_gate_code','derived_gate_calibration'):native_policy[key]['path']=str(b.mapped(native_policy[key]['path']))
  numeric.configure(native_policy);failure=b.read(output/'failure.json');diff=failure['trace_difference']
  if not numeric.equal(diff['actual'],diff['expected']):raise ValueError('Saved event fails authorized policy: '+repr(numeric.LAST_FAILURE))
  current=b.read(output/'isolated_durable/CURRENT');source=output/'isolated_durable/checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=b.read(source/'metadata.json')
  cursors=dict(candidates=31,commits=8,next_index=283,observed=283,rollbacks=23)
  if meta['scientific_cursors']!=cursors or b.sha(source/'metadata.json')!=current['metadata_sha256']:raise ValueError('Saved complete283 checkpoint changed')
  # Immutable hardlinks protect the complete saved recovery point from rolling
  # pruning without copying or modifying its state, metadata or checksum.
  preserved=V7/'saved_complete_checkpoint'
  preserved.mkdir(exist_ok=True)
  for name in ('state.bin','metadata.json','metadata.sha256'):
   dest=preserved/name
   if dest.exists():
    if not os.path.samefile(dest,source/name):raise ValueError('Unknown preserved recovery point')
   else:os.link(source/name,dest)
  artifacts=dict(candidate=b.read(output/'candidate_binding.json'),candidate_binding=b.bind(output/'candidate_binding.json'),progress_accept=b.bind(output/'progress_accept.json'),progress_rollback=b.bind(output/'progress_rollback.json'))
  for branch in ('accept','rollback'):
   doc=b.read(output/('progress_'+branch+'.json'))
   if len(doc['rows'])!=128 or doc['task_sha256']!=row['task_sha256'] or doc['candidate_learner']!=artifacts['candidate']:raise ValueError('Persisted panel identity changed')
  past=[x for x in state['attempts'] if x['run_id']==rid]
  if len(past)!=2 or {x['nonce'] for x in past}!={'91581c54d91148939048b5c8c5ca5369',active['nonce']}:raise ValueError('Unaccounted prior w31 attempts')
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
   b.immutable_bytes(V7/'before'/path.relative_to(D),path.read_bytes())
  manifest=dict(schema='spnv2-partial-saved-main-recovery-v1',run_id=rid,task=dict(path=row['task_path'],sha256=row['task_sha256']),actual_checkpoint=dict(directory=b.logical(preserved),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256']),saved_cursors=cursors,original_checkpoint_directory=b.logical(source),artifacts=artifacts,failed_attempts=failed,ledger_prefix=dict(path=b.logical(D/'budget/journal.jsonl'),bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()),source_execution=dict(nonce=active['nonce'],actual_exit_code=1,no_old_exit_rewritten=True),recompute_only_unpersisted_tail=True,no_saved_panel_model_replay=True)
  mp=b.immutable(V7/'partial_recovery_manifest.json',manifest);oldrev=b.read(b.mapped(before['numeric_acceptance']['path']));revision=copy.deepcopy(oldrev)
  revision.update(published_code_commit=commit,policy=b.bind(REV/'policy_v7.json'),comparator=b.bind(R/'numeric_compare.py'),overrides={name:b.bind(R/(name+'.py')) for name in ('pair_runner','verify_pair')},recovery_code=b.bind(R/'recover_saved_main.py'),partial_main_recovery=dict(run_id=rid,manifest=b.bind(mp)),supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(V7/'before/index.json'),derived_gate_authorization='explicit_user_reply_2026-10-28_only_derived_gate_score_self_consistency',existing_accepted_pairs_reused=28)
  rp=b.immutable(REV/'revision_v7.json',revision);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(rp);b.immutable(V7/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json')
  b.immutable_bytes(V7/'archived_w31_failure.json',(output/'failure.json').read_bytes());(output/'failure.json').unlink()
  state.update(index=ib,status='between_tasks',active=None,manager=None,reason='authorized_derived_gate_score_self_consistency_resume_saved_tail',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='derived_gate_score_self_consistency_20261028',revision=index['numeric_acceptance'],accepted_reused=28,all_failures_and_costs_preserved=True,saved_panels_reused=True));b.atomic(D/'status.json',b.raw(state))
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v6.json');handoff['diagnostic_index']=ib;hp=b.immutable(control/'budget_handoff_numeric_v7.json',handoff);template=b.read(control/'template_numeric_v6.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_numeric_v7.json',template);dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_numeric_v7.json',dep)
  config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_v7_validation_'+nonce+'.json');validator=R/'prepare_runtime.py';req=b.immutable(V7/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Metadata revision altered original budget')
  result=dict(index=ib,revision=index['numeric_acceptance'],policy=revision['policy'],config=version,deployment=b.bind(dp),validation_nonce=nonce,validation_output=b.logical(proof),validation_argv=va,manager_argv=ns.build_manager_bwrap(config,b.bind(dp),version),screen_socket_dir=config['screen_socket_dir'],accepted_reused=28,model_forwards=0,partial_main_recovery=b.bind(mp),published_code_commit=commit);b.immutable(V7/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))
if __name__=='__main__':
 if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise SystemExit('Published commit required')
 main(sys.argv[1])
