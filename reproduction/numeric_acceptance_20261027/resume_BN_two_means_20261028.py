"""Apply two explicitly authorized BN-mean bounds and finalize saved w54 work."""
import pathlib,json,copy,hashlib,sys,uuid,os
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
R=U/'spnv2/reproduction/numeric_acceptance_20261027';REV=D/'revisions/numeric_acceptance_20261027';V9=REV/'v9_BN_two_mean_tolerances'
def main(commit):
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!='a7460da6b7f717a91d1b3b0f549a184aa466163b5642ecc8f8b630001bdacbc0':raise ValueError('Exact V8 scientific baseline changed')
  state=b.read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0054'
  if state['status']!='blocked' or active['nonce']!='3d2a29a7f22d4fe48f23696be6114253' or active.get('exit_code')!=1 or active['outcome'].get('error')!='full model/optimizer/gradient/RNG/pool/provider scientific checkpoint differs' or len(state['accepted'])!=29 or len(state['main'])!=29:raise ValueError('Not the specifically authorized complete saved-state stop')
  if (REV/'v8_rotation_0_01_degree/manager.log.exitcode').read_text().strip()!='1':raise ValueError('Prior manager actual exit missing')
  proposal_path=REV/'v8_rotation_0_01_degree/w54_BN_mean_proposal.json';proposal=b.read(proposal_path)
  if proposal['affected_fields']!=['learner.learning.bn_mean','learner.window_base.bn_features'] or not proposal['all_other_categories_pass'] or not proposal['cached_mean_square_separately_confirmed'] or proposal['accumulator_worst_proposed_ratio']>1. or proposal['cache_mean_worst_proposed_ratio']>1.:raise ValueError('Saved complete category evidence changed')
  for ev in proposal['evidence']:
   for key in ('result','outcome','budget_closed'):
    item=ev[key];path=pathlib.Path(item['path'])
    if not path.is_relative_to(U) or b.sha(path)!=item['sha256']:raise ValueError('Saved CPU diagnosis binding changed')
   outcome=b.read(ev['outcome']['path'])
   if outcome['exit_code']!=0 or outcome['model_forwards']!=0:raise ValueError('Saved diagnosis was not a real zero-forward success')
  before=b.read(D/'index.json');row=next(x for x in before['tasks'] if x['run_id']==rid);task_path=b.mapped(row['task_path']);task=b.read(task_path);output=D/'pairs'/rid
  if rid in state['main'] or (output/'report.json').exists():raise ValueError('Completed producer must not replay')
  current=b.read(output/'isolated_durable/CURRENT');directory=output/'isolated_durable/checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=b.read(directory/'metadata.json')
  if meta['scientific_cursors']!=task['selection']['match_checkpoint']['cursors'] or meta['scientific_cursors']['next_index']!=512 or meta['state_sha256']!=proposal['actual_checkpoint']['state_sha256'] or current['metadata_sha256']!=b.sha(directory/'metadata.json'):raise ValueError('Saved complete512 checkpoint changed')
  for path in (D/'index.json',D/'status.json',output/'failure.json',output/'isolated_durable/CURRENT'):b.immutable_bytes(V9/'before'/path.relative_to(D),path.read_bytes())
  preserved=V9/'saved_complete_checkpoint';preserved.mkdir(exist_ok=True)
  for name in ('state.bin','metadata.json','metadata.sha256'):
   dest=preserved/name
   if dest.exists():
    if not os.path.samefile(dest,directory/name):raise ValueError('Unknown saved checkpoint alias')
   else:os.link(directory/name,dest)
  artifacts=dict(candidate=b.read(output/'candidate_binding.json'),candidate_binding=b.bind(output/'candidate_binding.json'),progress_accept=b.bind(output/'progress_accept.json'),progress_rollback=b.bind(output/'progress_rollback.json'))
  for branch in ('accept','rollback'):
   doc=b.read(output/('progress_'+branch+'.json'))
   if len(doc['rows'])!=128 or doc['task_sha256']!=row['task_sha256'] or doc['candidate_learner']!=artifacts['candidate']:raise ValueError('Saved128 panel/task/candidate identity changed')
  past=[x for x in state['attempts'] if x['run_id']==rid];expected={'968ebcce7bc3495fb80241059a92d9c7','3d2a29a7f22d4fe48f23696be6114253'}
  if {x['nonce'] for x in past}!=expected:raise ValueError('Historical failed pair attempts not exactly known')
  failed=[]
  for x in past:
   root=b.mapped(x['launch_directory']);outcome=b.read(root/'outcome.json');launch=b.read(root/'launch.json');exitfile=pathlib.Path(str(b.mapped(x['log_path']))+'.exitcode')
   if x['exit_code']!=1 or outcome!=x['outcome'] or outcome['exit_code']!=1 or exitfile.read_text().strip()!='1' or launch['run_nonce']!=x['nonce'] or x['task_sha256']!=row['task_sha256']:raise ValueError('Actual historical failed pair identity changed')
   failed.append(dict(nonce=x['nonce'],task_sha256=x['task_sha256'],launch_directory=x['launch_directory'],launch=b.bind(root/'launch.json'),outcome=b.bind(root/'outcome.json'),exit_code=b.bind(exitfile)))
  journal=(D/'budget/journal.jsonl').read_bytes();prefix=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes()
  if not journal.startswith(prefix):raise ValueError('Original budget prefix changed')
  events=[json.loads(x) for x in journal.splitlines()];starts={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'};reserved={e['payload']['token'] for e in events if e['kind']=='forward_reserved'};confirmed={e['payload']['token'] for e in events if e['kind']=='forward_confirmed'}
  if starts-closed or reserved-confirmed:raise ValueError('Outstanding allocation prevents safe finalization')
  oldrev=b.read(b.mapped(before['numeric_acceptance']['path']));archive=R/'versions/v8/pair_runner.py'
  if b.sha(archive)!=oldrev['overrides']['pair_runner']['sha256']:raise ValueError('Old executed branch/isolation code archive changed')
  source_execution=dict(nonce=active['nonce'],launch=b.bind(b.mapped(active['launch_directory'])/'launch.json'),outcome=b.bind(b.mapped(active['launch_directory'])/'outcome.json'),actual_exit_code=1,reached_error='full model/optimizer/gradient/RNG/pool/provider scientific checkpoint differs',source_index=b.bind(V9/'before/index.json'),source_pair_runner_archive=b.bind(archive),isolation_checks_precede_persisted_progress=True,model_work_reused_not_rerun=True)
  manifest=dict(schema='spnv2-complete-saved-main-recovery-v1',run_id=rid,task=dict(path=row['task_path'],sha256=row['task_sha256']),actual_checkpoint=dict(directory=b.logical(preserved),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256']),artifacts=artifacts,failed_attempts=failed,ledger_prefix=dict(path=b.logical(D/'budget/journal.jsonl'),bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()),source_execution=source_execution,no_old_exit_code_rewritten=True,no_new_model_forward_for_recovery=True)
  mp=b.immutable(V9/'recovery_manifest.json',manifest)
  # Create the immutable alias natively before namespace read-only revision
  # mounts would make a cross-mount link fail despite the same underlying disk.
  alias=output/'matched_scientific_state.bin';source_state=preserved/'state.bin'
  if alias.exists():
   if not os.path.samefile(alias,source_state):raise ValueError('Unknown matched-state alias exists')
  else:os.link(source_state,alias)
  revision=copy.deepcopy(oldrev);revision.pop('partial_main_recovery',None)
  revision.update(published_code_commit=commit,policy=b.bind(REV/'policy_v9.json'),comparator=b.bind(R/'numeric_compare.py'),overrides={name:b.bind(R/(name+'.py')) for name in ('pair_runner','verify_pair')},supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(V9/'before/index.json'),recovery_code=b.bind(R/'recover_saved_main.py'),saved_main_recovery=dict(run_id=rid,manifest=b.bind(mp)),two_BN_mean_authorization='explicit user reply: window mean accumulator atol1e-5 and cached channel mean atol2e-5; rtol1e-5 and all other rules unchanged',existing_accepted_pairs_reused=29)
  rp=b.immutable(REV/'revision_v9.json',revision);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(rp);b.immutable(V9/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json')
  b.immutable_bytes(V9/'archived_w54_failure.json',(output/'failure.json').read_bytes());(output/'failure.json').unlink();state.update(index=ib,status='between_tasks',active=None,manager=None,reason='authorized_two_BN_mean_bounds_finalize_saved_w54',next_resource_check=0.);state.setdefault('technical_revisions',[]).append(dict(name='two_BN_mean_tolerances_20261028',revision=index['numeric_acceptance'],accepted_reused=29,old_failures_and_costs_preserved=True,no_model_or_panel_replay=True));b.atomic(D/'status.json',b.raw(state))
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v8.json');handoff['diagnostic_index']=ib;hp=b.immutable(control/'budget_handoff_numeric_v9.json',handoff);template=b.read(control/'template_numeric_v8.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_numeric_v9.json',template);dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_numeric_v9.json',dep)
  config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_v9_validation_'+nonce+'.json');validator=R/'prepare_runtime.py';req=b.immutable(V9/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proof)));python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Metadata revision altered continuous budget')
  result=dict(index=ib,revision=index['numeric_acceptance'],policy=revision['policy'],config=version,deployment=b.bind(dp),validation_nonce=nonce,validation_output=b.logical(proof),validation_argv=va,manager_argv=ns.build_manager_bwrap(config,b.bind(dp),version),screen_socket_dir=config['screen_socket_dir'],accepted_reused=29,model_forwards=0,saved_main_recovery=b.bind(mp),published_code_commit=commit);b.immutable(V9/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))
if __name__=='__main__':
 if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise SystemExit('Explicit published commit required')
 main(sys.argv[1])
