"""Apply the explicit BN-cache statistics approval and recover persisted w23 work."""
import pathlib,json,hashlib,copy,sys,time,uuid
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
R=U/'spnv2/reproduction/numeric_acceptance_20261027';REV=D/'revisions/numeric_acceptance_20261027';V4=REV/'v4_bn_statistics'
def main(commit):
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!='12cd5da3b6ad9b8619bc19cd4ecc161f2e0ae46f014edc15916ed227b6ae5004':raise ValueError('BN amendment baseline changed')
  state=b.read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0023'
  if state['status']!='blocked' or active['nonce']!='e80ee9706c904b4091898253dddf16df' or active.get('exit_code')!=1 or active['outcome'].get('error')!='full model/optimizer/gradient/RNG/pool/provider scientific checkpoint differs' or len(state['accepted'])!=27 or len(state['main'])!=27:raise ValueError('Not the specifically approved BN-cache stop')
  if (REV/'v3_derived/manager.log.exitcode').read_text().strip()!='1':raise ValueError('Old manager exit is unresolved')
  diagnostic=D/'launches/5f108c3df92d4b2bb30245a5eb23a832/result.json';diagnosis=b.read(diagnostic)
  if diagnosis['model_forwards']!=0 or diagnosis['diagnostic_excluding_bn_features']['all_remaining_fields_pass_current_policy'] is not True:raise ValueError('Saved non-BN state diagnosis invalid')
  before=b.read(D/'index.json');row=next(x for x in before['tasks'] if x['run_id']==rid);task=b.read(b.mapped(row['task_path']));output=D/'pairs'/rid
  if rid in state['main'] or (output/'report.json').exists():raise ValueError('Completed main must not replay')
  for path in [D/'index.json',D/'status.json',output/'failure.json']:
   b.immutable_bytes(V4/'before'/path.relative_to(D),path.read_bytes())
  current=b.read(output/'isolated_durable/CURRENT');directory=output/'isolated_durable/checkpoints'/('generation-'+str(current['generation']).zfill(20));meta=b.read(directory/'metadata.json')
  if meta['scientific_cursors']!=task['selection']['match_checkpoint']['cursors'] or meta['state_sha256']!=diagnosis['actual_checkpoint']['state_sha256'] or current['metadata_sha256']!=b.sha(directory/'metadata.json'):raise ValueError('Saved matching recovery point changed')
  artifacts={'candidate':b.read(output/'candidate_binding.json'),'candidate_binding':b.bind(output/'candidate_binding.json'),'progress_accept':b.bind(output/'progress_accept.json'),'progress_rollback':b.bind(output/'progress_rollback.json')}
  for branch in ('accept','rollback'):
   doc=b.read(output/('progress_'+branch+'.json'))
   if len(doc['rows'])!=128 or doc['task_sha256']!=row['task_sha256'] or doc['candidate_learner']!=artifacts['candidate']:raise ValueError('Persisted branch identity/count changed')
  expected_nonces={'1f1b09d799f24328962e875eb4fad45f','1a0172d029da42ffaba702672cad2986','c3d81307bbba454aac4dfc0be91dd552','0c6be39dec624522acd9aea2de04d124','f41d56a3e1b9499d812ed952d34a4c7b','e80ee9706c904b4091898253dddf16df'}
  past=[x for x in state['attempts'] if x['run_id']==rid]
  if {x['nonce'] for x in past}!=expected_nonces:raise ValueError('Historical failed attempts not exactly known')
  failed=[]
  for x in past:
   directory_x=b.mapped(x['launch_directory']);outcome=b.read(directory_x/'outcome.json');launch=b.read(directory_x/'launch.json');exitfile=pathlib.Path(str(b.mapped(x['log_path']))+'.exitcode')
   if x['exit_code']!=1 or outcome!=x['outcome'] or outcome['exit_code']!=1 or exitfile.read_text().strip()!='1' or launch['run_nonce']!=x['nonce'] or x['task_sha256']!=row['task_sha256']:raise ValueError('Actual previous failure lineage changed')
   failed.append(dict(nonce=x['nonce'],task_sha256=x['task_sha256'],launch_directory=x['launch_directory'],launch=b.bind(directory_x/'launch.json'),outcome=b.bind(directory_x/'outcome.json'),exit_code=b.bind(exitfile)))
  journal=(D/'budget/journal.jsonl').read_bytes();source=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes()
  if not journal.startswith(source):raise ValueError('Original budget prefix changed')
  events=[json.loads(x) for x in journal.splitlines()];started={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if started-closed:raise ValueError('Outstanding computation prevents safe continuation')
  oldrev=b.read(b.mapped(before['numeric_acceptance']['path']));archive=R/'versions/v3/pair_runner.py'
  if b.sha(archive)!=oldrev['overrides']['pair_runner']['sha256']:raise ValueError('Original executed override archive changed')
  source_execution=dict(nonce=active['nonce'],launch=b.bind(b.mapped(active['launch_directory'])/'launch.json'),outcome=b.bind(b.mapped(active['launch_directory'])/'outcome.json'),actual_exit_code=1,reached_error='full model/optimizer/gradient/RNG/pool/provider scientific checkpoint differs',source_index=b.bind(V4/'before/index.json'),source_pair_runner_archive=b.bind(archive),isolation_checks_precede_persisted_progress=True,model_work_reused_not_rerun=True)
  manifest=dict(schema='spnv2-complete-saved-main-recovery-v1',run_id=rid,task=dict(path=row['task_path'],sha256=row['task_sha256']),actual_checkpoint=dict(directory=b.logical(directory),metadata_sha256=current['metadata_sha256'],state_sha256=meta['state_sha256']),artifacts=artifacts,failed_attempts=failed,ledger_prefix=dict(path=b.logical(D/'budget/journal.jsonl'),bytes=len(journal),sha256=hashlib.sha256(journal).hexdigest()),source_execution=source_execution,BN_acceptance_authorized_only=True,no_old_exit_code_rewritten=True,no_new_model_forward_for_recovery=True)
  mp=b.immutable(V4/'recovery_manifest.json',manifest)
  # The manifest protects the source generation before publishing this alias.
  import os
  alias=output/'matched_scientific_state.bin';source_state=directory/'state.bin'
  if alias.exists():
   if not os.path.samefile(alias,source_state):raise ValueError('Unknown matched-state output already exists')
  else:os.link(source_state,alias)
  rev=copy.deepcopy(oldrev);rev.update(published_code_commit=commit,policy=b.bind(REV/'policy_v4.json'),comparator=b.bind(R/'numeric_compare.py'),overrides={name:b.bind(R/(name+'.py')) for name in ('pair_runner','verify_pair')},supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(V4/'before/index.json'),bn_cache_authorization='explicit user reply: BN channel means and mean squares under unchanged atol1e-6/rtol1e-5; all other acceptance and exact internal restoration unchanged',recovery_code=b.bind(R/'recover_saved_main.py'),saved_main_recovery=dict(run_id=rid,manifest=b.bind(mp)))
  rp=b.immutable(REV/'revision_v4.json',rev);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(rp);b.immutable(V4/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));ib=b.bind(D/'index.json')
  b.immutable_bytes(V4/'archived_failure.json',(output/'failure.json').read_bytes());(output/'failure.json').unlink()
  state.update(index=ib,status='between_tasks',active=None,manager=None,reason='explicit_BN_statistics_approval_reuse_saved_model_work',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='BN_cache_statistics_and_saved_main_finalization',revision=index['numeric_acceptance'],accepted_reused=27,source_failed_work_and_costs_preserved=True,model_replay_not_required=True));b.atomic(D/'status.json',b.raw(state))
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_v3.json');handoff['diagnostic_index']=ib;hp=b.immutable(control/'budget_handoff_numeric_v4.json',handoff);template=b.read(control/'template_numeric_v3.json');template.update(diagnostic_index=ib,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_numeric_v4.json',template);dep=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=b.bind(tp),index=ib,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt']);dp=b.immutable(control/'deployment_numeric_v4.json',dep)
  task_path=b.mapped(row['task_path']);config=dc.compose_single_task(b.bind(dp),b.bind(task_path),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped);nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_v4_validation_'+nonce+'.json');validator=R/'prepare_runtime.py';req=b.immutable(V4/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=nonce,validator=b.bind(validator),config=version,task=b.bind(task_path),output=b.logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(req),'--request-sha256',b.sha(req)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Budget mutated in metadata preparation')
  result=dict(index=ib,revision=index['numeric_acceptance'],policy=rev['policy'],config=version,deployment=b.bind(dp),validation_nonce=nonce,validation_output=b.logical(proof),validation_argv=va,manager_argv=ns.build_manager_bwrap(config,b.bind(dp),version),screen_socket_dir=config['screen_socket_dir'],accepted_reused=27,model_forwards=0,saved_main_recovery=b.bind(mp),published_code_commit=commit);b.immutable(V4/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))
if __name__=='__main__':
 if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise SystemExit('Explicit published commit required')
 main(sys.argv[1])
