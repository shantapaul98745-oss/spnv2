"""Exact CUDA own-proc mount repair preparation and zero-model namespace validation.

--prepare requires the actual idle-manager exit receipt and free original locks.
It never stops or launches a process and never changes the budget journal.
"""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,os,sys,time,uuid
from contextlib import ExitStack,contextmanager
from pathlib import Path
try:
 from . import namespace_runner as ns
 from . import dispatch_config as dc
except ImportError:
 import namespace_runner as ns
 import dispatch_config as dc
U=Path('/home/ps/Downloads/UsrSourceCode/LuXinghai');J=Path(ns.LOGICAL_ROOT)
M=U/'spnv2_runs/migrations/sysu_20261027';D=U/ns.DIAGNOSTIC
REV=D/'revisions/shared_gpu_20261027';R=U/'spnv2/reproduction/shared_gpu_20261027/cuda_namespace_fix_20261027'
FIX=REV/'cuda_namespace_fix_20261027'
AUTH='user_2026-10-27_shared_gpu_capacity_admission'
BEFORE='8de2d8c94004f876fcf9b2d8344b9fa69f19db057d93b7ed6bfc0df16073bf04'
def raw(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mapped(p):return U/Path(p).relative_to(J)
def logical(p):return str(J/Path(p).relative_to(U))
def bind(p):return dict(path=logical(p),sha256=sha(p))
def immutable_bytes(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise ValueError('Immutable revision file changed: '+str(p))
 else:
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return p
def immutable(p,o):return immutable_bytes(p,raw(o))
def atomic(p,b):
 p=Path(p);tmp=p.with_name('.shared-pending-'+uuid.uuid4().hex);tmp.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
@contextmanager
def lock(p):
 with Path(p).open('a+b') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def validate_request(path,expected):
 if sha(path)!=expected:raise ValueError('Validation request binding changed')
 request=read(path)
 if request.get('schema')!='spnv2-cuda-proc-validation-request-v1':raise ValueError('Wrong validation request')
 if sha(__file__)!=request['validator']['sha256']:raise ValueError('Validation code changed')
 nonce=os.environ.get('SPNV2_RESOURCE_REVISION_NONCE','')
 if nonce!=request['nonce'] or not os.environ.get('STY') or os.getuid()!=1000:raise ValueError('Owned namespace validation screen required')
 config=read(request['config']['path'])
 if sha(request['config']['path'])!=request['config']['sha256']:raise ValueError('Validation config changed')
 ns.validate_config(config,scientific=True,inside_namespace=True)
 index=read(config['diagnostic_index']['path']);task=read(request['task']['path'])
 sys.path.insert(0,str(J/ns.DIAGNOSTIC/'implementation'))
 import host_entry
 checked_task,checked_index=host_entry.validate_task(request['task']['path'],config['diagnostic_index']['path'])
 if checked_task!=task or checked_index!=index:raise ValueError('Actual entry validation differs')
 gate,collector=host_entry.host_modules(index)
 import importlib.util
 spec=importlib.util.spec_from_file_location('spnv2_shared_manager_validation',J/ns.DIAGNOSTIC/'implementation/manager.py')
 manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager);manager.validate_index(index)
 if 'torch' in sys.modules:raise ValueError('Zero-model validation imported torch')
 proc=Path('/proc')/str(os.getpid());cmd=(proc/'cmdline').read_bytes()
 result=dict(schema='spnv2-cuda-proc-metadata-validation-v1',status='passed',nonce=nonce,
  config=request['config'],task=request['task'],index=config['diagnostic_index'],resource_revision=index['resource_revision'],
  resource_policy=index['resource_policy'],gate_policy_configured=True,host_entry_validation=True,preserved_A_pairs_and_main_validation=True,
  identity=dict(pid=os.getpid(),starttime=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline_sha256=hashlib.sha256(cmd).hexdigest(),uid=os.getuid(),sty=os.environ['STY']),
  model_forwards=0,torch_imported=False,scientific_acceptance=False,at=time.time())
 immutable(request['output'],result)
 print(json.dumps(dict(status='passed',proof=request['output'],model_forwards=0)))

def prepare(commit):
 if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact published code commit required')
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(lock(D/'manager.lock'));stack.enter_context(lock(M/'cuda_proc_repair.lock'))
  if sha(D/'index.json')!=BEFORE:raise ValueError('CUDA repair baseline index differs')
  before=read(D/'index.json');state=read(D/'status.json');active=state.get('active')
  rid='g50p_diag_v1_sunlamp_s2021_w0023';failed_nonce='1f1b09d799f24328962e875eb4fad45f'
  if state['status']!='blocked' or len(state['main'])!=27 or len(state['accepted'])!=27 or not active or active['run_id']!=rid or active['nonce']!=failed_nonce or active.get('exit_code')!=1:raise ValueError('Only exact observed CUDA304 stop may be repaired')
  manager_exit=REV/'shared_manager.log.exitcode'
  if manager_exit.read_text().strip()!='1':raise ValueError('Actual failed manager wrapper exit differs')
  directory=mapped(active['launch_directory']);outcome=read(directory/'outcome.json');launch=read(directory/'launch.json')
  logfile=mapped(active['log_path']);exitfile=Path(str(logfile)+'.exitcode')
  worker_budget_closed_present=(directory/'budget_closed.json').exists()
  row=next(row for row in before['tasks'] if row['run_id']==rid)
  if exitfile.read_text().strip()!='1' or outcome!=active['outcome'] or outcome.get('exit_code')!=1 or outcome.get('status')!='failed' or outcome.get('error')!='real paired-window diagnosis requires the admitted GPU':raise ValueError('Failed worker/outcome/exit evidence differs')
  if launch.get('run_nonce')!=failed_nonce or launch.get('run_id')!=rid or launch.get('index_sha256')!=BEFORE or launch.get('task_sha256')!=row['task_sha256'] or sha(mapped(row['task_path']))!=row['task_sha256']:raise ValueError('Failed launch/task identity differs')
  if '304' not in logfile.read_text():raise ValueError('Actual CUDA304 failure log missing')
  journal_raw=(D/'budget/journal.jsonl').read_bytes();events=[];previous='0'*64
  for sequence,line in enumerate(journal_raw.splitlines(keepends=True),1):
   if not line.endswith(b'\n'):raise ValueError('Torn actual budget journal; never invent closure')
   event=json.loads(line);digest=event.pop('sha256')
   canonical=json.dumps(event,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
   if event['sequence']!=sequence or event['previous_sha256']!=previous or hashlib.sha256(canonical).hexdigest()!=digest:raise ValueError('Actual budget chain changed')
   event['sha256']=digest;events.append(event);previous=digest
  attempts=[e['payload'] for e in events if e['kind']=='attempt_started' and e['payload']['launch'].get('run_nonce')==failed_nonce]
  if len(attempts)!=1:raise ValueError('Failed CUDA attempt allocation ambiguous')
  aid=attempts[0]['attempt_id'];reserved=[e for e in events if e['kind']=='forward_reserved' and e['payload'].get('attempt_id')==aid]
  ended=[e for e in events if e['kind']=='attempt_closed' and e['payload'].get('attempt_id')==aid]
  started={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if reserved or len(ended)!=1 or started-closed:raise ValueError('CUDA failure is not zero-forward fully settled attempt')
  w16='g50p_diag_v1_sunlamp_s2021_w0016';accepted=state['accepted'][w16]
  if read(mapped(accepted['proof']['path']))['status']!='passed' or sha(mapped(accepted['proof']['path']))!=accepted['proof']['sha256'] or accepted['verification_attempt']['exit_code']!=0 or mapped(accepted['verification_attempt']['log_path']+'.exitcode').read_text().strip()!='0':raise ValueError('Completed w16 independent acceptance is not exact')
  if rid in state['main'] or rid in state['accepted'] or mapped(read(mapped(row['task_path']))['output_dir']+'/report.json').exists():raise ValueError('A completed w23 producer must never replay')
  prefix=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes();journal=(D/'budget/journal.jsonl').read_bytes()
  if not journal.startswith(prefix):raise ValueError('Original budget prefix changed')
  journal_sha=sha(D/'budget/journal.jsonl');identity_raw=(D/'budget/identity.json').read_bytes()
  for path in [D/'index.json',D/'status.json']:
   immutable_bytes(FIX/'before'/path.relative_to(D),path.read_bytes())
  cost_events=[e for e in events if e['kind'] in ('attempt_started','attempt_closed','time_finalization_observed','finalization_bound_violated') and e['payload'].get('attempt_id')==aid]
  derived_cost=dict(schema='spnv2-derived-existing-closed-attempt-cost-v1',status='derived_from_actual_verified_ledger',
   derived_read_only=True,no_new_charge_or_settlement=True,actual_worker_budget_closed_receipt_present=worker_budget_closed_present,worker_budget_closed_receipt_absent=not worker_budget_closed_present,
   worker_budget_closed_file_was_not_created=True,attempt_id=aid,run_nonce=failed_nonce,run_id=rid,physical_forwards_reserved=0,
   unique_actual_closed_event=ended[0],actual_cost_events=cost_events,
   journal_prefix_observed=dict(path=logical(D/'budget/journal.jsonl'),bytes=len(journal_raw),sha256=hashlib.sha256(journal_raw).hexdigest(),events=len(events),tail_sha256=events[-1]['sha256']),
   original_source_prefix=dict(bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest()),
   original_recorded_charged_seconds_upper=ended[0]['payload']['charged_seconds_upper'],
   original_recorded_measured_seconds_before_finalization=ended[0]['payload'].get('measured_seconds_before_finalization'),
   unresolved_attempt_ids=[],evidence_only_not_a_worker_success_path_receipt=True)
  derived_cost_path=immutable(FIX/'from_actual_ledger_cost.json',derived_cost)
  failed=dict(run_id=rid,nonce=failed_nonce,launch=bind(directory/'launch.json'),outcome=bind(directory/'outcome.json'),
   exit_code=bind(exitfile),log=bind(logfile),closed_cost_evidence=bind(derived_cost_path),actual_worker_budget_closed_receipt_present=worker_budget_closed_present,worker_budget_closed_receipt_absent=not worker_budget_closed_present,
   attempt_id=aid,physical_forwards_reserved=0,actual_exit_code=1)
  probe=REV/'cuda_namespace_diagnosis/own_comm_mount_validation.json'
  probevalue=read(probe)
  if probevalue.get('status')!='passed' or probevalue.get('model_forwards')!=0:raise ValueError('Actual paired own-comm metadata validation missing')
  index=copy.deepcopy(before)
  for name in ('namespace_runner','bootstrap','dispatch_config'):index['migration_extensions'][name]=bind(R/(name+'.py'))
  revision=dict(schema='spnv2-cuda-proc-namespace-revision-v1',technical_repair_only=True,published_code_commit=commit,
   before_index=bind(FIX/'before/index.json'),before_state=bind(FIX/'before/status.json'),
   before_extensions=before['migration_extensions'],after_extensions=index['migration_extensions'],
   mount_delta=['--bind','/proc','/proc'],user_pid_uid_and_other_mounts_unchanged=True,own_comm_validation=bind(probe),failed_attempt=failed,
   resource_revision_unchanged=before['resource_revision'],resource_policy_unchanged=before['resource_policy'],
   implementation_sha256_unchanged=before['implementation_sha256'],protected_accepted_records=state['accepted'],protected_main_records=state['main'],
   original_budget_prefix_preserved=True,source_dispatch_fence_preserved=True,completed_producer_replay_forbidden=True)
  revision_path=immutable(FIX/'revision.json',revision);index['cuda_namespace_fix']=bind(revision_path)
  if {k:v for k,v in index.items() if k not in ('migration_extensions','cuda_namespace_fix')}!={k:v for k,v in before.items() if k!='migration_extensions'}:raise ValueError('CUDA repair changed other index fields')
  immutable(FIX/'index_candidate.json',index);atomic(D/'index.json',raw(index));new_index=bind(D/'index.json')
  state.update(index=new_index,status='between_tasks',active=None,manager=None,reason='exact_cuda_proc_mount_repair',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='cuda_proc_mount_repair_20261027',revision=index['cuda_namespace_fix'],failed_attempt=failed,
   accepted_reused=27,main_reused=27,resource_policy_preserved=True,implementation_code_preserved=True,budget_prefix_preserved=True))
  atomic(D/'status.json',raw(state))
  control=M/'runtime_control'
  old_handoff=control/'budget_handoff_shared_gpu_20261027.json';handoff=read(old_handoff);handoff['diagnostic_index']=new_index
  handoff_path=immutable(control/'budget_handoff_cuda_proc_20261027.json',handoff)
  template=read(control/'template_shared_gpu_20261027.json');template.update(diagnostic_index=new_index,code_bindings=index['migration_extensions'],budget_handoff_receipt=bind(handoff_path),cuda_namespace_fix=index['cuda_namespace_fix'])
  template_path=immutable(control/'template_cuda_proc_20261027.json',template)
  deployment=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=bind(template_path),index=new_index,
   controller=index['cuda_namespace_fix'],source_fence=template['source_fence_receipt'],resource_revision=index['resource_revision'],cuda_namespace_fix=index['cuda_namespace_fix'])
  deployment_path=immutable(control/'deployment_cuda_proc_20261027.json',deployment)
  task_path=mapped(row['task_path']);config=dc.compose_single_task(bind(deployment_path),bind(task_path),resolve=mapped)
  version=dc.write_config_version(config,resolve=mapped);nonce=uuid.uuid4().hex;proof=control/'config_versions'/('cuda_proc_validation_'+nonce+'.json')
  request=immutable(FIX/'validation_request.json',dict(schema='spnv2-cuda-proc-validation-request-v1',nonce=nonce,validator=bind(Path(__file__)),config=version,task=bind(task_path),output=logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True)
  validation_argv=base[:base.index(python)]+[python,'-B','-s','-u',logical(Path(__file__)),'--validate-request',logical(request),'--request-sha256',sha(request)]
  manager_argv=ns.build_manager_bwrap(config,bind(deployment_path),version)
  if sha(D/'budget/journal.jsonl')!=journal_sha or (D/'budget/identity.json').read_bytes()!=identity_raw:raise ValueError('CUDA metadata preparation mutated budget')
  result=dict(schema='spnv2-cuda-proc-runtime-preparation-v1',status='prepared_pending_actual_validation',at=time.time(),published_code_commit=commit,
   index=new_index,cuda_namespace_fix=index['cuda_namespace_fix'],resource_revision=index['resource_revision'],resource_policy=index['resource_policy'],
   template=bind(template_path),deployment=bind(deployment_path),budget_handoff=bind(handoff_path),task=bind(task_path),config=version,
   validation_request=bind(request),validation_nonce=nonce,validation_output=logical(proof),validation_argv=validation_argv,manager_argv=manager_argv,
   screen_socket_dir=config['screen_socket_dir'],model_forwards=0,budget_prefix_preserved=True,budget_journal_sha256=journal_sha,
   budget_identity_sha256=sha(D/'budget/identity.json'),preserved_main_count=27,preserved_accepted_count=27,
   no_main_or_accepted_replay=True,original_implementation_map_preserved=True,failed_attempt=failed,scientific_acceptance=False)
  immutable(FIX/'preparation_receipt.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv','failed_attempt')}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--published-code-commit')
 p.add_argument('--validate-request');p.add_argument('--request-sha256');args=p.parse_args()
 if args.validate_request:return validate_request(args.validate_request,args.request_sha256)
 if args.prepare:return prepare(args.published_code_commit or '')
 p.error('Explicit --prepare or bound --validate-request is required')
if __name__=='__main__':main()
