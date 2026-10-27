"""Bound resource-only amendment preparation and zero-model namespace validation.

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
REV=D/'revisions/shared_gpu_20261027';R=U/'spnv2/reproduction/shared_gpu_20261027'
AUTH='user_2026-10-27_shared_gpu_capacity_admission'
BEFORE='be27ea03fb71ca6f6b910c7612cf75d03e5edecfeeabcf35163d529865e4ad6c'
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
 if request.get('schema')!='spnv2-shared-resource-validation-request-v1':raise ValueError('Wrong validation request')
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
 result=dict(schema='spnv2-shared-resource-metadata-validation-v1',status='passed',nonce=nonce,
  config=request['config'],task=request['task'],index=config['diagnostic_index'],resource_revision=index['resource_revision'],
  resource_policy=index['resource_policy'],gate_policy_configured=True,host_entry_validation=True,preserved_A_pairs_and_main_validation=True,
  identity=dict(pid=os.getpid(),starttime=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline_sha256=hashlib.sha256(cmd).hexdigest(),uid=os.getuid(),sty=os.environ['STY']),
  model_forwards=0,torch_imported=False,scientific_acceptance=False,at=time.time())
 immutable(request['output'],result)
 print(json.dumps(dict(status='passed',proof=request['output'],model_forwards=0)))

def prepare(commit):
 if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact published code commit required')
 stop=read(REV/'old_manager_stop_receipt.json')
 if stop.get('authorization')!=AUTH or stop.get('actual_wrapper_exit_code')!=143 or stop.get('old_active') is not None or stop.get('native_publisher_and_transfer_not_signaled') is not True:raise ValueError('Missing actual safe idle-manager switch receipt')
 if Path(stop['exit_path']).read_text().strip()!='143':raise ValueError('Actual old wrapper exit changed')
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(lock(D/'manager.lock'))
  stack.enter_context(lock(M/'shared_gpu_amendment.lock'))
  if sha(D/'index.json')!=BEFORE:raise ValueError('Only exact current baseline index can be revised')
  before=read(D/'index.json');state=read(D/'status.json')
  if state['active'] is not None or len(state['main'])!=27 or len(state['accepted'])!=26:raise ValueError('Baseline advanced; preserve and reconcile instead of replay')
  started=set();closed=set()
  for line in (D/'budget/journal.jsonl').open():
   event=json.loads(line)
   if event['kind']=='attempt_started':started.add(event['payload']['attempt_id'])
   if event['kind']=='attempt_closed':closed.add(event['payload']['attempt_id'])
  if started-closed:raise ValueError('Unclosed budget allocations prevent control cutover')
  prefix=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes();journal=(D/'budget/journal.jsonl').read_bytes()
  if not journal.startswith(prefix):raise ValueError('Original budget prefix changed')
  journal_sha=sha(D/'budget/journal.jsonl');identity_raw=(D/'budget/identity.json').read_bytes()
  for path in [D/'index.json',D/'status.json',D/'implementation/host_entry.py',D/'implementation/manager.py']:
   immutable_bytes(REV/'before'/path.relative_to(D),path.read_bytes())
  old_control=M/'runtime_control'
  for name in ['template.json','deployment.json','budget_handoff.json']:
   immutable_bytes(REV/'before/runtime_control'/name,(old_control/name).read_bytes())
  policy=REV/'resource_policy.json'
  if not policy.exists():raise ValueError('Explicit reviewed resource policy missing')
  policy_value=read(policy)
  if policy_value.get('authorization')!=AUTH or policy_value.get('min_free_gpu_mib')!=6144 or policy_value.get('min_host_mem_available_mib')!=40960:raise ValueError('Reviewed resource thresholds changed')
  index=copy.deepcopy(before);index['sysu_task_revision']='shared_gpu_20261027'
  index['resource_policy']=bind(policy)
  index['host_modules']={name:bind(R/(name+'.py')) for name in ('host_gate','host_snapshot')}
  index['implementation_sha256']['host_entry.py']=sha(R/'host_entry.py')
  index['implementation_sha256']['manager.py']=sha(R/'manager_sysu.py')
  for name in ('namespace_runner','bootstrap','dispatch_config'):index['migration_extensions'][name]=bind(R/(name+'.py'))
  revision=dict(schema='spnv2-shared-gpu-resource-revision-v1',authorization=AUTH,resource_admission_only=True,
   published_code_commit=commit,validator=bind(R/'resource_amendment.py'),before_index=bind(REV/'before/index.json'),
   before_implementation_sha256=before['implementation_sha256'],after_implementation_sha256=index['implementation_sha256'],
   before_host_modules=before['host_modules'],after_host_modules=index['host_modules'],resource_policy=index['resource_policy'],
   before_host_entry=bind(REV/'before/implementation/host_entry.py'),after_host_entry_source=bind(R/'host_entry.py'),
   source_fence_preserved=True,original_budget_prefix_preserved=True,no_completed_producer_replay=True,original_scientific_code_unchanged=True)
  revision_path=immutable(REV/'resource_revision.json',revision);index['resource_revision']=bind(revision_path)
  remaining=[]
  for row in index['tasks']:
   if row['run_id'] in state['main']:continue
   original=read(mapped(row['task_path']));new=copy.deepcopy(original);new['implementation_sha256']=copy.deepcopy(index['implementation_sha256'])
   if {k:v for k,v in new.items() if k!='implementation_sha256'}!={k:v for k,v in original.items() if k!='implementation_sha256'}:raise ValueError('Scientific task changed')
   task_path=D/'tasks/shared_gpu_20261027'/(row['run_id']+'.json');immutable(task_path,new)
   row['task_path']=logical(task_path);row['task_sha256']=sha(task_path);remaining.append(row['run_id'])
  if len(remaining)!=21:raise ValueError('Must rebind exactly21 unstarted mains')
  for kind in ('pilot_admission','summary'):
   index[kind+'_template']['implementation_sha256']=copy.deepcopy(index['implementation_sha256'])
   index['audit_task_paths'][kind]=logical(D/'tasks/shared_gpu_20261027'/Path(before['audit_task_paths'][kind]).name)
  immutable(REV/'index_candidate.json',index)
  for src,dest in [(R/'host_entry.py',D/'implementation/host_entry.py'),(R/'manager_sysu.py',D/'implementation/manager.py')]:atomic(dest,src.read_bytes())
  atomic(D/'index.json',raw(index));new_index=bind(D/'index.json')
  state.update(index=new_index,status='between_tasks',active=None,manager=None,reason='authorized_shared_gpu_capacity_revision',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='shared_gpu_20261027',revision=index['resource_revision'],previous_index=bind(REV/'before/index.json'),
   original_main_reused=27,original_accepted_reused=26,source_fence_preserved=True,budget_prefix_preserved=True))
  atomic(D/'status.json',raw(state))
  handoff=read(old_control/'budget_handoff.json');handoff['diagnostic_index']=new_index
  handoff_path=immutable(old_control/'budget_handoff_shared_gpu_20261027.json',handoff)
  template=read(old_control/'template.json');template.update(diagnostic_index=new_index,new_task_revision=index['sysu_task_revision'],
   code_bindings=index['migration_extensions'],resource_policy=index['resource_policy'],resource_revision=index['resource_revision'],budget_handoff_receipt=bind(handoff_path))
  template_path=immutable(old_control/'template_shared_gpu_20261027.json',template)
  deployment=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=bind(template_path),index=new_index,
   controller=index['resource_revision'],source_fence=template['source_fence_receipt'],resource_revision=index['resource_revision'])
  deployment_path=immutable(old_control/'deployment_shared_gpu_20261027.json',deployment)
  rid='g50p_diag_v1_sunlamp_s2021_w0016';row=next(r for r in index['tasks'] if r['run_id']==rid)
  science=read(mapped(row['task_path']));receipt=state['main'][rid]
  task=dict(schema='g50p-risk-diagnostic-task-v1',action='verify_pair',run_id=rid+'_verify_'+new_index['sha256'][:12],
   implementation_sha256=index['implementation_sha256'],run_task=receipt['task'],execution_receipt=receipt,all_attempts=receipt['attempts'],
   proof_path=science['output_dir']+'/proof.json',index_path=new_index['path'],index_binding=science['index_binding'],budget_policy=copy.deepcopy(index['budget_policy']))
  task_path=immutable(D/'tasks/shared_gpu_20261027'/(task['run_id']+'.json'),task)
  config=dc.compose_single_task(bind(deployment_path),bind(task_path),resolve=mapped)
  version=dc.write_config_version(config,resolve=mapped)
  nonce=uuid.uuid4().hex;proof=old_control/'config_versions'/('shared_gpu_validation_'+nonce+'.json')
  request=immutable(REV/'validation_request.json',dict(schema='spnv2-shared-resource-validation-request-v1',nonce=nonce,
   validator=bind(Path(__file__)),config=version,task=bind(task_path),output=logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python'
  base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True)
  validation_argv=base[:base.index(python)]+[python,'-B','-s','-u',logical(Path(__file__)),'--validate-request',logical(request),'--request-sha256',sha(request)]
  manager_argv=ns.build_manager_bwrap(config,bind(deployment_path),version)
  if sha(D/'budget/journal.jsonl')!=journal_sha or (D/'budget/identity.json').read_bytes()!=identity_raw:raise ValueError('Resource metadata preparation mutated budget')
  result=dict(schema='spnv2-shared-gpu-runtime-preparation-v1',status='prepared_pending_actual_validation',at=time.time(),
   published_code_commit=commit,index=new_index,resource_revision=index['resource_revision'],resource_policy=index['resource_policy'],
   template=bind(template_path),deployment=bind(deployment_path),budget_handoff=bind(handoff_path),task=bind(task_path),
   config=version,validation_request=bind(request),validation_nonce=nonce,validation_output=logical(proof),validation_argv=validation_argv,manager_argv=manager_argv,
   screen_socket_dir=config['screen_socket_dir'],model_forwards=0,source_fence_preserved=True,native_publisher_not_restarted=True,
   budget_prefix_preserved=True,budget_journal_sha256=journal_sha,budget_identity_sha256=sha(D/'budget/identity.json'),
   preserved_main_count=27,preserved_accepted_count=26,unstarted_rebound=remaining,scientific_acceptance=False)
  immutable(REV/'preparation_receipt.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv','unstarted_rebound')}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--published-code-commit')
 p.add_argument('--validate-request');p.add_argument('--request-sha256');args=p.parse_args()
 if args.validate_request:return validate_request(args.validate_request,args.request_sha256)
 if args.prepare:return prepare(args.published_code_commit or '')
 p.error('Explicit --prepare or bound --validate-request is required')
if __name__=='__main__':main()
