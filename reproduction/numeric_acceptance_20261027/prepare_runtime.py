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
REV=D/'revisions/shared_gpu_20261027';R=U/'spnv2/reproduction/numeric_acceptance_20261027'
FIX=D/'revisions/numeric_acceptance_20261027'
AUTH='user_2026-10-27_shared_gpu_capacity_admission'
BEFORE='3dcd27d1f2150d962baa19d1b0272a12525e6dcbc571fdd83f089dff7087922f'
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
 if request.get('schema')!='spnv2-numeric-acceptance-validation-request-v1':raise ValueError('Wrong validation request')
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
 result=dict(schema='spnv2-numeric-acceptance-metadata-validation-v1',status='passed',nonce=nonce,
  config=request['config'],task=request['task'],index=config['diagnostic_index'],resource_revision=index['resource_revision'],
  resource_policy=index['resource_policy'],gate_policy_configured=True,host_entry_validation=True,preserved_A_pairs_and_main_validation=True,
  identity=dict(pid=os.getpid(),starttime=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline_sha256=hashlib.sha256(cmd).hexdigest(),uid=os.getuid(),sty=os.environ['STY']),
  model_forwards=0,torch_imported=False,scientific_acceptance=False,at=time.time())
 immutable(request['output'],result)
 print(json.dumps(dict(status='passed',proof=request['output'],model_forwards=0)))

def prepare(commit):
 if len(commit)!=40:raise ValueError('Published revision required')
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(lock(D/'manager.lock'));stack.enter_context(lock(M/'numeric_acceptance.lock'))
  if sha(D/'index.json')!=BEFORE:raise ValueError('Numeric amendment baseline differs')
  before=read(D/'index.json');state=read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0023';nonce='c3d81307bbba454aac4dfc0be91dd552'
  if state['status']!='blocked' or len(state['accepted'])!=27 or len(state['main'])!=27 or active['nonce']!=nonce or active['run_id']!=rid or active.get('exit_code')!=1 or active['outcome'].get('error_type')!='TraceMismatch':raise ValueError('Only exact saved numeric stop can resume')
  directory=mapped(active['launch_directory']);log=mapped(active['log_path']);exitfile=Path(str(log)+'.exitcode')
  if exitfile.read_text().strip()!='1':raise ValueError('Actual failed worker exit missing')
  managerlog=REV/'cuda_namespace_fix_20261027/verified_resume/manager.log.exitcode'
  if managerlog.read_text().strip()!='1':raise ValueError('Old manager has not actually stopped')
  failure=D/'pairs'/rid/'failure.json';f=read(failure)
  if f['type']!='TraceMismatch' or f['trace_difference'] is None:raise ValueError('Saved trace difference missing')
  sys.path.insert(0,str(R));import numeric_compare as numeric
  policy=read(FIX/'policy.json');numeric.configure(policy)
  diff=f['trace_difference']
  if not numeric.equal(diff['actual'],diff['expected']):raise ValueError('Saved difference exceeds the fixed new policy')
  import copy as cp
  changed=cp.deepcopy(diff['actual']);changed['event']['accepted']=not changed['event']['accepted']
  if numeric.equal(changed,diff['expected']):raise ValueError('Discrete decisions were not kept exact')
  journal=(D/'budget/journal.jsonl').read_bytes();prefix=(M/'source_handoff_snapshot_20261027/budget/journal.jsonl').read_bytes()
  if not journal.startswith(prefix):raise ValueError('Original ledger prefix changed')
  events=[json.loads(line) for line in journal.splitlines()];starts={e['payload']['attempt_id']:e['payload'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if set(starts)-closed:raise ValueError('Unclosed model allocation remains')
  attempts=[k for k,v in starts.items() if v['launch'].get('run_nonce')==nonce]
  if len(attempts)!=1:raise ValueError('Failed allocation ambiguous')
  reserves=[e for e in events if e['kind']=='forward_reserved' and e['payload'].get('attempt_id')==attempts[0]]
  confirms=[e for e in events if e['kind']=='forward_confirmed' and e['payload'].get('attempt_id')==attempts[0]]
  if sum(e['payload']['count'] for e in reserves)!=1 or sum(e['payload']['count'] for e in confirms)!=1:raise ValueError('Known first-forward cost differs')
  if rid in state['main'] or (D/'pairs'/rid/'report.json').exists():raise ValueError('Completed producer cannot replay')
  for path in [D/'index.json',D/'status.json',failure,directory/'outcome.json',directory/'admission.json']:
   immutable_bytes(FIX/'before'/path.relative_to(D),path.read_bytes())
  immutable(FIX/'authorization_and_scope.json',dict(authorization='user_2026-10-27_authorized_scientific_acceptance_revision',policy=bind(FIX/'policy.json'),scientific_math_unchanged=True,internal_rollback_exact=True,discrete_decisions_rng_exact=True,failed_forward_reserved=1,failed_forward_confirmed=1,no_refunds=True,accepted_reused=27,model_work_before_amendment_preserved=True))
  index=copy.deepcopy(before)
  for name in ('namespace_runner','bootstrap','dispatch_config'):index['migration_extensions'][name]=bind(R/(name+'.py'))
  revision=dict(schema='spnv2-numeric-scientific-acceptance-revision-v1',authorization='user_2026-10-27_authorized_scientific_acceptance_revision',published_code_commit=commit,
   before_index=bind(FIX/'before/index.json'),before_extensions=before['migration_extensions'],after_extensions=index['migration_extensions'],policy=bind(FIX/'policy.json'),
   overrides={name:bind(R/(name+'.py')) for name in ('pair_runner','verify_pair')},comparator=bind(R/'numeric_compare.py'),
   effective_protocol='Original frozen scientific protocol plus this explicit cross-GPU acceptance amendment; historical exactness document remains immutable',
   exact_math_loss_updates_protection_thresholds_unchanged=True,metric_recomputation_and_GT_rules_unchanged=True,existing_strict_acceptances_reused=27,
   boundary='Only cross-device floating comparisons are tolerant; same-run full rollback,branch isolation,RNG and discrete trajectory decisions remain exact',failed_trace=bind(FIX/'before'/failure.relative_to(D)))
  revision_path=immutable(FIX/'revision.json',revision);index['numeric_acceptance']=bind(revision_path)
  immutable(FIX/'index_candidate.json',index);atomic(D/'index.json',raw(index));new_index=bind(D/'index.json')
  state.update(index=new_index,status='between_tasks',active=None,manager=None,reason='user_authorized_numeric_acceptance_revision',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append(dict(name='numeric_acceptance_20261027',revision=index['numeric_acceptance'],accepted_reused=27,main_reused=27,old_failure_and_forward_cost_preserved=True))
  atomic(D/'status.json',raw(state))
  archived=FIX/'archived_trace_failure.json';immutable_bytes(archived,failure.read_bytes());failure.unlink()
  control=M/'runtime_control';handoff=read(control/'budget_handoff_cuda_proc_20261027.json');handoff['diagnostic_index']=new_index
  hp=immutable(control/'budget_handoff_numeric_20261027.json',handoff)
  template=read(control/'template_cuda_proc_20261027.json');template.update(diagnostic_index=new_index,code_bindings=index['migration_extensions'],budget_handoff_receipt=bind(hp),numeric_acceptance=index['numeric_acceptance'])
  tp=immutable(control/'template_numeric_20261027.json',template)
  deployment=dict(schema=dc.DEPLOYMENT_SCHEMA,migration_id=ns.MIGRATION,template=bind(tp),index=new_index,controller=index['numeric_acceptance'],source_fence=template['source_fence_receipt'])
  dp=immutable(control/'deployment_numeric_20261027.json',deployment)
  row=next(x for x in index['tasks'] if x['run_id']==rid);task=mapped(row['task_path']);config=dc.compose_single_task(bind(dp),bind(task),resolve=mapped);version=dc.write_config_version(config,resolve=mapped)
  validation_nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_validation_'+validation_nonce+'.json')
  req=immutable(FIX/'validation_request.json',dict(schema='spnv2-numeric-acceptance-validation-request-v1',nonce=validation_nonce,validator=bind(Path(__file__)),config=version,task=bind(task),output=logical(proof)))
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True)
  va=base[:base.index(python)]+[python,'-B','-s','-u',logical(Path(__file__)),'--validate-request',logical(req),'--request-sha256',sha(req)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Acceptance preparation changed historical cost')
  result=dict(schema='spnv2-numeric-acceptance-runtime-preparation-v1',status='prepared_pending_validation',published_code_commit=commit,index=new_index,revision=index['numeric_acceptance'],policy=bind(FIX/'policy.json'),deployment=bind(dp),config=version,task=bind(task),validation_nonce=validation_nonce,validation_output=logical(proof),validation_argv=va,manager_argv=ns.build_manager_bwrap(config,bind(dp),version),screen_socket_dir=config['screen_socket_dir'],accepted_reused=27,old_forward_cost_preserved=1,model_forwards=0)
  immutable(FIX/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--published-code-commit')
 p.add_argument('--validate-request');p.add_argument('--request-sha256');args=p.parse_args()
 if args.validate_request:return validate_request(args.validate_request,args.request_sha256)
 if args.prepare:return prepare(args.published_code_commit or '')
 p.error('Explicit --prepare or bound --validate-request is required')
if __name__=='__main__':main()
