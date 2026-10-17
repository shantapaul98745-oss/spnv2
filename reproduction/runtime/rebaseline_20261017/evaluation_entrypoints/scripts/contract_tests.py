"""Pure CPU entry contracts. No observer dependency, original dry runner only."""
from pathlib import Path
import sys,os,json,io,contextlib,types,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import entry_contract as e
import odr_contract as o
import capture_source as s
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();D=e.E/'tests'/a.run_id;D.mkdir(exist_ok=False);checks=[]
def reject(fn):
 try:fn()
 except (e.ContractError,FileNotFoundError,ValueError,SystemExit):return
 raise AssertionError('Invalid contract accepted')
# Compatible source arguments and preserving all original flags.
argv=['evaluate','--cfg','MOCK_cfg','--report','MOCK_raw','--source-report','MOCK_source','--output','MOCK_output','--expected-samples','3','--log','MOCK_log'];x=e.source_spec(argv);assert x.mode=='evaluate' and x.log=='MOCK_log' and x.expected_samples==3
reject(lambda:e.source_spec(['verify_train','--cfg','MOCK','--report','MOCK']));reject(lambda:e.source_spec(['verify_eval','--cfg','MOCK','--report','SAME','--source-report','SAME','--expected-samples','3']))
checks.append(dict(name='original_source_argument_contract',status='passed'))
# Actual unchanged original runner dry path, module-local job only, without execute/GPU query.
before=list(sys.path);args=list(sys.argv);sentinel=types.ModuleType('common');old=sys.modules.get('common');sys.modules['common']=sentinel
try:
 with o.original_odr_namespace() as common:
  original=common.job;task=next(t for t in e.read(o.O/'manifest.json')['tasks'] if t['group']=='fixed');parent=e.local_module(o.O/'run_task.py','_entry_base_odr_parent');assert parent.job is original;plan=o.planned_task(task);parent.job=lambda id:o.planned_task(original(id));buf=io.StringIO()
  try:
   with e.arguments(['BASE','--task',task['id']]),contextlib.redirect_stdout(buf):parent.main()
  finally:parent.job=original
  parsed=json.loads(buf.getvalue());assert parsed['base_command']==task['command'] and parsed['command']==plan['command'] and common.job is original and task==original(task['id'])
 assert sys.modules['common'] is sentinel
finally:
 if old is None:sys.modules.pop('common',None)
 else:sys.modules['common']=old
assert sys.path==before and sys.argv==args
checks.append(dict(name='actual_original_ODR_dry_runner_local_job_no_leak',status='passed',no_execute=True))
# Source pointer deterministically reattached after original verifier rewrites its base fields.
f=D/'MOCK_accepted.json';base=dict(status='passed',mode='verify_eval',MOCK=True,source_sha256='MOCK_SHA',metrics=dict(final_pose=None));f.write_text(json.dumps(base));pointer=dict(schema='spnv2-evaluation-extension-link-v1',report=dict(path='MOCK',sha256='MOCK'),verification=dict(path='MOCK',sha256='MOCK'),context_sha256='MOCK',applicability='not_applicable_no_heatmap',observer_status='not_applicable');s.replace_source_pointer(f,base,pointer);first=f.read_bytes();f.write_text(json.dumps(base));s.replace_source_pointer(f,base,pointer);assert f.read_bytes()==first and {k:v for k,v in e.read(f).items() if k!='extension'}==base
checks.append(dict(name='source_pointer_deterministic_recovery_and_original_fields',status='passed'))
# Stable evidence refuses overwrite; argument and namespace restoration on exceptions.
f=D/'MOCK_immutable.json';e.stable_write(f,dict(MOCK=True));before=f.read_bytes();reject(lambda:e.stable_write(f,dict(MOCK=False)));assert f.read_bytes()==before
before_path=list(sys.path);before_argv=list(sys.argv)
try:
 with e.namespace([D]),e.arguments(['MOCK_CHANGED']):raise LookupError('MOCK exception')
except LookupError:pass
assert sys.path==before_path and sys.argv==before_argv and os.environ['CUDA_VISIBLE_DEVICES']==''
checks.append(dict(name='immutable_sidecars_exception_and_env_restore',status='passed'))
proof=dict(schema='spnv2-evaluation-entrypoints-contract-cpu-v1',status='passed',run_id=a.run_id,passed=len(checks),checks=checks,no_GPU_no_model_no_observer=True,formal_scientific_results=False,protocol_manifest_sha256=e.sha(e.E/'protocol_manifest.json'),wrapper_code_sha256=e.read(e.E/'protocol_manifest.json')['wrapper_code_sha256'])
e.stable_write(e.E/'reports'/('contract_acceptance_'+a.run_id+'.json'),proof);print(json.dumps(proof))
