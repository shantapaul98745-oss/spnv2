"""Finite CPU integration tests, MOCK assets only, no CUDA or real adaptation."""
import argparse,contextlib,copy,importlib,io,json,os,pickle,random,shutil,sys,traceback,types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np,torch
import entry_contract as ec
import odr_contract as oc
import capture_source as cs
import capture_odr as co
import run_odr as ro
import verify_odr as vo
from cpu_fixture import toy,forward_fixture
ACTUAL_E=ec.E;ACTUAL_O=ec.O;ACTUAL_P=ec.P
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
D=ACTUAL_E/'tests'/a.run_id;D.mkdir(exist_ok=False);checks=[]
# Pin real observer API before fixture paths are substituted; all observer files remain untouched.
ec.extension_api()
entry=D/'MOCK_entrypoints';entry.mkdir()
for p in ACTUAL_E.glob('*.py'):shutil.copyfile(p,entry/p.name)
shutil.copyfile(ACTUAL_E/'protocol_manifest.json',entry/'protocol_manifest.json')
modules=[ec,oc,cs,co,ro,vo];stack=contextlib.ExitStack()
for module in modules:stack.enter_context(patch.object(module,'E',entry))
def reject(fn):
 try:fn()
 except (ec.ContractError,FileNotFoundError,ValueError,AssertionError):return
 raise AssertionError('Bad evidence was accepted')
def check(name,fn):
 data=fn();checks.append(dict(name=name,status='passed',details=data));print('PASS',name,json.dumps(data),flush=True)

# Only the GPU/source model-construction path is replaced. Verification invokes actual unmodified source_artifacts.py verify_eval.
original_invoke=cs.invoke_base
fixtures={}
def invoke(base,argv):
 spec=ec.source_spec(argv)
 if spec.mode=='verify_eval':return original_invoke(base,argv)
 f=fixtures[spec.cfg];output=Path(spec.output);output.mkdir(parents=True,exist_ok=False);inference=importlib.import_module('engine.inference');summary=forward_fixture(f,output,inference)
 raw=dict(status='awaiting_verification',mode='evaluate',cfg=spec.cfg,source=str(f.final),source_sha256=ec.sha(f.final),source_report=spec.source_report,expected_samples=len(f.ids),domain='lightbox',results=str(output/'lightbox'),summary=summary,code_sha256=ec.read(ACTUAL_P/'source_manifest.json')['core_tools_sha256'],csv=str(f.csv),mock_fixture=True)
 Path(spec.report).write_text(ec.payload(raw));return raw
stack.enter_context(patch.object(cs,'invoke_base',invoke))
source_cases={}
def source_case(label,heads,**options):
 f=toy(D/label,heads=heads,**options);fixtures[str(f.cfgpath)]=f;raw=f.root/(label+'_raw.json');out=f.root/'MOCK_base_output';accepted=f.root/(label+'_accepted.json')
 evaluate=['--base-entry',str(ACTUAL_P/'tasks/source_artifacts.py'),'evaluate','--cfg',str(f.cfgpath),'--report',str(raw),'--source-report',str(f.source_report),'--output',str(out),'--expected-samples',str(len(f.ids))]
 verify=['--base-entry',str(ACTUAL_P/'tasks/source_artifacts.py'),'verify_eval','--cfg',str(f.cfgpath),'--report',str(accepted),'--source-report',str(raw),'--expected-samples',str(len(f.ids))]
 before_path=list(sys.path);before_argv=list(sys.argv);sentinel=types.ModuleType('common');old=sys.modules.get('common');sys.modules['common']=sentinel
 try:cs.main(evaluate);assert sys.modules['common'] is sentinel
 finally:
  if old is None:sys.modules.pop('common',None)
  else:sys.modules['common']=old
 assert sys.path==before_path and sys.argv==before_argv
 cs.main(verify);first=accepted.read_bytes();paths=ec.source_paths(raw);extra_proof=paths['api_proof'];mtime=extra_proof.stat().st_mtime_ns;wrapper_sha=ec.sha(paths['wrapper_proof']);cs.main(verify)
 assert accepted.read_bytes()==first and extra_proof.stat().st_mtime_ns==mtime and ec.sha(paths['wrapper_proof'])==wrapper_sha
 value=ec.read(accepted);report=ec.read(paths['report']);assert value['extension']['report']['sha256']==ec.sha(paths['report']) and value['extension']['verification']['sha256']==ec.sha(paths['api_proof'])
 assert len(f.models)==1 and f.models[0].calls==len(f.ids);reject(lambda:cs.main(evaluate));assert len(f.models)==1
 source_cases[label]=(f,raw,accepted,verify,paths);return f,report

def source_same_forward_and_repeat():
 f,report=source_case('MOCK_EHS',('heatmap','efficientpose'));assert report['counts']['original_accepted']==1 and report['counts']['H_E_intersection']==2 and report['counts']['missing_H_rotation']==1
 assert report['extended_primary']['metrics']['pose']['denominator_scope']=='numeric_pose_available_subset'
 return dict(actual_frozen_inference_CPU_forwards=3,solver_calls=2,original_accepted=1,extended_numeric=2,missing=1,actual_source_verify_eval_executed_twice=True,accepted_report_repeat_byte_identical=True,API_proof_repeat_mtime_identical=True,namespace_and_argv_restored=True)
def source_head_modes():
 _,h=source_case('MOCK_H_only',('heatmap',));_,e=source_case('MOCK_E_only',('efficientpose',));_,zero=source_case('MOCK_zero_H',('heatmap','efficientpose'),zero_H=True);_,full=source_case('MOCK_all_numeric',('heatmap','efficientpose'),all_numeric=True)
 assert h['primary_mode']=='H_rotation_H_translation' and not h['protocols']['extended_hybrid']['applicable'];assert e['primary_mode']=='E_only_original' and e['counts']['missing_H_rotation'] is None
 assert zero['counts']['original_accepted']==0 and zero['extended_primary']['metrics']['pose']['mean'] is None;assert full['extended_primary']['metrics']['pose']['denominator_scope']=='all_samples'
 f,raw,accepted,verify,paths=source_cases['MOCK_E_only'];assert ec.read(accepted)['extension']['applicability']=='not_applicable_no_heatmap' and ec.read(paths['capture']/'capture_manifest.json')['solver_calls']==0
 return dict(H_only_E_only_zero_H_and_full_coverage=True,not_applicable_H_explicit=True,all_samples_only_N_over_N=True,actual_source_CPU_verifiers_used=True)
def source_failure_preserves_old():
 f,raw,accepted,verify,paths=source_cases['MOCK_EHS'];before=accepted.read_bytes();p=paths['extension']/'extended_per_sample.jsonl';original=p.read_bytes();p.write_bytes(original+b'CORRUPT\n')
 try:reject(lambda:cs.main(verify));assert accepted.read_bytes()==before
 finally:p.write_bytes(original)
 cs.main(verify);assert accepted.read_bytes()==before
 return dict(corrupt_extra_blocks_acceptance=True,existing_source_base_pointer_restored_byte_identical=True,no_evaluation_rerun=True)

# Real original run_task.py is loaded unchanged. Its process boundary is stubbed; child uses actual frozen inference + real extension API.
O=D/'MOCK_odr';O.mkdir();(O/'logs').mkdir();(O/'reports').mkdir()
for name in ['run_task.py','audited_entry.py','evidence.py','implementation_manifest.json','A_sha256.json']:shutil.copyfile(ACTUAL_O/name,O/name)
(O/'source_A').symlink_to(ACTUAL_O/'source_A',target_is_directory=True)
shutil.copyfile(ACTUAL_O/'reports/integration_cpu_acceptance.json',O/'reports/integration_cpu_acceptance.json')
fodr=toy(D/'MOCK_ODR_DATA');base_task=dict(id='MOCK_fixed0',phi=3,line='A',domain='lightbox',seed=2021,N=4,B=1,group='fixed',source_final=str(fodr.final),source_report=str(fodr.source_report),config=str(fodr.cfgpath),config_sha256=ec.sha(fodr.cfgpath),output=str(D/'MOCK_ODR_RESULT'),code_sha256_manifest='MOCK protocol')
def task(id,**changes):
 t=dict(base_task,id=id,output=str(D/('MOCK_OUTPUT_'+id)));t.update(changes);t['command']=[ec.PYTHON,'-B','-s','-u',str(O/'audited_entry.py'),'--task',id];return t
tasks=[task('MOCK_fixed'+str(i)) for i in range(40)];canonical=tasks[0];alias=task('MOCK_alias',reuse_of=canonical['id'],group='Bscan');scan=task('MOCK_scan',group='Nscan');tasks += [alias,scan];ec.stable_write(O/'manifest.json',dict(tasks=tasks))
for module in modules:stack.enter_context(patch.object(module,'O',O))
with ec.namespace([ACTUAL_O]):original_common=ec.local_module(ACTUAL_O/'common.py','_CPU_original_common_readonly')
common=types.ModuleType('common');common.__dict__.update(original_common.__dict__);common.__name__='common';common.__file__=str(O/'common.py');common.O=O;common.job=lambda id:copy.deepcopy(next(t for t in tasks if t['id']==id));job_identity=common.job;common.accepted_source=lambda phi:dict(status='passed',final=str(fodr.final),final_sha256=ec.sha(fodr.final),cfg=str(fodr.cfgpath),code_sha256=ec.read(ACTUAL_P/'source_manifest.json')['core_tools_sha256'])
common.verify_implementation=original_common.verify_implementation;common.verify_code=original_common.verify_code;common.gpu_available=lambda:(True,'MOCK gate; no GPU queried');common.fixed_barrier=lambda t:None;common.parent_gpu_spec=lambda:{'MOCK':True}
@contextlib.contextmanager
def mock_namespace():
 with ec.namespace([O],extra_names=('_entry_base_odr_parent','_entry_base_odr_verify')):
  sys.modules['common']=common;yield common
for module in [oc,co,ro,vo]:stack.enter_context(patch.object(module,'original_odr_namespace',mock_namespace))
# A deterministic mock baseline proof exercises the original verifier call boundary; adaptation/GPU correctness is not claimed.
(O/'verify_result.py').write_text("from common import *\nimport json\ndef verify(task_id):\n t=job(task_id);p=O/'reports'/(task_id+'.json')\n if t.get('reuse_of'):\n  value=verify(t['reuse_of']);r=dict(status='passed',task_id=task_id,reuse_of=t['reuse_of'],source_sha256=value['source_sha256'],canonical_report_sha256=sha(O/'reports'/(t['reuse_of']+'.json')),MOCK_CPU_contract=True)\n else:r=dict(status='passed',task_id=task_id,source_sha256=accepted_source(t['phi'])['final_sha256'],MOCK_CPU_contract=True)\n atomic(p,r);return r\n")
original_runpath=co.runpy.run_path
child_calls=[]
def mock_audited(path,run_name):
 assert Path(path)==O/'audited_entry.py';t=common.job(sys.argv[-1]);child_calls.append(t['id']);output=Path(t['output']);output.mkdir();torch.save(fodr.Tiny().state_dict(),output/'adapted_before_eval.pth.tar');inference=importlib.import_module('engine.inference');summary=forward_fixture(fodr,output,inference);ec.stable_write(output/'evaluation_summary.json',summary);ec.stable_write(output/'adaptation_audit.json',dict(task=t,MOCK_CPU_contract=True));return {}
class MockChild:
 def __init__(self,command,**kwargs):
  assert command==oc.planned_task(common.job(command[-1]))['command'];assert os.environ['CUDA_VISIBLE_DEVICES']=='';self.pid=os.getpid();buffer=io.StringIO();self.rc=0
  try:
   with contextlib.redirect_stdout(buffer),patch.object(co.runpy,'run_path',mock_audited):co.main(['--task',command[-1]])
  except BaseException:traceback.print_exc(file=buffer);self.rc=1
  self.stdout=io.StringIO(buffer.getvalue())
 def wait(self):return self.rc
def mock_verify_process(command,**kwargs):
 assert Path(command[4])==O/'verify_result.py' and kwargs['env']['CUDA_VISIBLE_DEVICES']==''
 module=ec.local_module(O/'verify_result.py','_MOCK_verifier');proof=module.verify(command[-1]);return NS(returncode=0,stdout=json.dumps(proof)+'\n')
original_loader=ro.local_module
def parent_loader(path,name):
 module=original_loader(path,name)
 if name=='_entry_base_odr_parent':module.subprocess=NS(Popen=MockChild,run=mock_verify_process,PIPE=module.subprocess.PIPE,STDOUT=module.subprocess.STDOUT)
 return module
stack.enter_context(patch.object(ro,'local_module',parent_loader))

def odr_parent_child_and_repeat():
 before=list(sys.path);argv=list(sys.argv);sentinel=types.ModuleType('common');old=sys.modules.get('common');sys.modules['common']=sentinel
 try:ro.main(['--task',canonical['id'],'--execute']);assert sys.modules['common'] is sentinel
 finally:
  if old is None:sys.modules.pop('common',None)
  else:sys.modules['common']=old
 assert sys.path==before and sys.argv==argv and common.job is job_identity
 paths=ec.odr_paths(canonical['id']);base=O/'reports'/(canonical['id']+'.json');first=base.read_bytes();proof_sha=ec.sha(paths['wrapper_proof']);mtime=paths['api_proof'].stat().st_mtime_ns;vo.main(['--task',canonical['id']]);vo.main(['--task',canonical['id']])
 assert base.read_bytes()==first and ec.sha(paths['wrapper_proof'])==proof_sha and paths['api_proof'].stat().st_mtime_ns==mtime;assert child_calls==[canonical['id']] and len(fodr.models)==1
 record=ec.read(O/'reports'/(canonical['id']+'.process.json'));assert record['command']==oc.planned_task(canonical)['command'];assert ec.read(Path(canonical['output'])/'adaptation_audit.json')['task']==canonical
 assert ec.read(paths['wrapper_proof'])['extension']['report']['path']==str(paths['report'])
 return dict(original_run_task_code_exercised=True,child_actual_frozen_inference_CPU=True,base_verifier_process_boundary_mocked=True,actual_forward_calls=3,adaptation_not_executed=True,base_report_repeat_byte_identical=True,extension_proof_repeat_SHA_and_mtime_identical=True,parent_process_actual_wrapper_command=True,audit_task_unmodified=True,common_job_identity_and_namespace_restored=True)
def odr_alias_and_barrier():
 count=len(child_calls);base=O/'reports'/(canonical['id']+'.json');before=base.read_bytes()
 reject(lambda:ro.main(['--task',alias['id'],'--execute']));reject(lambda:ro.main(['--task',scan['id'],'--execute']));assert len(child_calls)==count
 # Complete exactly40 explicitly MOCK fixed observations to test both sides of the real barrier.
 for fixed in tasks[1:40]:ro.main(['--task',fixed['id'],'--execute'])
 assert len(child_calls)==40
 count=len(child_calls);ro.main(['--task',alias['id'],'--execute']);paths=ec.odr_paths(alias['id']);proof=ec.read(paths['wrapper_proof']);canonicalproof=ec.read(ec.odr_paths(canonical['id'])['wrapper_proof']);assert proof['extension']==canonicalproof['extension'] and proof['canonical_task']==canonical['id'] and not paths['capture'].exists() and len(child_calls)==count and base.read_bytes()==before
 sha_alias=ec.sha(paths['wrapper_proof']);vo.main(['--task',alias['id']]);assert ec.sha(paths['wrapper_proof'])==sha_alias and base.read_bytes()==before and len(child_calls)==count
 ro.main(['--task',scan['id'],'--execute']);assert len(child_calls)==41
 bad=copy.deepcopy(alias);bad['B']=2;reject(lambda:oc.verify_extension(bad,common));assert len(child_calls)==41
 canonical_api=ec.odr_paths(canonical['id'])['api_proof'];original=canonical_api.read_bytes();canonical_api.write_bytes(original+b' ')
 try:reject(lambda:vo.main(['--task',alias['id']]))
 finally:canonical_api.write_bytes(original)
 assert ec.sha(paths['wrapper_proof'])==sha_alias and base.read_bytes()==before
 return dict(alias_no_forward_or_capture=True,canonical_extension_SHA_dedup=True,alias_reverify_deterministic=True,changed_alias_protocol_and_canonical_SHA_rejected=True,missing_fixed40_extensions_blocks_before_child=True,all40_mock_fixed_positive_barrier_verified=True,one_mock_scan_after_barrier=True,actual_CPU_tiny_forward_calls=41*3,no_GPU_or_adaptation=True)

def capture_error_and_namespace():
 CaptureError=ec.extension_api()[0].CaptureError
 f=toy(D/'MOCK_capture_failure');original_path=list(sys.path);original_argv=list(sys.argv)
 with ec.namespace([ACTUAL_P/'source/core']):
  inf=importlib.import_module('engine.inference');original=inf.do_valid;paths=ec.source_paths(D/'MOCK_capture_failure_raw.json')
  ctx=lambda v,h:ec.context(kind='source',identifier=paths['key'],cfg_path=f.cfgpath,weights=dict(source_final=f.final),base_entry=ACTUAL_P/'tasks/source_artifacts.py',source_proof=f.source_report,heads=h)
  try:
   with ec.EvaluationCapture(inf,paths=paths,context_factory=ctx,csv_factory=lambda v:f.csv,heads_factory=lambda v:['heatmap'],result_factory=lambda v:D/'MOCK_unused'):
    # Missing observer solver calls after body return must fail at observer exit, then restore both symbols.
    inf.do_valid(0,f.cfg,f.Tiny(),[],f.camera,f.keypts,log_dir=str(D/'MOCK_unused'),return_summary=True)
  except (ValueError,CaptureError):pass
  else:raise AssertionError('Empty original evaluation accepted')
  assert inf.do_valid is original
 assert sys.path==original_path and sys.argv==original_argv and os.environ['CUDA_VISIBLE_DEVICES']==''
 return dict(original_body_error_propagates=True,do_valid_and_solver_restore=True,CUDA_stays_hidden=True,no_extra_forward=True)
try:
 check('source_actual_inference_same_forward_and_repeat',source_same_forward_and_repeat)
 check('source_H_E_zero_and_N_over_N_modes',source_head_modes)
 check('source_corruption_preserves_old_accepted_pointer',source_failure_preserves_old)
 check('ODR_original_parent_same_forward_child_and_repeat',odr_parent_child_and_repeat)
 check('ODR_alias_dedup_and_fixed40_extension_barrier',odr_alias_and_barrier)
 check('exception_and_namespace_no_leak',capture_error_and_namespace)
 proof=dict(schema='spnv2-evaluation-entrypoints-cpu-acceptance-v1',status='passed',run_id=a.run_id,passed=len(checks),checks=checks,scope='MOCK CPU integration only; source GPU model construction and ODR adaptation/base-state-verifier process mocked; actual frozen inference, actual source verify_eval and actual extended independent CPU verifier exercised',real_GPU_executed=False,real_adaptation_executed=False,formal_scientific_results=False,protocol_manifest_sha256=ec.sha(ACTUAL_E/'protocol_manifest.json'),wrapper_code_sha256={p.name:ec.sha(p) for p in ACTUAL_E.glob('*.py')})
 ec.stable_write(ACTUAL_E/'reports'/('cpu_acceptance_'+a.run_id+'.json'),proof);print(json.dumps(proof,ensure_ascii=False))
except BaseException as error:
 ec.stable_write(ACTUAL_E/'reports'/('cpu_acceptance_'+a.run_id+'.json'),dict(status='failed',checks=checks,error=repr(error),traceback=traceback.format_exc(),formal_scientific_results=False));raise
finally:stack.close()
