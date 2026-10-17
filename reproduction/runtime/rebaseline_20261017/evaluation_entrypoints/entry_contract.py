"""Isolated compatibility adapters. No launch, CUDA or model work at import time."""
from pathlib import Path
from contextlib import contextmanager
import argparse,copy,csv,hashlib,importlib,importlib.util,inspect,json,os,runpy,sys
E=Path(__file__).resolve().parent;P=E.parent;O=P/'odr';X=P/'extended_evaluation';PYTHON='/home/ps/anaconda3/envs/SPNv2/bin/python'
class ContractError(RuntimeError):pass
def need(ok,message):
 if not ok:raise ContractError(message)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def payload(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
def stable_write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=payload(value)
 if path.exists():need(path.read_text()==data,'Refuse overwrite of differing accepted/previous evidence: '+str(path));return
 with path.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
def source_spec(argv):
 p=argparse.ArgumentParser(add_help=False);p.add_argument('mode',choices=['evaluate','verify_eval']);p.add_argument('--cfg',required=True);p.add_argument('--report',required=True);p.add_argument('--log');p.add_argument('--source-report');p.add_argument('--output');p.add_argument('--expected-samples',type=int)
 a=p.parse_args(argv);need(Path(a.report).resolve()!=Path(a.source_report).resolve() if a.source_report else False,'Report must not overwrite the source/input report');need(a.source_report and a.expected_samples and a.expected_samples>0,'Original source report and positive expected sample count required');need(a.mode!='evaluate' or a.output,'Original evaluate output required');return a

def verify_protocol():
 path=E/'protocol_manifest.json';m=read(path);need(m['schema']=='spnv2-evaluation-entrypoints-protocol-v1','Unknown wrapper protocol schema')
 for name,value in m['wrapper_code_sha256'].items():need(sha(E/name)==value,'Wrapper implementation changed: '+name)
 for name,value in m['base_entries_sha256'].items():need(sha(P/name)==value,'Original base entry changed: '+name)
 need(sha(P/'source_manifest.json')==m['source_manifest_sha256'],'Frozen source manifest changed');need(sha(X/'preparation_manifest.json')==m['observer_preparation_manifest_sha256'],'Observer validated release manifest changed')
 for name,value in m['observer_code_sha256'].items():need(sha(X/name)==value,'Observer implementation changed: '+name)
 need({p.name for p in X.glob('*.py')}==set(m['observer_code_sha256']),'Observer Python file set changed')
 return m

def event(name,**fields):
 print('SPNV2_EVENT '+json.dumps(dict(event=name,phase='evaluation_extension',**fields),sort_keys=True,allow_nan=False),flush=True)

def local_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

@contextmanager
def namespace(paths,extra_names=()):
 previous_path=list(sys.path);prefixes=('config','nets','dataset','engine','utils','solver','loss','common','evidence','audit_core','verify_heads','_init_paths')+tuple(extra_names)
 selected=lambda name:any(name==q or name.startswith(q+'.') for q in prefixes)
 saved={k:v for k,v in sys.modules.items() if selected(k)}
 for name in saved:sys.modules.pop(name,None)
 roots=[Path(p).resolve() for p in paths];before=set(sys.modules)
 sys.path[:0]=[str(p) for p in roots]
 try:yield
 finally:
  for name,module in list(sys.modules.items()):
   file=getattr(module,'__file__',None)
   under=bool(file and any(Path(file).resolve().is_relative_to(root) for root in roots))
   if selected(name) or (name not in before and under):sys.modules.pop(name,None)
  sys.modules.update(saved);sys.path[:]=previous_path

@contextmanager
def arguments(args):
 before=sys.argv;sys.argv=list(args)
 try:yield
 finally:sys.argv=before

@contextmanager
def CPU_environment():
 before=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=''
 try:yield
 finally:
  if before is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=before

def extension_api():
 # Own package name preserves relative imports without selecting a foreign generic package.
 prefix='_spnv2_evaluation_entry_observer'
 if prefix not in sys.modules:
  spec=importlib.util.spec_from_file_location(prefix,X/'__init__.py',submodule_search_locations=[str(X)]);package=importlib.util.module_from_spec(spec);sys.modules[prefix]=package
  try:spec.loader.exec_module(package)
  except BaseException:sys.modules.pop(prefix,None);raise
 need(Path(sys.modules[prefix].__file__).resolve()==(X/'__init__.py').resolve(),'Foreign observer package namespace')
 modules=[]
 for name in ['capture','report','verify']:
  module=importlib.import_module(prefix+'.'+name);need(Path(module.__file__).resolve()==(X/(name+'.py')).resolve(),'Foreign observer implementation selected');modules.append(module)
 return tuple(modules)

def source_key(raw_report):return Path(raw_report).stem+'_'+hashlib.sha256(str(Path(raw_report).resolve()).encode()).hexdigest()[:12]
def source_paths(raw_report):
 key=source_key(raw_report);root=E/'runs/source'/key
 return dict(key=key,root=root,capture=root/'capture',extension=root/'extension',report=root/'extension/extended_report.json',api_proof=root/'independent_extension_verification.json',wrapper_proof=E/'reports/source'/(key+'.json'),plan=root/'entrypoint_plan.json')
def odr_paths(task_id):
 need(Path(task_id).name==task_id and task_id not in ['.','..'],'Invalid task ID');root=E/'runs/odr'/task_id
 return dict(key=task_id,root=root,capture=root/'capture',extension=root/'extension',report=root/'extension/extended_report.json',api_proof=root/'independent_extension_verification.json',wrapper_proof=E/'reports'/(task_id+'.json'),plan=root/'entrypoint_plan.json')

def context(*,kind,identifier,cfg_path,weights,base_entry,source_proof,heads,extra=None):
 m=verify_protocol();weights={name:dict(path=str(Path(path).resolve()),sha256=sha(path)) for name,path in weights.items()}
 result=dict(schema='spnv2-evaluation-entrypoint-context-v1',kind=kind,identifier=identifier,configuration=dict(path=str(Path(cfg_path).resolve()),sha256=sha(cfg_path)),weights=weights,source_training_acceptance=dict(path=str(Path(source_proof).resolve()),sha256=sha(source_proof)),base_entry=dict(path=str(Path(base_entry).resolve()),sha256=sha(base_entry)),frozen_source_manifest=dict(path=str(P/'source_manifest.json'),sha256=sha(P/'source_manifest.json')),observer_code_sha256=m['observer_code_sha256'],observer_preparation_manifest=dict(path=str(X/'preparation_manifest.json'),sha256=m['observer_preparation_manifest_sha256']),wrapper_code_sha256=m['wrapper_code_sha256'],wrapper_protocol_sha256=sha(E/'protocol_manifest.json'),heads=list(heads),extension_applicability='applicable' if 'heatmap' in heads else 'not_applicable_no_heatmap')
 result.update(extra or {});digest(result);return result

def verify_context(value):
 need(value['schema']=='spnv2-evaluation-entrypoint-context-v1','Unknown entrypoint context');m=verify_protocol();need(value['wrapper_protocol_sha256']==sha(E/'protocol_manifest.json') and value['wrapper_code_sha256']==m['wrapper_code_sha256'] and value['observer_code_sha256']==m['observer_code_sha256'],'Context implementation mismatch')
 for item in [value['configuration'],value['source_training_acceptance'],value['base_entry'],value['frozen_source_manifest'],value['observer_preparation_manifest'],*value['weights'].values()]:need(sha(item['path'])==item['sha256'],'Context cfg/weights/base/source proof changed: '+item['path'])
 if value.get('line_manifest'):need(sha(value['line_manifest']['path'])==value['line_manifest']['sha256'],'ODR line manifest changed')
 return value

class EvaluationCapture:
 def __init__(self,inference_module,*,paths,context_factory,csv_factory,heads_factory,result_factory):
  self.module=inference_module;self.paths=paths;self.context_factory=context_factory;self.csv_factory=csv_factory;self.heads_factory=heads_factory;self.result_factory=result_factory;self.calls=0;self.original=inference_module.do_valid;self.captured=None
 def __enter__(self):
  signature=inspect.signature(self.original)
  def wrapped(*args,**kwargs):
   need(self.calls==0,'Exactly one original evaluation invocation is allowed');self.calls+=1;bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments;heads=list(self.heads_factory(values));csv_path=Path(self.csv_factory(values)).resolve();ctx=self.context_factory(values,heads);verify_context(ctx)
   self.captured=dict(context=ctx,csv_path=str(csv_path),heads=heads,result_dir=str(Path(self.result_factory(values)).resolve()))
   capture,_,_=extension_api();event('capture_start',identifier=ctx['identifier'],heads=heads,csv=str(csv_path),context_sha256=digest(ctx))
   with capture.observe_heatmap_poses(self.module,csv_path=csv_path,output_dir=self.paths['capture'],heads=heads,context=ctx):result=self.original(*args,**kwargs)
   verify_context(ctx);event('capture_complete',identifier=ctx['identifier'],context_sha256=digest(ctx));return result
  self.wrapped=wrapped;self.module.do_valid=wrapped;return self
 def __exit__(self,*exc):
  current=self.module.do_valid;self.module.do_valid=self.original;need(current is self.wrapped,'Inference do_valid symbol changed unexpectedly');return False
 def build(self,cfg_path):
  need(self.calls==1 and self.captured is not None,'Original evaluation did not run exactly once');captured=self.captured;verify_context(captured['context']);_,report,_=extension_api()
  value=report.build_extension(capture_dir=self.paths['capture'],result_dir=captured['result_dir'],csv_path=captured['csv_path'],cfg_path=cfg_path,output_dir=self.paths['extension'],context=captured['context'])
  need(self.paths['report'].is_file(),'Extension API did not create its documented report');return value

def extension_verify(paths):
 capture=read(paths['capture']/'capture_manifest.json');ctx=verify_context(capture['context']);need(capture['status'] in ['complete','not_applicable'],'Observer capture failed')
 need(capture['context_sha256']==digest(ctx),'Capture context mismatch');_,_,verifier=extension_api()
 with CPU_environment():proof=verifier.independent_cpu_verify(paths['report'],output_path=paths['api_proof'])
 need(paths['api_proof'].is_file(),'Independent extension proof missing');stored=read(paths['api_proof']);need(stored==proof,'Returned independent proof differs from saved proof');need(proof['status']=='passed','Extension independent CPU verification did not pass')
 need(proof['report_sha256']==sha(paths['report']) and proof['context_sha256']==digest(ctx),'Extension report/context SHA mismatch')
 return dict(context=ctx,context_sha256=digest(ctx),report=str(paths['report']),report_sha256=sha(paths['report']),independent_proof=str(paths['api_proof']),independent_proof_sha256=sha(paths['api_proof']),observer_status=capture['status'],extension_applicability=ctx['extension_applicability'])


def extension_link(extra):
 return dict(schema='spnv2-evaluation-extension-link-v1',report=dict(path=extra['report'],sha256=extra['report_sha256']),verification=dict(path=extra['independent_proof'],sha256=extra['independent_proof_sha256']),context_sha256=extra['context_sha256'],applicability=extra['extension_applicability'],observer_status=extra['observer_status'])
