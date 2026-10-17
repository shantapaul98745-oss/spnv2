"""Compatible source evaluate/verify_eval CLI with independent same-forward evidence."""
import argparse,sys
from entry_contract import *

def invoke_base(base_entry,argv):
 with arguments([str(base_entry),*argv]):return runpy.run_path(str(base_entry),run_name='__main__')

def source_evaluate(base_entry,argv,a):
 paths=source_paths(a.report);need(not paths['root'].exists(),'Existing source extension attempt is preserved; no automatic rerun');need(not Path(a.report).exists() and not Path(a.output).exists(),'Existing original source outputs preserved')
 plan=dict(schema='spnv2-source-evaluation-entrypoint-plan-v1',mode='evaluate',base_entry=str(base_entry),base_entry_sha256=sha(base_entry),base_command=[PYTHON,'-B','-s','-u',str(base_entry),*argv],wrapper_entry=str(Path(__file__).resolve()),raw_report=str(Path(a.report).resolve()),configuration=dict(path=str(Path(a.cfg).resolve()),sha256=sha(a.cfg)),wrapper_protocol_sha256=sha(E/'protocol_manifest.json'))
 stable_write(paths['plan'],plan)
 def csv_factory(v):return Path(v['cfg'].DATASET.ROOT)/v['cfg'].DATASET.DATANAME/v['cfg'].TEST.TEST_CSV
 def context_factory(v,heads):return context(kind='source',identifier=paths['key'],cfg_path=a.cfg,weights=dict(source_final=v['cfg'].TEST.MODEL_FILE),base_entry=base_entry,source_proof=a.source_report,heads=heads)
 with namespace([P/'source/core',P/'source/tools',P/'tasks']):
  inference=importlib.import_module('engine.inference')
  need(Path(inference.__file__).resolve()==P/'source/core/engine/inference.py','Wrong source inference module')
  with EvaluationCapture(inference,paths=paths,context_factory=context_factory,csv_factory=csv_factory,heads_factory=lambda v:list(v['cfg'].TEST.HEAD),result_factory=lambda v:Path(v['log_dir'])/str(v['cfg'].TEST.TEST_CSV).split('/')[0]) as captured:
   invoke_base(base_entry,argv)
  value=captured.build(a.cfg)
 raw=read(a.report);need(raw['status']=='awaiting_verification' and Path(raw['results']).resolve()==Path(captured.captured['result_dir']) and Path(raw['csv']).resolve()==Path(captured.captured['csv_path']) and raw['source_sha256']==captured.captured['context']['weights']['source_final']['sha256'],'Original source evaluation identity/result path differs from observer capture')
 stable_write(paths['root']/'execution_record.json',dict(schema='spnv2-source-evaluation-entrypoint-execution-v1',raw_report=str(Path(a.report).resolve()),raw_report_sha256=sha(a.report),extension_report=str(paths['report']),extension_report_sha256=sha(paths['report']),capture_manifest_sha256=sha(paths['capture']/'capture_manifest.json'),context_sha256=digest(captured.captured['context']),base_entry_sha256=sha(base_entry),wrapper_protocol_sha256=sha(E/'protocol_manifest.json')))
 event('source_extension_produced',identifier=paths['key'],output=str(paths['report']),status=value['status']);return value

def replace_source_pointer(path,base,pointer):
 path=Path(path);need(read(path)==base,'Base source report changed before pointer attachment');value=dict(base,extension=pointer);data=payload(value)
 if path.read_text()==data:return value
 temporary=path.with_name('.'+path.name+'.extension.tmp')
 with temporary.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temporary,path);return value

def source_verify(base_entry,argv,a):
 paths=source_paths(a.source_report);record=read(paths['root']/'execution_record.json');need(record['raw_report_sha256']==sha(a.source_report) and record['extension_report_sha256']==sha(paths['report']) and record['capture_manifest_sha256']==sha(paths['capture']/'capture_manifest.json'),'Original source/extension evidence changed before verification')
 ctx=verify_context(read(paths['capture']/'capture_manifest.json')['context']);need(ctx['configuration']['path']==str(Path(a.cfg).resolve()) and ctx['base_entry']['path']==str(base_entry),'Source verify cfg/base identity mismatch')
 target=Path(a.report);previous=target.read_bytes() if target.exists() else None;old=read(target) if previous is not None else None
 try:
  with namespace([P/'source/core',P/'source/tools',P/'tasks']),CPU_environment():invoke_base(base_entry,argv)
  base=read(target);need(base['status']=='passed' and base['mode']=='verify_eval','Original CPU verifier did not pass');need('extension' not in base,'Original source verifier unexpectedly emitted extension fields')
  if old is not None:need({k:v for k,v in old.items() if k!='extension'}==base,'Repeated base source verification changed original fields')
  extra=extension_verify(paths);proof=dict(schema='spnv2-source-evaluation-extension-entrypoint-verification-v1',status='passed',identifier=paths['key'],base_report=str(target.resolve()),base_fields_sha256=digest(base),raw_report=str(Path(a.source_report).resolve()),raw_report_sha256=sha(a.source_report),source_sha256=base['source_sha256'],wrapper_protocol_sha256=sha(E/'protocol_manifest.json'),extension=extension_link(extra),**extra)
  stable_write(paths['wrapper_proof'],proof)
  pointer=extension_link(extra);pointer['entrypoint_verification']=dict(path=str(paths['wrapper_proof']),sha256=sha(paths['wrapper_proof']))
  final=replace_source_pointer(target,base,pointer)
  if previous is not None and 'extension' in old:need(target.read_bytes()==previous,'Repeated accepted source report SHA drifted')
  event('source_extension_verified',identifier=paths['key'],output=str(paths['wrapper_proof']),accepted_report=str(target),accepted_report_sha256=sha(target),observer_status=extra['observer_status']);return final
 except BaseException:
  # Restore existing accepted evidence if a repeated verifier failed after base rewrote it.
  if previous is not None and (not target.exists() or target.read_bytes()!=previous):
   temporary=target.with_name('.'+target.name+'.restore.tmp')
   with temporary.open('xb') as f:f.write(previous);f.flush();os.fsync(f.fileno())
   os.replace(temporary,target)
  raise

def main(argv=None):
 env_guard=os.environ.get('STY');need(env_guard,'Run through unique guarded screen');p=argparse.ArgumentParser();p.add_argument('--base-entry',required=True);args,rest=p.parse_known_args(argv);base=Path(args.base_entry).resolve();need(base==P/'tasks/source_artifacts.py','Only original source_artifacts.py may be used as base entry');verify_protocol();a=source_spec(rest)
 return source_evaluate(base,rest,a) if a.mode=='evaluate' else source_verify(base,rest,a)
if __name__=='__main__':
 try:main()
 except BaseException as error:event('error',error=repr(error),message='Source extension entrypoint failed; preserve evidence and block downstream');raise
