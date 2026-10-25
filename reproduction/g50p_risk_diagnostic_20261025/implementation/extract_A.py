"""Zero-forward extraction of accepted saved scientific records after admission."""
import json, pathlib, hashlib, pickle, sys, os, time
from budget import atomic_json
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def binding(p):return dict(path=str(p),sha256=sha(p))
def checked(b):
 if sha(b['path'])!=b['sha256']:raise ValueError('input binding changed: '+b['path'])
 return read(b['path'])
def execute(task_path,*,admission,launch_record,conflict_check,budget):
 started=time.monotonic();task=read(task_path);frozen=checked(task['frozen_inputs']);D=pathlib.Path(task['output_dir']);D.mkdir(parents=True,exist_ok=True)
 producer=frozen['producer']
 for name,digest in producer['implementation_sha256'].items():
  if sha(pathlib.Path(producer['implementation_dir'])/name)!=digest:raise ValueError('producer changed')
 # No model is constructed and no forward function is called. CPU torch is
 # imported only inside the already admitted entry to decode saved CPU tensors.
 import torch
 if torch.cuda.is_initialized():raise RuntimeError('A extraction must not initialize CUDA')
 outputs=[];traces=[];parents={}
 for key,run in frozen['runs'].items():
  budget.check_time('A_before_checkpoint_read')
  if conflict_check():return dict(status='waiting_resources',exit_code=75,model_forwards=0)
  cp=run['original_final_checkpoint'];gp=pathlib.Path(cp['directory']);mp=gp/'metadata.json'
  if sha(mp)!=cp['metadata_sha256'] or sha(gp/'state.bin')!=cp['state_sha256']:raise ValueError('original final checkpoint changed')
  md=read(mp)
  if md['identity']['config_sha256']!=run['original_task']['sha256']:raise ValueError('checkpoint task identity changed')
  with (gp/'state.bin').open('rb') as h:state=pickle.load(h)
  if state['identity']!=md['identity'] or state['counters']['observed']!=1024:raise ValueError('incomplete original science')
  doc=dict(run_key=run['run_key'],run_id=run['run_id'],identity=state['identity'],source_sha256=run['source_sha256'],protocol_sha256=run['protocol_sha256'],state_sha256=cp['state_sha256'],scientific_trace=state['scientific_trace'],pools=state['pools'],stream_records=state['stream_records'],counters=state['counters'])
  out=D/(run['run_id']+'.json')
  if out.exists():
   if read(out)!=doc:raise ValueError('existing A extraction mismatch')
  else:atomic_json(out,doc)
  b=binding(out);outputs.append(b);traces.append(dict(run_key=run['run_key'],**b));parents[run['run_key']]=cp['state_sha256']
  del state,doc
  budget.check_time('A_after_metadata_fsync')
 manifest=dict(summary=frozen['old_summary'],literature_section=task['literature_section'],traces=traces,offline_predictions=frozen['offline_predictions'],static_diagnostic=checked(frozen['protocol'])['static_diagnostic'])
 ap=D/'analysis_manifest.json';atomic_json(ap,manifest)
 proof=dict(schema='g50p-risk-diagnostic-A-extraction-proof-v1',status='passed',model_forwards=0,frozen_inputs=task['frozen_inputs'],analysis_manifest=binding(ap),parent_final_state_sha256=parents,outputs=outputs,elapsed_seconds=time.monotonic()-started,CUDA_initialized=False,models_constructed=0)
 pp=D/'extraction_proof.json';atomic_json(pp,proof)
 return dict(status='a_extracted',exit_code=0,proof=str(pp),proof_sha256=sha(pp),model_forwards=0)
