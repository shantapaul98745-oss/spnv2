"""Independent CPU verification of source/adapted states, inputs, optimizer/RNG and metrics."""
import json,sys,argparse,os
from pathlib import Path
from common import *
# Set before importing torch or model modules, even when parent invokes CLI directly.
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TORCH_HOME']=str(RUNS/'env/cache/torch')
from evidence import emit_event,verify_input_evidence,validate_optimizer,validate_rng,equal_tree,validate_weights,anchor_input_evidence

def verify(task_id):
 import torch,csv
 from types import SimpleNamespace
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 torch.set_num_threads(2);torch.hub.set_dir(str(RUNS/'env/cache/torch/hub'))
 implementation=verify_implementation();t=job(task_id);canonical_t=canonical(t)
 verify_code(t['line']);accepted=accepted_source(t['phi']);assert sha(t['config'])==t['config_sha256']
 if t.get('reuse_of'):
  assert all(t[k]==canonical_t[k] for k in ['phi','line','domain','seed','N','B','source_final','source_report','code_sha256_manifest'])
  proof=verify(canonical_t['id']);assert proof['source_sha256']==accepted['final_sha256'] and proof['protocol']==PROTOCOL
  report=dict(status='passed',task_id=t['id'],reuse_of=canonical_t['id'],canonical_report=str(O/'reports'/(canonical_t['id']+'.json')),canonical_report_sha256=sha(O/'reports'/(canonical_t['id']+'.json')),source_sha256=proof['source_sha256'],protocol=PROTOCOL,implementation_manifest_sha256=sha(O/'implementation_manifest.json'),reason='Same new source SHA, line, observation seed/domain, N/B and protocol; independently verified canonical output only')
  atomic(O/'reports'/(t['id']+'.json'),report);return report
 out=Path(t['output']);audit=json.loads((out/'adaptation_audit.json').read_text());summary=json.loads((out/'evaluation_summary.json').read_text());inputs=json.loads((out/'input_evidence.json').read_text())
 assert audit['status']=='adaptation_passed_evaluation_pending' and audit['task']==t
 assert audit['implementation_manifest_sha256']==sha(O/'implementation_manifest.json') and audit['helpers_sha256']==implementation['helpers_sha256']
 assert audit['source_sha256']==accepted['final_sha256'] and audit['code_sha256']==verify_code(t['line']) and audit['protocol']==PROTOCOL
 assert audit['config_sha256']==sha(t['config']) and audit['target_labels_used_for_adaptation'] is False and audit['raw_image_content_hashed'] is False
 assert sha(out/'input_evidence.json')==audit['input_evidence_sha256'];verify_input_evidence(inputs)
 assert inputs['domain']==t['domain'] and Path(inputs['csv']['path'])==DATA/t['domain']/'labels/test.csv'
 before=torch.load(t['source_final'],map_location='cpu',weights_only=True);saved=torch.load(out/'adapted_before_eval.pth.tar',map_location='cpu',weights_only=True);final=torch.load(out/'model_final.pth.tar',map_location='cpu',weights_only=True);checkpoint=torch.load(out/'checkpoint.pth.tar',map_location='cpu',weights_only=True);early=torch.load(out/'adaptation_checkpoint_before_eval.pth.tar',map_location='cpu',weights_only=True)
 for state in [saved,final,checkpoint['state_dict'],early['state_dict']]:validate_weights(state,before);assert equal_tree(saved,state),'Adapted/final/checkpoint tensor mismatch'
 assert sha(out/'adapted_before_eval.pth.tar')==audit['adapted_sha256'] and sha(out/'adaptation_checkpoint_before_eval.pth.tar')==audit['checkpoint_sha256']
 assert early['source_sha256']==accepted['final_sha256'] and early['code_sha256']==verify_code(t['line']) and early['task']==t and early['input_evidence']==inputs
 assert early['format']=='spnv2-odr-v2' and checkpoint['epoch']==1 and equal_tree(checkpoint['optimizer'],early['optimizer'])
 for stage in ['rng_initial','rng_final']:validate_rng(early[stage],cuda_count=1)
 assert [x.shape for x in early['rng_initial']['torch_cuda']]==[x.shape for x in early['rng_final']['torch_cuda']]
 ids=json.loads((out/'observations.json').read_text());assert sha(out/'observations.json')==audit['observations_sha256']
 csvpath=DATA/t['domain']/'labels/test.csv'
 with csvpath.open() as stream:names=[f'{t["domain"]}/{r[0]}' for r in csv.reader(stream)]
 size=len(names);assert size==(6740 if t['domain']=='lightbox' else 2791)==inputs['csv']['sample_count']
 assert len(ids)==audit['N_seen']==audit['forward_calls']==t['N'] and len(set(ids))==audit['N_unique']==min(size,t['N']) and set(ids)<=set(names)
 for start in range(0,len(ids),size):assert len(set(ids[start:start+size]))==len(ids[start:start+size]),'Sampler pass is without replacement'
 assert audit['optimizer_steps']==t['N']//t['B'] and audit['step_observation_counts']==early['steps']==list(range(t['B'],t['N']+1,t['B']))
 root=O/('source_'+t['line']);sys.path[:0]=[str(root/'tools'),str(root/'core')]
 if t['phi']==3:assert (RUNS/'env/cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth').is_file(),'No implicit cache download'
 from config import cfg as defaults,update_config
 from nets import build_spnv2
 from solver import get_optimizer
 from audit_core import check_scope
 cfg=defaults.clone();update_config(cfg,SimpleNamespace(cfg=t['config'],opts=[]))
 anchor_input_evidence(inputs,Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/cfg.TRAIN.TRAIN_CSV,Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/t['domain']/'images_768x512_RGB',cfg.DATASET.CAMERA,cfg.DATASET.KEYPOINTS,t['domain'])
 model=build_spnv2(cfg)
 # Independently strictly reload original source and compare actual tensor values/schema.
 validate_weights(before,model.state_dict());model.load_state_dict(before,strict=True)
 assert equal_tree(before,{k:v.detach().cpu() for k,v in model.state_dict().items()}),'Original source strict reload differs'
 model.eval();model.requires_grad_(False)
 for m in model.backbone.modules():
  if isinstance(m,(torch.nn.BatchNorm2d,torch.nn.GroupNorm)):m.requires_grad_(True)
  if isinstance(m,torch.nn.BatchNorm2d) and t['line']=='B2':m.momentum=.1
 model.load_state_dict(saved,strict=True);assert equal_tree(saved,{k:v.detach().cpu() for k,v in model.state_dict().items()})
 _,scope=check_scope(model,before,t['line'])
 for k,v in scope.items():assert audit[k]==v,k
 assert audit['strict_reload'] and audit['heads_unchanged']
 optimizer_names=validate_optimizer(early['optimizer'],model,t['N']//t['B'],cfg);assert optimizer_names==audit['optimizer_state_names']
 validate_optimizer(checkpoint['optimizer'],model,t['N']//t['B'],cfg)
 optimizer=get_optimizer(cfg,model);optimizer.load_state_dict(early['optimizer']);assert equal_tree(optimizer.state_dict(),early['optimizer'])
 from verify_heads import verify as verify_heads
 metrics=verify_heads(out/t['domain'],csvpath,['heatmap','efficientpose'])
 assert metrics['n_samples']==size==summary['n_samples'] and metrics['n_reject']==summary['heatmap_rejections'] and metrics['final_pose']==summary['score']
 assert all(summary['metrics'][k]['count']==n for k,n in metrics['metric_denominators'].items())
 verify_input_evidence(inputs)
 report=dict(status='passed',task_id=t['id'],source_sha256=accepted['final_sha256'],protocol=PROTOCOL,implementation_manifest_sha256=sha(O/'implementation_manifest.json'),adaptation_audit_sha256=sha(out/'adaptation_audit.json'),input_evidence_sha256=sha(out/'input_evidence.json'),strict_reload=True,source_strict_reload=True,optimizer_and_rng_schema_verified=True,N_seen=len(ids),N_unique=len(set(ids)),optimizer_steps=t['N']//t['B'],heads_unchanged=True,metrics=metrics,summary=summary,result_directory=str(out),raw_image_content_hashed=False,scope='Full domain/stable IDs/physical metrics/explicit accepted denominators; missing predictions retained as NaN')
 atomic(O/'reports'/(t['id']+'.json'),report);return report
if __name__=='__main__':
 try:
  assert os.environ.get('STY'),'CPU verification requires guarded screen'
  p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args();proof=verify(a.task);emit_event('verification_complete',task=a.task,message='Independent CPU verification passed',output=proof.get('result_directory'))
  print(json.dumps(proof,allow_nan=False))
 except BaseException as error:
  emit_event('error',error=repr(error),message='Independent CPU verification failed; block downstream tasks');raise
