"""Run reviewed frozen ODR math with standard progress and pre-evaluation evidence."""
import argparse,sys,os,json,importlib.util,time,random
from pathlib import Path
from common import *
from evidence import emit_event,input_evidence,verify_input_evidence,validate_rng,validate_optimizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);args=p.parse_args()
 assert os.environ.get('STY'),'Use guarded screen'
 t=job(args.task);assert not t.get('reuse_of'),'Alias requires verified canonical result through run_task.py'
 assert sha(t['config'])==t['config_sha256']
 implementation=verify_implementation();code=verify_code(t['line']);accepted=accepted_source(t['phi'])
 assert os.environ.get('TORCH_HOME')==str(RUNS/'env/cache/torch')
 root=O/('source_'+t['line']);sys.path[:0]=[str(root/'tools'),str(root/'core')]
 import torch,numpy as np
 from audit_core import IndexedDataset,LoggedLoader,norm_scope,check_scope
 torch.set_num_threads(2);torch.hub.set_dir(str(RUNS/'env/cache/torch/hub'))
 if t['phi']==3:assert (RUNS/'env/cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth').is_file(),'Own cache missing; do not download implicitly'
 output=Path(t['output']);assert not output.exists(),'Refuse overwrite/restart of task output'
 assert os.environ.get('SPNV2_PILOT_EVAL_LIMIT','0')=='0','Full target domain evaluation required'
 spec=importlib.util.spec_from_file_location('frozen_odr_cli',root/'tools/odr.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
 original_loader=cli.get_dataloader;inputs={};counters=dict(samples_seen=0,update_attempts=0,optimizer_updates=0,amp_skips=0)
 def event(name,**fields):
  defaults=dict(task=t['id'],domain=t['domain'],source=accepted['final'],config=t['config'],output=str(output),epoch=1,epochs=1,batch=counters['samples_seen'],batches=t['N'],N=t['N'],B=t['B'],physical_batch=1,actual_batch_size=1,lr=PROTOCOL['lr'],**counters);defaults.update(fields)
  return emit_event(name,**defaults)
 event('odr_start',message='One observation per forward; ODR window B controls affine/statistic updates')
 def get_loader(cfg,split='train',**kwargs):
  assert cfg.TRAIN.REPRODUCIBLE is False and cfg.TRAIN.ACCUM_STEPS==1 and cfg.TRAIN.IMAGES_PER_GPU==1
  assert cfg.FP16 is False and cfg.AUGMENT.P==0 and cfg.AUGMENT.APPLY_TEXTURE_RANDOMIZATION is False
  assert cfg.MODEL.PRETRAIN_FILE==accepted['final'] and cfg.SEED==t['seed']
  assert cfg.ODR.NUM_TRAIN_SAMPLES==t['N'] and cfg.ODR.IMAGES_PER_BATCH==t['B'] and t['N']%t['B']==0
  loader=original_loader(cfg,split=split,**kwargs);dataset=loader.dataset
  csv_path=DATA/(cfg.TRAIN.TRAIN_CSV if split=='train' else cfg.TRAIN.VAL_CSV)
  captured=input_evidence(csv_path,Path(dataset.root)/dataset.domain/dataset.imagefolder,cfg.DATASET.CAMERA,cfg.DATASET.KEYPOINTS,dataset.domain)
  assert captured['domain']==t['domain'] and captured['csv']['sample_count']==(6740 if t['domain']=='lightbox' else 2791)
  if 'start' not in inputs:
   inputs['start']=captured;atomic(output/'input_evidence.json',captured)
   event('input_evidence',message='CSV/geometry contents SHA256; all image paths/sizes/mtimes captured; raw image bytes not hashed',input_evidence_sha256=sha(output/'input_evidence.json'))
  else:assert captured==inputs['start'],'Train/evaluation target input evidence differs'
  if split=='train':
   assert kwargs.get('load_labels') is False and dataset.load_labels is False and loader.batch_size==1 and loader.drop_last
   object.__setattr__(loader,'dataset',IndexedDataset(dataset))
  return LoggedLoader(loader,split,t['N'])
 cli.get_dataloader=get_loader;original_adapt=cli.do_adapt
 def rng():
  state=np.random.get_state()
  return dict(python=random.getstate(),numpy=dict(kind=state[0],keys=state[1].tolist(),pos=state[2],has_gauss=state[3],cached_gaussian=state[4]),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
 def save_weights(path,state):
  tmp=path.with_name('.'+path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)
 def adapt(epoch,cfg,model,bn_features,data_loader,optimizer,**kwargs):
  assert kwargs.get('scaler') is None
  verify_input_evidence(inputs['start'])
  before=torch.load(accepted['final'],map_location='cpu',weights_only=True)
  assert before.keys()==model.state_dict().keys() and all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
  allowed,running,momenta=norm_scope(model);assert {n for n,p in model.named_parameters() if p.requires_grad}==allowed
  if t['phi']==3:assert set(momenta.values())==({.1} if t['line']=='B2' else {.1,.01})
  assert not model.training and all(not m.training for m in model.modules()) and not bn_features
  steps=[];losses=[];start_rng=rng();validate_rng(start_rng);started=time.time()
  def check_step(opt,unused_args,unused_kwargs):
   assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for g in opt.param_groups for p in g['params']),'Nonfinite adaptation gradient'
  def count_step(*unused):
   steps.append(len(data_loader.ids));counters.update(samples_seen=len(data_loader.ids),update_attempts=len(steps),optimizer_updates=len(steps))
   event('update',loss=sum(losses[-t['B']:])/t['B'])
  def check_loss(module,images,result):
   loss,items=result;assert bool(torch.isfinite(loss).all()),'Nonfinite adaptation loss'
   losses.append(float(loss.detach().cpu()));counters['samples_seen']=len(data_loader.ids)
   if len(losses)==1 or len(losses)%128==0 or len(losses)==t['N']:event('observation',loss=losses[-1])
  hooks=[optimizer.register_step_pre_hook(check_step),optimizer.register_step_post_hook(count_step),model.register_forward_hook(check_loss)]
  torch.cuda.reset_peak_memory_stats()
  try:original_adapt(epoch,cfg,model,bn_features,data_loader,optimizer,**kwargs)
  finally:
   for h in hooks:h.remove()
  state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};path=output/'adapted_before_eval.pth.tar'
  save_weights(path,state);atomic(output/'observations.json',data_loader.ids)
  end_rng=rng();early=output/'adaptation_checkpoint_before_eval.pth.tar'
  save_weights(early,dict(format='spnv2-odr-v2',state_dict=state,optimizer=optimizer.state_dict(),rng_initial=start_rng,rng_final=end_rng,task=t,source_sha256=accepted['final_sha256'],code_sha256=code,steps=steps,input_evidence=inputs['start']))
  event('checkpoint',path=str(path),checkpoint=str(path),message='Adapted weights and optimizer/RNG evidence saved before audit and evaluation')
  checked,scope=check_scope(model,before,t['line']);assert momenta==scope['bn_momenta']
  assert len(data_loader.ids)==len(losses)==t['N'] and len(set(data_loader.ids))==min(t['N'],6740 if t['domain']=='lightbox' else 2791)
  assert steps==list(range(t['B'],t['N']+1,t['B']))
  optimizer_names=validate_optimizer(optimizer.state_dict(),model,t['N']//t['B'],cfg);validate_rng(end_rng);verify_input_evidence(inputs['start'])
  model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
  audit=dict(status='adaptation_passed_evaluation_pending',task=t,protocol=PROTOCOL,implementation_manifest_sha256=sha(O/'implementation_manifest.json'),helpers_sha256=implementation['helpers_sha256'],source_sha256=accepted['final_sha256'],source_final=accepted['final'],source_training_config=accepted['cfg'],source_training_code_sha256=accepted['code_sha256'],code_sha256=code,config_sha256=sha(t['config']),adapted_sha256=sha(path),checkpoint_sha256=sha(early),observations_sha256=sha(output/'observations.json'),input_evidence_sha256=sha(output/'input_evidence.json'),N_seen=len(data_loader.ids),N_unique=len(set(data_loader.ids)),optimizer_steps=len(steps),optimizer_state_names=optimizer_names,step_observation_counts=steps,forward_calls=len(losses),strict_reload=True,adaptation_seconds=time.time()-started,target_labels_used_for_adaptation=False,raw_image_content_hashed=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),**scope)
  atomic(output/'adaptation_audit.json',audit);event('adaptation_complete',loss=sum(losses[-t['B']:])/t['B'],checkpoint=str(path))
 cli.do_adapt=adapt;original_valid=cli.do_valid
 def valid(*a,**kw):
  verify_input_evidence(inputs['start']);event('evaluation_start',batch=0,batches=inputs['start']['csv']['sample_count'])
  def progress(fields):event('evaluation_progress',**fields)
  kw.update(return_summary=True,event_callback=progress);summary=original_valid(*a,**kw)
  atomic(output/'evaluation_summary.json',summary);verify_input_evidence(inputs['start']);event('evaluation_complete',summary=summary,n_samples=summary['n_samples'],score=summary['score'],batch=summary['n_samples'],batches=inputs['start']['csv']['sample_count'])
  # Frozen CLI only stores this score; None preserves zero accepted H without inventing zero.
  return summary['score']
 cli.do_valid=valid;sys.argv=[str(root/'tools/odr.py'),'--cfg',t['config']];cli.main(cli.cfg)
 event('odr_complete',checkpoint=str(output/'adapted_before_eval.pth.tar'))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  emit_event('error',error=repr(error),message='ODR stopped; retain existing adaptation/evaluation evidence, no retry');raise
