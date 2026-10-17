"""Final-only source evaluation and independent CPU artifact verification."""
import argparse,csv,hashlib,json,math,os,sys
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify_train','evaluate','verify_eval']);p.add_argument('--cfg',required=True);p.add_argument('--report',required=True);p.add_argument('--log');p.add_argument('--source-report');p.add_argument('--output');p.add_argument('--expected-samples',type=int)
a=p.parse_args();plan=Path(__file__).resolve().parents[1];source=plan/'source';sys.path.insert(0,str(source/'core'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from config import cfg as defaults,update_config
from nets import build_spnv2
from utils.training_state import sha256_file,assert_finite_weights,emit_event,model_digest,atomic_torch_save
from utils.utils import set_seeds_cudnn,load_camera_intrinsics,load_tango_3d_keypoints
cfg=defaults.clone();update_config(cfg,SimpleNamespace(cfg=a.cfg,opts=[]))
torch.set_num_threads(2)
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('w') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,path)
def codes():return {str(f.relative_to(source)):sha256_file(f) for folder in ('core','tools') for f in sorted((source/folder).rglob('*.py'))}
def verify_frozen():
 expected=json.loads((plan/'source_manifest.json').read_text())['core_tools_sha256'];assert codes()==expected,'Frozen source code changed'
 return expected
verify_frozen()
if a.mode=='verify_train':
 output=Path(cfg.OUTPUT_DIR)/cfg.MODEL.BACKBONE.NAME/cfg.EXP_NAME
 final=output/'model_final.pth.tar';checkpoint=output/'checkpoint.pth.tar';initial=output/'initial_state.pth.tar'
 for f in [final,checkpoint,initial,output/'manifest.json',output/'reproducible_run.json']:assert f.is_file(),str(f)
 payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
 start=torch.load(initial,map_location='cpu',weights_only=True)
 assert payload['format']==start['format']=='spnv2-boundary-v1'
 assert payload['manifest']==start['manifest']==json.loads((output/'manifest.json').read_text())
 assert payload['manifest']['code']==codes()
 normalized=cfg.clone();normalized.defrost();normalized.TRAIN.RESUME_FILE='';normalized.AUTO_RESUME=False
 assert payload['manifest']['config_sha256']==hashlib.sha256(normalized.dump().encode()).hexdigest()
 n=sum(1 for _ in csv.reader(open(Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/cfg.TRAIN.TRAIN_CSV)))
 physical=cfg.TRAIN.IMAGES_PER_GPU;k=cfg.TRAIN.ACCUM_STEPS;windows=n//(physical*k);epochs=cfg.TRAIN.END_EPOCH
 state=payload['training_state'];assert (state['epoch'],state['next_batch'])==(epochs,0),state
 assert state['update_attempts']==windows*epochs and state['microbatches_seen']==windows*k*epochs,state
 assert state['samples_seen']==windows*k*physical*epochs,state
 assert state['optimizer_updates']+state['amp_skips']==state['update_attempts'] and state['optimizer_updates']>0,state
 assert state['consecutive_amp_skips']<cfg.TRAIN.AMP_MAX_CONSECUTIVE_SKIPS
 initial_state=start['training_state'];assert initial_state['epoch']==0 and initial_state['next_batch']==0 and initial_state['update_attempts']==0
 set_seeds_cudnn(cfg,seed=cfg.SEED);model=build_spnv2(cfg)
 assert model_digest(model)==start['manifest']['initial_state_sha256'],'Initialization differs from independent seed/config reconstruction'
 model.load_state_dict(start['state_dict'],strict=True);assert_finite_weights(model)
 assert model_digest(model)==start['manifest']['initial_state_sha256']
 final_state=torch.load(final,map_location='cpu',weights_only=True);model.load_state_dict(final_state,strict=True);assert_finite_weights(model)
 assert set(final_state)==set(payload['state_dict'])
 for key,value in final_state.items():assert torch.equal(value,payload['state_dict'][key]),key
 for file,digest in payload['manifest']['csv'].items():assert sha256_file(file)==digest,file
 from solver import get_optimizer
 from utils.training_state import EpochBatchSampler,TrainingState,tensors_are_finite
 required={'optimizer','scaler','rng','sampler','manifest','training_state','state_dict','format'}
 assert required.issubset(payload),'Incomplete recovery checkpoint'
 sampler=EpochBatchSampler(n,physical,k,cfg.SEED,cfg.TRAIN.SHUFFLE);sampler.set_epoch(epochs,0)
 assert payload['sampler']==sampler.state_dict(),'Final sampler state mismatch'
 optimizer=get_optimizer(cfg,model);optimizer.load_state_dict(payload['optimizer'])
 for param,values in optimizer.state.items():
  for name,value in values.items():
   if torch.is_tensor(value):
    assert bool(torch.isfinite(value).all()),name
    if name!='step':assert value.shape==param.shape,(name,value.shape,param.shape)
 assert optimizer.state,'Missing optimizer moments'
 assert isinstance(payload['scaler'],dict) and {'scale','growth_factor','backoff_factor','growth_interval','_growth_tracker'}.issubset(payload['scaler'])
 scaler=torch.amp.GradScaler('cpu');scaler.load_state_dict(payload['scaler']);assert scaler.state_dict()==payload['scaler']
 rng=payload['rng'];assert {'python','numpy','torch_cpu','torch_cuda'}.issubset(rng)
 import random,numpy as np
 random.Random().setstate(rng['python'])
 npstate=rng['numpy'];np.random.RandomState().set_state((npstate['kind'],np.array(npstate['keys'],dtype=np.uint32),npstate['pos'],npstate['has_gauss'],npstate['cached_gaussian']))
 torch.Generator().set_state(rng['torch_cpu']);assert len(rng['torch_cuda'])==1
 assert all(torch.is_tensor(v) and v.dtype==torch.uint8 and v.numel()>0 for v in rng['torch_cuda'])
 for geometry in payload['manifest']['auxiliary_geometry'].values():
  path=Path(geometry['path']);assert path.is_file()==geometry['exists']
  if geometry['exists']:assert sha256_file(path)==geometry['sha256']
 inventory=payload['manifest']['data_inventory'];assert inventory and inventory['raw_content_hashed'] is False
 assert hashlib.sha256(json.dumps(inventory['records'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==inventory['sha256']
 for split,row,file,size,mtime in inventory['records']:
  stat=Path(file).stat();assert (stat.st_size,stat.st_mtime_ns)==(size,mtime),file
 for name,initialization in payload['manifest']['initialization'].items():
  if isinstance(initialization,dict):assert sha256_file(initialization['path'])==initialization['sha256']
  else:assert sha256_file(name)==initialization
 if a.log:
  events=[]
  with open(a.log) as stream:
   for line in stream:
    if line.startswith('SPNV2_EVENT '):events.append(json.loads(line[len('SPNV2_EVENT '):]))
  complete=[e for e in events if e['event']=='training_complete'];assert len(complete)==1
  assert complete[0]['final_sha256']==sha256_file(final)
  validation=[e for e in events if e['event']=='validation_end'];assert len(validation)==epochs
  updates=[e for e in events if e['event']=='update'];assert len(updates)==windows*epochs
  with open(Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/cfg.TRAIN.TRAIN_CSV) as stream:ids=[r[0] for r in csv.reader(stream)]
  for epoch in range(epochs):
   sampler.set_epoch(epoch,0);order=sampler.indices()
   for window in range(windows):
    event=updates[epoch*windows+window];offset=window*physical*k
    assert event['epoch']==epoch+1 and event['batch']==(window+1)*k and event['update_attempts']==epoch*windows+window+1
    assert event['sample_ids']==[ids[i] for i in order[offset:offset+physical*k]],'Observed training sample sequence mismatch'
  for e in validation:
   assert e['n_samples']==1199 and e['metrics']['final_pose']['count']==e['heatmap_valid'] if 'heatmap' in cfg.TEST.HEAD else e['n_samples']==1199
   for m in e['metrics'].values():assert (m['mean'] is None)==(m['count']==0)
 report=dict(status='passed',mode=a.mode,source=str(source),cfg=a.cfg,final=str(final),final_sha256=sha256_file(final),initial_state_sha256=start['manifest']['initial_state_sha256'],checkpoint_sha256=sha256_file(checkpoint),state=state,physical_batch=physical,accum_steps=k,logical_windows_per_epoch=windows,samples_per_epoch=windows*k*physical,code_sha256=codes(),strict_reload=True,final_matches_checkpoint=True)
 final.chmod(0o444)
 atomic(a.report,report);print(json.dumps(report))
elif a.mode=='evaluate':
 from dataset import get_dataloader
 from engine.inference import do_valid
 final=Path(cfg.TEST.MODEL_FILE);accepted=json.loads(Path(a.source_report).read_text())
 assert accepted['status']=='passed' and str(final)==accepted['final'] and sha256_file(final)==accepted['final_sha256']
 output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
 assert cfg.TEST.IMAGES_PER_GPU==1 and not cfg.TRAIN.REPRODUCIBLE
 set_seeds_cudnn(cfg,seed=cfg.SEED);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 device=torch.device('cuda:0');model=build_spnv2(cfg);model.load_state_dict(torch.load(final,map_location='cpu',weights_only=True),strict=True);assert_finite_weights(model);model=model.to(device)
 data=get_dataloader(cfg,split='test',load_labels=True);assert len(data.dataset)==a.expected_samples
 domain=cfg.TEST.TEST_CSV.split('/')[0]
 def progress(fields):emit_event('evaluation_progress',phase='evaluation',domain=domain,source=str(final),output=str(output),**fields)
 emit_event('evaluation_start',phase='evaluation',domain=domain,source=str(final),config=a.cfg,output=str(output),batches=len(data),epochs=1,epoch=1)
 summary=do_valid(0,cfg,model,data,load_camera_intrinsics(cfg.DATASET.CAMERA),load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS),device=device,log_dir=str(output),return_summary=True,event_callback=progress)
 assert summary['n_samples']==a.expected_samples
 report=dict(status='awaiting_verification',mode=a.mode,cfg=a.cfg,source=str(final),source_sha256=accepted['final_sha256'],source_report=a.source_report,expected_samples=a.expected_samples,domain=domain,results=str(output/domain),summary=summary,code_sha256=codes(),csv=str(Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/cfg.TEST.TEST_CSV))
 atomic(a.report,report);emit_event('evaluation_complete',phase='evaluation',domain=domain,report=a.report,summary=summary,source=str(final))
else:
 from verify_heads import verify
 report=json.loads(Path(a.source_report).read_text());assert report['status']=='awaiting_verification'
 assert report['code_sha256']==codes() and sha256_file(report['source'])==report['source_sha256']
 result=verify(report['results'],report['csv'],list(cfg.TEST.HEAD))
 assert result['n_samples']==a.expected_samples==report['summary']['n_samples']
 assert result['n_reject']==report['summary']['heatmap_rejections'] if 'heatmap' in cfg.TEST.HEAD else result['n_reject'] is None
 assert result['final_pose']==report['summary']['score']
 atomic(a.report,dict(status='passed',mode=a.mode,source_sha256=report['source_sha256'],metrics=result,summary=report['summary'],prediction_scope='Original acceptance protocol; missing H predictions retained as NaN; no zero filling.',source_report=a.source_report))
 print(json.dumps(result))
