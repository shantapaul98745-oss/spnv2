"""CPU-only audit of an immutable capture from the live atomic checkpoint."""
import argparse,csv,hashlib,json,os,random,shutil,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--cfg',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--capture',required=True);p.add_argument('--log',required=True);p.add_argument('--report',required=True);a=p.parse_args()
plan=Path(__file__).resolve().parents[1];source=plan/'source';sys.path.insert(0,str(source/'core'))
import numpy as np,torch
from config import cfg as defaults,update_config
from nets import build_spnv2
from solver import get_optimizer
from utils.training_state import EpochBatchSampler,model_digest,sha256_file,assert_finite_weights
from utils.utils import set_seeds_cudnn
torch.set_num_threads(2)
result=dict(status='failed',started_at=time.time(),capture=a.capture)
try:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU isolation required'
 capture=Path(a.capture);assert not capture.exists();capture.parent.mkdir(parents=True,exist_ok=True)
 try:os.link(a.checkpoint,capture)
 except OSError:shutil.copyfile(a.checkpoint,capture)
 payload=torch.load(capture,map_location='cpu',weights_only=True)
 assert {'format','state_dict','optimizer','scaler','rng','sampler','manifest','training_state'}.issubset(payload)
 assert payload['format']=='spnv2-boundary-v1'
 cfg=defaults.clone();update_config(cfg,SimpleNamespace(cfg=a.cfg,opts=[]))
 normalized=cfg.clone();normalized.defrost();normalized.TRAIN.RESUME_FILE='';normalized.AUTO_RESUME=False
 manifest=payload['manifest'];assert hashlib.sha256(normalized.dump().encode()).hexdigest()==manifest['config_sha256']
 codes={str(f.relative_to(source)):sha256_file(f) for folder in ('core','tools') for f in sorted((source/folder).rglob('*.py'))}
 assert codes==manifest['code']==json.loads((plan/'source_manifest.json').read_text())['core_tools_sha256']
 for file,sha in manifest['csv'].items():assert sha256_file(file)==sha,file
 for g in manifest['auxiliary_geometry'].values():
  path=Path(g['path']);assert path.is_file()==g['exists']
  if g['exists']:assert sha256_file(path)==g['sha256']
 for name,init in manifest['initialization'].items():
  if isinstance(init,dict):assert sha256_file(init['path'])==init['sha256']
  else:assert sha256_file(name)==init
 inventory=manifest['data_inventory'];assert inventory['raw_content_hashed'] is False
 assert hashlib.sha256(json.dumps(inventory['records'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==inventory['sha256']
 for split,row,file,size,mtime in inventory['records']:
  stat=Path(file).stat();assert (stat.st_size,stat.st_mtime_ns)==(size,mtime),file
 with open(Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/cfg.TRAIN.TRAIN_CSV) as stream:ids=[row[0] for row in csv.reader(stream)]
 physical=cfg.TRAIN.IMAGES_PER_GPU;k=cfg.TRAIN.ACCUM_STEPS;state=payload['training_state']
 sampler=EpochBatchSampler(len(ids),physical,k,cfg.SEED,cfg.TRAIN.SHUFFLE);sampler.set_epoch(state['epoch'],state['next_batch'])
 assert sampler.state_dict()==payload['sampler']
 expected_micro=state['epoch']*sampler.total_batches+state['next_batch']
 assert state['microbatches_seen']==expected_micro and state['samples_seen']==expected_micro*physical
 assert state['next_batch']%k==0 and state['update_attempts']==expected_micro//k
 assert state['optimizer_updates']+state['amp_skips']==state['update_attempts'] and state['optimizer_updates']>0
 set_seeds_cudnn(cfg,seed=cfg.SEED);model=build_spnv2(cfg)
 assert model_digest(model)==manifest['initial_state_sha256'],'Initial model differs from independent cfg/seed reconstruction'
 model.load_state_dict(payload['state_dict'],strict=True);assert_finite_weights(model)
 optimizer=get_optimizer(cfg,model);optimizer.load_state_dict(payload['optimizer']);assert optimizer.state
 moment_tensors=0
 for param,values in optimizer.state.items():
  for name,value in values.items():
   if torch.is_tensor(value):
    assert bool(torch.isfinite(value).all()),name
    if name!='step':assert value.shape==param.shape
    moment_tensors+=1
 scaler=torch.amp.GradScaler('cpu');scaler.load_state_dict(payload['scaler']);assert scaler.state_dict()==payload['scaler']
 rng=payload['rng'];assert {'python','numpy','torch_cpu','torch_cuda'}.issubset(rng)
 random.Random().setstate(rng['python']);n=rng['numpy'];np.random.RandomState().set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
 torch.Generator().set_state(rng['torch_cpu']);assert len(rng['torch_cuda'])==1 and all(v.dtype==torch.uint8 and v.numel()>0 for v in rng['torch_cuda'])
 updates=[]
 with open(a.log) as stream:
  for line in stream:
   if line.startswith('SPNV2_EVENT '):
    event=json.loads(line[len('SPNV2_EVENT '):])
    if event['event']=='update' and event['update_attempts']<=state['update_attempts']:updates.append(event)
 assert len(updates)==state['update_attempts']
 expected_ids=[]
 for epoch in range(state['epoch']+1):
  sampler.set_epoch(epoch,0);order=sampler.indices()
  if epoch==state['epoch']:order=order[:state['next_batch']*physical]
  expected_ids.extend(ids[i] for i in order)
 observed=[i for e in updates for i in e['sample_ids']];assert observed==expected_ids
 assert not torch.cuda.is_initialized(),'CPU audit initialized CUDA unexpectedly'
 result.update(status='passed',checkpoint_sha256=sha256_file(capture),state=state,strict_model_reload=True,independent_initialization_match=True,optimizer_state_loaded=True,optimizer_tensors_checked=moment_tensors,scaler_state_loaded=True,rng_schema_valid=True,actual_sample_ids_match=True,data_inventory_sha256=inventory['sha256'],data_coverage=inventory['coverage'],code_sha256=codes,raw_image_contents_hashed=False)
except Exception as exc:
 result.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());print(result['traceback'],file=sys.stderr)
finally:
 result['finished_at']=time.time();out=Path(a.report);assert not out.exists();temp=out.with_name('.'+out.name+'.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));os.replace(temp,out)
print(json.dumps({k:v for k,v in result.items() if k not in ['code_sha256','traceback']}));sys.exit(0 if result['status']=='passed' else 1)
