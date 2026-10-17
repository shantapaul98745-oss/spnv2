"""Table6 PARTIAL: actual frozen-source CPU construction and static parameter scopes."""
import argparse,csv,datetime,gc,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parents[1];P=R.parent;SOURCE=P/'source';RUNS=P.parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,obj):
 tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)
assert os.environ.get('STY'),'Run CPU model construction in own unique guarded screen'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('TORCH_HOME')==str(RUNS/'env/cache/torch')
sys.path.insert(0,str(SOURCE/'core'))
import numpy as np,torch
from torch import nn
from config import cfg as defaults,update_config
from nets import build_spnv2
torch.set_num_threads(2)
CONFIGS=['table4_phi0_bn','phi3_bn_ehs','table4_phi3_gn','phi6_gn_ehs']
FROZEN=json.loads((P/'source_manifest.json').read_text())['core_tools_sha256']
def code():return {str(f.relative_to(SOURCE)):sha(f) for d in ['core','tools'] for f in sorted((SOURCE/d).rglob('*.py'))}
def cache_hashes():
 paths=[RUNS/'cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth',RUNS/'env/cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth',RUNS/'env/cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth']
 return {str(p):sha(p) if p.is_file() else None for p in paths}
def inventory(module,prefix=''):
 return [dict(name=prefix+n,shape=list(p.shape),numel=p.numel(),dtype=str(p.dtype),element_size=p.element_size()) for n,p in module.named_parameters()]
def amount(items):return sum(x['numel'] for x in items)
def buffers(module,prefix=''):
 return [dict(name=prefix+n,shape=list(b.shape),numel=b.numel(),dtype=str(b.dtype),element_size=b.element_size()) for n,b in module.named_buffers()]
def norm_inventory(module):
 rows=[]
 for name,m in module.named_modules():
  if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.GroupNorm)):
   params=[dict(name=name+'.'+n,shape=list(p.shape),numel=p.numel()) for n,p in m.named_parameters(recurse=False)]
   bufs=[dict(name=name+'.'+n,shape=list(b.shape),numel=b.numel(),dtype=str(b.dtype)) for n,b in m.named_buffers(recurse=False)]
   row=dict(name=name,type=type(m).__name__,affine_parameters=amount(params),parameter_names=[x['name'] for x in params],buffers=bufs,buffer_elements=amount(bufs))
   if isinstance(m,nn.GroupNorm):row.update(groups=m.num_groups,channels=m.num_channels)
   else:row.update(momentum=m.momentum,eps=m.eps,channels=m.num_features,track_running_stats=m.track_running_stats)
   rows.append(row)
 return rows

def collect(config_name):
 assert code()==FROZEN,'Frozen source changed'
 cfg_path=P/'configs'/(config_name+'.yaml');cfg=defaults.clone();update_config(cfg,SimpleNamespace(cfg=str(cfg_path),opts=[]))
 assert cfg.SEED==2021 and not cfg.MODEL.PRETRAIN_FILE,'Static source initialization must not load an experiment result'
 assert cfg.MODEL.BACKBONE.PRETRAINED is None,'Author GN initialization was not established'
 if not cfg.MODEL.USE_GROUPNORM_BACKBONE:
  filename='efficientnet_b0_rwightman-3dd342df.pth' if cfg.MODEL.EFFICIENTDET_PHI==0 else 'efficientnet_b3_rwightman-cf984f9c.pth'
  assert (RUNS/'env/cache/torch/hub/checkpoints'/filename).is_file(),'Required local asset missing; forbid implicit download'
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021)
 model=build_spnv2(cfg);assert all(p.device.type=='cpu' for p in model.parameters())
 whole=inventory(model);backbone=inventory(model.backbone,'backbone.');encoder=inventory(model.backbone.efficientnet,'backbone.efficientnet.');bifpn=inventory(model.backbone.bifpn,'backbone.bifpn.')
 heads={name:inventory(head,f'heads.{i}.') for i,(name,head) in enumerate(zip(model.head_names,model.heads))}
 assert len(heads)==len(model.head_names)==len(model.heads)
 names=lambda rows:{x['name'] for x in rows}
 assert names(encoder).isdisjoint(names(bifpn)) and names(backbone)==names(encoder)|names(bifpn)
 assert names(backbone).isdisjoint(set().union(*(names(x) for x in heads.values())))
 assert names(whole)==names(backbone)|set().union(*(names(x) for x in heads.values()))
 assert amount(whole)==amount(backbone)+sum(amount(x) for x in heads.values())
 assert amount(backbone)==amount(encoder)+amount(bifpn)
 # Mirror the public ODR scope: all network params frozen, backbone BN/GN affine enabled.
 model.eval();model.requires_grad_(False);allowed=set();norm_type=nn.GroupNorm if cfg.MODEL.USE_GROUPNORM_BACKBONE else nn.BatchNorm2d
 for name,m in model.backbone.named_modules():
  if isinstance(m,norm_type):
   m.requires_grad_(True);allowed.update('backbone.'+name+'.'+n for n,_ in m.named_parameters(recurse=False))
 actual={n for n,p in model.named_parameters() if p.requires_grad}
 assert allowed==actual and actual and actual<=names(backbone)
 odr=[x for x in whole if x['name'] in actual];assert all(not p.requires_grad for head in model.heads for p in head.parameters())
 assert not any(n.startswith('heads.') for n in actual)
 norm_rows=norm_inventory(model);buf_rows=buffers(model)
 zero_residual=[]
 if cfg.MODEL.USE_GROUPNORM_BACKBONE:
  for n,m in model.backbone.efficientnet.named_modules():
   if type(m).__name__=='MBConv' and m.use_res_connect:
    for sub,layer in m.named_modules():
     if isinstance(layer,(nn.BatchNorm2d,nn.GroupNorm)) and sub=='block.3.1':
      assert bool((layer.weight==0).all());zero_residual.append('backbone.efficientnet.'+n+'.'+sub+'.weight')
  assert zero_residual,'Expected public GN zero-residual construction not found'
  initialization=dict(kind='public random GN construction plus zero residual weights',seed=2021,author_initialization_verified=False,pretrained_backbone=None,zero_residual_norm_weight_names=zero_residual,disclosure='No author GN ImageNet-100 initialization file established; do not describe these initialized CPU models as trained final models')
 else:
  name='efficientnet_b0_rwightman-3dd342df.pth' if cfg.MODEL.EFFICIENTDET_PHI==0 else 'efficientnet_b3_rwightman-cf984f9c.pth';asset=RUNS/'env/cache/torch/hub/checkpoints'/name
  initialization=dict(kind='official ImageNet pretrained BN EfficientNet; other modules public seeded initialization',seed=2021,backbone_asset=str(asset),backbone_sha256=sha(asset),author_initialization_verified=False)
 row=dict(status='parameters_verified',config=config_name,config_path=str(cfg_path),config_sha256=sha(cfg_path),resolved_config_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(),phi=cfg.MODEL.EFFICIENTDET_PHI,backbone_norm='GN' if cfg.MODEL.USE_GROUPNORM_BACKBONE else 'BN',seed=2021,source=str(SOURCE),code_sha256=FROZEN,initialization=initialization,
  parameter_counts=dict(whole_spnv2=amount(whole),efficientdet_backbone=amount(backbone),efficientnet_encoder=amount(encoder),bifpn=amount(bifpn),heads={name:amount(v) for name,v in heads.items()},odr_updated_backbone_norm_affine=amount(odr),odr_heads=0),
  parameter_tensors=dict(whole=len(whole),backbone=len(backbone),encoder=len(encoder),bifpn=len(bifpn),odr=len(odr)),
  parameter_storage_bytes=sum(x['numel']*x['element_size'] for x in whole),buffer_storage_bytes=sum(x['numel']*x['element_size'] for x in buf_rows),buffers=buf_rows,normalization=norm_rows,
  full_parameter_inventory=whole,odr_allowed_parameters=odr,odr_allowed_names=sorted(actual),head_parameter_names={name:sorted(names(v)) for name,v in heads.items()},
  validations=dict(whole_equals_backbone_plus_heads=True,backbone_equals_encoder_plus_bifpn=True,scopes_disjoint=True,odr_subset_backbone=True,heads_odr_zero=True,all_parameters_cpu=True,no_forward_executed=True),
  gpu_resource_measurements=None,gpu_resource_status='pending; CPU parameter/storage counts are not latency, allocated VRAM or peak memory measurements')
 atomic(R/'reports'/(config_name+'_parameters.json'),row)
 del model;gc.collect();return row

started=datetime.datetime.now().astimezone().isoformat();rows=[];limitations=[];before=cache_hashes();report={}
try:
 download=json.loads((R/'reports/b0_download.json').read_text());assert download['B3_preserved_byte_identical']
 for name in CONFIGS:
  if name=='table4_phi0_bn' and download['status']!='passed':
   limitations.append(dict(config=name,status='unavailable',parameter_counts=None,reason='Official B0 download did not match required filename SHA prefix; no installation or alternate initialization',download_report=str(R/'reports/b0_download.json')));print('LIMITED',name,flush=True);continue
  print('CONSTRUCT_CPU',name,flush=True);rows.append(collect(name));print('COUNTS',name,json.dumps(rows[-1]['parameter_counts']),flush=True)
 assert cache_hashes()==before and code()==FROZEN
 report=dict(status='PARTIAL_parameters_only',started_at=started,completed_at=datetime.datetime.now().astimezone().isoformat(),models=rows,cache_before_sha256=before,cache_after_sha256=cache_hashes(),source_manifest_sha256=sha(P/'source_manifest.json'),B3_cache_preserved=True,code_frozen_unchanged=True,all_four_parameter_scopes_verified=len(rows)==4,limitations=limitations,gpu_measurements=None,gpu_status='pending; no GPU forward, latency or peak-resource experiment performed',scientific_accuracy_results_created=False)
 fields=['config','status','phi','backbone_norm','whole_spnv2','efficientdet_backbone','efficientnet_encoder','bifpn','heatmap','efficientpose','segmentation','odr_updated_backbone_norm_affine','odr_heads']
 with (R/'reports/table6_parameters_partial.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:
   counts=row['parameter_counts'];w.writerow(dict(config=row['config'],status=row['status'],phi=row['phi'],backbone_norm=row['backbone_norm'],**{k:counts[k] for k in fields if k in counts},**counts['heads']))
  for limitation in limitations:w.writerow(dict(config=limitation['config'],status='unavailable',phi=0,backbone_norm='BN'))
except BaseException as error:
 report=dict(status='failed_or_limited',started_at=started,completed_models=[r['config'] for r in rows],error=repr(error),traceback=traceback.format_exc(),cache_before_sha256=before,cache_after_sha256=cache_hashes(),gpu_measurements=None,scientific_accuracy_results_created=False)
 raise
finally:
 atomic(R/'reports/table6_parameters_partial.json',report);print(json.dumps({k:v for k,v in report.items() if k!='models'},indent=2),flush=True)
