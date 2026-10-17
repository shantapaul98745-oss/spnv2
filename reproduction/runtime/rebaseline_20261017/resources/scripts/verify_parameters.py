"""CPU acceptance of provenance, parameter partitions, ODR scope and explicit gaps."""
from pathlib import Path
import hashlib,json,math,os,datetime
R=Path(__file__).resolve().parents[1];P=R.parent
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def total(rows):
 for x in rows:assert math.prod(x['shape'])==x['numel']
 return sum(x['numel'] for x in rows)
try:
 report=json.loads((R/'reports/table6_parameters_partial.json').read_text());assert report['status']=='PARTIAL_parameters_only' and report['gpu_measurements'] is None
 assert report['code_frozen_unchanged'] and report['cache_before_sha256']==report['cache_after_sha256'] and report['B3_cache_preserved']
 assert report['source_manifest_sha256']==sha(P/'source_manifest.json')
 code=json.loads((P/'source_manifest.json').read_text())['core_tools_sha256']
 actual={str(f.relative_to(P/'source')):sha(f) for d in ['core','tools'] for f in sorted((P/'source'/d).rglob('*.py'))}
 assert actual==code
 checks=[]
 for model in report['models']:
  name=model['config'];counts=model['parameter_counts'];inventory=model['full_parameter_inventory'];byname={r['name']:r for r in inventory};assert len(byname)==len(inventory)
  assert model['code_sha256']==code and model['config_sha256']==sha(model['config_path'])
  whole=total(inventory);backbone=total([r for r in inventory if r['name'].startswith('backbone.')]);encoder=total([r for r in inventory if r['name'].startswith('backbone.efficientnet.')]);bifpn=total([r for r in inventory if r['name'].startswith('backbone.bifpn.')]);head_names=set().union(*(set(v) for v in model['head_parameter_names'].values()))
  assert set(byname)=={n for n in byname if n.startswith('backbone.')}|head_names
  assert whole==counts['whole_spnv2']==backbone+sum(counts['heads'].values())
  assert backbone==counts['efficientdet_backbone']==encoder+bifpn and encoder==counts['efficientnet_encoder'] and bifpn==counts['bifpn']
  for head,names in model['head_parameter_names'].items():assert total([byname[n] for n in names])==counts['heads'][head]
  selected=set(model['odr_allowed_names']);assert selected=={r['name'] for r in model['odr_allowed_parameters']}
  norm_names=set()
  wanted='GroupNorm' if model['backbone_norm']=='GN' else 'BatchNorm2d'
  for norm in model['normalization']:
   assert sum(byname[n]['numel'] for n in norm['parameter_names'])==norm['affine_parameters']
   if norm['name'].startswith('backbone.') and (norm['type'] in {'GroupNorm','MyGroupNorm8','MyGroupNorm16'} if wanted=='GroupNorm' else norm['type']==wanted):norm_names.update(norm['parameter_names'])
  assert selected==norm_names and selected<=set(byname) and selected.isdisjoint(head_names)
  assert total(model['odr_allowed_parameters'])==sum(byname[n]['numel'] for n in selected)==counts['odr_updated_backbone_norm_affine'] and counts['odr_heads']==0
  assert model['parameter_storage_bytes']==sum(r['numel']*r['element_size'] for r in inventory)
  assert model['buffer_storage_bytes']==sum(r['numel']*r['element_size'] for r in model['buffers'])
  assert all(model['validations'].values()) and model['gpu_resource_measurements'] is None
  if model['backbone_norm']=='GN':
   init=model['initialization'];assert init['author_initialization_verified'] is False and init['pretrained_backbone'] is None
   assert set(init['zero_residual_norm_weight_names'])<=selected and init['zero_residual_norm_weight_names']
  checks.append(dict(config=name,status='passed',parameter_counts=counts,normalization_modules={kind:sum((n['type'] in {'GroupNorm','MyGroupNorm8','MyGroupNorm16'} if kind=='GroupNorm' else n['type']==kind) for n in model['normalization']) for kind in ['BatchNorm2d','GroupNorm']}))
 available={r['config'] for r in report['models']};limited={r['config'] for r in report['limitations']}
 assert available|limited=={'table4_phi0_bn','phi3_bn_ehs','table4_phi3_gn','phi6_gn_ehs'} and available.isdisjoint(limited)
 download=json.loads((R/'reports/b0_download.json').read_text());assert download['B3_preserved_byte_identical']
 if download['status']=='failed':assert limited=={'table4_phi0_bn'} and report['all_four_parameter_scopes_verified'] is False and not Path(download['target']).exists()
 else:assert report['all_four_parameter_scopes_verified'] is True and len(available)==4
 for item in report['limitations']:assert item['parameter_counts'] is None
 for file,expected in report['cache_before_sha256'].items():assert (sha(file) if Path(file).is_file() else None)==expected
 acceptance=dict(status='passed_with_explicit_limitations' if limited else 'passed_parameters_only',scope='Table6 PARTIAL parameters only; no GPU measurement or accuracy evidence',checks=checks,available_models=sorted(available),limited_models=sorted(limited),table_report_sha256=sha(R/'reports/table6_parameters_partial.json'),download_report_sha256=sha(R/'reports/b0_download.json'),gpu_resources_pending=True,checked_at=datetime.datetime.now().astimezone().isoformat())
 (R/'reports/parameter_acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n');print(json.dumps(acceptance,indent=2))
except BaseException as error:
 import traceback
 failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),scope='CPU parameter-report verifier; no scientific results generated')
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
 (R/'reports'/('parameter_acceptance_failed_'+stamp+'.json')).write_text(json.dumps(failure,indent=2)+'\n')
 raise
