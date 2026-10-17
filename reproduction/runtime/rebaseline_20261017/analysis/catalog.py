"""Resolve expected fixed-final evidence from actual configs and parent queue commands."""
import yaml
from core import *
CONFIGS=['phi3_bn_ehs','phi6_gn_ehs','table2_h','table2_e','table2_eh','table3_base','table3_erase','table3_flare','table4_phi0_bn','table4_phi3_gn']
def option(command,name,default=None):return command[command.index(name)+1] if name in command else default
def discover():
 queue=read(PLAN/'queue_v2.json');trains={};evals={}
 for t in queue['tasks']:
  command=t.get('command',[]);verify=t.get('verify_command',[])
  if 'verify_train' in verify:trains[Path(option(verify,'--cfg')).stem]=option(verify,'--report')
  if 'evaluate' in command and any('source_artifacts.py' in x for x in command):evals[str(Path(option(command,'--cfg')).resolve())]=(command,verify)
 sources=[];rows=[]
 for name in CONFIGS:
  path=PLAN/'configs'/(name+'.yaml');cfg=yaml.safe_load(required(path).read_text());final=Path(cfg['OUTPUT_DIR'])/cfg['MODEL']['BACKBONE']['NAME']/cfg['EXP_NAME']/'model_final.pth.tar'
  proof=Path(trains.get(name,str(PLAN/'reports'/('source_phi3_verified.json' if name=='phi3_bn_ehs' else 'source_phi6_verified.json' if name=='phi6_gn_ehs' else 'source_'+name+'_verified.json'))))
  source=dict(config=name,config_path=str(path),final=str(final),source_report=str(proof),phi=cfg['MODEL']['EFFICIENTDET_PHI'],heads=cfg['TEST']['HEAD'],source_training_seed=cfg['SEED'])
  sources.append(source)
  for domain,total in [('synthetic',11994),('lightbox',6740),('sunlamp',2791)]:
   evalcfg=PLAN/'configs'/f'{name}_eval_{domain}.yaml';ecfg=yaml.safe_load(required(evalcfg).read_text());match=evals.get(str(evalcfg.resolve()))
   command,verify=match if match else ([],[])
   raw=option(command,'--report',str(PLAN/'reports'/f'eval_{name}_{domain}.json'));accepted=option(verify,'--report',str(PLAN/'reports'/f'eval_{name}_{domain}_verified.json'));output=option(command,'--output',str(PLAN/'evaluations'/name/domain))
   rows.append(dict(id=f'source_{name}_{domain}',stage='source',cohort='new',source_config=name,source_final=str(final),source_report=str(proof),source_config_path=str(path),domain=domain,heads=ecfg['TEST']['HEAD'],expected_total=total,accepted_report=accepted,pending_report=raw,folder=str(Path(output)/domain),csv=str(DATA/domain/'labels'/('validation.csv' if domain=='synthetic' else 'test.csv')),config=str(evalcfg)))
 odr=read(PLAN/'odr/manifest.json');odrrows=[]
 for t in odr['tasks']:
  name='phi3_bn_ehs' if t['phi']==3 else 'phi6_gn_ehs'
  odrrows.append(dict(id=t['id'],stage='odr',cohort='new',source_config=name,source_final=t['source_final'],source_report=t['source_report'],source_config_path=str(PLAN/'configs'/(name+'.yaml')),domain=t['domain'],heads=['heatmap','efficientpose'],expected_total=6740 if t['domain']=='lightbox' else 2791,accepted_report=str(PLAN/'odr/reports'/(t['id']+'.json')),pending_report=str(Path(t['output'])/'adaptation_audit.json'),folder=str(Path(t['output'])/t['domain']),csv=str(DATA/t['domain']/'labels/test.csv'),config=t['config'],line=t['line'],N=t['N'],B=t['B'],seed=t['seed'],group=t['group'],task=t,is_alias=bool(t.get('reuse_of')),canonical_id=t.get('reuse_of',t['id'])))
 return dict(sources=sources,records=rows+odrrows,expected_source_evaluations=30,expected_odr_requests=400,expected_unique_odr=370,protocol=odr['protocol'],source_policy='fixed final; first accepted SHA pinned, no target-dependent checkpoint choice',inputs=dict(queue=str(PLAN/'queue_v2.json'),queue_sha256=sha(PLAN/'queue_v2.json'),ODR_manifest_sha256=sha(PLAN/'odr/manifest.json')))
