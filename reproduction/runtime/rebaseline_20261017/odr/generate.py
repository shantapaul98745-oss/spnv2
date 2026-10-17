"""Freeze 40 fixed +120 B +240 N requests without reading target scores."""
import json,sys,copy,hashlib
from pathlib import Path
import yaml
from common import *
def generate(refresh=False):
 tasks=[];first={}
 def add(group,phi,line,domain,seed,n,b):
  tid=f'{group}_phi{phi}_{line}_{domain}_s{seed}_N{n}_B{b}'
  key=(phi,line,domain,seed,n,b)
  task=dict(id=tid,group=group,phi=phi,line=line,domain=domain,seed=seed,N=n,B=b,source_final=str(source_path(phi)),source_report=str(source_report(phi)))
  if key in first:task['reuse_of']=first[key]
  else:first[key]=tid
  task['config']=str(O/'configs'/(tid+'.yaml'));task['output']=str(O/'runs'/f'efficientdet_d{phi}'/tid)
  task['command']=[PYTHON,'-B','-s','-u',str(O/'audited_entry.py'),'--task',tid]
  task['queue_command']=[PYTHON,'-B','-s','-u',str(O/'run_task.py'),'--task',tid,'--execute']
  task['verify_command']=[PYTHON,'-B','-s','-u',str(O/'verify_result.py'),'--task',tid]
  task['requires_source_three_domain_acceptance']=True
  base=yaml.safe_load((PLAN.parent/'odr_seeds_20261005'/('odr_phi3_B4_N1024.yaml' if phi==3 else 'odr_phi6_GN8_N1024.yaml')).read_text())
  base.update(ROOT=str(O/('source_'+line)),OUTPUT_DIR=str(O/'runs'),LOG_DIR=str(O/'training_logs'),EXP_NAME=tid,SEED=seed,AUTO_RESUME=False,FP16=False,VERBOSE=False)
  base['DATASET']['ROOT']=str(DATA.parent);base['DATASET']['KEYPOINTS']=str(O/('source_'+line)/'core/utils/models/tangoPoints.mat')
  base['MODEL']['PRETRAIN_FILE']=task['source_final']
  base['ODR'].update(NUM_TRAIN_SAMPLES=n,IMAGES_PER_BATCH=b)
  base['TRAIN'].update(TRAIN_CSV=f'{domain}/labels/test.csv',VAL_CSV=f'{domain}/labels/test.csv',IMAGES_PER_GPU=1,ACCUM_STEPS=1,REPRODUCIBLE=False,SHUFFLE=True)
  base['TEST']['TEST_CSV']=f'{domain}/labels/test.csv'
  text=yaml.safe_dump(base,sort_keys=False)
  cfg=Path(task['config'])
  if cfg.exists():assert cfg.read_text()==text,'Refuse config regeneration drift'
  else:cfg.write_text(text)
  task['config_sha256']=sha(cfg);task['code_sha256_manifest']=sha(O/(line+'_sha256.json'))
  tasks.append(task)
 for phi,line,n,b in [(3,'A',1024,4),(6,'A',4096,1),(3,'B1',1024,4),(3,'B2',1024,4)]:
  for domain in ['lightbox','sunlamp']:
   for seed in range(2021,2026):add('fixed',phi,line,domain,seed,n,b)
 for line in ['A','B2']:
  for domain in ['lightbox','sunlamp']:
   for seed in range(2021,2026):
    for b in [1,2,4,8,16,32]:add('Bscan',3,line,domain,seed,1024,b)
 for phi in [3,6]:
  for domain in ['lightbox','sunlamp']:
   for seed in range(2021,2026):
    for k in range(2,14):add('Nscan',phi,'A',domain,seed,2**k,4)
 fixed_ids=[t['id'] for t in tasks if t['group']=='fixed']
 for task in tasks:task['depends_on_fixed'] = fixed_ids if task['group']!='fixed' else []
 manifest=dict(gpu_gate=parent_gpu_spec(),execution_order='all fixed40 accepted before any scan; source three-domain dependencies owned by parent',status='prepared_not_executed',protocol=PROTOCOL,source_frozen_manifest_sha256=sha(PLAN/'source_manifest.json'),requested=400,unique_executions=len(first),reuse_requests=400-len(first),counts={g:sum(t['group']==g for t in tasks) for g in ['fixed','Bscan','Nscan']},tasks=tasks)
 path=O/'manifest.json'
 if path.exists() and json.loads(path.read_text())!=manifest:
  assert refresh,'Refuse manifest regeneration drift without explicit reviewed refresh and history'
  assert (O/'history/preintegration_20261017/manifest.json').is_file()
  atomic(path,manifest)
 elif not path.exists():atomic(path,manifest)
 return manifest
if __name__=='__main__':
 assert __import__('os').environ.get('STY'),'Use unique guarded screen even for CPU acceptance'
 m=generate(refresh=('--refresh-reviewed' in sys.argv));print(json.dumps({k:v for k,v in m.items() if k!='tasks'}))
 if '--dry-list' in sys.argv:
  for t in m['tasks']:print(t['id']+(' REUSE '+t['reuse_of'] if t.get('reuse_of') else ' EXECUTE')+' '+str(t['command']))
