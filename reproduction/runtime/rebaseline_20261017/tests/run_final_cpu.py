import hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parents[1];repo=p.parents[2]/'spnv2'
def hashes():
 return {str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('core','tools') for f in sorted((repo/folder).rglob('*.py'))}
before=hashes();reports=[]
for test in ['training_acceptance','training_boundaries','inference_summary']:
 output=p/'reports'/('final_cpu_'+test+'.json')
 if output.exists():raise FileExistsError(output)
 subprocess.run([sys.executable,'-B','-s','-u',str(p/'tests'/('test_'+test+'.py')),'--source',str(repo),'--report',str(output)],check=True)
 result=json.loads(output.read_text());assert result['status']=='passed',result
 reports.append(dict(test=test,tests=result['tests'],report=str(output),report_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
assert before==hashes(),'Source changed during final acceptance'
sys.path.insert(0,str(repo/'core'))
from config import cfg,update_config
matrix=json.loads((p/'protocol_matrix.json').read_text());epochs=0
for row in matrix['source_configs']:
 c=cfg.clone();update_config(c,SimpleNamespace(cfg=row['config'],opts=[]))
 assert c.TRAIN.ACCUM_STEPS==4 and c.TRAIN.REPRODUCIBLE and c.TRAIN.RESUME_FILE==''
 assert c.SEED==2021 and list(c.DATASET.INPUT_SIZE)==[768,512]
 assert c.TRAIN.LR==.001 and c.TRAIN.WD==.0001 and c.TRAIN.OPTIMIZER=='AdamW'
 assert c.TRAIN.LR_STEP==[15,18] and c.TRAIN.IMAGES_PER_GPU==row['physical_batch']
 assert list(c.MODEL.HEAD.NAMES)==row['heads'] and c.TRAIN.END_EPOCH==row['epochs']
 epochs+=c.TRAIN.END_EPOCH
assert epochs==190 and len(matrix['source_configs'])==10
result=dict(status='passed',tests=sum(r['tests'] for r in reports),reports=reports,code_sha256=before,source_configs=10,source_epochs=190)
(p/'reports/final_cpu_acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
