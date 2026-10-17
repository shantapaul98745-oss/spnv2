from pathlib import Path
import json,os,hashlib
import numpy as np
from scipy.io import loadmat
A=Path(__file__).resolve().parents[1];P=A.parent;RUNS=P.parents[1]
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
source=RUNS/'outputs/odr/efficientdet_d3/phi3_A_lightbox_N1024_B4_seed2021/lightbox'
fields={}
for name in ['mean_performance','per_sample_performance','predictions_pose']:
 f=source/(name+'.mat');m=loadmat(f,simplify_cells=True)
 fields[name]={k:dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype),first_ids=[str(x) for x in np.atleast_1d(v)[:3]] if k=='sample_id' else None) for k,v in m.items() if not k.startswith('__')}
queue=json.loads((P/'queue_v2.json').read_text());tasks=[]
for t in queue['tasks']:
 cmd=t.get('command',[])
 if 'source_artifacts.py' not in ' '.join(cmd):continue
 tasks.append(dict(id=t['id'],mode='evaluate' if 'evaluate' in cmd else 'train',verified_outputs=t.get('expected_outputs')))
report=dict(status='schema_observed_only_not_new_scientific_acceptance',historical_schema_folder=str(source),historical_only=True,fields=fields,source_artifact_task_summaries=tasks,ODR_CPU_acceptance=json.loads((P/'odr/reports/integration_cpu_acceptance.json').read_text())['status'],new_source_reports_present=[str(f) for f in (P/'reports').glob('source_*_verified.json')],camera=json.loads(Path('/home/DATA/Downloads/DataSet/speedplus/camera.json').read_text()))
(A/'reports/schema_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
