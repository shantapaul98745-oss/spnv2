"""Shared ODR protocol, hash boundaries, paths; no launch side effects."""
from pathlib import Path
import hashlib,json,os
O=Path(__file__).resolve().parent
PLAN=O.parent
RUNS=PLAN.parents[1]
PYTHON='/home/ps/anaconda3/envs/SPNv2/bin/python'
DATA=Path('/home/DATA/Downloads/DataSet/speedplus')
PROTOCOL=dict(version='new-source-odr-v2',physical_loader_batch=1,accum_steps=1,reproducible_training=False,fp16=False,lr=1e-4,wd=0.0,optimizer='AdamW',augment_p=0.0,texture=False,cudnn_benchmark=True,cudnn_deterministic=False,seed_meaning='ODR observation stream; one independently trained source per architecture',nscan_line='A')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp')
 with tmp.open('w') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def tree(root):return {str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
def source_path(phi):
 import yaml
 c=yaml.safe_load((PLAN/'configs'/('phi3_bn_ehs.yaml' if phi==3 else 'phi6_gn_ehs.yaml')).read_text())
 return Path(c['OUTPUT_DIR'])/c['MODEL']['BACKBONE']['NAME']/c['EXP_NAME']/'model_final.pth.tar'
def source_report(phi):return PLAN/'reports'/f'source_phi{phi}_verified.json'
def verify_code(line):
 root=O/('source_'+line);expected=json.loads((O/(line+'_sha256.json')).read_text());assert tree(root)==expected,'ODR source snapshot changed'
 frozen=json.loads((PLAN/'source_manifest.json').read_text())['core_tools_sha256']
 actual={str(f.relative_to(PLAN/'source')):sha(f) for d in ['core','tools'] for f in sorted((PLAN/'source'/d).rglob('*.py'))}
 assert actual==frozen,'Parent frozen source changed'
 for rel,value in frozen.items():
  if line!='A' and rel=='core/engine/adapter.py':continue
  if line=='B2' and rel=='tools/odr.py':continue
  assert expected[rel]==value,rel
 return expected
def accepted_source(phi):
 r=json.loads(source_report(phi).read_text());f=source_path(phi)
 assert r['status']=='passed' and r['final']==str(f) and r['final_sha256']==sha(f),'New source final not accepted or changed'
 assert r['code_sha256']==json.loads((PLAN/'source_manifest.json').read_text())['core_tools_sha256']
 assert r['physical_batch']==(4 if phi==3 else 1) and r['accum_steps']==4
 return r

def job(task_id):
 m=json.loads((O/'manifest.json').read_text());return next(t for t in m['tasks'] if t['id']==task_id)
def canonical(task):return job(task['reuse_of']) if task.get('reuse_of') else task

def environ():
 env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',TORCH_HOME=str(RUNS/'env/cache/torch'),MPLCONFIGDIR=str(O/'cache/matplotlib'))
 return env

def verify_implementation():
 m=json.loads((O/'implementation_manifest.json').read_text())
 assert m['manifest_sha256']==sha(O/'manifest.json'),'Task manifest drift'
 assert m['helpers_sha256']=={str(f.relative_to(O)):sha(f) for f in sorted(O.glob('*.py'))},'ODR helper changed since reviewed manifest'
 assert m['parent_frozen_manifest_sha256']==sha(PLAN/'source_manifest.json')
 assert m['source_line_manifests_sha256']=={line:sha(O/(line+'_sha256.json')) for line in ['A','B1','B2']}
 assert m['configs_sha256']=={str(f.relative_to(O)):sha(f) for f in sorted((O/'configs').glob('*.yaml'))}
 assert m['gpu_gate_function_sha256']==gpu_gate_function_sha() and m['gpu_gate_spec']==parent_gpu_spec()
 return m


def parent_runner():
 import importlib.util,sys
 path=PLAN/'pipeline/runner.py'
 if '_odr_parent_runner' in sys.modules:return sys.modules['_odr_parent_runner']
 sys.path.insert(0,str(path.parent))
 spec=importlib.util.spec_from_file_location('_odr_parent_runner',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 return module

def parent_gpu_spec():
 # Same resource limits and bounded desktop handling as the existing parent queue.
 manifest=json.loads((PLAN/'queue.json').read_text())
 specs=[t['gpu'] for t in manifest['tasks'] if isinstance(t.get('gpu'),dict)]
 assert specs and all(s==specs[0] for s in specs),'Parent GPU resource specifications differ; root review required'
 return specs[0]

def gpu_available():return parent_runner().gpu_available(parent_gpu_spec())

def fixed_barrier(task):
 if task['group']=='fixed':return
 m=json.loads((O/'manifest.json').read_text());sources={phi:accepted_source(phi)['final_sha256'] for phi in [3,6]}
 for fixed in [x for x in m['tasks'] if x['group']=='fixed']:
  path=O/'reports'/(fixed['id']+'.json');proof=json.loads(path.read_text())
  assert proof['status']=='passed' and proof['task_id']==fixed['id']
  assert proof['source_sha256']==sources[fixed['phi']] and proof['protocol']==PROTOCOL
  assert proof['implementation_manifest_sha256']==sha(O/'implementation_manifest.json'),'All 40 fixed tasks must be accepted under this reviewed implementation before scans'


def gpu_gate_function_sha():
 import ast
 source=(PLAN/'pipeline/runner.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='gpu_available')
 return hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
