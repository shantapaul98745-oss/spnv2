"""Exact resource-only amendment bridge; scientific historical evidence remains bound."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT='/home/DATA/Downloads/UsrSourceCode/LuXinghai'
PLAN=ROOT+'/spnv2_runs/plans/g50p_risk_diagnostic_20261025'
PACKAGE=ROOT+'/spnv2/reproduction/shared_gpu_20261027'
SCHEMA='spnv2-shared-gpu-resource-revision-v1'
AUTH='user_2026-10-27_shared_gpu_capacity_admission'
BEFORE_INDEX='be27ea03fb71ca6f6b910c7612cf75d03e5edecfeeabcf35163d529865e4ad6c'

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolved(path,resolve):return Path(path) if resolve is None else Path(resolve(path))
def checked(item,resolve=None):
 require(isinstance(item,dict) and set(item)=={'path','sha256'},'Invalid resource amendment binding')
 path=resolved(item['path'],resolve);require(digest(path)==item['sha256'],'Resource amendment binding changed: '+item['path'])
 return json.loads(path.read_text())
def load(index,resolve=None):
 binding=index.get('resource_revision');require(isinstance(binding,dict) and binding.get('path')==PLAN+'/revisions/shared_gpu_20261027/resource_revision.json','Exact resource revision required')
 rev=checked(binding,resolve)
 require(rev.get('schema')==SCHEMA and rev.get('authorization')==AUTH and rev.get('resource_admission_only') is True,'Unapproved resource amendment')
 require(rev['validator']['path']==PACKAGE+'/resource_amendment.py' and digest(resolved(rev['validator']['path'],resolve))==rev['validator']['sha256']==digest(__file__),'Resource bridge code changed')
 require(rev['before_index']['path']==PLAN+'/revisions/shared_gpu_20261027/before/index.json' and rev['before_index']['sha256']==BEFORE_INDEX,'Resource before-index differs')
 before=checked(rev['before_index'],resolve)
 require(before['implementation_sha256']==rev['before_implementation_sha256'] and index['implementation_sha256']==rev['after_implementation_sha256'],'Resource control implementation maps differ')
 require(set(index['implementation_sha256'])==set(before['implementation_sha256']) and all(index['implementation_sha256'][k]==before['implementation_sha256'][k] for k in before['implementation_sha256'] if k not in ('manager.py','host_entry.py')),'Resource amendment changed scientific code')
 require(before['host_modules']==rev['before_host_modules'] and index['host_modules']==rev['after_host_modules'],'Resource host maps differ')
 require(set(index['host_modules'])=={'host_gate','host_snapshot'},'Resource host module scope differs')
 for name,item in index['host_modules'].items():
  require(item['path']==PACKAGE+'/'+name+'.py' and digest(resolved(item['path'],resolve))==item['sha256'],'Resource host module bytes/path differ')
 require(index.get('resource_policy')==rev['resource_policy'] and rev['resource_policy']['path']==PLAN+'/revisions/shared_gpu_20261027/resource_policy.json','Resource policy binding differs')
 policy=checked(rev['resource_policy'],resolve)
 require(policy.get('schema')=='spnv2-shared-gpu-resource-policy-v1' and policy.get('authorization')==AUTH and policy.get('resource_admission_only') is True,'Resource policy authorization differs')
 require(index['frozen_inputs']==before['frozen_inputs'] and index['protocol']==before['protocol'] and index['budget_policy']==before['budget_policy'] and index['complete_ODR_runs']==before['complete_ODR_runs']==0,'Resource amendment changed science/input/budget bindings')
 require([r['run_id'] for r in index['tasks']]==[r['run_id'] for r in before['tasks']] and len(index['tasks'])==48,'Resource amendment changed frozen order')
 require(rev['before_host_entry']['sha256']==before['implementation_sha256']['host_entry.py'] and digest(resolved(rev['before_host_entry']['path'],resolve))==rev['before_host_entry']['sha256'],'Original pre-resource host entry archive changed')
 require(rev['after_host_entry_source']['path']==PACKAGE+'/host_entry.py' and rev['after_host_entry_source']['sha256']==index['implementation_sha256']['host_entry.py'] and digest(resolved(rev['after_host_entry_source']['path'],resolve))==rev['after_host_entry_source']['sha256'],'Resource entry source changed')
 for name in ('host_entry.py','manager.py'):
  require(digest(resolved(PLAN+'/implementation/'+name,resolve))==index['implementation_sha256'][name],'Live resource control implementation differs')
 return rev

def host_modules_compatible(index,original,resolve=None):
 if index['host_modules']==original['host_modules']:return True
 rev=load(index,resolve)
 return original['host_modules']==rev['before_host_modules'] and index['host_modules']==rev['after_host_modules']

def code_compatible(index,actual,original,resolve=None):
 if not isinstance(actual,dict) or set(actual)!=set(original):return False
 rev=load(index,resolve)
 for name in original:
  if name=='manager.py':continue
  if actual[name]==original[name]:continue
  if name=='host_entry.py' and actual[name]==rev['after_implementation_sha256'][name] and original[name]==rev['before_implementation_sha256'][name]:continue
  return False
 return True
