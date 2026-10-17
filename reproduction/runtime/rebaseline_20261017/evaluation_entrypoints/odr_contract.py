"""ODR parent/child contract helpers; frozen common.job is never patched."""
from entry_contract import *

def planned_task(task):
 t=copy.deepcopy(task);base=list(t['command']);need(len(base)>=6 and Path(base[4]).resolve()==O/'audited_entry.py' and base[-2:]==['--task',t['id']],'Unexpected original ODR child command')
 t['base_command']=base;t['command']=[*base[:4],str(E/'capture_odr.py'),'--task',t['id']];return t

def ensure_plan(task):
 paths=odr_paths(task['id']);planned=planned_task(task)
 value=dict(schema='spnv2-odr-evaluation-entrypoint-plan-v1',task=task,task_sha256=digest(task),base_command=planned['base_command'],actual_command=planned['command'],wrapper_protocol_sha256=sha(E/'protocol_manifest.json'))
 stable_write(paths['plan'],value);return paths

def quick_verified(task):
 paths=odr_paths(task['id']);proof=read(paths['wrapper_proof']);need(proof['schema']=='spnv2-odr-evaluation-extension-entrypoint-verification-v1' and proof['status']=='passed' and proof['task_id']==task['id'],'ODR extension wrapper proof missing/wrong')
 need(proof['wrapper_protocol_sha256']==sha(E/'protocol_manifest.json'),'ODR extension proof from another wrapper implementation')
 need(sha(proof['base_report'])==proof['base_report_sha256'],'Original ODR base report SHA drift')
 need(sha(proof['report'])==proof['report_sha256'] and sha(proof['independent_proof'])==proof['independent_proof_sha256'],'ODR extension report/proof changed')
 return proof

def fixed_extension_barrier(task,common):
 if task['group']=='fixed':return
 tasks=read(O/'manifest.json')['tasks'];fixed=[t for t in tasks if t['group']=='fixed'];need(len(fixed)==40,'Expected40 fixed ODR requests')
 source={phi:common.accepted_source(phi)['final_sha256'] for phi in [3,6]}
 for t in fixed:
  proof=quick_verified(t);need(not proof.get('reuse_of') and proof['source_sha256']==source[t['phi']],'Fixed extension source hash or canonical identity wrong')

@contextmanager
def original_odr_namespace():
 with namespace([O],extra_names=('_entry_base_odr_parent','_entry_base_odr_verify')):
  common=importlib.import_module('common');need(Path(common.__file__).resolve()==O/'common.py','Foreign common namespace selected');yield common

def original_verify(task_id):
 with original_odr_namespace() as common,CPU_environment():
  task=common.job(task_id);chain=[task]
  if task.get('reuse_of'):chain.append(common.job(task['reuse_of']))
  previous={str(O/'reports'/(t['id']+'.json')):(O/'reports'/(t['id']+'.json')).read_bytes() for t in chain if (O/'reports'/(t['id']+'.json')).exists()}
  try:
   verifier=local_module(O/'verify_result.py','_entry_base_odr_verify');result=verifier.verify(task_id)
   for name,data in previous.items():need(Path(name).read_bytes()==data,'Original ODR verification changed previously accepted base report bytes')
   need(result['status']=='passed','Original ODR verifier failed');return result
  except BaseException:
   for name,data in previous.items():
    path=Path(name)
    if not path.exists() or path.read_bytes()!=data:
     temporary=path.with_name('.'+path.name+'.entrypoint_restore.tmp')
     with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
     os.replace(temporary,path)
   raise

def verify_extension(task,common):
 verify_protocol();base_report=O/'reports'/(task['id']+'.json');base=read(base_report);need(base['status']=='passed' and base['task_id']==task['id'],'Original ODR proof required before extension proof')
 source=common.accepted_source(task['phi']);need(base['source_sha256']==source['final_sha256'],'ODR extension source differs from original base proof');need(sha(task['config'])==task['config_sha256'],'ODR alias/canonical config changed')
 paths=odr_paths(task['id'])
 if task.get('reuse_of'):
  canonical=common.job(task['reuse_of']);need(not canonical.get('reuse_of'),'Nested alias disallowed')
  for key in ['phi','line','domain','seed','N','B','source_final','source_report','code_sha256_manifest']:need(task[key]==canonical[key],'Alias protocol mismatch: '+key)
  canonical_proof=verify_extension(canonical,common);canonical_paths=odr_paths(canonical['id']);need(base['canonical_report_sha256']==sha(O/'reports'/(canonical['id']+'.json')),'Base alias canonical report changed')
  proof=dict(schema='spnv2-odr-evaluation-extension-entrypoint-verification-v1',status='passed',task_id=task['id'],reuse_of=canonical['id'],canonical_entrypoint_proof=str(canonical_paths['wrapper_proof']),canonical_entrypoint_proof_sha256=sha(canonical_paths['wrapper_proof']),base_report=str(base_report),base_report_sha256=sha(base_report),source_sha256=source['final_sha256'],requested_config=dict(path=task['config'],sha256=sha(task['config'])),wrapper_protocol_sha256=sha(E/'protocol_manifest.json'),report=canonical_proof['report'],report_sha256=canonical_proof['report_sha256'],independent_proof=canonical_proof['independent_proof'],independent_proof_sha256=canonical_proof['independent_proof_sha256'],context_sha256=canonical_proof['context_sha256'],observer_status=canonical_proof['observer_status'],extension_applicability=canonical_proof['extension_applicability'],scope='Verified canonical same-forward extension only; alias performs no model forward or adaptation')
 else:
  plan=read(paths['plan']);need(plan['task']==task and plan['base_command']==task['command'] and plan['actual_command']==planned_task(task)['command'],'ODR task/command plan changed')
  process=O/'reports'/(task['id']+'.process.json');record=read(process);need(record['command']==planned_task(task)['command'],'Parent process record must expose actual observer wrapper command')
  extra=extension_verify(paths);ctx=extra['context'];need(ctx['kind']=='odr' and ctx['identifier']==task['id'] and ctx['configuration']['sha256']==task['config_sha256'] and ctx['weights']['source_final']['sha256']==source['final_sha256'],'ODR captured context identity mismatch')
  need(ctx['weights']['adapted_before_evaluation']['sha256']==sha(Path(task['output'])/'adapted_before_eval.pth.tar'),'ODR captured adapted weight changed')
  proof=dict(schema='spnv2-odr-evaluation-extension-entrypoint-verification-v1',status='passed',task_id=task['id'],base_report=str(base_report),base_report_sha256=sha(base_report),source_sha256=source['final_sha256'],parent_process_record=str(process),parent_process_record_sha256=sha(process),base_command=task['command'],actual_command=planned_task(task)['command'],wrapper_protocol_sha256=sha(E/'protocol_manifest.json'),**extra)
 proof['extension']=extension_link(proof)
 if task.get('reuse_of'):proof.update(canonical_task=task['reuse_of'],canonical_proof=dict(path=proof['canonical_entrypoint_proof'],sha256=proof['canonical_entrypoint_proof_sha256']))
 stable_write(paths['wrapper_proof'],proof);event('odr_extension_verified',task=task['id'],output=str(paths['wrapper_proof']),reuse_of=task.get('reuse_of'));return proof
