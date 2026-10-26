"""Admit only the two included fixed pilots after independent acceptance."""
import json, pathlib, hashlib, time
from budget import atomic_json
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def checked(b):
 if sha(b['path'])!=b['sha256']:raise ValueError('binding changed: '+b['path'])
 return read(b['path'])
def execute(task_path,*,admission,launch_record,conflict_check,budget):
 task=read(task_path);idx=read(task['index_path']);frozen=checked(idx['frozen_inputs']);rec=checked(task['execution_receipts'])
 expected=[r['run_id'] for r in idx['tasks'][:2]]
 if set(rec['accepted'])!=set(expected) or len(rec['runs'])!=2:raise ValueError('Both fixed included pilots must have independent proofs')
 evidence=[]
 for run in rec['runs']:
  budget.check_time('pilot independent receipt review')
  proof=checked(rec['accepted'][run['run_id']]['proof']);report=checked(run['report']);rt=checked(run['task'])
  if proof['status']!='passed' or proof['report_sha256']!=run['report']['sha256'] or proof['task_binding']!=run['task']:raise ValueError('pilot independent binding mismatch')
  if not report['full_scientific_checkpoint_equal'] or not report['original_trace_reproduced_stepwise']:raise ValueError('pilot original trajectory not strictly equal')
  if not all(report['branch_results'][x]['learning_unchanged'] and report['branch_results'][x]['producer_provider_unchanged'] for x in ['accept','rollback']):raise ValueError('pilot branch polluted state')
  if pathlib.Path(run['exit_code_path']).read_text().strip()!='0':raise ValueError('pilot main did not exit0')
  va=rec['accepted'][run['run_id']]['verification_attempt']
  if pathlib.Path(va['log_path']+'.exitcode').read_text().strip()!='0':raise ValueError('pilot verifier did not exit0')
  if report['matched_scientific_sha256']!=report['expected_scientific_sha256']:raise ValueError('pilot whole-state digest mismatch')
  evidence.append(dict(run_id=run['run_id'],window_id=rt['selection']['window_id'],report=run['report'],proof=rec['accepted'][run['run_id']]['proof'],forward_categories=report['forward_categories'],attempt_costs=report['attempt_costs']))
 if [next(e['window_id'] for e in evidence if e['run_id']==rid) for rid in expected]!=frozen['execution_order'][:2]:raise ValueError('pilot selection changed')
 snap=budget.snapshot();budget.check_time('pilot cost acceptance')
 if snap['physical_forwards_reserved_upper']>40000 or (snap.get('time_hard_stop_enabled',True) and snap.get('active_time_bound_violation_attempt_ids')):raise ValueError('pilot budget boundary violated')
 if conflict_check():return dict(status='waiting_resources',exit_code=75)
 p=pathlib.Path(task['proof_path']);atomic_json(p,dict(schema='g50p-risk-diagnostic-pilot-proof-v1',status='passed',allow_remaining_46=True,pilots=evidence,execution_receipts=task['execution_receipts'],budget_snapshot=snap,model_forwards_added=0,scope='correctness and actual bounded cost, not effects; remaining work still subject to all global caps'))
 return dict(status='pilot_admitted',exit_code=0,proof=str(p),proof_sha256=sha(p),model_forwards=0)
