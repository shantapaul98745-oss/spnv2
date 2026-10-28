"""Two in-memory regression cases for the exact existing audit admission function.

Read source code only; no runtime status, models, budget, processes or GPU.
"""
import ast,copy,importlib.util,json,re,unittest
from pathlib import Path,PurePosixPath
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('final_cost_guard_v15_candidate',HERE/'final_cost_guard_v15.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

# Compile only the unchanged checked-in function. Its metadata reader is an
# in-memory fixture, keeping all exact cohort/template/denominator predicates.
source=ast.parse((HERE/'dispatch_config.py').read_text())
node=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='check_audit_task')
function_code=compile(ast.Module(body=[node],type_ignores=[]),'dispatch_config.py::check_audit_task','exec')

def require(ok,message):
 if not ok:raise ValueError(message)

class AuditHandoffTests(unittest.TestCase):
 def setUp(self):
  root='/home/DATA/Downloads/UsrSourceCode/LuXinghai';diagnostic='spnv2_runs/plans/g50p_risk_diagnostic_20261025';plan=root+'/'+diagnostic
  ids=['window_'+str(i) for i in range(48)]
  tasks=[dict(run_id=rid,task_path=plan+'/tasks/'+rid+'.json',task_sha256='a'*64) for rid in ids]
  self.live=dict(main={row['run_id']:dict(run_id=row['run_id'],task=dict(path=row['task_path'],sha256=row['task_sha256'])) for row in tasks},accepted={rid:dict(proof=dict(path=plan+'/pairs/'+rid+'/proof.json',sha256='b'*64)) for rid in ids})
  self.disk=copy.deepcopy(self.live);self.disk['accepted'].pop(ids[-1]);self.events=[]
  template=dict(action='summarize_diagnostic',run_id='summary',frozen_inputs=dict(path=plan+'/frozen_inputs.json',sha256='c'*64))
  task_path=plan+'/tasks/summary.json'
  self.index=dict(summary_template=template,audit_task_paths=dict(summary=task_path),frozen_inputs=template['frozen_inputs'],tasks=tasks)
  self.receipts=dict(schema='g50p-diagnostic-execution-receipts-v1',planned_run_ids=ids,runs=list(copy.deepcopy(self.live['main']).values()),accepted=copy.deepcopy(self.live['accepted']),missing_main_run_ids=[],missing_independent_acceptance_run_ids=[])
  self.task=dict(template,execution_receipts=dict(path=plan+'/final_receipts_fixture.json',sha256='d'*64),index_path=plan+'/index.json')
  self.binding=dict(path=task_path,sha256='e'*64)
  ns=SimpleNamespace(LOGICAL_ROOT=root,DIAGNOSTIC=diagnostic,NAME=re.compile(r'[A-Za-z0-9_.-]+'),absolute=lambda path:PurePosixPath(path),require=require)
  scope=dict(ns=ns,PurePosixPath=PurePosixPath,read_bound=lambda binding,resolve=None:self.receipts)
  exec(function_code,scope);self.audit=scope['check_audit_task']
  def write_state(state):
   self.events.append('atomic_state_write');self.disk=copy.deepcopy(state)
  def original_launch(state,task_path,run_id,index_path):
   self.events.append('original_launch');return self.audit(self.index,self.task,self.binding,self.disk)
  self.manager=SimpleNamespace(read=lambda path:self.task,write_state=write_state,launch=original_launch)

 def test_stale47_is_persisted_before_original_audit48(self):
  with self.assertRaisesRegex(ValueError,'Audit receipts omit/change'):
   self.audit(self.index,self.task,self.binding,self.disk)
  original=self.manager.launch;guard.install_audit_handoff_guard(self.manager)
  result=self.manager.launch(self.live,self.binding['path'],'summary',self.task['index_path'])
  self.assertEqual(result,self.receipts['planned_run_ids'])
  self.assertEqual(self.events,['atomic_state_write','original_launch'])
  self.assertEqual(self.disk,self.live)
  self.assertIsNot(self.manager.launch,original)

 def test_changed_receipt_still_rejected_after_fresh_state(self):
  last=self.receipts['planned_run_ids'][-1]
  self.receipts['accepted'][last]['proof']['sha256']='f'*64
  guard.install_audit_handoff_guard(self.manager)
  with self.assertRaisesRegex(ValueError,'Audit receipts omit/change'):
   self.manager.launch(self.live,self.binding['path'],'summary',self.task['index_path'])
  self.assertEqual(self.events,['atomic_state_write','original_launch'])
  self.assertEqual(self.disk,self.live)

if __name__=='__main__':unittest.main()
