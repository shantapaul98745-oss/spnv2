import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
V=Path(__file__).parent
class ManagerLaunchContracts(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir='/tmp/spnv2_sysu_20261026');self.p=Path(self.tmp.name);self.plan=self.p/'plan';self.plan.mkdir();self.writes=[]
  budget=types.ModuleType('budget');budget.GlobalBudget=object;budget.atomic_json=lambda *a:self.writes.append(a)
  host=types.ModuleType('host_entry');host.PLAN=self.plan;host.BASE=self.p;host.PROJECT='g50p_risk_diagnostic_20261025';host.host_modules=lambda x:None;host.project_lock=lambda x:None;host.read=lambda p:json.loads(Path(p).read_text());host.sha=lambda p:'b'*64
  self.patch=mock.patch.dict(sys.modules,{'budget':budget,'host_entry':host});self.patch.start();spec=importlib.util.spec_from_file_location('test_sysu_manager',V/'manager_sysu.py');self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
 def tearDown(self):self.patch.stop();self.tmp.cleanup()
 def test_real_host_entry_and_nonce_remain_in_worker_argv(self):
  cfg=dict(path='/bound/config.json',sha256='c'*64);argv=[self.m.PYTHON,'-B','-s','-u','/bound/bootstrap.py','--config',cfg['path'],'--execute-host-entry',str(self.plan/'implementation/host_entry.py'),'--task-json',str(self.plan/'tasks/task.json')]
  state=dict(active=None,attempts=[],waiting_since=None)
  result=types.SimpleNamespace(returncode=0,stdout='launched',stderr='')
  with mock.patch.object(self.m,'sysu_worker_command',return_value=(argv,cfg)),mock.patch.object(self.m,'write_state'),mock.patch.object(self.m.subprocess,'run',return_value=result) as run:
   self.m.launch(state,self.plan/'tasks/task.json','run',self.plan/'index.json')
  command=run.call_args.args[0];self.assertEqual(command[4:],argv);self.assertIn(str(self.plan/'implementation/host_entry.py'),command);self.assertEqual(len(run.call_args.kwargs['env']['SPNV2_RUN_NONCE']),32);self.assertEqual(state['status'],'running')
 def test_bad_runtime_config_prevents_launch_intent(self):
  state=dict(active=None,attempts=[],waiting_since=None)
  with mock.patch.object(self.m,'sysu_worker_command',side_effect=ValueError('bad config')),mock.patch.object(self.m.subprocess,'run') as run:
   with self.assertRaises(ValueError):self.m.launch(state,self.plan/'tasks/task.json','run',self.plan/'index.json')
  self.assertIsNone(state['active']);self.assertEqual(state['attempts'],[]);run.assert_not_called()
 def test_verification_is_versioned_without_replaying_main(self):
  rid='g50p_diag_v1_sunlamp_s2021_w0016';row=dict(run_id=rid,task_path=str(self.plan/'tasks/old.json'),task_sha256='d'*64);binding=dict(path=row['task_path'],sha256=row['task_sha256']);receipt=dict(task=binding,attempts=[dict(exit_code=0)])
  idx=dict(implementation_sha256={'manager.py':'e'*64},budget_policy={'path':'/original/policy','sha256':'f'*64},sysu_task_revision='sysu_migration_20261027');state=dict(index={'sha256':'a'*64},main={rid:receipt})
  with mock.patch.object(self.m,'checked',return_value={'output_dir':str(self.plan/'pairs'/rid)}),mock.patch.object(self.m,'immutable',side_effect=lambda p,v:p) as write:
   path,runid=self.m.verification_task(idx,row,state)
  self.assertEqual(path.parent,self.plan/'tasks/sysu_migration_20261027');self.assertEqual(write.call_args.args[1]['run_task'],binding);self.assertEqual(write.call_args.args[1]['execution_receipt'],receipt);self.assertEqual(write.call_args.args[1]['budget_policy'],idx['budget_policy']);self.assertTrue(runid.endswith('_verify_'+'a'*12))
if __name__=='__main__':unittest.main()
