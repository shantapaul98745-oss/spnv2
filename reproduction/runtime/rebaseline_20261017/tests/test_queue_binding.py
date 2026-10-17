"""CPU integration for manager startup plus durable revision evidence."""
import argparse,copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args()
PLAN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PLAN/'pipeline'))
import runner
from revision import canonical_sha256
class Tests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix='binding_',dir=str(PLAN/'reports/pipeline_tests')));self.patch=patch.object(runner,'PLAN',self.root);self.patch.start()
 def tearDown(self):self.patch.stop()
 def task(self,ident):return dict(id=ident,kind='cpu',command=[sys.executable,'-c','print(1)'],cwd=str(runner.BASE/'spnv2'),session=self.root.name+'_'+ident,log=str(self.root/(ident+'.log')),gpu=False,verify_command=[sys.executable,'-c','print(2)'])
 def inputs(self):
  old=dict(version=1,plan_id=self.root.name,tasks=[self.task('first')]);new=copy.deepcopy(old);new['tasks'].append(self.task('second'));new['backlog']=['remaining goal']
  prior=runner.initialize_state(dict(tasks=[],notes=[]),old);prior.update(active='first',current=dict(task_id='first',pid=123,starttime='456'))
  prior['tasks'][0].update(status='running',runtime=dict(phase='main',record='/retained-record',pid=123,starttime='456',tail_offset=789))
  return old,new,prior
 def test_live_evidence_preserved_and_audit_archived(self):
  old,new,prior=self.inputs();original=copy.deepcopy(prior)
  state=runner.bind_execution_state(prior,new,old,caller_context={'test':'actual binding'})
  self.assertEqual(prior,original);self.assertEqual(state['manifest_sha256'],canonical_sha256(new));self.assertEqual(state['tasks'][0]['runtime'],original['tasks'][0]['runtime']);self.assertEqual(state['current']['pid'],123);self.assertEqual(state['tasks'][1]['status'],'queued')
  audit=json.loads(Path(state['manifest_revisions'][-1]).read_text());self.assertEqual(audit['old_manifest_sha256'],canonical_sha256(old));self.assertEqual(audit['active_task_id'],'first');self.assertEqual(audit['new_task_count'],2)
 def test_edited_prior_task_rejected_before_audit_write(self):
  old,new,prior=self.inputs();new['tasks'][0]['command']=[sys.executable,'-c','wrong()']
  with self.assertRaises(ValueError):runner.bind_execution_state(prior,new,old)
  self.assertFalse((self.root/'revisions').exists())
 def test_plain_restart_refuses_unapproved_extension(self):
  old,new,prior=self.inputs()
  with self.assertRaises(ValueError):runner.bind_execution_state(prior,new)
 def test_plain_restart_keeps_original_manifest(self):
  old,new,prior=self.inputs();state=runner.bind_execution_state(prior,old)
  self.assertEqual(state['manifest_sha256'],canonical_sha256(old));self.assertEqual(state['tasks'][0]['runtime'],prior['tasks'][0]['runtime']);self.assertNotIn('manifest_revisions',state)
 def test_duplicate_or_invalid_append_rejected(self):
  old,new,prior=self.inputs();new['tasks'][1]['session']=new['tasks'][0]['session']
  with self.assertRaises(ValueError):runner.bind_execution_state(prior,new,old)
  self.assertFalse((self.root/'revisions').exists())
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));Path(a.report).write_text(json.dumps(dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=[dict(test=str(t),trace=s) for t,s in r.failures+r.errors]),indent=2));sys.exit(0 if r.wasSuccessful() else 1)
