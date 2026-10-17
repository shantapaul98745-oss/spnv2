"""Durable restart publication and pinned launch assets, CPU only."""
import argparse, hashlib, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'pipeline'))
import runner, progress, task_entry

class Tests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix='takeover_',dir=str(P/'reports/pipeline_tests')))
  self.plan_patch=patch.object(runner,'PLAN',self.root);self.plan_patch.start()
 def tearDown(self): self.plan_patch.stop()
 def queue(self):
  task=dict(id='live_task',kind='cpu',command=[sys.executable,'-c','print(1)'],cwd=str(runner.BASE/'spnv2'),session=self.root.name,log=str(self.root/'task.log'),gpu=False,verify_command=[sys.executable,'-c','print(2)'])
  manifest=dict(version=1,plan_id=self.root.name,tasks=[task]);progress.atomic_json(self.root/'manifest.json',manifest)
  state=runner.initialize_state(dict(tasks=[]),manifest)
  reporter=progress.Reporter(self.root/'status.json',self.root/'progress.md',state,interval=60)
  return runner.Queue(self.root/'manifest.json',manifest,state,reporter),reporter,state
 def test_restart_reverification_is_durable_without_new_events(self):
  queue,reporter,state=self.queue();task=state['tasks'][0]
  record=self.root/'process.json';progress.atomic_json(record,task_entry.process_identity(os.getpid()))
  task.update(status='launching',runtime=dict(phase='main',record=str(record),result=str(self.root/'none.result'),log=task['log'],observed_live=True,tail_offset=0))
  state.update(status='monitor_stopped',current=dict(process_state='unverified after manager restart'))
  reporter.publish(force=True)
  with patch.object(queue,'prepare_launch') as launch:queue.step();launch.assert_not_called()
  disk=progress.read_json(self.root/'status.json');self.assertEqual(disk['status'],'running');self.assertEqual(disk['tasks'][0]['status'],'running');self.assertIn('verified',disk['current']['process_state']);self.assertIn('live PID', (self.root/'progress.md').read_text())
 def pin(self,task,path):
  task['metadata']=dict(asset_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
 def test_matching_asset_is_accepted(self):
  queue,reporter,state=self.queue();path=self.root/'helper.py';path.write_text('pass\n');self.pin(state['tasks'][0],path)
  self.assertIsNone(runner.asset_hash_error(state['tasks'][0]))
 def test_changed_asset_prevents_process_launch(self):
  queue,reporter,state=self.queue();task=state['tasks'][0];path=self.root/'helper.py';path.write_text('pass\n');self.pin(task,path);path.write_text('raise Exception()\n')
  # Bind the final metadata exactly as a published manifest does.
  manifest=dict(queue.manifest,tasks=[dict(task)]);progress.atomic_json(queue.manifest_path,manifest);state['manifest_sha256']=runner.manifest_hash(manifest)
  with patch('runner.subprocess.run') as spawn:self.assertFalse(queue.prepare_launch(task,'main'));spawn.assert_not_called()
  self.assertEqual(task['status'],'failed');self.assertIn('Pinned asset changed',task['error']);self.assertNotIn('runtime',task)
 def test_missing_asset_rejected(self):
  queue,reporter,state=self.queue();state['tasks'][0]['metadata']={'asset_sha256':{str(self.root/'missing'):'0'*64}}
  self.assertIn('unavailable',runner.asset_hash_error(state['tasks'][0]))
 def test_outside_project_asset_rejected(self):
  self.assertIn('unavailable',runner.asset_hash_error({'metadata':{'asset_sha256':{'/etc/passwd':'0'*64}}}))
 def test_malformed_hash_rejected(self):
  self.assertIn('Invalid',runner.asset_hash_error({'metadata':{'asset_sha256':{str(self.root/'x'):'wrong'}}}))
 def test_verification_phase_rechecks_assets(self):
  queue,reporter,state=self.queue();task=state['tasks'][0];path=self.root/'verifier.py';path.write_text('pass\n');self.pin(task,path)
  manifest=dict(queue.manifest,tasks=[dict(task)]);progress.atomic_json(queue.manifest_path,manifest);state['manifest_sha256']=runner.manifest_hash(manifest)
  task['runtime']={'phase':'main'};path.write_text('modified\n')
  with patch('runner.subprocess.run') as spawn:self.assertFalse(queue.prepare_launch(task,'verify'));spawn.assert_not_called()
  self.assertEqual(task['status'],'failed');self.assertEqual(task['runtime'],{'phase':'main'})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 Path(args.report).write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,errors=[dict(test=str(t),trace=x) for t,x in result.failures+result.errors]),indent=2));sys.exit(0 if result.wasSuccessful() else 1)
