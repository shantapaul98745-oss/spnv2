"""Publication cadence and mandatory immediate transitions; CPU only."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'pipeline'))
import runner,progress
class Tests(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix='cadence_',dir=str(P/'reports/pipeline_tests')));self.mock=patch.object(runner,'PLAN',self.root);self.mock.start()
  self.task=dict(id='train',runtime=dict(log=str(self.root/'task.log'),tail_offset=0))
  self.state=dict(status='running',tasks=[self.task],current=dict(phase='train'),notes=[])
  self.reporter=progress.Reporter(self.root/'status.json',self.root/'progress.md',self.state,interval=60);self.reporter.publish(force=True)
  self.queue=runner.Queue(self.root/'manifest.json',{},self.state,self.reporter)
 def tearDown(self):self.mock.stop()
 def emit(self,**event):
  with Path(self.task['runtime']['log']).open('a') as f:f.write('SPNV2_EVENT '+json.dumps(event)+'\n')
  self.queue.tail_events(self.task)
 def test_update_kept_in_memory_and_audit_without_rewriting_full_state(self):
  before=(self.root/'status.json').read_bytes();rev=self.state['revision']
  for i in range(10):self.emit(event='update',phase='train',batch=i,loss=1.0)
  self.assertEqual(self.state['current']['batch'],9);self.assertEqual((self.root/'status.json').read_bytes(),before);self.assertEqual(self.state['revision'],rev)
  self.assertEqual(len((self.root/'events/queue.events.jsonl').read_text().splitlines()),10)
 def test_sixty_second_interval_publishes_latest_metrics(self):
  self.reporter.last_success-=61;self.emit(event='update',phase='train',batch=42)
  self.assertEqual(progress.read_json(self.root/'status.json')['current']['batch'],42)
 def test_checkpoint_immediately_flushes_current_progress(self):
  self.emit(event='update',phase='train',batch=12);self.emit(event='checkpoint',phase='train',path='/actual/checkpoint')
  state=progress.read_json(self.root/'status.json');self.assertEqual(state['current']['batch'],12);self.assertEqual(state['current']['checkpoint'],'/actual/checkpoint')
 def test_phase_change_is_immediate_even_for_progress(self):
  self.emit(event='progress',phase='validation',batch=0);self.assertEqual(progress.read_json(self.root/'status.json')['current']['phase'],'validation')
 def test_error_and_reporting_warning_are_immediate(self):
  self.emit(event='error',phase='train',error='failure');self.assertIn('failure',progress.read_json(self.root/'status.json')['error'])
  self.emit(event='progress_write_warning',phase='train',message='write failed');self.assertTrue(progress.read_json(self.root/'status.json')['downstream_blocked'])
 def test_evaluation_completion_is_immediate(self):
  self.emit(event='evaluation_complete',phase='validation',n_samples=3,score=None,summary={'n_samples':3,'metrics':{}})
  self.assertEqual(progress.read_json(self.root/'status.json')['last_evaluation']['n_samples'],3)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 Path(a.report).write_text(json.dumps(dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,errors=[dict(test=str(t),trace=x) for t,x in r.failures+r.errors]),indent=2));sys.exit(0 if r.wasSuccessful() else 1)
