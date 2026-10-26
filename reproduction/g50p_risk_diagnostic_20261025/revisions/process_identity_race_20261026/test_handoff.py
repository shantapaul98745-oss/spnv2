from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest import mock
import manager
import host_entry


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.launch=self.root/'launch';self.launch.mkdir()
        self.log=self.root/'worker.log'
        self.task=self.root/'task.json';self.task.write_text(json.dumps(dict(action='run_pair',run_id='test')))
        self.report=self.root/'report.json';self.report.write_text('{}')
        (self.launch/'launch.json').write_text('{"pid":123}')
        self.item=dict(run_id='test',nonce='n'*32,task_path=str(self.task),task_sha256=manager.sha(self.task),
            log_path=str(self.log),launch_directory=str(self.launch),queued_at=time.time()-121)
        self.state=dict(active=self.item,attempts=[self.item],main={},accepted={},manager=dict(boot_id='test-boot'),status='running')
        self.outcome=dict(exit_code=0,status='pair_main_complete',run_id='test',report=str(self.report),report_sha256=manager.sha(self.report))
    def tearDown(self):self.tmp.cleanup()
    def test_permission_error_with_terminal_arrival_collects_without_accepting_or_dispatching(self):
        def race(record):
            Path(str(self.log)+'.exitcode').write_text('0\n')
            (self.launch/'outcome.json').write_text(json.dumps(self.outcome))
            raise PermissionError('exit-time proc/environ race')
        with mock.patch.object(host_entry,'live_identity',side_effect=race),\
             mock.patch.object(manager,'launch',side_effect=AssertionError('no downstream launch')):
            manager.finish(self.state)
        self.assertIsNone(self.state['active'])
        self.assertIn('test',self.state['main'])
        self.assertEqual(self.state['accepted'],{})
    def test_persistent_unreadability_keeps_same_attempt_then_blocks_without_ownership_assumption(self):
        now=[1000.]
        with mock.patch.object(host_entry,'live_identity',side_effect=PermissionError('unreadable')) as probe,\
             mock.patch.object(manager.time,'monotonic',side_effect=lambda:now[0]),\
             mock.patch.object(manager,'launch',side_effect=AssertionError('no downstream launch')):
            manager.finish(self.state)
            self.assertEqual(self.state['status'],'awaiting_terminal_evidence')
            self.assertIs(self.state['active'],self.item)
            self.assertEqual(self.item['terminal_evidence_handoff']['identity_status'],'unverified')
            now[0]+=manager.TERMINAL_HANDOFF_GRACE_SECONDS+1
            manager.finish(self.state)
            self.assertEqual(self.state['status'],'blocked')
            self.assertIs(self.state['active'],self.item)
            self.assertEqual(probe.call_count,1)
            self.assertEqual(self.state['main'],{})


if __name__=='__main__':
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HandoffTests))
    proof=dict(schema='g50p-terminal-handoff-local-contract-v1',status='passed' if result.wasSuccessful() else 'failed',
        scenarios=result.testsRun,model_forwards=0,models_constructed=False,measured_seconds=time.monotonic()-start)
    (Path(__file__).parent/'local_validation.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
