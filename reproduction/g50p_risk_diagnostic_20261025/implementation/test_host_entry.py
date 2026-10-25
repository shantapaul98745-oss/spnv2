import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import host_entry as entry
from budget import atomic_json


class HostEntryContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.plan=Path(self.tmp.name)
        (self.plan/'tasks').mkdir();(self.plan/'implementation').mkdir();(self.plan/'launches').mkdir()
        self.patch=patch.object(entry,'PLAN',self.plan);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def test_task_and_frozen_code_must_match_exact_index(self):
        code=self.plan/'implementation/fake.py';code.write_text('# no model\n')
        task=dict(schema='g50p-risk-diagnostic-task-v1',action='run_pair',run_id='fixed1',
                  implementation_sha256={'fake.py':entry.sha(code)})
        path=self.plan/'tasks/one.json';atomic_json(path,task)
        index=dict(schema='g50p-risk-diagnostic-index-v1',implementation_sha256=task['implementation_sha256'],
                   tasks=[dict(run_id='fixed1',task_path=str(path),task_sha256=entry.sha(path))])
        atomic_json(self.plan/'index.json',index)
        self.assertEqual(entry.validate_task(path,self.plan/'index.json')[0],task)
        code.write_text('# changed\n')
        with self.assertRaises(ValueError):entry.validate_task(path,self.plan/'index.json')

    def test_nonplan_task_and_extra_action_cannot_enter(self):
        outside=self.plan/'elsewhere.json';atomic_json(outside,{'action':'run_pair'})
        atomic_json(self.plan/'index.json',{})
        with self.assertRaises(ValueError):entry.validate_task(outside,self.plan/'index.json')
        path=self.plan/'tasks/bad.json'
        atomic_json(path,dict(schema='g50p-risk-diagnostic-task-v1',action='retrain_source',implementation_sha256={}))
        atomic_json(self.plan/'index.json',dict(schema='g50p-risk-diagnostic-index-v1',implementation_sha256={}))
        with self.assertRaises(ValueError):entry.validate_task(path,self.plan/'index.json')

    def test_watchdog_rejects_any_unverified_identity_before_pidfd_or_signal(self):
        path=self.plan/'launches/nonce/watchdog.json'
        atomic_json(path,dict(launch_record={'project_id':entry.PROJECT},kill_at_monotonic=0.))
        with patch.object(entry,'live_identity',return_value=False), \
             patch.object(entry.os,'pidfd_open',create=True,side_effect=AssertionError('No unverified pidfd')), \
             patch.object(entry.signal,'pidfd_send_signal',create=True,side_effect=AssertionError('No signal')):
            with self.assertRaises(ValueError):entry.watchdog(path)

    def test_deadline_signal_uses_only_preverified_pinned_pidfd(self):
        path=self.plan/'launches/nonce/watchdog.json'
        record=dict(project_id=entry.PROJECT,pid=123,starttime=456)
        atomic_json(path,dict(launch_record=record,kill_at_monotonic=100.))
        with patch.object(entry,'live_identity',return_value=True), \
             patch.object(entry.os,'pidfd_open',create=True,return_value=77) as opened, \
             patch.object(entry.os,'close') as closed, \
             patch.object(entry.signal,'SIGKILL',create=True,new=9), \
             patch.object(entry.signal,'pidfd_send_signal',create=True) as sent, \
             patch.object(entry.select,'select',return_value=([],[],[])), \
             patch.object(entry.time,'monotonic',return_value=101.):
            self.assertEqual(entry.watchdog(path),0)
        opened.assert_called_once_with(123,0)
        sent.assert_called_once_with(77,9,None,0)
        closed.assert_called_once_with(77)
        receipt=json.loads((path.parent/'watchdog_timeout.json').read_text())
        self.assertTrue(receipt['unknown_work_preserved'])
        self.assertFalse(receipt['exact_cost_claim'])

    def test_wrong_project_command_fails_without_reading_foreign_proc(self):
        record=dict(project_id='someone_else',pid=123,argv=['other_training.py'])
        self.assertFalse(entry.live_identity(record))


if __name__=='__main__':unittest.main(verbosity=2)
