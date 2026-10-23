"""Filesystem-only wake tests: no signals, GPU queries, or subprocesses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_control as control


class WakeContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan = Path(self.tmp.name)
        self.manager = dict(pid=17, starttime=99, boot_id='owned-boot',
                            cmdline_sha256='c'*64, manager_nonce='n'*32,
                            manager_lock=str(self.plan/'manager.lock'))
        self.state = dict(status='waiting_resources', manager=self.manager)
        self.patches = [patch.object(control, 'atomic_json', side_effect=self.write),
                        patch.object(control, 'private_read', side_effect=self.read)]
        for item in self.patches: item.start()
        self.write(self.plan/'status.json', self.state)

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.tmp.cleanup()

    @staticmethod
    def write(path, data):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf8')

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf8'))

    def request(self):
        with patch.object(control, 'verify_manager_live', return_value=self.manager), \
             patch.object(control.time, 'time', return_value=100.):
            return control.request_refresh(self.plan)

    def test_request_only_writes_mailbox_and_preserves_status(self):
        original = (self.plan/'status.json').read_bytes()
        request = self.request()
        self.assertTrue(request['no_signal_sent'])
        self.assertTrue(request['no_manager_restarted'])
        self.assertEqual((self.plan/'status.json').read_bytes(), original)
        rows = control.pending_refreshes(self.plan, self.manager, now=101.)
        self.assertEqual([row['request_id'] for row in rows], [request['request_id']])

    def test_exact_request_once_ack_uses_real_sample_time(self):
        request = self.request()
        rows = control.pending_refreshes(self.plan, self.manager, now=101.)
        control.acknowledge(self.plan, rows, snapshot={'snapshot_id':'fresh-1', 'captured_at':102.},
                            status='sampled_busy', now=103.)
        ack = self.read(self.plan/'control/acks'/(request['request_id']+'.json'))
        self.assertEqual(ack['sampled_at'], 102.)
        self.assertTrue(ack['sampled_resources'])
        self.assertEqual(control.pending_refreshes(self.plan, self.manager, now=104.), [])

    def test_other_manager_or_old_request_cannot_wake_new_manager(self):
        request = self.request()
        changed = {**self.manager, 'manager_nonce':'x'*32}
        self.assertEqual(control.pending_refreshes(self.plan, changed, now=101.), [])
        ack = self.read(self.plan/'control/acks'/(request['request_id']+'.json'))
        self.assertFalse(ack['sampled_resources'])
        request = self.request()
        self.assertEqual(control.pending_refreshes(self.plan, self.manager, now=4000.), [])

    def test_terminal_state_or_failed_live_identity_prevents_mailbox(self):
        for status in ('blocked', 'complete', 'engineering_held'):
            self.write(self.plan/'status.json', {**self.state, 'status':status})
            with patch.object(control, 'verify_manager_live', side_effect=AssertionError('Do not probe terminal manager')):
                with self.assertRaises(ValueError): control.request_refresh(self.plan)
        self.write(self.plan/'status.json', self.state)
        with patch.object(control, 'verify_manager_live', side_effect=ValueError('Wrong nonce')):
            with self.assertRaises(ValueError): control.request_refresh(self.plan)
        self.assertFalse((self.plan/'control/requests').exists())

    def test_target_requires_launch_nonce(self):
        broken = copy.deepcopy(self.manager); broken['manager_nonce'] = ''
        with self.assertRaises(ValueError): control.target_identity(broken)


if __name__ == '__main__':
    unittest.main(verbosity=2)
