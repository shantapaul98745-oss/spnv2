"""CPU fixtures only. No real signals, CUDA, subprocesses, or production file writes."""
import copy
import datetime as dt
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import pause_controller as pc


class PauseSafetyTests(unittest.TestCase):
    def setUp(self):
        self.worker = dict(pc.TRAIN, state='S', ppid=10, command=['python', 'train.py', '--cfg', 'fixed.yaml'])
        self.manager = dict(pc.MANAGER, state='S', ppid=11,
                            command=list(pc.MANAGER_COMMAND))
        self.manifest = {'tasks': [], 'fixture': True}
        self.state = dict(status='running', active='train_phi6_EHS', error=None,
                          heartbeat_at=dt.datetime.fromtimestamp(1000, dt.timezone.utc).isoformat(),
                          manifest_sha256=pc.canonical(self.manifest), manager=self.manager,
                          current=pc.TRAIN,
                          tasks=[dict(id='train_phi6_EHS', status='running', command=self.worker['command'],
                                      runtime={'phase': 'main'})])

    def gate(self, **changes):
        value = copy.deepcopy(self.state); value.update(changes)
        with mock.patch.object(pc, 'PRODUCTION_SHA', pc.canonical(self.manifest)):
            return pc.production_gate(value, self.worker, self.manager, self.manifest, clock=1000)

    def test_healthy_and_stopped_process_are_recognized(self):
        self.assertEqual(self.gate()['id'], 'train_phi6_EHS')
        self.worker['state'] = 'T'
        self.assertEqual(self.gate()['id'], 'train_phi6_EHS')

    def test_wrong_phase_error_stale_or_drift_block(self):
        for changes in [dict(active='another'), dict(status='failed'), dict(error='NaN'),
                        dict(downstream_blocked=True), dict(reporting_warning='write failed'),
                        dict(manifest_sha256='changed'), dict(heartbeat_at='1970-01-01T00:00:00+00:00')]:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.gate(**changes)

    def test_recycled_pid_never_signaled(self):
        sent = []
        changed = dict(self.worker, starttime='another')
        self.assertFalse(pc.signal_exact(pc.TRAIN, signal.SIGCONT, reader=lambda _: changed,
                                         sender=lambda *x: sent.append(x)))
        self.assertEqual(sent, [])

    def test_wrong_command_never_signaled(self):
        sent = []
        expected = dict(self.worker, command=['different'])
        self.assertFalse(pc.signal_exact(expected, signal.SIGSTOP, reader=lambda _: self.worker,
                                         sender=lambda *x: sent.append(x)))
        self.assertEqual(sent, [])

    def test_exact_signal_dispatch_is_mocked(self):
        sent = []
        self.assertTrue(pc.signal_exact(self.worker, signal.SIGCONT, reader=lambda _: self.worker,
                                        sender=lambda *x: sent.append(x)))
        self.assertEqual(sent, [(self.worker, signal.SIGCONT)])

    def test_gpu_gate_does_not_treat_stopped_trainer_as_an_intruder(self):
        value = dict(free_mb=5000, processes=[dict(pid=pc.TRAIN['pid'], name='python', memory='9600')])
        pc.gpu_gate(value, {pc.TRAIN['pid']})
        with self.assertRaises(RuntimeError):
            pc.gpu_gate(dict(value, free_mb=4499), {pc.TRAIN['pid']})
        with self.assertRaises(RuntimeError):
            pc.gpu_gate(dict(value, processes=value['processes'] + [dict(pid=42, name='python', memory='100')]),
                        {pc.TRAIN['pid']})

    def test_cleanup_targets_exact_run_and_descendants_only(self):
        config = dict(config=str(pc.Q / 'events/task.main.config.json'), record=str(pc.Q / 'events/task.main.process.json'),
                      result=str(pc.Q / 'events/task.main.result.json'), log=str(pc.Q / 'logs/task.log'),
                      command=['python', 'evaluation.py'])
        owned = dict(pid=123, starttime='555', state='S', ppid=1, command=config['command'])
        child = dict(pid=124, starttime='556', state='S', ppid=123, command=['python', 'child.py'])
        other = dict(pid=125, starttime='557', state='S', ppid=1, command=['python', 'unrelated.py'])
        inventory = {p['pid']: p for p in [owned, child, other, self.worker, self.manager]}
        picked = pc.owned_processes([config], inventory, {config['record']: owned})
        self.assertEqual({p['pid'] for p in picked}, {123, 124})
        stale = dict(owned, starttime='old')
        self.assertEqual(pc.owned_processes([config], inventory, {config['record']: stale}), [])

    def test_unregistered_entry_is_found_without_matching_another_run(self):
        config = dict(config=str(pc.Q / 'events/task.main.config.json'), record=str(pc.Q / 'events/task.main.process.json'),
                      result=str(pc.Q / 'events/task.main.result.json'), log=str(pc.Q / 'logs/task.log'),
                      command=['python', 'evaluation.py'])
        entry = dict(pid=123, starttime='555', state='S', ppid=1,
                     command=[pc.PYTHON, '-B', '-s', str(pc.ENTRY), '--config', config['config']])
        other = dict(entry, pid=124, command=[pc.PYTHON, str(pc.ENTRY), '--config', '/other/config.json'])
        picked = pc.owned_processes([config], {123: entry, 124: other}, {})
        self.assertEqual([p['pid'] for p in picked], [123])

    def test_outside_scope_record_rejected(self):
        config = dict(config='/tmp/unrelated', record=str(pc.Q / 'record.json'), result=str(pc.Q / 'result.json'),
                      log=str(pc.Q / 'log'), command=['python', 'eval.py'])
        with self.assertRaises(RuntimeError):
            pc.owned_processes([config], {}, {})

    def test_cpu_verification_hides_gpu(self):
        task = dict(id='a', session='unique_a', log=str(pc.Q / 'logs/a.log'), cwd=str(pc.P / 'source'),
                    env={'CUDA_VISIBLE_DEVICES': '0'}, command=['python', 'eval.py'], verify_command=['python', 'verify.py'])
        self.assertEqual(pc.phase_config(task, 'main')['env']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(pc.phase_config(task, 'verify')['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_recovery_cleans_before_resume_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(pc, 'Q', Path(directory)):
            pc.atomic(pc.Q / 'pause_intent.json', {'trainer': self.worker})
            order = []
            with mock.patch.object(pc, 'cleanup_owned', side_effect=lambda _: order.append('clean') or []), \
                 mock.patch.object(pc, 'identity', return_value=self.worker), \
                 mock.patch.object(pc, 'signal_exact', side_effect=lambda *a: order.append('continue') or True):
                result = pc.recover({'tasks': []}, 'fixture')
                self.assertEqual(result['status'], 'resumed')
                self.assertEqual(order, ['clean', 'continue'])
                pc.recover({'tasks': []}, 'fixture again')
                self.assertEqual(order, ['clean', 'continue'])

    def test_cleanup_failure_never_resumes_concurrently(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(pc, 'Q', Path(directory)):
            pc.atomic(pc.Q / 'pause_intent.json', {'trainer': self.worker})
            with mock.patch.object(pc, 'cleanup_owned', side_effect=RuntimeError('owned GPU child still alive')), \
                 mock.patch.object(pc, 'signal_exact') as sender:
                with self.assertRaises(RuntimeError):
                    pc.recover({'tasks': []}, 'fixture')
                sender.assert_not_called()


if __name__ == '__main__':
    unittest.main()
