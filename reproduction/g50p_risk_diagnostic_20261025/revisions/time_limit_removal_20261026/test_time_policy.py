"""Targeted metadata contracts; no model is imported or constructed."""
from pathlib import Path
import hashlib
import importlib.util
import json
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import budget
import host_entry


class Clock:
    def __init__(self): self.now=1000.
    def monotonic(self): return self.now
    def time(self): return 1793000000.+self.now


def binding(p):
    return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


class TimePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.p=Path(self.tmp.name)
        self.clock=Clock()
        self.identity=dict(project_id='test-diagnostic',frozen_inputs_sha256='a'*64)
        self.record=dict(pid=123,starttime=456,boot_id='test-boot',cmdline_sha256='b'*64,
                         run_nonce='c'*32,run_id='metadata-test')
        cost=self.p/'initial.json'
        budget.atomic_json(cost,dict(schema='g50p-diagnostic-initial-cost-v1',status='passed',
            physical_forwards_upper=39999,active_seconds_upper=14400.))
        self.initial=binding(cost)
        self.old=budget.GlobalBudget(self.p/'ledger',self.identity,clock=self.clock,initial_cost_receipt=self.initial)
        protocol=self.p/'old_protocol.json'
        budget.atomic_json(protocol,dict(max_physical_forwards=40000,max_active_seconds=14400))
        authorization=self.p/'authorization.json'
        budget.atomic_json(authorization,dict(authorization='user_2026-10-26_remove_cumulative_time_limit'))
        self.prefix=(self.p/'ledger/journal.jsonl').read_bytes()
        policy=self.p/'policy.json'
        budget.atomic_json(policy,dict(schema='g50p-diagnostic-budget-policy-v2',
            project_id=self.identity['project_id'],frozen_inputs_sha256=self.identity['frozen_inputs_sha256'],
            max_physical_forwards=40000,max_active_seconds=None,time_hard_stop_enabled=False,
            preserve_all_past_charges=True,original_protocol=binding(protocol),
            original_budget_identity=binding(self.p/'ledger/identity.json'),authorization=binding(authorization),
            historical_journal_prefix=dict(events=len(self.old.rows),tail_sha256=self.old.rows[-1]['sha256'])))
        self.policy=binding(policy)
    def tearDown(self): self.tmp.cleanup()
    def amended(self):
        return budget.GlobalBudget(self.p/'ledger',self.identity,clock=self.clock,
            initial_cost_receipt=self.initial,budget_policy=self.policy,activate_policy=True)
    def test_explicit_activation_preserves_legacy_identity_and_full_charges(self):
        identity_bytes=(self.p/'ledger/identity.json').read_bytes()
        with self.assertRaises(budget.BudgetExhausted): self.old.begin_attempt(self.record)
        with self.assertRaises(ValueError):
            budget.GlobalBudget(self.p/'ledger',self.identity,clock=self.clock,budget_policy=self.policy)
        revised=self.amended()
        self.assertTrue((self.p/'ledger/journal.jsonl').read_bytes().startswith(self.prefix))
        self.assertEqual((self.p/'ledger/identity.json').read_bytes(),identity_bytes)
        snap=revised.snapshot()
        self.assertEqual(snap['active_seconds_charged_upper'],14400.)
        self.assertEqual(snap['physical_forwards_reserved_upper'],39999)
        self.assertFalse(snap['time_hard_stop_enabled'])
        self.assertIsNone(snap['remaining_active_seconds_lower'])
    def test_past_old_deadline_allowed_but_40000_forward_cap_still_strict(self):
        revised=self.amended();attempt=revised.begin_attempt(self.record)
        self.assertIsNone(attempt.deadline_monotonic)
        self.clock.now+=100000.
        attempt.check_time('beyond historical deadline')
        token=attempt.reserve_forward('mock-accounting-token-only')
        attempt.finish_forward(token)
        with self.assertRaises(budget.BudgetExhausted): attempt.reserve_forward('over-cap')
        attempt.close();self.clock.now+=25;attempt.check_final_receipt()
        self.assertGreater(revised.snapshot()['active_seconds_charged_upper'],114400.)
        with self.assertRaises(budget.BudgetExhausted): revised.begin_attempt(self.record)
        verify=revised.begin_attempt(self.record,allow_model_forwards=False)
        with self.assertRaises(ValueError): verify.reserve_forward('forbidden-verifier-forward')
        verify.close();verify.check_final_receipt()
    def test_unclosed_unlimited_attempt_is_unknown_not_zero_and_blocks_duplicates(self):
        revised=self.amended();revised.begin_attempt(self.record)
        self.clock.now+=999999.
        snap=revised.snapshot()
        self.assertIsNone(snap['active_seconds_charged_upper'])
        self.assertEqual(snap['settled_active_seconds_charged_upper'],14400.)
        self.assertTrue(snap['unresolved_attempt_ids'])
        loaded=budget.GlobalBudget(self.p/'ledger',self.identity,clock=self.clock)
        with self.assertRaises(budget.BudgetExhausted): loaded.begin_attempt(self.record)
    def test_forward_cap_and_frozen_identity_cannot_be_revised(self):
        doc=json.loads(Path(self.policy['path']).read_text());doc['max_physical_forwards']=40001
        bad=self.p/'bad_policy.json';budget.atomic_json(bad,doc)
        with self.assertRaises(ValueError):
            budget.GlobalBudget(self.p/'ledger',self.identity,clock=self.clock,budget_policy=binding(bad),activate_policy=True)
        self.assertEqual((self.p/'ledger/journal.jsonl').read_bytes(),self.prefix)
    def test_auto_loads_applied_policy_and_reports_measured_vs_conservative(self):
        revised=self.amended();a=revised.begin_attempt(self.record,allow_model_forwards=False)
        self.clock.now+=3;a.close();self.clock.now+=1;a.check_final_receipt()
        loaded=budget.GlobalBudget(self.p/'ledger',self.identity,clock=self.clock)
        snap=loaded.snapshot();self.assertFalse(snap['time_hard_stop_enabled'])
        self.assertEqual(snap['time_accounting']['closed_measured_seconds'],4.)
        self.assertGreaterEqual(snap['active_seconds_charged_upper'],14406.)
        self.assertEqual(snap['physical_forwards_reserved_upper'],39999)
    def test_slow_close_fsync_is_accounted_without_a_time_stop(self):
        revised=self.amended();a=revised.begin_attempt(self.record,allow_model_forwards=False)
        append=revised._append
        def slow(kind,payload):
            row=append(kind,payload)
            if kind=='attempt_closed':self.clock.now+=25.
            return row
        revised._append=slow
        a.close('slow_fsync')
        a.check_final_receipt()
        snap=revised.snapshot()
        self.assertFalse(snap['active_time_bound_violation_attempt_ids'])
        self.assertGreaterEqual(snap['time_accounting']['closed_measured_seconds'],25.)
        self.assertGreaterEqual(snap['active_seconds_charged_upper'],14427.)


class WatchdogTests(unittest.TestCase):
    def test_no_deadline_monitors_pinned_identity_without_sending_any_signal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'launches'/'test'/'watchdog.json';path.parent.mkdir(parents=True)
            budget.atomic_json(path,dict(launch_record=dict(project_id=host_entry.PROJECT,pid=123,starttime=456),
                                         kill_at_monotonic=None))
            with mock.patch.object(host_entry,'PLAN',root),mock.patch.object(host_entry,'live_identity',return_value=True),\
                 mock.patch.object(host_entry.os,'pidfd_open',return_value=99,create=True),\
                 mock.patch.object(host_entry.os,'close'),\
                 mock.patch.object(host_entry.signal,'pidfd_send_signal',create=True) as kill,\
                 mock.patch.object(host_entry.select,'select',return_value=([99],[],[])):
                self.assertEqual(host_entry.watchdog(path),0)
                kill.assert_not_called()
                ready=json.loads((path.parent/'watchdog_ready.json').read_text())
                self.assertIsNone(ready['deadline_monotonic'])
    def test_start_watchdog_serializes_null_deadline_instead_of_old_14400(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            class B:
                deadline_monotonic=None
                def check_time(self,*args): pass
            class Child:
                def poll(self): return None
            def spawn(*args,**kwargs):
                budget.atomic_json(p/'watchdog_ready.json',dict(status='ready'))
                return Child()
            with mock.patch.object(host_entry.subprocess,'Popen',side_effect=spawn):
                child,log=host_entry.start_watchdog(p,dict(pid=123,starttime=456),B())
                log.close()
            spec=json.loads((p/'watchdog.json').read_text())
            self.assertIsNone(spec['kill_at_monotonic'])
            self.assertIsNone(spec['active_time_deadline_monotonic'])


if __name__=='__main__':
    started=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    proof=dict(schema='g50p-time-limit-targeted-local-validation-v1',status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun,physical_model_forwards=0,models_constructed=False,
        mocked_accounting_tokens_not_physical_forwards=True,measured_seconds=time.monotonic()-started)
    (HERE/'local_validation.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
