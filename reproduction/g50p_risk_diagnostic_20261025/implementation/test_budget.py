import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from budget import GlobalBudget, BudgetExhausted, MAX_FORWARDS, MAX_ACTIVE_SECONDS, atomic_json


class Clock:
    def __init__(self):self.now=1000.
    def monotonic(self):return self.now
    def time(self):return self.now+100000.


class PhysicalBudgetContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.clock=Clock()
        self.identity={'index_sha256':'a'*64}
        self.record=dict(pid=7,starttime=99,boot_id='own-boot',cmdline_sha256='c'*64,run_nonce='n'*32,run_id='window1')
        self.b=GlobalBudget(self.root/'budget',self.identity,clock=self.clock)
    def tearDown(self):self.tmp.cleanup()

    def test_forward_is_durable_before_dispatch_and_unknown_never_refunded(self):
        attempt=self.b.begin_attempt(self.record)
        token=attempt.reserve_forward('pre',count=2)
        replay=GlobalBudget(self.root/'budget',self.identity,clock=self.clock)
        self.assertEqual(replay.snapshot()['physical_forwards_reserved_upper'],2)
        self.assertEqual(replay.snapshot()['physical_forwards_unknown_upper'],2)
        with self.assertRaises(BudgetExhausted):replay.begin_attempt(self.record)
        attempt.finish_forward(token,count=2)
        self.assertEqual(self.b.snapshot()['physical_forwards_confirmed'],2)
        with self.assertRaises(ValueError):attempt.finish_forward(token)

    def test_exact_global_forward_ceiling_blocks_before_next_forward(self):
        attempt=self.b.begin_attempt(self.record)
        token=attempt.reserve_forward('synthetic-cap',count=MAX_FORWARDS)
        attempt.finish_forward(token)
        with self.assertRaises(BudgetExhausted):attempt.reserve_forward('would-exceed')
        self.assertEqual(self.b.snapshot()['physical_forwards_reserved_upper'],MAX_FORWARDS)
        attempt.close()
        verify=self.b.begin_attempt(self.record,allow_model_forwards=False)
        with self.assertRaises(ValueError):verify.reserve_forward('forbidden_independent_replay')
        verify.close('zero_forward_independent_verification')

    def test_active_time_includes_setup_io_and_close_upper_tail_wait_separate(self):
        self.b.record_queue_wait(600.,reason='foreign_compute')
        attempt=self.b.begin_attempt(self.record)
        self.clock.now+=120.5
        result=attempt.close()
        self.assertEqual(result['active_seconds_charged_upper'],122.5)
        self.assertEqual(result['queue_wait_seconds'],600.)
        second=self.b.begin_attempt({**self.record,'run_id':'window2'})
        self.assertEqual(second.record['allocated_seconds_upper'],MAX_ACTIVE_SECONDS-122.5)

    def test_60_second_stop_margin_and_only_light_final_close_remains(self):
        receipt=self.root/'initial.json'
        atomic_json(receipt,dict(schema='g50p-diagnostic-initial-cost-v1',status='passed',
                                physical_forwards_upper=0,active_seconds_upper=14000.))
        binding=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        other=GlobalBudget(self.root/'with_initial',self.identity,clock=self.clock,initial_cost_receipt=binding)
        attempt=other.begin_attempt(self.record)
        self.clock.now+=350.
        with self.assertRaises(BudgetExhausted):attempt.check_time('checkpoint_before_torch_save')
        result=attempt.close('stopped_before_heavy_save')
        self.assertEqual(result['active_seconds_charged_upper'],14352.)
        with self.assertRaises(BudgetExhausted):other.begin_attempt(self.record)

    def test_initial_cost_imported_once_and_binding_change_rejected(self):
        receipt=self.root/'initial.json'
        atomic_json(receipt,dict(schema='g50p-diagnostic-initial-cost-v1',status='passed',
                                physical_forwards_upper=3,active_seconds_upper=12.))
        item=dict(path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
        b=GlobalBudget(self.root/'initial_budget',self.identity,clock=self.clock,initial_cost_receipt=item)
        self.assertEqual(b.snapshot()['physical_forwards_reserved_upper'],3)
        self.assertEqual(b.snapshot()['physical_forwards_unknown_upper'],0)
        self.assertEqual(b.snapshot()['active_seconds_charged_upper'],12.)
        b=GlobalBudget(self.root/'initial_budget',self.identity,clock=self.clock,initial_cost_receipt=item)
        self.assertEqual(len(b.rows),1)
        wrong={**item,'sha256':'b'*64}
        with self.assertRaises(ValueError):GlobalBudget(self.root/'initial_budget',self.identity,initial_cost_receipt=wrong)

    def test_crash_upper_is_entire_assigned_time_and_no_blind_resume(self):
        attempt=self.b.begin_attempt(self.record)
        attempt.reserve_forward('may-have-dispatched')
        self.clock.now+=2.
        b=GlobalBudget(self.root/'budget',self.identity,clock=self.clock)
        self.assertEqual(b.snapshot()['active_seconds_charged_upper'],MAX_ACTIVE_SECONDS)
        self.assertEqual(len(b.snapshot()['unresolved_attempt_ids']),1)
        with self.assertRaises(BudgetExhausted):b.begin_attempt(self.record)

    def test_torn_or_modified_journal_never_resets_budget(self):
        self.b.begin_attempt(self.record)
        original=self.b.path.read_bytes()
        self.b.path.write_bytes(original+b'{"incomplete":')
        with self.assertRaises(ValueError):GlobalBudget(self.root/'budget',self.identity)
        self.b.path.write_bytes(original.replace(b'attempt_started',b'attempt_starteX'))
        with self.assertRaises(ValueError):GlobalBudget(self.root/'budget',self.identity)

    def test_identity_change_blocks(self):
        with self.assertRaises(ValueError):GlobalBudget(self.root/'budget',{'index_sha256':'x'*64})


if __name__=='__main__':unittest.main(verbosity=2)
