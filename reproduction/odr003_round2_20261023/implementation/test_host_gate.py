"""Synthetic local-only unittest: no torch, host query, SSH, launch or signals."""
import copy
import unittest

import host_gate as h


def launch(pid=123, nonce="0123456789abcdef0123456789abcdef"):
    argv=["/opt/python", "run.py", "--run-id", "g50_protected"]
    return dict(schema=h.LAUNCH_SCHEMA,project_id="odr003_round2",run_id="g50_protected",role="worker",
        run_nonce=nonce,boot_id="boot-A",pid=pid,starttime=900,uid=1000,argv=argv,
        cmdline_sha256=h.command_sha256(argv),cwd="/project",session="unique_screen",
        log_path="/project/unique.log",launched_at=50.)


def process(record=None, **changes):
    record=record or launch()
    obj={key:copy.deepcopy(record[key]) for key in h.IDENTITY_FIELDS}
    obj.update(argv=list(record["argv"]),identity_complete=True,state="S",exe=record["argv"][0],role="compute",memory_mib=100)
    obj.update(changes)
    return obj


def desktop(pid=5, **changes):
    argv=[h.DESKTOP_EXE, "--system"]
    p=process(pid=pid,argv=argv,cmdline_sha256=h.command_sha256(argv),exe=h.DESKTOP_EXE,
              role="desktop",memory_mib=256,run_nonce=None)
    p.update(changes)
    return p


def snapshot(name="a", at=100., processes=None, checks=None):
    checks=checks or []
    return dict(snapshot_id=name,captured_at=at,boot_id="boot-A",gpu_uuid="GPU-1",gpu_index=0,
        free_mib=14500,used_mib=1200,query_ok=True,inventory_complete=True,
        gpu_processes=[] if processes is None else processes,worker_inventory_complete=True,
        registered_run_nonces=[x["expected_record"]["run_nonce"] for x in checks],worker_checks=checks)


def worker_check(record=None, actual="live"):
    record=record or launch()
    return dict(expected_record=record,observed_record=copy.deepcopy(record),probe_succeeded=True,
                process=process(record) if actual=="live" else None,absence_confirmed=actual!="live")


def boundary(now=102.):
    return h.assess_launch_boundary(snapshot("a",now-2),snapshot("b",now-1),now=now,project_lock_held=True)


class OwnershipTests(unittest.TestCase):
    def test_exact_complete_launch_record_is_own(self):
        record=launch()
        self.assertTrue(h.verify_ownership(process(record),record,copy.deepcopy(record)).is_own)

    def test_every_identity_component_is_required(self):
        record=launch()
        changes={"boot_id":"boot-B","pid":124,"starttime":901,"run_nonce":"f"*32,"uid":1001}
        for key,value in changes.items():
            with self.subTest(key=key):
                self.assertFalse(h.verify_ownership(process(record,**{key:value}),record,record).is_own)
        changed=process(record,argv=record['argv']+["--different"])
        changed['cmdline_sha256']=h.command_sha256(changed['argv'])
        self.assertFalse(h.verify_ownership(changed,record,record).is_own)

    def test_same_uid_name_or_prefix_is_not_ownership(self):
        record=launch();p=process(record);p.pop('run_nonce')
        self.assertFalse(h.verify_ownership(p,record,record).is_own)
        self.assertFalse(h.verify_ownership(process(record),{},{}).is_own)

    def test_complete_record_extra_fields_are_not_dropped(self):
        record=launch();observed=copy.deepcopy(record);observed['unreviewed_extra']='changed'
        self.assertFalse(h.verify_ownership(process(record),record,observed).is_own)
        observed=copy.deepcopy(record);observed['log_path']='/other.log'
        self.assertFalse(h.verify_ownership(process(record),record,observed).is_own)

    def test_full_argv_hash_checks_arguments_and_boundaries(self):
        self.assertNotEqual(h.command_sha256(['ab','c']),h.command_sha256(['a','bc']))
        self.assertNotEqual(h.command_sha256(['python','x.py']),h.command_sha256(['python','x.py','']))
        with self.assertRaises(ValueError):h.command_sha256(['python','bad\x00arg'])
        record=launch();record['cmdline_sha256']='0'*64
        self.assertFalse(h.verify_ownership(process(record),record,record).is_own)


class ResourceTests(unittest.TestCase):
    def test_idle_resource_thresholds_are_inclusive(self):
        self.assertTrue(h.assess_snapshot(snapshot(),now=102).allow_start)
        for key,value in [('free_mib',14499),('used_mib',1201)]:
            s=snapshot();s[key]=value
            self.assertFalse(h.assess_snapshot(s,now=102).allow_start)

    def test_unknown_compute_blocks_even_at_zero_utilization(self):
        p=process(pid=3365933);p['utilization']=0
        for kind in ('gpu','heavy_cpu'):
            with self.subTest(kind=kind):
                result=h.assess_snapshot(snapshot(processes=[p]),now=102,task_kind=kind)
                self.assertFalse(result.allow_start)
                self.assertIn('unknown_GPU_process_blocks_start:3365933',result.reasons)

    def test_verified_desktop_is_exempt_but_not_owned(self):
        result=h.assess_snapshot(snapshot(processes=[desktop()]),now=102)
        self.assertTrue(result.allow_start)
        self.assertEqual(result.desktop_pids,(5,))
        self.assertEqual(result.owned_worker_pids,())

    def test_desktop_needs_exact_role_path_and_small_memory(self):
        for change in ({'memory_mib':257},{'role':'compute'},{'exe':'/tmp/gnome-remote-desktop-daemon'},
                       {'identity_complete':False},{'memory_mib':None},{'state':'Z'}):
            with self.subTest(change=change):
                self.assertFalse(h.assess_snapshot(snapshot(processes=[desktop(**change)]),now=102).allow_start)

    def test_existing_owned_gpu_or_cpu_worker_blocks_duplicate(self):
        check=worker_check()
        for processes in ([],[check['process']]):
            result=h.assess_snapshot(snapshot(processes=processes,checks=[check]),now=102)
            self.assertFalse(result.allow_start)
            self.assertEqual(result.owned_worker_pids,(123,))

    def test_confirmed_absence_allows_new_task_but_missing_probe_does_not(self):
        check=worker_check(actual='absent')
        self.assertTrue(h.assess_snapshot(snapshot(checks=[check]),now=102).allow_start)
        check['absence_confirmed']=False
        self.assertFalse(h.assess_snapshot(snapshot(checks=[check]),now=102).allow_start)

    def test_recycled_pid_never_becomes_ours(self):
        check=worker_check();check['process']['starttime']+=1
        self.assertTrue(h.assess_snapshot(snapshot(checks=[check]),now=102).allow_start)
        result=h.assess_snapshot(snapshot(checks=[check],processes=[check['process']]),now=102)
        self.assertFalse(result.allow_start)
        self.assertEqual(result.owned_worker_pids,())

    def test_old_boot_gpu_observation_is_not_owned_on_current_boot(self):
        record=launch();record['boot_id']='boot-B'
        check=worker_check(record,actual='absent')
        result=h.assess_snapshot(snapshot(checks=[check],processes=[process(record)]),now=102)
        self.assertFalse(result.allow_start)
        self.assertEqual(result.owned_worker_pids,())

    def test_nonce_registry_must_be_fully_probed(self):
        s=snapshot();s['registered_run_nonces']=['a'*32]
        self.assertFalse(h.assess_snapshot(s,now=102).allow_start)
        s=snapshot();s['worker_inventory_complete']=False
        self.assertFalse(h.assess_snapshot(s,now=102).allow_start)

    def test_query_failure_or_stale_or_wrong_boot_blocks(self):
        for changes in ({'query_ok':False},{'inventory_complete':False},{'captured_at':0.},{'captured_at':103.}):
            s=snapshot();s.update(changes)
            self.assertFalse(h.assess_snapshot(s,now=102).allow_start)
        s=snapshot(processes=[desktop(boot_id='boot-B')])
        self.assertFalse(h.assess_snapshot(s,now=102).allow_start)

    def test_double_boundary_requires_fresh_distinct_snapshots_and_lock(self):
        self.assertTrue(boundary().allow_start)
        self.assertIn('project_lock_is_not_a_global_GPU_reservation',boundary().notes)
        self.assertFalse(h.assess_launch_boundary(snapshot(),snapshot(),now=102,project_lock_held=True).allow_start)
        self.assertFalse(h.assess_launch_boundary(snapshot('a',100),snapshot('b',101),now=102,project_lock_held=False).allow_start)
        self.assertFalse(h.assess_launch_boundary(snapshot('a',90),snapshot('b',91),now=102,project_lock_held=True).allow_start)
        second=snapshot('b',101);second['gpu_uuid']='GPU-2'
        self.assertFalse(h.assess_launch_boundary(snapshot(),second,now=102,project_lock_held=True).allow_start)

    def test_process_arriving_at_final_boundary_blocks(self):
        after=snapshot('b',101,processes=[process(pid=3365933)])
        self.assertFalse(h.assess_launch_boundary(snapshot(),after,now=102,project_lock_held=True).allow_start)

    def test_wait_is_advice_only_at_least_thirty_minutes(self):
        blocked=h.assess_snapshot(snapshot(processes=[process()]),now=102)
        advice=h.wait_advice(blocked,requested_seconds=10)
        self.assertEqual(advice['retry_after_seconds'],1800)
        self.assertFalse(advice['poll_loop']);self.assertFalse(advice['sleep_performed'])
        self.assertTrue(advice['heavy_CPU_model_validation_must_also_wait'])
        self.assertEqual(h.wait_advice(blocked,3600)['retry_after_seconds'],3600)


class AccountingTests(unittest.TestCase):
    def test_wait_and_compute_never_share_a_denominator(self):
        ledger=h.RunAccounting('run',0)
        ledger.begin_compute(1800,boundary(1800))
        ledger.begin_wait(1810,'GPU phase ended; wait for CPU verification admission')
        ledger.begin_compute(3610,boundary(3610))
        ledger.finish(3615,'completed')
        result=ledger.snapshot()
        self.assertEqual(result['queue_wait_seconds'],3600)
        self.assertEqual(result['compute_seconds'],15)
        self.assertEqual(result['accounted_seconds'],3615)

    def test_running_snapshot_does_not_double_count(self):
        ledger=h.RunAccounting('run',10)
        self.assertEqual(ledger.snapshot(20)['queue_wait_seconds'],10)
        self.assertEqual(ledger.snapshot(30)['queue_wait_seconds'],20)
        ledger.begin_compute(40,boundary(40))
        self.assertEqual(ledger.snapshot(45)['queue_wait_seconds'],30)
        self.assertEqual(ledger.snapshot(45)['compute_seconds'],5)

    def test_invalid_or_unadmitted_accounting_transitions_rejected(self):
        ledger=h.RunAccounting('run',100)
        with self.assertRaises(ValueError):ledger.snapshot(99)
        with self.assertRaises(ValueError):ledger.finish(101,'completed')
        with self.assertRaises(ValueError):ledger.begin_compute(101,h.assess_snapshot(snapshot(),now=102))
        blocked=h.assess_launch_boundary(snapshot(),snapshot('b',101),now=102,project_lock_held=False)
        with self.assertRaises(ValueError):ledger.begin_compute(101,blocked)
        ledger.finish(102,'cancelled')
        with self.assertRaises(ValueError):ledger.finish(103,'cancelled')

    def test_expired_admission_is_not_reusable(self):
        ledger=h.RunAccounting('run',0)
        with self.assertRaisesRegex(ValueError,'expired'):
            ledger.begin_compute(108,boundary(102))
        self.assertEqual(ledger.snapshot(108)['compute_seconds'],0)
        self.assertEqual(ledger.snapshot(108)['queue_wait_seconds'],108)

    def test_invalid_wait_reason_does_not_mutate_totals(self):
        ledger=h.RunAccounting('run',0);ledger.begin_compute(10,boundary(10))
        with self.assertRaises(ValueError):ledger.begin_wait(20,'')
        ledger.begin_wait(20,'phase naturally ended')
        self.assertEqual(ledger.snapshot()['compute_seconds'],10)


if __name__=='__main__':
    unittest.main(verbosity=2)
