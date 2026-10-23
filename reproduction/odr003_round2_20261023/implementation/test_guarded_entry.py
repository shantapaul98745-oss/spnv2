"""Pure/mock entry wiring tests. No real /proc, Linux locks, GPU, SSH or model."""
from contextlib import contextmanager
import copy
import hashlib
import json
import posixpath
from pathlib import PurePosixPath
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import guarded_entry as entry
import host_gate as gate
import host_snapshot as collector

ROOT='/project/round2'
PROJECT='odr003_round2_20261023'
NONCE='0123456789abcdef0123456789abcdef'
REGISTRY=ROOT+'/launches/registry.json'


def launch_record(pid=123,nonce=NONCE):
    argv=['/opt/python','guarded_entry.py','--task-json',ROOT+'/tasks/test.json']
    return dict(schema=gate.LAUNCH_SCHEMA,project_id=PROJECT,run_id='test',role='worker',
        run_nonce=nonce,boot_id='boot-A',pid=pid,starttime=900,uid=1000,argv=argv,
        cmdline_sha256=gate.command_sha256(argv),cwd=ROOT,session='123.unique',log_path=ROOT+'/test.log',launched_at=102.)


def live_process(record):
    return {**{key:copy.deepcopy(record[key]) for key in gate.IDENTITY_FIELDS},
        'argv':copy.deepcopy(record['argv']),'identity_complete':True,'state':'R','role':'experiment_worker','memory_mib':700}


def resource_snapshot(record=None, *, own_gpu=True):
    checks=[] if record is None else [dict(expected_record=copy.deepcopy(record),observed_record=copy.deepcopy(record),
        probe_succeeded=True,absence_confirmed=False,process=live_process(record))]
    return dict(snapshot_id='snapshot',captured_at=101.,boot_id='boot-A',gpu_uuid='GPU-1',gpu_index=0,
        free_mib=15000,used_mib=700,query_ok=True,inventory_complete=True,worker_inventory_complete=True,
        registered_run_nonces=[] if record is None else [record['run_nonce']],worker_checks=checks,
        gpu_processes=[] if record is None or not own_gpu else [live_process(record)],
        verified_own_gpu_pids=[] if record is None or not own_gpu else [record['pid']],
        foreign_gpu_pids=[],desktop_gpu_pids=[],collection_errors=[])


def proc_stat():
    return ('123 (guarded entry) '+' '.join(['R']+['0']*18+['900']+['0']*10)).encode()


class FakePath:
    fs=None
    def __init__(self,*parts):self.value=posixpath.normpath(posixpath.join(*(str(x) for x in parts)))
    def __str__(self):return self.value
    def __fspath__(self):return self.value
    def __truediv__(self,other):return FakePath(self.value,str(other))
    @property
    def parent(self):return FakePath(posixpath.dirname(self.value))
    @property
    def name(self):return posixpath.basename(self.value)
    @property
    def suffix(self):return PurePosixPath(self.value).suffix
    def is_relative_to(self,root):return PurePosixPath(self.value).is_relative_to(PurePosixPath(str(root)))
    def resolve(self,strict=False):
        if strict and not self.exists():raise FileNotFoundError(self.value)
        return self
    def exists(self):return self.value in self.fs.files or self.value in self.fs.directories
    def mkdir(self,parents=False,exist_ok=False):
        if self.exists() and not exist_ok:raise FileExistsError(self.value)
        self.fs.directories.add(self.value);self.fs.created_dirs.append(self.value)
    def read_bytes(self):
        self.fs.reads.append(self.value)
        if self.value not in self.fs.files:raise FileNotFoundError(self.value)
        return self.fs.files[self.value]
    def read_text(self,*args,**kwargs):return self.read_bytes().decode()
    @classmethod
    def cwd(cls):return cls(ROOT)


class MockEntryWorld:
    def __init__(self, *, initial_foreign=False, runtime_foreign=False, missing_registration=False,
                 fail_lock=None, nonce=NONCE, fail_registration_write=False):
        self.files={};self.directories=set();self.created_dirs=[];self.reads=[];self.writes=[]
        self.initial_foreign=initial_foreign;self.runtime_foreign=runtime_foreign
        self.missing_registration=missing_registration;self.fail_lock=fail_lock
        self.fail_registration_write=fail_registration_write
        self.calls=0;self.held=[];self.acquired=[];self.released=[];self.executed=False;self.execute_count=0
        self.env={'STY':'123.unique','SPNV2_RUN_NONCE':nonce}
        record=launch_record()
        self.files['/proc/123/stat']=proc_stat()
        self.files['/proc/123/cmdline']=('\x00'.join(record['argv'])+'\x00').encode()
        self.files['/proc/123/environ']=('SPNV2_RUN_NONCE='+NONCE+'\x00').encode()
        self.files['/proc/sys/kernel/random/boot_id']=b'boot-A\n'
        self.task={'schema':'odr003-round2-task-v1','run_id':'test','implementation_sha256':{}}
        self.files[ROOT+'/tasks/test.json']=json.dumps(self.task).encode()
        self.lock_paths=('/old/pipeline.lock','/first003/pipeline.lock',ROOT+'/pipeline.lock')

    @contextmanager
    def lock(self,path):
        path=str(path)
        if path==self.fail_lock:raise BlockingIOError('another manager holds this lock')
        self.held.append(path);self.acquired.append(path)
        try:yield
        finally:self.held.remove(path);self.released.append(path)

    def atomic(self,path,value):
        path=str(path)
        if self.fail_registration_write and path.endswith('/launch.json'):
            raise PermissionError('synthetic registration write failure')
        self.writes.append((path,copy.deepcopy(value)));self.files[path]=json.dumps(value).encode()

    def collect(self,path,**kwargs):
        self.calls+=1
        if str(path)!=REGISTRY or str(kwargs['project_root'])!=ROOT or kwargs['project_id']!=PROJECT:
            raise AssertionError('Wrong registry/root/project passed to collector')
        registry=json.loads(self.files[REGISTRY])
        record=None
        if registry['records'] and not self.missing_registration:
            item=registry['records'][-1];record=json.loads(self.files[item['record_path']])
        snapshot=resource_snapshot(record,own_gpu=self.executed)
        if record is not None:
            actual_nonce=collector._nonce(self.files['/proc/123/environ'])
            snapshot['worker_checks'][0]['process']['run_nonce']=actual_nonce
            if self.executed:
                snapshot['gpu_processes'][0]['run_nonce']=actual_nonce
                if actual_nonce!=record['run_nonce']:
                    snapshot['verified_own_gpu_pids']=[]
                    snapshot['foreign_gpu_pids']=[record['pid']]
        snapshot.update(snapshot_id='snap'+str(self.calls),captured_at=min(99.+self.calls,102.),
            registry_path=REGISTRY,registry_sha256=hashlib.sha256(self.files[REGISTRY]).hexdigest(),project_id=PROJECT)
        if registry.get('schema')!=collector.REGISTRY_SCHEMA or registry.get('project_id')!=PROJECT:
            snapshot['worker_inventory_complete']=False
        if self.initial_foreign or (self.runtime_foreign and self.executed):
            snapshot['foreign_gpu_pids']=[3365933]
            snapshot['gpu_processes'].append({'pid':3365933,'boot_id':'boot-A','role':'foreign','identity_complete':False,'memory_mib':9000})
        return snapshot

    def execute(self,task_path,*,admission,launch_record,conflict_check):
        self.executed=True;self.execute_count+=1
        if self.held!=list(self.lock_paths):raise AssertionError('All three caller locks must remain held during execution')
        if not admission['allow_start'] or not admission['boundary_verified']:raise AssertionError('Model entered without admission')
        conflict=conflict_check()
        return {'exit_code':75 if conflict else 0,'status':'waiting_resources' if conflict else 'main_complete','reason':conflict}

    @contextmanager
    def patched(self, *, bypass_task_path=True):
        from contextlib import ExitStack
        FakePath.fs=self
        model_module=ModuleType('run_round2');model_module.execute=self.execute
        verify_module=ModuleType('verify_engineering_phase');verify_module.execute=self.execute
        fake_os=SimpleNamespace(name='posix',environ=self.env,getpid=lambda:123,getuid=lambda:1000)
        with ExitStack() as stack:
            stack.enter_context(patch.object(entry,'Path',FakePath))
            stack.enter_context(patch.object(entry,'PLAN',FakePath(ROOT)))
            stack.enter_context(patch.object(entry,'LOCKS',tuple(FakePath(x) for x in self.lock_paths)))
            stack.enter_context(patch.object(entry,'os',fake_os))
            stack.enter_context(patch.object(entry,'atomic_json',self.atomic))
            stack.enter_context(patch.object(entry,'project_lock',self.lock))
            stack.enter_context(patch.object(entry,'sha256',lambda path:'a'*64))
            stack.enter_context(patch.object(entry.time,'time',lambda:102.))
            stack.enter_context(patch.object(collector,'collect_snapshot',self.collect))
            stack.enter_context(patch.dict(sys.modules,{'run_round2':model_module,'verify_engineering_phase':verify_module}))
            if bypass_task_path:stack.enter_context(patch.object(entry,'assert_task_path',lambda path:self.task))
            yield


class ResourceConflictContractTests(unittest.TestCase):
    def setUp(self):
        self.clock=patch.object(entry.time,'time',lambda:102.)
        self.clock.start();self.addCleanup(self.clock.stop)

    def test_current_worker_before_GPU_allocation_is_not_foreign(self):
        own=launch_record()
        self.assertIsNone(entry.resource_conflict(resource_snapshot(own,own_gpu=False),own))

    def test_current_worker_own_GPU_is_not_foreign(self):
        own=launch_record()
        self.assertIsNone(entry.resource_conflict(resource_snapshot(own),own))

    def test_absent_self_registration_fails_closed(self):
        own=launch_record()
        self.assertIsNotNone(entry.resource_conflict(resource_snapshot(),own))

    def test_forged_classification_does_not_replace_nonce_proof(self):
        own=launch_record();snapshot=resource_snapshot(own)
        snapshot['worker_checks'][0]['process']['run_nonce']='wrong-nonce-of-another-run'
        self.assertIsNotNone(entry.resource_conflict(snapshot,own))

    def test_changed_full_launch_record_fails_even_if_core_identity_matches(self):
        own=launch_record();snapshot=resource_snapshot(own)
        for key in ('expected_record','observed_record'):snapshot['worker_checks'][0][key]['log_path']='/other.log'
        self.assertIsNotNone(entry.resource_conflict(snapshot,own))

    def test_another_registered_live_CPU_worker_is_a_conflict(self):
        own=launch_record();snapshot=resource_snapshot(own)
        other=launch_record(pid=456,nonce='f'*32)
        snapshot['worker_checks'].append(dict(expected_record=other,observed_record=copy.deepcopy(other),
            probe_succeeded=True,absence_confirmed=False,process=live_process(other)))
        snapshot['registered_run_nonces'].append(other['run_nonce'])
        self.assertIsNotNone(entry.resource_conflict(snapshot,own))

    def test_foreign_arrival_or_incomplete_observation_blocks(self):
        own=launch_record()
        snapshot=resource_snapshot(own);snapshot['foreign_gpu_pids']=[3365933]
        self.assertIsNotNone(entry.resource_conflict(snapshot,own))
        for key in ('query_ok','inventory_complete','worker_inventory_complete'):
            snapshot=resource_snapshot(own);snapshot[key]=False
            self.assertIsNotNone(entry.resource_conflict(snapshot,own))


class OwnRecordContractTests(unittest.TestCase):
    def test_record_uses_exact_proc_command_bytes_including_empty_last_arg(self):
        world=MockEntryWorld()
        raw=world.files['/proc/123/cmdline']+b'\x00'
        world.files['/proc/123/cmdline']=raw
        with world.patched():record=entry.own_launch_record(world.task,ROOT+'/test.log')
        self.assertEqual(record['argv'][-1],'')
        self.assertEqual(record['cmdline_sha256'],hashlib.sha256(raw).hexdigest())

    def test_initial_proc_environment_nonce_must_match_declared_nonce(self):
        world=MockEntryWorld();world.files['/proc/123/environ']=b'SPNV2_RUN_NONCE=different-initial-value\x00'
        with world.patched():
            try:rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
            except (RuntimeError,ValueError):rc=1
        self.assertNotEqual(rc,0)
        self.assertFalse(world.executed)


class MainEntryContractTests(unittest.TestCase):
    def test_busy_host_never_enters_heavy_execution(self):
        world=MockEntryWorld(initial_foreign=True)
        with world.patched():rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
        self.assertEqual(rc,75);self.assertFalse(world.executed)
        self.assertEqual(world.acquired,list(world.lock_paths));self.assertEqual(world.held,[])
        self.assertEqual(json.loads(world.files[REGISTRY])['records'],[])

    def test_idle_admission_registers_and_holds_three_locks_until_execution_ends(self):
        world=MockEntryWorld()
        with world.patched():rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
        self.assertEqual(rc,0);self.assertEqual(world.execute_count,1)
        self.assertEqual(world.held,[])
        registry=json.loads(world.files[REGISTRY]);self.assertEqual(registry['schema'],collector.REGISTRY_SCHEMA)
        self.assertEqual(registry['project_id'],PROJECT);self.assertEqual(len(registry['records']),1)
        item=registry['records'][0];record=json.loads(world.files[item['record_path']])
        self.assertEqual(item['record_sha256'],gate.launch_record_sha256(record))

    def test_registration_must_be_observed_before_heavy_execution(self):
        world=MockEntryWorld(missing_registration=True)
        with world.patched():
            try:rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
            except (ValueError,RuntimeError):rc=1
        self.assertNotEqual(rc,0);self.assertFalse(world.executed)

    def test_registration_write_failure_cannot_enter_heavy_execution(self):
        world=MockEntryWorld(fail_registration_write=True)
        with world.patched():
            with self.assertRaises(PermissionError):entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
        self.assertFalse(world.executed)
        self.assertEqual(world.held,[])

    def test_heavy_CPU_verifier_waits_for_foreign_GPU_too(self):
        world=MockEntryWorld(initial_foreign=True);world.task['action']='verify_engineering'
        with world.patched():rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
        self.assertEqual(rc,75);self.assertFalse(world.executed)

    def test_conflict_yields_cooperatively_without_any_signal_surface(self):
        world=MockEntryWorld(runtime_foreign=True)
        with world.patched():rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
        self.assertEqual(rc,75);self.assertTrue(world.executed)
        self.assertEqual(world.held,[])
        # Fake OS deliberately has no kill/signal/process-termination APIs.

    def test_contended_lock_only_releases_locks_this_attempt_acquired(self):
        world=MockEntryWorld(fail_lock='/first003/pipeline.lock')
        with world.patched():rc=entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
        self.assertEqual(rc,75);self.assertFalse(world.executed);self.assertEqual(world.calls,0)
        self.assertEqual(world.acquired,['/old/pipeline.lock'])
        self.assertEqual(world.released,['/old/pipeline.lock'])

    def test_invalid_nonce_rejected_before_attempt_directory_write(self):
        for nonce in ('','missing','/tmp/escape_nonce_123456789','../escape_nonce_123456789'):
            world=MockEntryWorld(nonce=nonce)
            with world.patched():
                with self.assertRaises((RuntimeError,ValueError)):
                    entry.main(ROOT+'/tasks/test.json',ROOT+'/test.log')
            self.assertEqual(world.created_dirs,[])
            self.assertFalse(world.executed)

    def test_task_must_bind_admission_dependencies(self):
        world=MockEntryWorld();world.task['implementation_sha256']={'guarded_entry.py':'a'*64}
        world.files[ROOT+'/tasks/test.json']=json.dumps(world.task).encode()
        with world.patched(bypass_task_path=False):
            with self.assertRaises(ValueError):entry.assert_task_path(ROOT+'/tasks/test.json')


if __name__=='__main__':
    unittest.main(verbosity=2)
