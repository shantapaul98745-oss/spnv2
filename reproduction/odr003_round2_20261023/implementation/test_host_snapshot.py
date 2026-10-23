"""Injected filesystem/command tests only; no real collector, /proc, GPU or SSH."""
from contextlib import contextmanager
import copy
import json
import posixpath
import unittest

import host_gate as gate
import host_snapshot as hs

ROOT = "/project/round2"
REGISTRY = ROOT + "/launch_registry.json"
PROJECT = "odr003_round2_20261023"


def record(pid=123):
    argv=["/opt/python", "run_round2.py", "--task", "candidate.json"]
    return dict(schema=gate.LAUNCH_SCHEMA,project_id=PROJECT,run_id="protected",role="worker",
        run_nonce="0123456789abcdef0123456789abcdef",boot_id="boot-A",pid=pid,starttime=900,uid=1000,
        argv=argv,cmdline_sha256=gate.command_sha256(argv),cwd=ROOT,session="unique",log_path=ROOT+"/run.log",launched_at=50.)


def stat_bytes(pid=123, starttime=900, state="S"):
    fields=[state]+["0"]*18+[str(starttime)]+["0"]*10
    return (str(pid)+" (name (with) parens) "+" ".join(fields)).encode()


def proc_data(rec=None):
    rec=rec or record()
    return {"stat":stat_bytes(rec['pid'],rec['starttime']),
            "cmdline":("\x00".join(rec['argv'])+"\x00").encode(),
            "status":("Name:\tpython\nUid:\t"+"\t".join([str(rec['uid'])]*4)+"\n").encode(),
            "environ":b"UNRELATED_SECRET=not_retained\x00SPNV2_RUN_NONCE="+rec['run_nonce'].encode()+b"\x00",
            "cwd":rec['cwd'],"exe":"/opt/python-real"}


class FakeProc:
    def __init__(self, fs, pid, values):self.fs,self.pid,self.values=fs,pid,values
    def _value(self,key):
        self.fs.reads.append((self.pid,key))
        value=self.values.get(key,FileNotFoundError(key))
        if isinstance(value,list):
            value=value.pop(0) if len(value)>1 else value[0]
        if isinstance(value,BaseException):raise value
        return value
    def read_bytes(self,name):return self._value(name)
    def readlink(self,name):return self._value(name)


class FakeFS:
    def __init__(self, records=()):
        self.files={};self.procs={};self.reads=[];self.opened=[];self.secure_reads=[];self.denied=set()
        entries=[]
        for rec in records:
            path=ROOT+"/launches/"+str(rec['pid'])+".json"
            self.files[path]=json.dumps(rec).encode()
            entries.append(dict(record_path=path,record_sha256=gate.launch_record_sha256(rec)))
            self.procs[rec['pid']]=proc_data(rec)
        self.files[REGISTRY]=json.dumps(dict(schema=hs.REGISTRY_SCHEMA,project_id=PROJECT,records=entries)).encode()
        self.boot=b"boot-A\n"
    def realpath(self,path):return posixpath.normpath(path)
    def read_bytes(self,path):
        if path!=hs.BOOT_ID_PATH:raise AssertionError("unexpected public file read")
        return self.boot
    def read_owned_bytes(self,path,uid):
        self.secure_reads.append(path)
        if path in self.denied:raise PermissionError(path)
        if path not in self.files:raise FileNotFoundError(path)
        value=self.files[path]
        if isinstance(value,list):value=value.pop(0) if len(value)>1 else value[0]
        if isinstance(value,BaseException):raise value
        return value
    @contextmanager
    def open_proc(self,pid):
        self.opened.append(pid)
        values=self.procs.get(pid,FileNotFoundError(str(pid)))
        if isinstance(values,BaseException):raise values
        yield FakeProc(self,pid,values)


class FakeCommands:
    def __init__(self, processes="", gpu="GPU-1, 15000, 700\n"):
        self.processes,self.gpu,self.calls=processes,gpu,[]
    def __call__(self,argv):
        self.calls.append(argv)
        value=self.gpu if any(x.startswith('--query-gpu=') for x in argv) else self.processes
        if isinstance(value,BaseException):raise value
        return value if isinstance(value,hs.CommandResult) else hs.CommandResult(0,value)


def collect(fs=None, commands=None, at=100., name="sample"):
    return hs.collect_snapshot(REGISTRY,project_root=ROOT,project_id=PROJECT,fs=fs or FakeFS(),
        command_runner=commands or FakeCommands(),clock=lambda:at,snapshot_id_factory=lambda:name,current_uid=1000)


class CollectorTests(unittest.TestCase):
    def test_empty_trusted_registry_and_idle_GPU(self):
        s=collect()
        self.assertTrue(s['query_ok']);self.assertTrue(s['worker_inventory_complete'])
        self.assertTrue(gate.assess_snapshot(s,now=101).allow_start)

    def test_foreign_GPU_uses_only_nvidia_metadata(self):
        fs=FakeFS();commands=FakeCommands('GPU-1, 3365933, /opt/python, 9000\n')
        s=collect(fs,commands)
        self.assertEqual(fs.opened,[]);self.assertEqual(fs.reads,[])
        self.assertEqual(s['foreign_gpu_pids'],[3365933])
        for kind in ('gpu','heavy_cpu'):
            self.assertFalse(gate.assess_snapshot(s,now=101,task_kind=kind).allow_start)
        self.assertEqual(len(commands.calls),2)

    def test_own_nonce_is_observed_and_only_that_env_key_retained(self):
        rec=record();fs=FakeFS([rec]);s=collect(fs,FakeCommands('GPU-1, 123, /opt/python, 700\n'))
        self.assertEqual(s['verified_own_gpu_pids'],[123]);self.assertEqual(s['foreign_gpu_pids'],[])
        self.assertEqual(s['worker_checks'][0]['process']['run_nonce'],rec['run_nonce'])
        self.assertNotIn('UNRELATED_SECRET',json.dumps(s))
        self.assertEqual([pid for pid,key in fs.reads if key=='environ'],[123])
        self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_missing_or_wrong_nonce_not_filled_from_expected(self):
        for environment in (b'OTHER=1\x00',b'SPNV2_RUN_NONCE=wrong\x00'):
            fs=FakeFS([record()]);fs.procs[123]['environ']=environment
            s=collect(fs,FakeCommands('GPU-1, 123, /opt/python, 700\n'))
            self.assertEqual(s['verified_own_gpu_pids'],[])
            self.assertEqual(s['foreign_gpu_pids'],[123])
            self.assertNotIn((123,'cwd'),fs.reads)
            self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_recycled_PID_only_reads_stat_never_private_fields(self):
        fs=FakeFS([record()]);fs.procs[123]['stat']=stat_bytes(starttime=901)
        s=collect(fs,FakeCommands('GPU-1, 123, /other/program, 100\n'))
        self.assertEqual(fs.reads,[(123,'stat')])
        self.assertTrue(s['worker_checks'][0]['absence_confirmed'])
        self.assertEqual(s['foreign_gpu_pids'],[123])
        self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_old_boot_record_is_absent_without_reading_that_PID(self):
        rec=record();rec['boot_id']='old-boot';fs=FakeFS([rec]);s=collect(fs)
        self.assertEqual(fs.opened,[])
        self.assertTrue(s['worker_checks'][0]['absence_confirmed'])
        self.assertTrue(gate.assess_snapshot(s,now=101).allow_start)

    def test_public_command_or_uid_mismatch_skips_environment(self):
        for change in ({'cmdline':b'/other/program\x00'},{'status':b'Uid:\t2000\t2000\t2000\t2000\n'}):
            fs=FakeFS([record()]);fs.procs[123].update(change);s=collect(fs)
            self.assertNotIn((123,'environ'),fs.reads);self.assertNotIn((123,'cwd'),fs.reads)
            self.assertFalse(s['worker_inventory_complete'])
            self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_missing_proc_is_different_from_permission_failure(self):
        for failure,confirmed in ((FileNotFoundError(),True),(PermissionError(),False)):
            fs=FakeFS([record()]);fs.procs[123]=failure;s=collect(fs)
            self.assertEqual(s['worker_checks'][0]['absence_confirmed'],confirmed)
            self.assertEqual(gate.assess_snapshot(s,now=101).allow_start,confirmed)

    def test_missing_env_alone_never_means_process_absent(self):
        fs=FakeFS([record()]);fs.procs[123]['environ']=FileNotFoundError();s=collect(fs)
        self.assertFalse(s['worker_checks'][0]['absence_confirmed'])
        self.assertFalse(s['worker_inventory_complete'])

    def test_nonce_permission_error_is_uncertainty_not_absence(self):
        fs=FakeFS([record()]);fs.procs[123]['environ']=PermissionError();s=collect(fs)
        self.assertFalse(s['worker_checks'][0]['absence_confirmed'])
        self.assertFalse(s['worker_inventory_complete'])

    def test_process_birth_change_during_probe_blocks(self):
        fs=FakeFS([record()]);fs.procs[123]['stat']=[stat_bytes(),stat_bytes(starttime=901)]
        s=collect(fs)
        self.assertFalse(s['worker_inventory_complete'])
        self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_known_desktop_reads_public_metadata_only(self):
        fs=FakeFS();rec=record(pid=5);rec['argv']=[gate.DESKTOP_EXE,'--system'];rec['cmdline_sha256']=gate.command_sha256(rec['argv'])
        fs.procs[5]=proc_data(rec);fs.procs[5]['exe']=gate.DESKTOP_EXE
        s=collect(fs,FakeCommands('GPU-1, 5, '+gate.DESKTOP_EXE+', 120\n'))
        self.assertEqual(s['desktop_gpu_pids'],[5]);self.assertEqual(s['verified_own_gpu_pids'],[])
        self.assertNotIn((5,'environ'),fs.reads);self.assertNotIn((5,'cwd'),fs.reads)
        self.assertTrue(gate.assess_snapshot(s,now=101).allow_start)

    def test_desktop_basename_high_memory_or_other_exe_not_whitelisted(self):
        for name,memory in [('gnome-remote-desktop-daemon',120),(gate.DESKTOP_EXE,257),('/usr/lib/xorg/Xorg',100)]:
            fs=FakeFS();s=collect(fs,FakeCommands(f'GPU-1, 5, {name}, {memory}\n'))
            self.assertEqual(fs.opened,[])
            self.assertEqual(s['foreign_gpu_pids'],[5])
            self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_desktop_exe_spoof_fails_public_verification(self):
        fs=FakeFS();rec=record(5);rec['argv']=[gate.DESKTOP_EXE];fs.procs[5]=proc_data(rec)
        s=collect(fs,FakeCommands('GPU-1, 5, '+gate.DESKTOP_EXE+', 120\n'))
        self.assertEqual(s['desktop_gpu_pids'],[]);self.assertEqual(s['foreign_gpu_pids'],[5])
        self.assertNotIn((5,'environ'),fs.reads)

    def test_nvidia_failure_or_bad_csv_fails_closed(self):
        for command in (FakeCommands(gpu=hs.CommandResult(1,'')),FakeCommands('bad,row\n'),FakeCommands(gpu='GPU-1, N/A, 2\n')):
            s=collect(commands=command)
            self.assertFalse(s['query_ok']);self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)

    def test_registry_missing_untrusted_or_digest_mismatch_blocks(self):
        fs=FakeFS();del fs.files[REGISTRY]
        self.assertFalse(collect(fs)['worker_inventory_complete'])
        fs=FakeFS();fs.denied.add(REGISTRY)
        self.assertFalse(collect(fs)['worker_inventory_complete'])
        fs=FakeFS([record()]);r=json.loads(fs.files[REGISTRY]);r['records'][0]['record_sha256']='0'*64;fs.files[REGISTRY]=json.dumps(r).encode()
        s=collect(fs);self.assertFalse(s['worker_inventory_complete']);self.assertEqual(fs.opened,[])

    def test_record_path_outside_round_root_is_not_read(self):
        fs=FakeFS([record()]);r=json.loads(fs.files[REGISTRY]);r['records'][0]['record_path']='/other-project/private.json';fs.files[REGISTRY]=json.dumps(r).encode()
        s=collect(fs);self.assertFalse(s['worker_inventory_complete'])
        self.assertNotIn('/other-project/private.json',fs.secure_reads)

    def test_registry_mutating_during_snapshot_blocks(self):
        fs=FakeFS();first=fs.files[REGISTRY];later=json.loads(first);later['new_field']='changed'
        fs.files[REGISTRY]=[first,json.dumps(later).encode()]
        s=collect(fs)
        self.assertFalse(s['worker_inventory_complete'])

    def test_two_collections_admit_only_with_caller_lock_and_stable_registry(self):
        before=collect(at=100,name='a');after=collect(at=101,name='b')
        self.assertTrue(hs.assess_collected_launch_boundary(before,after,project_lock_held=True,now=102).allow_start)
        self.assertFalse(hs.assess_collected_launch_boundary(before,after,project_lock_held=False,now=102).allow_start)
        after['registry_sha256']='changed'
        self.assertFalse(hs.assess_collected_launch_boundary(before,after,project_lock_held=True,now=102).allow_start)


if __name__=='__main__':
    unittest.main(verbosity=2)
