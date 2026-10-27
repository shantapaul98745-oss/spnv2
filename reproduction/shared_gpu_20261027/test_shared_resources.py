"""Targeted zero-model contracts for explicitly shared capacity admission."""
from __future__ import annotations
from contextlib import contextmanager
import copy, hashlib, importlib.util, json, pathlib, posixpath, sys, tempfile, time, unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import host_gate as gate
import host_snapshot as hs
ROOT='/project/diagnostic';REGISTRY=ROOT+'/registry.json';PROJECT='g50p_risk_diagnostic_20261025'

def policy():
 return dict(schema='spnv2-shared-gpu-resource-policy-v1',authorization='user_2026-10-27_shared_gpu_capacity_admission',project_id=PROJECT,
  resource_admission_only=True,allow_foreign_gpu_processes=True,ignore_gpu_utilization=True,
  gpu_peak_estimate_mib=4096,gpu_safety_margin_mib=2048,min_free_gpu_mib=6144,
  host_memory_peak_estimate_mib=32768,host_memory_safety_margin_mib=8192,min_host_mem_available_mib=40960,
  snapshot_max_age_seconds=60,launch_snapshot_max_age_seconds=5,resource_retry_seconds=300)

def record(pid=123):
 argv=['/opt/python','host_entry.py','--task','one.json']
 return dict(schema=gate.LAUNCH_SCHEMA,project_id=PROJECT,run_id='saved_w16_verify',role='worker',run_nonce='0123456789abcdef0123456789abcdef',
  boot_id='boot-A',pid=pid,starttime=900,uid=1000,argv=argv,cmdline_sha256=gate.command_sha256(argv),cwd=ROOT,session='unique',log_path=ROOT+'/run.log',launched_at=50.)
def process(rec=None):
 rec=rec or record();p={key:copy.deepcopy(rec[key]) for key in gate.IDENTITY_FIELDS};p.update(argv=list(rec['argv']),identity_complete=True,state='S',role='experiment_worker');return p

def snapshot(name='a',at=100.,free=7000,available=65536):
 return dict(snapshot_id=name,captured_at=at,collection_finished_at=at,boot_id='boot-A',gpu_uuid='GPU-A',gpu_index=0,free_mib=free,used_mib=40000,
  query_ok=True,inventory_complete=True,gpu_processes=[dict(pid=991,process_name='/other/user/python',memory_mib=30000,boot_id='boot-A',identity_complete=False,role='foreign',utilization=100)],
  worker_inventory_complete=True,registered_run_nonces=[],worker_checks=[],collection_errors=[],foreign_gpu_pids=[991],
  host_mem_available_mib=available,host_mem_total_mib=131072,host_memory_query_ok=True,host_memory_sampled_at=at,resource_policy_sha256=gate.RESOURCE_POLICY_BINDING['sha256'])

class PublicFS:
 def __init__(self,mem=None):self.reads=[];self.opened=[];self.mem=mem if mem is not None else b'MemTotal: 134217728 kB\nMemAvailable: 67108864 kB\n'
 def realpath(self,path):return posixpath.normpath(str(path))
 def read_bytes(self,path):
  self.reads.append(path)
  if path==hs.BOOT_ID_PATH:return b'boot-A\n'
  if path==hs.MEMINFO_PATH:
   if isinstance(self.mem,Exception):raise self.mem
   return self.mem
  raise AssertionError('nonpublic read')
 def read_owned_bytes(self,path,uid):
  self.reads.append(path);assert path==REGISTRY and uid==1000
  return json.dumps(dict(schema=hs.REGISTRY_SCHEMA,project_id=PROJECT,records=[])).encode()
 @contextmanager
 def open_proc(self,pid):
  self.opened.append(pid);raise AssertionError('Foreign process must never be opened')
  yield
class Commands:
 def __init__(self):self.calls=[]
 def __call__(self,argv):
  self.calls.append(argv)
  return hs.CommandResult(0,'GPU-A, 7000, 40000\n' if any(x.startswith('--query-gpu=') for x in argv) else 'GPU-A, 991, /other/user/python, 30000\n')

class SharedCapacityTests(unittest.TestCase):
 def setUp(self):
  gate.RESOURCE_POLICY=None;gate.RESOURCE_POLICY_BINDING=None
  self.tmp=tempfile.TemporaryDirectory(prefix='shared-gpu-contract-',dir='/tmp/spnv2_sysu_20261026');self.path=pathlib.Path(self.tmp.name)/'policy.json';self.path.write_text(json.dumps(policy()))
  self.binding=dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest());gate.configure_policy(self.binding)
 def tearDown(self):self.tmp.cleanup();gate.RESOURCE_POLICY=None;gate.RESOURCE_POLICY_BINDING=None
 def test_foreign_presence_nonzero_utilization_and_large_used_memory_allow_capacity(self):
  for kind in ('gpu','heavy_cpu'):
   self.assertTrue(gate.assess_snapshot(snapshot(),now=101,task_kind=kind).allow_start)
 def test_explicit_capacity_thresholds_are_inclusive(self):
  self.assertTrue(gate.assess_snapshot(snapshot(free=6144,available=40960),now=101).allow_start)
  self.assertIn('insufficient_free_GPU_memory',gate.assess_snapshot(snapshot(free=6143),now=101).reasons)
  self.assertIn('insufficient_available_host_memory',gate.assess_snapshot(snapshot(available=40959),now=101).reasons)
 def test_both_fresh_samples_must_have_capacity_and_original_lock(self):
  before=snapshot('first',100);after=snapshot('second',101)
  self.assertTrue(gate.assess_launch_boundary(before,after,now=102,project_lock_held=True).allow_start)
  for sample,key,val in ((before,'free_mib',6143),(after,'host_mem_available_mib',40959)):
   old=sample[key];sample[key]=val;self.assertFalse(gate.assess_launch_boundary(before,after,now=102,project_lock_held=True).allow_start);sample[key]=old
  self.assertFalse(gate.assess_launch_boundary(before,after,now=102,project_lock_held=False).allow_start)
 def test_stale_incomplete_or_policy_mismatched_snapshots_reject(self):
  for key,val in [('captured_at',0),('query_ok',False),('inventory_complete',False),('worker_inventory_complete',False),('host_memory_query_ok',False),('host_memory_sampled_at',99),('resource_policy_sha256','0'*64)]:
   s=snapshot();s[key]=val;self.assertFalse(gate.assess_snapshot(s,now=101).allow_start,key)
  self.assertFalse(gate.assess_launch_boundary(snapshot('a',100),snapshot('b',101),now=107,project_lock_held=True).allow_start)
 def test_same_snapshot_or_changed_registry_boot_cannot_admit(self):
  a=snapshot('same',100);b=snapshot('same',101);self.assertFalse(gate.assess_launch_boundary(a,b,now=102,project_lock_held=True).allow_start)
  a['snapshot_id']='a';b['snapshot_id']='b';b['boot_id']='other';self.assertFalse(gate.assess_launch_boundary(a,b,now=102,project_lock_held=True).allow_start)
 def test_exact_owned_worker_still_blocks_duplicate_and_ambiguity_is_not_foreign(self):
  rec=record();s=snapshot();s['registered_run_nonces']=[rec['run_nonce']];s['worker_checks']=[dict(expected_record=rec,observed_record=copy.deepcopy(rec),probe_succeeded=True,process=process(rec),absence_confirmed=False)]
  self.assertIn('existing_owned_worker_blocks_duplicate:123',gate.assess_snapshot(s,now=101).reasons)
  s['worker_checks'][0]['process']['run_nonce']='f'*32
  self.assertTrue(any('identity_ambiguous' in r for r in gate.assess_snapshot(s,now=101).reasons))
 def test_missing_or_tampered_policy_cannot_fallback_to_unbound_rules(self):
  self.path.write_text('{}')
  with self.assertRaises(ValueError):gate.configure_policy(self.binding)
  gate.RESOURCE_POLICY=None;gate.RESOURCE_POLICY_BINDING=None
  s=dict(snapshot_id='x')
  self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)
  p=policy();p['min_free_gpu_mib']-=1;self.path.write_text(json.dumps(p));b=dict(path=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())
  with self.assertRaises(ValueError):gate.configure_policy(b)
 def test_collector_reads_public_meminfo_and_not_foreign_private_files(self):
  fs=PublicFS();commands=Commands();s=hs.collect_snapshot(REGISTRY,project_root=ROOT,project_id=PROJECT,fs=fs,command_runner=commands,clock=lambda:100.,snapshot_id_factory=lambda:'a',current_uid=1000)
  self.assertEqual(s['host_mem_available_mib'],65536);self.assertEqual(s['host_mem_total_mib'],131072);self.assertTrue(s['host_memory_query_ok']);self.assertEqual(s['foreign_gpu_pids'],[991]);self.assertEqual(fs.opened,[]);self.assertFalse(s['foreign_private_files_read']);self.assertEqual(len(commands.calls),2);self.assertTrue(gate.assess_snapshot(s,now=101).allow_start)
 def test_collector_missing_memavailable_fails_closed(self):
  for mem in (b'MemTotal: 134217728 kB\n',b'MemTotal: 100 kB\nMemAvailable: 101 kB\n',PermissionError('meminfo')):
   fs=PublicFS(mem);s=hs.collect_snapshot(REGISTRY,project_root=ROOT,project_id=PROJECT,fs=fs,command_runner=Commands(),clock=lambda:100.,snapshot_id_factory=lambda:'a',current_uid=1000)
   self.assertFalse(s['host_memory_query_ok']);self.assertFalse(gate.assess_snapshot(s,now=101).allow_start)
 def test_postregistration_conflict_allows_foreign_but_still_requires_exact_self(self):
  path=pathlib.Path(__file__).parent/'host_entry.py';sys.path.insert(1,str(pathlib.Path('/home/ps/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025/implementation')))
  spec=importlib.util.spec_from_file_location('shared_host_entry',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  rec=record();s=snapshot();s['worker_checks']=[dict(expected_record=rec,observed_record=copy.deepcopy(rec),process=process(rec))]
  self.assertIsNone(module.conflict(s,rec,gate));s['free_mib']=0;s['host_mem_available_mib']=0;self.assertIsNone(module.conflict(s,rec,gate));s['worker_checks'][0]['process']['run_nonce']='bad'
  self.assertEqual(module.conflict(s,rec,gate),'own_worker_registration_unverified')

if __name__=='__main__':unittest.main()
