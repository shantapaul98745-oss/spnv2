"""Narrow resource bridge contracts against immutable real baseline metadata; no model IO."""
import copy,hashlib,importlib.util,json,pathlib,tempfile,unittest
R=pathlib.Path(__file__).resolve().parent
U=pathlib.Path('/home/ps/Downloads/UsrSourceCode/LuXinghai')
D=U/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
J=pathlib.Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
spec=importlib.util.spec_from_file_location('resource_bridge_under_test',R/'resource_amendment.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,sort_keys=True))
class ResourceBridgeContracts(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='shared-resource-contract-',dir='/tmp/spnv2_sysu_20261026');self.root=pathlib.Path(self.tmp.name)
  self.resolve=lambda value:self.root/pathlib.Path(value).relative_to(J)
  def cp(source,logical):
   target=self.resolve(logical);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(pathlib.Path(source).read_bytes());return dict(path=str(logical),sha256=sha(target))
  baseline=D/'revisions/shared_gpu_20261027/before/index.json'
  self.before=json.loads(baseline.read_text());self.index=copy.deepcopy(self.before)
  before=cp(baseline,J/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/revisions/shared_gpu_20261027/before/index.json')
  hostold=cp(D/'revisions/shared_gpu_20261027/before/implementation/host_entry.py',J/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/revisions/shared_gpu_20261027/before/implementation/host_entry.py')
  sources={name:cp(R/name,J/'spnv2/reproduction/shared_gpu_20261027'/name) for name in ['resource_amendment.py','host_gate.py','host_snapshot.py','host_entry.py','manager_sysu.py']}
  cp(R/'host_entry.py',J/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/implementation/host_entry.py');cp(R/'manager_sysu.py',J/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/implementation/manager.py')
  policy=cp(D/'revisions/shared_gpu_20261027/resource_policy.json',J/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/revisions/shared_gpu_20261027/resource_policy.json')
  self.index['implementation_sha256']['host_entry.py']=sources['host_entry.py']['sha256'];self.index['implementation_sha256']['manager.py']=sources['manager_sysu.py']['sha256']
  self.index['host_modules']={n:sources[n+'.py'] for n in ['host_gate','host_snapshot']};self.index['resource_policy']=policy
  self.rev=dict(schema=bridge.SCHEMA,authorization=bridge.AUTH,resource_admission_only=True,validator=sources['resource_amendment.py'],before_index=before,
   before_implementation_sha256=self.before['implementation_sha256'],after_implementation_sha256=self.index['implementation_sha256'],before_host_modules=self.before['host_modules'],after_host_modules=self.index['host_modules'],resource_policy=policy,before_host_entry=hostold,after_host_entry_source=sources['host_entry.py'])
  self.revpath=J/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/revisions/shared_gpu_20261027/resource_revision.json';self.persist()
 def persist(self):
  target=self.resolve(self.revpath);write(target,self.rev);self.index['resource_revision']=dict(path=str(self.revpath),sha256=sha(target))
 def tearDown(self):self.tmp.cleanup()
 def test_exact_resource_only_bridge_accepts_original_hosts_and_old_w16_code(self):
  self.assertEqual(bridge.load(self.index,self.resolve)['schema'],bridge.SCHEMA)
  self.assertTrue(bridge.host_modules_compatible(self.index,self.before,self.resolve))
  self.assertTrue(bridge.code_compatible(self.index,self.index['implementation_sha256'],self.before['implementation_sha256'],self.resolve))
  self.assertTrue(bridge.code_compatible(self.index,self.before['implementation_sha256'],self.before['implementation_sha256'],self.resolve))
 def test_noncontrol_scientific_change_rejected_even_with_rebound_manifest(self):
  self.index['implementation_sha256']['pair_runner.py']='0'*64;self.persist()
  with self.assertRaisesRegex(ValueError,'changed scientific code'):bridge.load(self.index,self.resolve)
 def test_unrelated_historical_host_map_is_not_waived(self):
  wrong=copy.deepcopy(self.before);wrong['host_modules']['host_gate']['sha256']='0'*64
  self.assertFalse(bridge.host_modules_compatible(self.index,wrong,self.resolve))
 def test_budget_or_window_order_change_rejected(self):
  self.index['tasks'][0],self.index['tasks'][1]=self.index['tasks'][1],self.index['tasks'][0]
  with self.assertRaisesRegex(ValueError,'frozen order'):bridge.load(self.index,self.resolve)
 def test_old_entry_archive_tamper_rejected(self):
  self.resolve(self.rev['before_host_entry']['path']).write_text('changed')
  with self.assertRaisesRegex(ValueError,'archive changed'):bridge.load(self.index,self.resolve)
if __name__=='__main__':unittest.main()
