import unittest,tempfile,json
from pathlib import Path
from durable_state import DurableRunStore
from storage_policy import StoragePolicyError,pin,after_publish
I=dict(run_id='storage-contract',source_sha256='a'*64,config_sha256='b'*64,manifest_sha256='c'*64,code_sha256='d'*64)
P=dict(schema='spnv2-storage-policy-v1',retain_latest=2,estimated_peak_bytes=1000000,protected_generations={})
class RetentionContracts(unittest.TestCase):
 def make(self,p):return DurableRunStore(p,I,state_validator=lambda _:None,retention_policy=dict(P,protected_generations={}))
 def save(self,s,n):return s.save({'value':n},scientific_cursors={'observed':n,'next_index':n,'candidates':n,'commits':n,'rollbacks':0})
 def test_roll_two_and_exact_recovery(self):
  with tempfile.TemporaryDirectory() as p:
   s=self.make(p)
   for n in range(1,6):self.save(s,n)
   self.assertEqual([n for n,_ in s._paths()],[5,4]);self.assertEqual(s.recover().state,{'value':5})
   self.assertEqual(json.loads((Path(p)/'storage_usage.json').read_text())['retained_generations'],[5,4])
 def test_frozen_pin_survives_roll(self):
  with tempfile.TemporaryDirectory() as p:
   s=self.make(p);self.save(s,1);pin(s,1,'frozen diagnostic parent')
   for n in range(2,6):self.save(s,n)
   self.assertEqual([n for n,_ in s._paths()],[5,4,1]);self.assertEqual(s.recover().state,{'value':5})
 def test_failed_publication_does_not_prune(self):
  with tempfile.TemporaryDirectory() as p:
   s=self.make(p);self.save(s,1);self.save(s,2)
   def crash(point):
    if point=='after_state_fsync':raise RuntimeError('injected disk interruption')
   s.failure_injector=crash
   with self.assertRaises(RuntimeError):self.save(s,3)
   self.assertEqual([n for n,_ in s._paths()],[2,1]);self.assertEqual(s.recover().state,{'value':2})
 def test_missing_policy_and_budget_fail_closed(self):
  with tempfile.TemporaryDirectory() as p:
   with self.assertRaises(StoragePolicyError):DurableRunStore(p,I,state_validator=lambda _:None)
  with tempfile.TemporaryDirectory() as p:
   s=DurableRunStore(p,I,state_validator=lambda _:None,retention_policy=dict(P,estimated_peak_bytes=1,protected_generations={}))
   with self.assertRaises(StoragePolicyError):self.save(s,1)
   self.assertEqual(s._paths(),[])
 def test_protected_missing_stops_pruning(self):
  with tempfile.TemporaryDirectory() as p:
   s=self.make(p);self.save(s,1);self.save(s,2)
   v=json.loads((Path(p)/'storage_policy.json').read_text());v['protected_generations']['generation-00000000000000000099']=dict(state_sha256='e'*64,reason='missing referenced parent');(Path(p)/'storage_policy.json').write_text(json.dumps(v))
   with s._lock():
    with self.assertRaises(StoragePolicyError):after_publish(s)
   self.assertEqual([n for n,_ in s._paths()],[2,1])
if __name__=='__main__':unittest.main()
