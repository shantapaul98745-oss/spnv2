import copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE));import streaming_assets_v1 as stream

def raw(o):return json.dumps(o,sort_keys=True).encode()
def put(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw(o));return dict(path=str(p),sha256=stream.digest(p))
class StreamingTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.m=self.root/'migration';self.stage=self.root/'stage';self.target=self.root/'target'
  self.m.mkdir();self.stage.mkdir();self.target.mkdir();self.rel='spnv2_runs/plans/frozen/state.bin';self.source=self.stage/self.rel
  self.source.parent.mkdir(parents=True);self.source.write_bytes(b'unchanged scientific state bytes')
  self.row=dict(relative_path=self.rel,bytes=self.source.stat().st_size,sha256=stream.digest(self.source))
 def tearDown(self):self.t.cleanup()
 def receipt(self):
  name=hashlib.sha256(self.rel.encode()).hexdigest()+'.json'
  return put(self.m/'asset_receipts'/name,dict(status='verified',relative_path=self.rel,sha256=self.row['sha256'],bytes=self.row['bytes'],file_stat=stream.signature(self.source)))
 def test_unverified_file_is_not_published(self):
  result=stream.publish_verified(dict(files=[self.row]),self.m,self.stage,self.target)
  self.assertEqual(result['missing'],[self.rel]);self.assertFalse((self.target/self.rel).exists())
 def test_verified_publish_preserves_stage_and_bytes_without_second_copy(self):
  self.receipt();before=stream.signature(self.source)
  result=stream.publish_verified(dict(files=[self.row]),self.m,self.stage,self.target)
  self.assertEqual(result['published'],[self.rel]);self.assertTrue(self.source.exists());self.assertTrue(os.path.samefile(self.source,self.target/self.rel));self.assertEqual(before,stream.signature(self.source))
  self.assertEqual(stream.digest(self.target/self.rel),self.row['sha256'])
 def test_already_published_idempotent(self):
  self.receipt();manifest=dict(files=[self.row]);stream.publish_verified(manifest,self.m,self.stage,self.target)
  self.assertEqual(stream.publish_verified(manifest,self.m,self.stage,self.target)['published'],[self.rel])
 def test_existing_different_copy_not_adopted_or_overwritten(self):
  self.receipt();dest=self.target/self.rel;dest.parent.mkdir(parents=True);dest.write_bytes(self.source.read_bytes())
  with self.assertRaisesRegex(ValueError,'Refusing different'):stream.publish_verified(dict(files=[self.row]),self.m,self.stage,self.target)
  self.assertFalse(os.path.samefile(self.source,dest))
 def test_receipt_changed_hash_rejected(self):
  self.receipt();wrong=dict(self.row,sha256='f'*64)
  with self.assertRaisesRegex(ValueError,'binding differs'):stream.checked_receipt(self.m,wrong,staging_root=self.stage)
 def test_actual_file_changed_rejected(self):
  self.receipt();self.source.write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'stat differs'):stream.checked_receipt(self.m,self.row,staging_root=self.stage)
 def test_escape_path_rejected(self):
  for value in ('/etc/passwd','spnv2_runs/../other/state.bin','unowned/a/b'):
   with self.assertRaises(ValueError):stream.relative(value)
 def test_verify_uses_original_saved_main_identity_and_summary_not_archive(self):
  original=put(self.root/'original_task.json',dict(run_id='fixed_window',action='run_pair'))
  plan=dict(common_relative_paths=['common'],summary_relative_paths=[],windows=dict(fixed_window=dict(run_pair=['parent'],verify_pair=['match','saved_main'])))
  verify=dict(action='verify_pair',run_id='fixed_window_verify_newrevision',run_task=original)
  self.assertEqual(stream.required_paths(plan,verify),['common','match','saved_main'])
  self.assertEqual(stream.required_paths(plan,dict(action='summarize_diagnostic',run_id='summary')),['common'])
  with self.assertRaisesRegex(ValueError,'outside'):stream.required_paths(plan,dict(action='run_pair',run_id='different'))
 def make_plan(self):
  self.m=self.target/'spnv2_runs/migrations/sysu_20261027';self.m.mkdir(parents=True)
  manifest=put(self.root/'manifest.json',dict(files=[self.row]))
  plan=dict(schema=stream.SCHEMA,logical_target_root=str(self.target),asset_manifest=manifest,frozen_order=['window'],common_relative_paths=[],summary_relative_paths=[],windows=dict(window=dict(run_pair=[self.rel],verify_pair=[self.rel])))
  plan_binding=put(self.root/'plan.json',plan)
  index=dict(tasks=[dict(run_id='window')],streaming_migration=dict(code=dict(path=stream.__file__,sha256=stream.digest(stream.__file__)),dependencies=plan_binding))
  return plan,index
 def test_valid_staging_receipt_does_not_admit_missing_canonical_file(self):
  plan,index=self.make_plan();self.receipt();task=dict(action='run_pair',run_id='window')
  self.assertFalse(stream.readiness(index,task)['ready'])
  stream.publish_verified(dict(files=[self.row]),self.m,self.stage,self.target)
  self.assertTrue(stream.readiness(index,task)['ready'])
 def test_dependency_plan_cannot_change_frozen_order(self):
  plan,index=self.make_plan();plan['frozen_order']=['another_window']
  index['streaming_migration']['dependencies']=put(self.root/'plan.json',plan)
  with self.assertRaisesRegex(ValueError,'frozen order'):stream.load_plan(index)
 def test_canonical_parent_symlink_rejected(self):
  self.receipt();(self.target/'spnv2_runs').symlink_to(self.stage/'spnv2_runs',target_is_directory=True)
  with self.assertRaisesRegex(ValueError,'alias'):stream.publish_verified(dict(files=[self.row]),self.m,self.stage,self.target)
if __name__=='__main__':unittest.main()

