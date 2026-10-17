"""Meaningful queued observer revision invariants; no GPU or real queue mutation."""
import copy,json,os,sys,unittest,tempfile,subprocess
from unittest.mock import patch
from pathlib import Path
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'pipeline'))
import runner
from pending_observers import revise_queued_observers
from revision import canonical_sha256
from task_entry import process_identity,process_alive

class Tests(unittest.TestCase):
 def setUp(self):
  base={'id':'train','kind':'source_training_20_epochs','command':['python','train.py'],'status':'queued'}
  evaluation={'id':'eval','kind':'full_source_evaluation','command':['python','-B','-s','-u','/plan/tasks/source_artifacts.py','evaluate','--cfg','cfg'],
   'verify_command':['python','-B','-s','-u','/plan/tasks/source_artifacts.py','verify_eval','--cfg','cfg'],
   'env':{'CUDA_VISIBLE_DEVICES':'0'},'session':'unique_eval','log':'log','requires':['train'],
   'expected_outputs':['accepted.json'],'gpu':{'min_free_mb':14500},'metadata':{'final_only':True,'asset_sha256':{'/original':'a'*64}}}
  self.old={'version':1,'plan_id':'fixture','tasks':[base,evaluation],'scope':'partial','backlog':['future']}
  self.prev={'plan_id':'fixture','manifest_sha256':canonical_sha256(self.old),'tasks':copy.deepcopy(self.old['tasks']),
    'completed':[],'active':'train','current':{'task_id':'train','pid':123,'starttime':'456'},'status':'running','manager':{'pid':789}}
  self.prev['tasks'][0].update(status='running',runtime={'record':'same','tail_offset':100},started_at='same')
  self.prev['tasks'][1]['status']='queued'
  self.new=copy.deepcopy(self.old);self.new['tasks'].append({'id':'later','command':['python','later.py']});self.new['scope']='full'
  e=self.new['tasks'][1];w='/plan/extended_evaluation/capture_source.py'
  for key in ['command','verify_command']:
   cmd=e[key];e[key]=cmd[:4]+[w,'--base-entry',cmd[4]]+cmd[5:]
  e['metadata'].update(pending_observer_entry=w);e['metadata']['asset_sha256'][w]='b'*64
 def revise(self,ids=None):return revise_queued_observers(self.prev,self.old,self.new,task_ids=['eval'] if ids is None else ids)
 def test_preserves_running_identity_and_all_evidence(self):
  originals=copy.deepcopy((self.prev,self.old,self.new));a,r=self.revise()
  self.assertEqual(a['tasks'][0],self.prev['tasks'][0]);self.assertEqual(a['current'],self.prev['current'])
  self.assertEqual(a['manager'],self.prev['manager']);self.assertEqual(a['manifest_sha256'],canonical_sha256(self.new))
  self.assertEqual(r['pending_observer_revisions'][0]['old'],self.old['tasks'][1]);self.assertEqual((self.prev,self.old,self.new),originals)
  state=runner.initialize_state(a,self.new);self.assertEqual(state['tasks'][0]['runtime'],self.prev['tasks'][0]['runtime'])
  self.assertEqual(state['current']['pid'],123);self.assertEqual(state['tasks'][0]['status'],'launching');self.assertEqual(state['tasks'][1]['status'],'queued')
 def test_reject_started_or_selected_task(self):
  for field,value in [('status','running'),('status','awaiting_verification'),('runtime',{}),('history',[]),('started_at','past'),('error','failed')]:
   with self.subTest(field=field,value=value):
    p=copy.deepcopy(self.prev);p['tasks'][1][field]=value
    with self.assertRaises(ValueError):revise_queued_observers(p,self.old,self.new,task_ids=['eval'])
  self.prev['active']='eval'
  with self.assertRaises(ValueError):self.revise()
 def test_reject_other_execution_changes(self):
  for field,value in [('env',{'CUDA_VISIBLE_DEVICES':''}),('requires',[]),('session','new'),('expected_outputs',[]),('gpu',False),('log','new')]:
   with self.subTest(field=field):
    n=copy.deepcopy(self.new);n['tasks'][1][field]=value
    with self.assertRaises(ValueError):revise_queued_observers(self.prev,self.old,n,task_ids=['eval'])
 def test_reject_arguments_interpreter_or_base_entry_changes(self):
  for key,index,value in [('command',0,'otherpython'),('command',6,'/other/base.py'),('command',8,'--bad'),('verify_command',7,'evaluate')]:
   n=copy.deepcopy(self.new);n['tasks'][1][key][index]=value
   with self.assertRaises(ValueError):revise_queued_observers(self.prev,self.old,n,task_ids=['eval'])
 def test_reject_changed_old_asset_or_unbound_wrapper(self):
  for mapping in [{'/original':'c'*64},{'/original':'a'*64,'/plan/extended_evaluation/capture_source.py':'bad'}]:
   n=copy.deepcopy(self.new);n['tasks'][1]['metadata']['asset_sha256']=mapping
   with self.assertRaises(ValueError):revise_queued_observers(self.prev,self.old,n,task_ids=['eval'])
 def test_reject_original_metadata_and_kind_change(self):
  self.new['tasks'][1]['metadata']['final_only']=False
  with self.assertRaises(ValueError):self.revise()
  self.new['tasks'][1]['metadata']['final_only']=True;self.new['tasks'][1]['kind']='training'
  with self.assertRaises(ValueError):self.revise()
 def test_reject_undeclared_and_unused_declared_ids(self):
  for ids in [[],['train'],['eval','train'],['eval','eval']]:
   with self.subTest(ids=ids):
    with self.assertRaises(ValueError):self.revise(ids)
 def test_reject_stale_binding_reorder_and_no_append(self):
  p=copy.deepcopy(self.prev);p['manifest_sha256']='stale'
  with self.assertRaises(ValueError):revise_queued_observers(p,self.old,self.new,task_ids=['eval'])
  n=copy.deepcopy(self.new);n['tasks'][:2]=reversed(n['tasks'][:2])
  with self.assertRaises(ValueError):revise_queued_observers(self.prev,self.old,n,task_ids=['eval'])
  self.new['tasks'].pop()
  with self.assertRaises(ValueError):self.revise()
 def test_existing_asset_bindings_and_audit_deep_copy(self):
  a,r=self.revise();r['pending_observer_revisions'][0]['new']['command'][0]='mutated'
  self.assertEqual(self.new['tasks'][1]['command'][0],'python');self.assertEqual(a['tasks'][1]['metadata']['asset_sha256']['/original'],'a'*64)
 def test_requires_paired_verification_wrapper(self):
  self.new['tasks'][1]['verify_command']=self.old['tasks'][1]['verify_command']
  with self.assertRaises(ValueError):self.revise()
 def test_bound_revision_audits_and_reobserves_exact_live_child(self):
  with tempfile.TemporaryDirectory(prefix='pending_observer_live_',dir=P/'reports') as directory:
   work=Path(directory);release=work/'release'
   child=subprocess.Popen([sys.executable,'-B','-s','-c',
    "import pathlib,time,sys;p=pathlib.Path(sys.argv[1]);end=time.monotonic()+8;exec('while not p.exists() and time.monotonic()<end: time.sleep(0.02)')",str(release)],
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'))
   try:
    record=process_identity(child.pid);self.assertTrue(process_alive(record));record['screen_id']=os.environ['STY']
    record_path=work/'process.json';record_path.write_text(json.dumps(record))
    self.old['plan_id']=self.new['plan_id']=self.prev['plan_id']=work.name
    for manifest in [self.old,self.new]:
     for task in manifest['tasks']:
      task.update(cwd=str(P),session='pending_live_'+task['id'],log=str(work/(task['id']+'.log')),
                  expected_outputs=[str(work/(task['id']+'.json'))],gpu=False)
    self.prev['tasks'][0]=copy.deepcopy(self.old['tasks'][0]);self.prev['tasks'][0].update(status='running',started_at='past',
     runtime=dict(phase='main',record=str(record_path),result=str(work/'absent.json'),log=str(work/'empty.log'),tail_offset=0,observed_live=True))
    self.prev['tasks'][1]=copy.deepcopy(self.old['tasks'][1]);self.prev['tasks'][1]['status']='queued'
    self.prev['manifest_sha256']=canonical_sha256(self.old)
    class Reporter:
     write_failed=False
     def publish(self,force=False):return True
    with patch.object(runner,'PLAN',work):
     state=runner.bind_execution_state(self.prev,self.new,self.old,pending_task_ids=['eval'],caller_context={'cpu_fixture':True})
     audit=json.loads(Path(state['manifest_revisions'][-1]).read_text())
     self.assertEqual(audit['operation'],'queued_source_observer_revision_and_append')
     self.assertEqual(audit['pending_observer_revisions'][0]['id'],'eval')
     queue=runner.Queue(work/'manifest.json',self.new,state,Reporter());queue.events_path=work/'queue.events.jsonl'
     with patch.object(queue,'prepare_launch',side_effect=AssertionError('duplicate launch')), \
          patch.object(runner,'gpu_available',side_effect=AssertionError('GPU query')), \
          patch.object(runner.subprocess,'run',side_effect=AssertionError('process operation')):
      queue.observe(state['tasks'][0])
    self.assertEqual(state['tasks'][0]['status'],'running');self.assertEqual(state['current']['pid'],child.pid)
    self.assertEqual(state['current']['starttime'],record['starttime']);self.assertEqual(state['tasks'][1]['status'],'queued')
    self.assertIsNone(child.poll())
   finally:
    release.touch();self.assertEqual(child.wait(timeout=10),0)

if __name__=='__main__':
 assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
 path=P/'reports/pending_observers_cpu_20261017_b.json';path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps({'status':'passed' if r.wasSuccessful() else 'failed','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)},indent=2))
 raise SystemExit(0 if r.wasSuccessful() else 1)
