"""CPU full-plan negative cases; mutate only deep copies, never runtime configs."""
import copy,json,os,sys,unittest,hashlib
from pathlib import Path
import yaml
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from final_reporting.requirements import validate_scope,CONFIG_EPOCHS

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.q=json.loads((P.parent/'queue_proposals/v3b/full_main_v3_proposal.json').read_text())
  cls.odr=json.loads((P.parent/'odr/manifest.json').read_text())
  cls.configs={n:yaml.safe_load((P.parent/'configs'/(n+'.yaml')).read_text()) for n in CONFIG_EPOCHS}
 def check(self,q=None,odr=None,cfg=None):return validate_scope(q or self.q,odr or self.odr,cfg or self.configs)
 def test_complete_core_scope_is_not_completion(self):
  r=self.check();self.assertEqual(r['formal_epochs'],190);self.assertEqual(r['formal_configs'],10)
  self.assertEqual(r['expected_training_update_attempts'],749310);self.assertEqual(r['expected_training_sample_appearances'],9111120)
  self.assertEqual(r['status'],'full_scope_validated_not_experiment_completion')
 def test_reject_removed_source_and_duplicate_config(self):
  q=copy.deepcopy(self.q);q['tasks']=[t for t in q['tasks'] if t['id']!='train_table4_phi0_bn']
  with self.assertRaises(ValueError):self.check(q=q)
  q=copy.deepcopy(self.q);t=next(t for t in q['tasks'] if t['id']=='train_table4_phi0_bn');t['command'][-2:]=['--cfg',str(P.parent/'configs/table4_phi3_gn.yaml')]
  with self.assertRaises(ValueError):self.check(q=q)
 def test_reject_physical_batch_and_lr_changes(self):
  for key,value in [('IMAGES_PER_GPU',16),('ACCUM_STEPS',1),('LR',.004),('END_EPOCH',19)]:
   c=copy.deepcopy(self.configs);c['phi3_bn_ehs']['TRAIN'][key]=value
   with self.subTest(key=key):
    with self.assertRaises(ValueError):self.check(cfg=c)
 def test_reject_resume_or_seed(self):
  c=copy.deepcopy(self.configs);c['phi6_gn_ehs']['TRAIN']['RESUME_FILE']='pilot.pth'
  with self.assertRaises(ValueError):self.check(cfg=c)
  c=copy.deepcopy(self.configs);c['table2_e']['SEED']=2022
  with self.assertRaises(ValueError):self.check(cfg=c)
 def test_reject_fewer_epochs_even_if_label_agrees(self):
  c=copy.deepcopy(self.configs);c['table2_h']['TRAIN']['END_EPOCH']=9
  with self.assertRaises(ValueError):self.check(cfg=c)
 def test_reject_wrong_resolution(self):
  c=copy.deepcopy(self.configs);c['phi6_gn_ehs']['DATASET']['INPUT_SIZE']=[384,256]
  with self.assertRaises(ValueError):self.check(cfg=c)
 def test_reject_missing_or_duplicated_domain(self):
  q=copy.deepcopy(self.q);q['tasks']=[t for t in q['tasks'] if t['id']!='eval_phi3_sunlamp']
  with self.assertRaises(ValueError):self.check(q=q)
 def test_reject_old_source_scan(self):
  m=copy.deepcopy(self.odr);m['tasks'][0]['source_final']='/history/old.pth'
  with self.assertRaises(ValueError):self.check(odr=m)
 def test_reject_scan_grid_hole_and_seed_drift(self):
  for key,value in [('N',123),('seed',2026)]:
   m=copy.deepcopy(self.odr);next(t for t in m['tasks'] if t['group']=='Nscan')[key]=value
   with self.subTest(key=key):
    with self.assertRaises(ValueError):self.check(odr=m)
 def test_reject_cross_protocol_alias_and_gpu_alias(self):
  m=copy.deepcopy(self.odr);next(t for t in m['tasks'] if t.get('reuse_of'))['B']=32
  with self.assertRaises(ValueError):self.check(odr=m)
  q=copy.deepcopy(self.q);alias=next(t for t in m['tasks'] if t.get('reuse_of'));next(t for t in q['tasks'] if t['id']==alias['id'])['gpu']={'index':0,'min_free_mb':14500,'max_used_mb':1200}
  with self.assertRaises(ValueError):self.check(q=q)
 def test_reject_batch16_or_extra_gpu(self):
  for token in ['true_batch16','batch_16','batch16']:
   q=copy.deepcopy(self.q);q['tasks'][0]['command'].append(token)
   with self.assertRaises(ValueError):self.check(q=q)
  q=copy.deepcopy(self.q);q['tasks'][0]['gpu']['index']=1
  with self.assertRaises(ValueError):self.check(q=q)
 def test_reporting_requirements_cannot_be_omitted(self):
  with self.assertRaises(ValueError):validate_scope(self.q,self.odr,self.configs,require_reporting=True)
 def test_reject_deleted_fixed_source_and_scan_barriers(self):
  for ident in ['pilot_table2_h','fixed_phi3_A_lightbox_s2021_N1024_B4','Bscan_phi3_A_lightbox_s2021_N1024_B1']:
   q=copy.deepcopy(self.q);next(t for t in q['tasks'] if t['id']==ident)['requires']=[]
   with self.subTest(ident=ident):
    with self.assertRaises(ValueError):self.check(q=q)
 def test_reject_verify_config_or_count_drift(self):
  for field,value in [('--cfg','wrong.yaml'),('--expected-samples','1')]:
   q=copy.deepcopy(self.q);t=next(t for t in q['tasks'] if t['id']=='eval_phi3_synthetic');c=t['verify_command'];c[c.index(field)+1]=value
   with self.assertRaises(ValueError):self.check(q=q)
 def test_reject_shortened_pilot_or_wrong_report(self):
  for field,value in [('--attempts','1'),('--checkpoint','1'),('--cfg','wrong.yaml')]:
   q=copy.deepcopy(self.q);t=next(t for t in q['tasks'] if t['id']=='pilot_phi6_EHS');c=t['command'];c[c.index(field)+1]=value
   with self.assertRaises(ValueError):self.check(q=q)

if __name__=='__main__':
 assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 output=P/'reports/requirements_cpu_20261017_c.json';output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=[{'test':str(t),'trace':s} for t,s in r.errors+r.failures],
  scope='Full-plan static/negative CPU checks only; does not assert real experiments completed',code_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),P/'requirements.py']}),indent=2))
 raise SystemExit(0 if r.wasSuccessful() else 1)
