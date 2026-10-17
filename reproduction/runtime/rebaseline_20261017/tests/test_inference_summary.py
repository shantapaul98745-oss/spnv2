import argparse,json,sys,tempfile,unittest,warnings
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset,DataLoader
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--report',required=True);args=p.parse_args()
sys.path.insert(0,str(Path(args.source)/'core'))
from config import cfg as defaults
from engine.inference import do_valid
from utils.training_state import sha256_file
torch.set_num_threads(2)
class Data(Dataset):
 def __len__(self):return 3
 def __getitem__(self,i):
  return torch.zeros(3,4,4),dict(domain='synthetic',imgpath=f'img{i:03d}',quaternion=torch.tensor([1.,0,0,0]),rotationmatrix=torch.eye(3),translation=torch.tensor([0.,0,5.]))
class Model(torch.nn.Module):
 def forward(self,*a,**kw):return [torch.zeros(1,11,4,4)]
def cfg():
 c=defaults.clone();c.defrost();c.TEST.HEAD=['heatmap'];c.VERBOSE=False;c.freeze();return c
class Tests(unittest.TestCase):
 def test_rejected_subset_missing_not_zero(self):
  events=[]
  with tempfile.TemporaryDirectory() as d, patch('engine.inference.solve_pose_from_heatmaps',return_value=(None,None,None,True)):
   s=do_valid(0,cfg(),Model(),DataLoader(Data(),batch_size=1),{},np.zeros((11,3)),log_dir=d,return_summary=True,event_callback=events.append)
   self.assertIsNone(s['score']);self.assertEqual(s['n_samples'],3);self.assertEqual(s['heatmap_rejections'],3)
   self.assertEqual(s['metrics']['final_pose']['count'],0)
   m=loadmat(str(next(Path(d).rglob('mean_performance.mat'))))
   self.assertTrue(np.isnan(m['final_pose']).all());self.assertEqual(int(m['final_pose_valid'][0,0]),0)
   self.assertEqual(events[-1]['n_samples'],3)
 def test_valid_and_rejected_denominators(self):
  pred=[(None,np.array([1.,0,0,0]),np.array([0.,0,5.]),False),(None,None,None,True),(None,np.array([1.,0,0,0]),np.array([0.,0,5.]),False)]
  with patch('engine.inference.solve_pose_from_heatmaps',side_effect=pred):
   s=do_valid(0,cfg(),Model(),DataLoader(Data(),batch_size=1),{},np.zeros((11,3)),return_summary=True)
   self.assertEqual(s['score'],0.);self.assertEqual(s['n_samples'],3);self.assertEqual(s['metrics']['final_pose']['count'],2);self.assertEqual(s['heatmap_rejections'],1)
 def test_fraction_and_empty_dataset(self):
  with patch('engine.inference.solve_pose_from_heatmaps',return_value=(None,None,None,True)):
   s=do_valid(0,cfg(),Model(),DataLoader(Data(),batch_size=1),{},np.zeros((11,3)),valid_fraction=.1,return_summary=True)
   self.assertEqual(s['n_samples'],1)
   with self.assertRaises(ValueError):do_valid(0,cfg(),Model(),DataLoader([],batch_size=1),{},np.zeros((11,3)))
suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
Path(args.report).write_text(json.dumps(dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=[dict(test=str(t),trace=s) for t,s in r.failures+r.errors],source=args.source,code_sha256=sha256_file(Path(args.source)/'core/engine/inference.py')),indent=2))
sys.exit(0 if r.wasSuccessful() else 1)
