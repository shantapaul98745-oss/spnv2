import argparse,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from scipy.io import loadmat,savemat
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--report',required=True);a=p.parse_args()
plan=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(a.source)/'core'));sys.path.insert(0,str(plan/'tasks'))
from config import cfg as defaults
from engine.inference import do_valid
from verify_heads import verify
torch.set_num_threads(2)
class Data(Dataset):
 def __len__(self):return 3
 def __getitem__(self,i):return torch.zeros(3,4,4),dict(domain='synthetic',imgpath=f'img{i:03d}',quaternion=torch.tensor([1.,0,0,0]),rotationmatrix=torch.eye(3),translation=torch.tensor([0.,0,5.]),boundingbox=torch.tensor([0.,0,1.,1.]))
class Model(torch.nn.Module):
 def forward(self,*a,**kw):return [torch.zeros(1,11,4,4),(torch.tensor([[.9]]),torch.tensor([[[0.,0,1.,1.]]]),torch.tensor([[[1.,0,0,0,1.,0]]]),torch.tensor([[[0.,0,5.]]]))]
def config():
 c=defaults.clone();c.defrost();c.TEST.HEAD=['heatmap','efficientpose'];c.VERBOSE=False;c.freeze();return c
def generate(temp,all_reject=False):
 csv=Path(temp)/'ids.csv';csv.write_text('img000\nimg001\nimg002\n')
 good=(None,np.array([1.,0,0,0]),np.array([0.,0,5.]),False);bad=(None,None,None,True)
 with patch('engine.inference.solve_pose_from_heatmaps',side_effect=[bad,bad,bad] if all_reject else [good,bad,good]):
  summary=do_valid(0,config(),Model(),DataLoader(Data(),batch_size=1),{},np.zeros((11,3)),log_dir=temp,return_summary=True)
 return Path(temp)/'synthetic',csv,summary
class Tests(unittest.TestCase):
 def test_mixed_valid_and_rejected_branches(self):
  with tempfile.TemporaryDirectory() as temp:
   path,csv,summary=generate(temp);v=verify(path,csv,['heatmap','efficientpose'])
   self.assertEqual(v['n_samples'],3);self.assertEqual(v['n_reject'],1)
   self.assertEqual(v['metric_denominators']['final_pose'],2);self.assertEqual(v['metric_denominators']['heat_pose'],2);self.assertEqual(v['metric_denominators']['effi_pose'],3)
   self.assertEqual(v['final_pose'],summary['score'])
 def test_zero_heatmap_coverage_keeps_efficientpose(self):
  with tempfile.TemporaryDirectory() as temp:
   path,csv,summary=generate(temp,True);v=verify(path,csv,['heatmap','efficientpose'])
   self.assertIsNone(v['final_pose']);self.assertIsNone(v['heat_pose']);self.assertIsNotNone(v['effi_pose']);self.assertEqual(v['metric_denominators']['final_pose'],0);self.assertEqual(v['metric_denominators']['effi_pose'],3)
 def test_missing_physical_metric_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   path,csv,_=generate(temp);file=path/'per_sample_performance.mat';data=loadmat(file);del data['heat_eR'];savemat(file,{k:v for k,v in data.items() if not k.startswith('__')})
   with self.assertRaises(AssertionError):verify(path,csv,['heatmap','efficientpose'])
 def test_corrupt_valid_metric_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   path,csv,_=generate(temp);file=path/'per_sample_performance.mat';data=loadmat(file);data['effi_eT'][0,0]=np.nan;savemat(file,{k:v for k,v in data.items() if not k.startswith('__')})
   with self.assertRaises(AssertionError):verify(path,csv,['heatmap','efficientpose'])
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
Path(a.report).write_text(json.dumps(dict(status='passed' if r.wasSuccessful() else 'failed',tests=r.testsRun,failures=[dict(test=str(t),trace=s) for t,s in r.failures+r.errors]),indent=2))
sys.exit(0 if r.wasSuccessful() else 1)
