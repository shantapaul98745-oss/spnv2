"""MOCK-only fixture helpers; execute actual frozen inference on tiny CPU tensors."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import csv,copy,hashlib,json,math,os,sys
import numpy as np
import torch,yaml
import entry_contract as contract
P=contract.P

def toy(root,heads=('heatmap','efficientpose'),count=3,all_numeric=False,zero_H=False):
 root=Path(root);root.mkdir(parents=True);csvpath=root/'data/speedplus/lightbox/labels/test.csv';csvpath.parent.mkdir(parents=True)
 ids=['MOCK_image_%03d.jpg'%i for i in range(count)]
 with csvpath.open('w') as f:csv.writer(f).writerows([[id,0,0,1,1,1,0,0,0,.1,.1,10] for id in ids])
 class Tiny(torch.nn.Module):
  def __init__(self):
   super().__init__();self.weight=torch.nn.Parameter(torch.tensor([2.]));self.register_buffer('counter',torch.tensor([7]));self.calls=0
  def forward(self,images,**kwargs):
   self.calls+=1;i=int(images[0,0,0,0]);values=[]
   for head in heads:
    if head=='heatmap':
     h=torch.zeros((1,11,12,16))
     if not zero_H and (i!=2 or all_numeric):h[:,:,5,6]=1
     values.append(h)
    elif head=='efficientpose':values.append((torch.tensor([[1.]]),torch.tensor([[[0.,0.,10.,10.]]]),torch.tensor([[[1.,0.,0.,0.,1.,0.]]]),torch.tensor([[[.2,.1,10.]]])))
   return values
 final=root/'MOCK_source_final.pth.tar';torch.save(Tiny().state_dict(),final);source_report=root/'MOCK_source_training_acceptance.json';source_report.write_text(json.dumps(dict(status='passed',final=str(final),final_sha256=contract.sha(final),cfg='MOCK source training',mock_fixture=True)))
 cfgdict=dict(DATASET=dict(ROOT=str(root/'data'),DATANAME='speedplus',IMAGE_SIZE=[64,48],NUM_KEYPOINTS=11),MODEL=dict(PRETRAIN_FILE=str(final)),TEST=dict(MODEL_FILE=str(final),TEST_CSV='lightbox/labels/test.csv',HEAD=list(heads),IMAGES_PER_GPU=1,HEATMAP_THRESHOLD=.5,SPEED_THRESHOLD_Q=.169,SPEED_THRESHOLD_T=.002173),TRAIN=dict(REPRODUCIBLE=False,VAL_CSV='lightbox/labels/test.csv'),SEED=2021,VERBOSE=False)
 cfgpath=root/'MOCK_cfg.yaml';cfgpath.write_text(yaml.safe_dump(cfgdict));cfg=NS(DATASET=NS(**cfgdict['DATASET']),MODEL=NS(**cfgdict['MODEL']),TEST=NS(**cfgdict['TEST']),TRAIN=NS(**cfgdict['TRAIN']),SEED=2021,VERBOSE=False)
 qgt=np.array([1,0,0,0],np.float32);tgt=np.array([.1,.1,10],np.float32);batches=[]
 for i,id in enumerate(ids):batches.append((torch.full((1,3,48,64),float(i)),dict(domain=['lightbox'],imgpath=[id],quaternion=torch.from_numpy(qgt[None].copy()),rotationmatrix=torch.eye(3).unsqueeze(0),translation=torch.from_numpy(tgt[None].copy()),boundingbox=torch.tensor([[0.,0.,10.,10.]]))))
 camera=dict(cameraMatrix=np.array([[50.,0,32.],[0,50.,24.],[0,0,1.]]),distCoeffs=np.zeros(5),horizontalFOV=1.2,Nu=64,Nv=48)
 keypts=np.array([[-1,-1,0],[1,-1,0],[-1,1,0],[1,1,0],[0,0,1],[.2,.5,-1],[-.4,.2,.3],[.6,-.2,.2],[-.6,-.8,1],[.1,.8,-.4],[.4,.6,.8]])
 return NS(root=root,ids=ids,csv=csvpath,cfg=cfg,cfgpath=cfgpath,final=final,source_report=source_report,Tiny=Tiny,batches=batches,camera=camera,keypts=keypts,models=[],solver_calls=[])

def forward_fixture(f,output,inference):
 import importlib
 post=importlib.import_module('utils.postprocess');model=f.Tiny();model.load_state_dict(torch.load(f.final,map_location='cpu',weights_only=True),strict=True);before={k:v.clone() for k,v in model.state_dict().items()};calls=[]
 def pnp(*a,**kw):
  i=len(calls);calls.append(i);return np.array([math.cos(.1),math.sin(.1),0.,0.]),np.array([.2,.1,10. if i==0 else 60.])
 with patch.object(post,'pnp',pnp):summary=inference.do_valid(0,f.cfg,model,f.batches,f.camera,f.keypts,log_dir=str(output),return_summary=True)
 assert model.calls==len(f.ids) and all(torch.equal(v,before[k]) for k,v in model.state_dict().items());f.models.append(model);f.solver_calls.append(len(calls));return summary
