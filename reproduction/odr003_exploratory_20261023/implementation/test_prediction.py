"""Known-geometry and threshold boundary checks, no real target labels."""
import os
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
sys.path.insert(0,'/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017/odr/source_A/core')
from prediction import developer_metrics, require_finite_heads, rotation_valid
from utils.postprocess import quat2dcm,project_keypoints
from utils.metrics import speed_score

class Tests(unittest.TestCase):
    def test_nonfinite_raw_is_fatal(self):
        for x in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(FloatingPointError):
                require_finite_heads({'heatmap':torch.tensor([x])})
            with self.assertRaises(FloatingPointError):
                require_finite_heads({'efficientpose':(torch.ones(2),torch.tensor([x]))})
    def test_rotation_matches_original_metric_tolerance(self):
        self.assertTrue(rotation_valid(np.eye(3,dtype=np.float32)))
        x=np.eye(3,dtype=np.float32);x[0,0]+=2e-6
        self.assertFalse(rotation_valid(x))
    def test_hil_dtype_and_rejected_extension(self):
        cfg=NS(TEST=NS(SPEED_THRESHOLD_Q=.169,SPEED_THRESHOLD_T=.002173))
        q=np.array([1.,0.,0.,0.]); gt=np.array([0,0,10],dtype=np.float32)
        for displacement in (0.,.021729,.02173,.021731,.3):
            et=np.array([displacement,0,10],dtype=np.float32)
            pr=dict(h_available=True,e_available=True,heat_q=q.tolist(),heat_t=[0,0,10],
                    effi_R=np.eye(3,dtype=np.float32).tolist(),effi_t=et.tolist(),original_h_rejected=True)
            result=developer_metrics(pr,dict(quaternion=q.tolist(),translation=gt.tolist()),cfg)
            st,sr,_=speed_score(et,np.eye(3,dtype=np.float32),gt,quat2dcm(q.astype(np.float32)),representation='rotationmatrix',applyThreshold=True,theta_q=.169,theta_t=.002173)
            self.assertEqual(result['error'],float(st))
            self.assertTrue(result['valid']);self.assertFalse(result['original_valid']);self.assertIsNone(result['original_error'])
    def test_missing_h_is_never_replaced_by_e(self):
        cfg=NS(TEST=NS(SPEED_THRESHOLD_Q=.169,SPEED_THRESHOLD_T=.002173))
        pr=dict(h_available=False,e_available=True,effi_R=np.eye(3,dtype=np.float32).tolist(),effi_t=[0,0,10])
        r=developer_metrics(pr,dict(quaternion=[1,0,0,0],translation=[0,0,10]),cfg)
        self.assertFalse(r['valid']);self.assertIsNone(r['error'])
    def test_camera_quaternion_convention(self):
        angle=np.pi/2;q=np.array([np.cos(angle/2),0,0,np.sin(angle/2)])
        K=np.array([[100,0,10],[0,100,20],[0,0,1.]])
        xy=project_keypoints(q,np.array([0,0,10]),K,np.zeros(5),np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0.]])).T
        np.testing.assert_allclose(xy[:2],[[10,30],[0,20]],atol=1e-5)

if __name__=='__main__':
    assert os.environ.get('STY'),'Use guarded screen'
    unittest.main(verbosity=2)
