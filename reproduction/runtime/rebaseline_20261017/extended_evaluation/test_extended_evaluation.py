"""CPU-only invariance, real postprocess, baseline MAT reconciliation and corruption tests."""
import argparse
import contextlib
import copy
import csv
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.io import loadmat, savemat
import torch
import yaml

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
sys.path.insert(0, str(PLAN))
sys.path.insert(0, str(PLAN / "source/core"))
from extended_evaluation.capture import observe_heatmap_poses, CaptureError, sha, digest, implementation, atomic
from extended_evaluation.report import build_extension
from extended_evaluation.verify import independent_cpu_verify
from engine import inference
from utils import postprocess

ARTIFACTS = HERE / "test_artifacts"
BOUNDARY_CASES = [{'kind': 'H_PnP_float64', 'threshold': 0.002173, 't_gt': {'dtype': 'float32', 'hex_bytes': 'ce0877bd73cb153e559f2c40', 'values': [-0.06031113117933273, 0.14628390967845917, 2.6972248554229736]}, 't_pr': {'dtype': 'float64', 'hex_bytes': '9bb0fdc48edfabbf000000606eb9c23f000000a0ea930540', 'values': [-0.054439984843015675, 0.14628390967845917, 2.6972248554229736]}}, {'kind': 'E_float32', 'threshold': 0.002173, 't_gt': {'dtype': 'float32', 'hex_bytes': 'c18f793d50777d3efd3e1041', 'values': [0.0609281100332737, 0.24752545356750488, 9.01537799835205]}, 't_pr': {'dtype': 'float32', 'hex_bytes': 'c18f793d70c4883efd3e1041', 'values': [0.0609281100332737, 0.2671236991882324, 9.01537799835205]}}, {'kind': 'H_PnP_float64', 'threshold': 0.005, 't_gt': {'dtype': 'float32', 'hex_bytes': 'ce0877bd73cb153e559f2c40', 'values': [-0.06031113117933273, 0.14628390967845917, 2.6972248554229736]}, 't_pr': {'dtype': 'float64', 'hex_bytes': '2a5c8f8268f6a7bf000000606eb9c23f000000a0ea930540', 'values': [-0.046801820695400245, 0.14628390967845917, 2.6972248554229736]}}, {'kind': 'E_float32', 'threshold': 0.005, 't_gt': {'dtype': 'float32', 'hex_bytes': 'ce0877bd73cb153e559f2c40', 'values': [-0.06031113117933273, 0.14628390967845917, 2.6972248554229736]}, 't_pr': {'dtype': 'float32', 'hex_bytes': '44b33fbd73cb153e559f2c40', 'values': [-0.046801820397377014, 0.14628390967845917, 2.6972248554229736]}}]


def fixture(heads=("heatmap", "efficientpose"), count=3, all_numeric=False, domain="lightbox", zero_heatmaps=False, accepted_first=True, pose_case=None):
    root = Path(tempfile.mkdtemp(prefix="cpu_", dir=ARTIFACTS))
    csvpath = root / "data/speedplus" / domain / "labels/test.csv"
    csvpath.parent.mkdir(parents=True)
    ids = ["image_%03d.jpg" % i for i in range(count)]
    gt_translation=np.asarray(pose_case["t_gt_values"],np.float32) if pose_case else np.array([.1,.1,10],np.float32)
    with csvpath.open("w", newline="") as stream:
        csv.writer(stream).writerows([[ident, 0, 0, 1, 1, 1, 0, 0, 0]+gt_translation.tolist() for ident in ids])
    cfgdict = dict(DATASET=dict(ROOT=str(root/"data"), DATANAME="speedplus",
                                IMAGE_SIZE=[64, 48], NUM_KEYPOINTS=11),
                   TEST=dict(TEST_CSV=domain+"/labels/test.csv", HEAD=list(heads),
                             HEATMAP_THRESHOLD=.5, SPEED_THRESHOLD_Q=.169, SPEED_THRESHOLD_T=pose_case["threshold"] if pose_case else .002173),
                   VERBOSE=False)
    cfgpath = root/"cfg.yaml"; cfgpath.write_text(yaml.safe_dump(cfgdict))
    cfg = NS(DATASET=NS(**cfgdict["DATASET"]), TEST=NS(**cfgdict["TEST"]), VERBOSE=False)
    qgt = np.array([1, 0, 0, 0], dtype=np.float32)
    tgt = gt_translation
    batches = []
    for i, ident in enumerate(ids):
        batches.append((torch.full((1,3,48,64), float(i)), {
            "domain":[domain], "imgpath":[ident], "quaternion":torch.from_numpy(qgt[None].copy()),
            "rotationmatrix":torch.from_numpy(postprocess.quat2dcm(qgt)[None]),
            "translation":torch.from_numpy(tgt[None].copy()),
            "boundingbox":torch.tensor([[0., 0., 10., 10.]])}))
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.]))
            self.register_buffer("counter", torch.tensor([7]))
            self.calls = 0
            self.eval()
        def forward(self, images, **kwargs):
            i = int(images[0,0,0,0]); self.calls += 1
            values = []
            for head in heads:
                if head == "heatmap":
                    heat = torch.zeros((1,11,12,16))
                    if not zero_heatmaps and (i != 2 or all_numeric):
                        heat[:,:,5,6] = 1
                    values.append(heat)
                elif head == "efficientpose":
                    values.append((torch.tensor([[1.]]), torch.tensor([[[0.,0.,10.,10.]]]),
                                   torch.tensor([[[1.,0.,0.,0.,1.,0.]]]),
                                   torch.from_numpy(pose_case["t_pr_array"].astype(np.float32).reshape(1,1,3).copy()) if pose_case else torch.tensor([[[.2,.1,10.]]])))
            return values
    camera = dict(cameraMatrix=np.array([[50.,0,32.],[0,50.,24.],[0,0,1.]]),
                  distCoeffs=np.zeros(5), horizontalFOV=1.2, Nu=64, Nv=48)
    keypts = np.array([[x,y,z] for x,y,z in
                      [(-1,-1,0),(1,-1,0),(-1,1,0),(1,1,0),(0,0,1),
                       (.2,.5,-1),(-.4,.2,.3),(.6,-.2,.2),(-.6,-.8,1),(.1,.8,-.4),(.4,.6,.8)]])
    def run(output, capture=False):
        model = Tiny()
        before = {k:v.detach().clone() for k,v in model.state_dict().items()}
        state_before = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
        pnp_calls = []
        def controlled_pnp(*args, **kwargs):
            i = len(pnp_calls); pnp_calls.append(i)
            return np.array([math.cos(.1),math.sin(.1),0.,0.]), pose_case["t_pr_array"].copy() if pose_case else np.array([.2,.1,10. if i == 0 and accepted_first else 60.])
        observer = observe_heatmap_poses(inference, csv_path=csvpath, output_dir=root/"capture",
                                         heads=heads, context={"fixture":True, "final_sha256":"f"*64}) if capture else contextlib.nullcontext()
        original = inference.solve_pose_from_heatmaps
        with patch.object(postprocess, "pnp", controlled_pnp), observer:
            summary = inference.do_valid(0,cfg,model,batches,camera,keypts,log_dir=str(output),return_summary=True)
        assert inference.solve_pose_from_heatmaps is original
        assert model.calls == count and not model.training
        for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
        state_after = (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
        assert state_before[:2] == state_after[:2] and torch.equal(state_before[2],state_after[2])
        return summary, model.calls, pnp_calls
    baseline = run(root/"baseline")
    observed = run(root/"observed",True)
    assert baseline == observed
    resultdir = root/"observed"/domain
    for name in ("mean_performance.mat","per_sample_performance.mat","predictions_pose.mat"):
        a,b = loadmat(root/"baseline"/domain/name),loadmat(resultdir/name)
        for key in a:
            if key.startswith("__"): continue
            if a[key].dtype.kind in "fc":
                assert np.array_equal(a[key], b[key], equal_nan=True), key
            else:
                assert repr(a[key].tolist()) == repr(b[key].tolist()), key
    original_sha = {str(p):sha(p) for p in resultdir.glob("*")}
    kwargs = dict(capture_dir=root/"capture", result_dir=resultdir, csv_path=csvpath,cfg_path=cfgpath,output_dir=root/"extension")
    report = build_extension(**kwargs)
    proof = independent_cpu_verify(report["report_path"],output_path=root/"extension/proof.json")
    assert original_sha == {str(p):sha(p) for p in resultdir.glob("*")}
    return NS(root=root,ids=ids,csv=csvpath,cfg=cfgpath,resultdir=resultdir,kwargs=kwargs,report=report,proof=proof)


def update_mat(path, mutation):
    value={k:v for k,v in loadmat(path).items() if not k.startswith("__")}
    mutation(value); savemat(path,value)


def proof_again(f):
    return independent_cpu_verify(f.report["report_path"],output_path=f.root/"extension/proof_new.json")


class Tests(unittest.TestCase):
    def test_frozen_dtype_hil_four_real_gt_boundaries_and_original_mat(self):
        from extended_evaluation.report import pose_metrics
        from extended_evaluation.verify import independent_metrics
        from utils.metrics import speed_score, error_translation
        cases=BOUNDARY_CASES
        for case in cases:
            with self.subTest(kind=case["kind"],threshold=case["threshold"]):
                gt=np.frombuffer(bytes.fromhex(case["t_gt"]["hex_bytes"]),dtype=case["t_gt"]["dtype"]).copy()
                pred=np.frombuffer(bytes.fromhex(case["t_pr"]["hex_bytes"]),dtype=case["t_pr"]["dtype"]).copy()
                q=np.array([1.,0,0,0]);qgt=np.array([1.,0,0,0],np.float32)
                expected=float(speed_score(pred,q,gt,qgt,applyThreshold=True,theta_t=case["threshold"])[0])
                options=dict(threshold=True,theta_q=.169,theta_t=case["threshold"])
                produced=pose_metrics(q,pred,qgt,gt,**options)
                verified=independent_metrics(q,pred,qgt,gt,**options)
                self.assertEqual(produced["eT_norm_HIL"],expected)
                self.assertEqual(verified["eT_norm_HIL"],expected)
                self.assertEqual(produced["eT"],float(error_translation(pred,gt)))
                self.assertEqual(verified["eT"],float(error_translation(pred,gt)))
                full=fixture(heads=("heatmap",) if case["kind"].startswith("H") else ("efficientpose",),count=1,
                             pose_case=dict(t_gt_values=gt.tolist(),t_pr_array=pred,threshold=case["threshold"]))
                self.assertEqual(full.proof["status"],"passed")
                self.assertEqual(full.report["extended_primary"]["metrics"]["eT_norm_HIL"]["mean"],expected)

    def test_zero_original_acceptance_and_missing_h_remain_missing(self):
        for empty in (False, True):
            with self.subTest(all_H_missing=empty):
                f=fixture(zero_heatmaps=empty,accepted_first=False)
                original=f.report["protocols"]["original_H_E_intersection"]["metrics"]["pose"]
                self.assertEqual(original["count"],0);self.assertIsNone(original["mean"])
                self.assertEqual(f.report["counts"]["original_accepted"],0)
                self.assertEqual(f.report["extended_primary"]["metrics"]["eT"]["count"],3)
                if empty:
                    self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["count"],0)
                    self.assertIsNone(f.report["extended_primary"]["metrics"]["pose"]["mean"])
                    self.assertEqual(f.report["counts"]["missing_H_rotation"],3)
                    self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["denominator_scope"],"numeric_pose_available_subset")

    def test_nontrivial_orientation_independent_math_and_frozen_convention(self):
        from extended_evaluation.report import matrix_from_q, pose_metrics
        from extended_evaluation.verify import independent_metrics, equal
        from utils.metrics import error_orientation
        rng=np.random.default_rng(2021)
        for _ in range(30):
            qgt=rng.normal(size=4);qgt=(qgt/np.linalg.norm(qgt)).astype(np.float32)
            q=rng.normal(size=4);q=q/np.linalg.norm(q)
            self.assertTrue(np.array_equal(matrix_from_q(qgt),postprocess.quat2dcm(qgt)))
            for matrix in (None,postprocess.quat2dcm(q.astype(np.float32))):
                opts=dict(rotation_matrix=matrix,threshold=True,theta_q=.169,theta_t=.002173)
                a=pose_metrics(q,[1.,2.,10.],qgt,np.array([.1,.1,10.],np.float32),**opts)
                b=independent_metrics(q,[1.,2.,10.],qgt,np.array([.1,.1,10.],np.float32),**opts)
                equal(a,b,"random metrics")
                expected=error_orientation(q,qgt,'quaternion') if matrix is None else error_orientation(matrix,postprocess.quat2dcm(qgt),'rotationmatrix')
                self.assertAlmostEqual(a["eR"],expected,places=5)

    def test_actual_postprocess_heatmaps_numeric_and_none(self):
        points = np.array([[-1,-1,0],[1,-1,0],[-1,1,0],[1,1,0],[0,0,1],
                           [.2,.5,-1],[-.4,.2,.3],[.6,-.2,.2],[-.6,-.8,1],[.1,.8,-.4],[.4,.6,.8]],np.float64)
        camera = dict(cameraMatrix=np.array([[100.,0.,64.],[0.,100.,48.],[0,0,1.]]),
                      distCoeffs=np.zeros(5),horizontalFOV=1.2,Nu=128,Nv=96)
        heat = torch.zeros((11,96,128))
        projected,_ = cv2.projectPoints(points,np.zeros(3),np.array([0.,0.,10.]),camera["cameraMatrix"],np.zeros(5))
        for i,xy in enumerate(projected[:,0]): heat[i,int(round(xy[1])),int(round(xy[0]))]=1.
        key,q,t,reject=postprocess.solve_pose_from_heatmaps(heat,[128,96],.5,camera,points)
        self.assertFalse(reject); self.assertTrue(np.isfinite(q).all()); self.assertLess(abs(t[2]-10),.5)
        empty=postprocess.solve_pose_from_heatmaps(torch.zeros_like(heat),[128,96],.5,camera,points)
        self.assertEqual(empty,(None,None,None,True))
        with patch.object(postprocess,"pnp",return_value=(np.array([1.,0,0,0]),np.array([0.,0.,60.]))):
            rejected=postprocess.solve_pose_from_heatmaps(heat,[128,96],.5,camera,points)
        self.assertTrue(rejected[3]); self.assertIsNotNone(rejected[1]); self.assertIsNotNone(rejected[2])

    def test_real_inference_invariance_and_partial_hybrid(self):
        f=fixture()
        self.assertEqual(f.proof["status"],"passed")
        self.assertEqual(f.report["counts"]["original_accepted"],1)
        self.assertEqual(f.report["counts"]["geometry_rejected_numeric_H"],1)
        self.assertEqual(f.report["counts"]["missing_H_rotation"],1)
        self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["count"],2)
        self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["denominator_scope"],"numeric_pose_available_subset")
        self.assertEqual(f.report["protocols"]["original_H_E_intersection"]["metrics"]["pose"]["count"],1)
        self.assertEqual(f.report["protocols"]["original_E"]["metrics"]["pose"]["count"],3)

    def test_h_only_never_uses_e(self):
        f=fixture(("heatmap",))
        self.assertEqual(f.report["primary_mode"],"H_rotation_H_translation")
        self.assertFalse(f.report["protocols"]["extended_hybrid"]["applicable"])
        self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["count"],2)

    def test_e_only_not_applicable_h(self):
        f=fixture(("efficientpose",))
        cap=json.loads((f.root/"capture/capture_manifest.json").read_text())
        self.assertEqual(cap["status"],"not_applicable"); self.assertEqual(cap["solver_calls"],0)
        self.assertEqual(cap["record_count"],0)
        self.assertEqual(f.report["primary_mode"],"E_only_original")
        self.assertIsNone(f.report["counts"]["missing_H_rotation"])
        self.assertIsNone(f.report["protocols"]["extended_H"]["metrics"])
        self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["count"],3)

    def test_all_samples_only_when_full_numeric_coverage(self):
        f=fixture(count=2,all_numeric=True,domain="synthetic")
        self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["denominator_scope"],"all_samples")
        self.assertEqual(f.report["extended_primary"]["metrics"]["pose"]["coverage"],1.)
        self.assertEqual(f.report["counts"]["original_accepted"],1)
        self.assertEqual(f.report["protocols"]["original_accepted_H"]["metrics"]["pose"]["denominator_scope"],"original_accepted_subset")

    def test_exact_return_identity_and_rng(self):
        root=Path(tempfile.mkdtemp(prefix="identity_",dir=ARTIFACTS)); csvpath=root/"ids.csv";csvpath.write_text("one\n")
        value=[torch.zeros(11,2),np.array([1.,0,0,0]),np.array([0.,0.,60.]),True]
        def original(*a,**kw): return value
        module=NS(solve_pose_from_heatmaps=original,__file__=__file__)
        before=(pickle.dumps(random.getstate()),pickle.dumps(np.random.get_state()),torch.get_rng_state().clone())
        with observe_heatmap_poses(module,csv_path=csvpath,output_dir=root/"cap",heads=["heatmap"]):
            self.assertIs(module.solve_pose_from_heatmaps(None),value)
        self.assertIs(module.solve_pose_from_heatmaps,original)
        self.assertEqual(before[0],pickle.dumps(random.getstate()));self.assertEqual(before[1],pickle.dumps(np.random.get_state()))
        self.assertTrue(torch.equal(before[2],torch.get_rng_state()))

    def test_exception_restores_original_and_keeps_failure_evidence(self):
        root=Path(tempfile.mkdtemp(prefix="exception_",dir=ARTIFACTS));csvpath=root/"ids.csv";csvpath.write_text("one\n")
        problem=LookupError("original failure")
        def original(*a,**kw): raise problem
        module=NS(solve_pose_from_heatmaps=original,__file__=__file__)
        with self.assertRaises(LookupError) as caught:
            with observe_heatmap_poses(module,csv_path=csvpath,output_dir=root/"cap",heads=["heatmap"]):
                module.solve_pose_from_heatmaps(None)
        self.assertIs(caught.exception,problem);self.assertIs(module.solve_pose_from_heatmaps,original)
        self.assertEqual(json.loads((root/"cap/capture_manifest.json").read_text())["status"],"failed")

    def test_missing_extra_and_e_only_solver_calls_fail(self):
        for calls,heads in ((0,["heatmap"]),(2,["heatmap"]),(1,["efficientpose"])):
            with self.subTest(calls=calls,heads=heads):
                root=Path(tempfile.mkdtemp(prefix="calls_",dir=ARTIFACTS));csvpath=root/"ids.csv";csvpath.write_text("one\n")
                result=(None,None,None,True)
                def original(*a,**kw): return result
                module=NS(solve_pose_from_heatmaps=original,__file__=__file__)
                with self.assertRaises(CaptureError):
                    with observe_heatmap_poses(module,csv_path=csvpath,output_dir=root/"cap",heads=heads):
                        for _ in range(calls): self.assertIs(module.solve_pose_from_heatmaps(None),result)
                self.assertIs(module.solve_pose_from_heatmaps,original)

    def test_nested_observer_refused(self):
        root=Path(tempfile.mkdtemp(prefix="nested_",dir=ARTIFACTS));csvpath=root/"ids.csv";csvpath.write_text("one\n")
        def original(): return None,None,None,True
        module=NS(solve_pose_from_heatmaps=original,__file__=__file__)
        with observe_heatmap_poses(module,csv_path=csvpath,output_dir=root/"cap",heads=["heatmap"]):
            with self.assertRaises(CaptureError):
                with observe_heatmap_poses(module,csv_path=csvpath,output_dir=root/"nested",heads=["heatmap"]): pass
            module.solve_pose_from_heatmaps()

    def test_report_and_proof_idempotence_no_mtime_or_sha_drift(self):
        f=fixture(); path=f.root/"extension/proof.json";before=(sha(path),path.stat().st_mtime_ns)
        self.assertEqual(f.proof,independent_cpu_verify(f.report["report_path"],output_path=path))
        self.assertEqual(before,(sha(path),path.stat().st_mtime_ns))
        rp=Path(f.report["report_path"]);before=(sha(rp),rp.stat().st_mtime_ns)
        self.assertEqual(f.report,build_extension(**f.kwargs));self.assertEqual(before,(sha(rp),rp.stat().st_mtime_ns))
        path.write_text("{}")
        with self.assertRaises(ValueError): independent_cpu_verify(f.report["report_path"],output_path=path)

    def test_corrupted_original_ids_pose_mask_and_means_rejected(self):
        mutations=[
            ("predictions_pose.mat",lambda d:d.__setitem__("sample_id",d["sample_id"][:,::-1])),
            ("predictions_pose.mat",lambda d:d["heat_q"].__setitem__((1,0),0.)),
            ("predictions_pose.mat",lambda d:d["heat_t"].__setitem__((0,0),99.)),
            ("per_sample_performance.mat",lambda d:d["final_pose"].__setitem__((0,0),9.)),
            ("per_sample_performance.mat",lambda d:d["final_pose_valid"].__setitem__((0,0),2)),
            ("mean_performance.mat",lambda d:d.__setitem__("final_pose_valid",np.array([[1.5]]))),
            ("mean_performance.mat",lambda d:d["heat_eR"].__setitem__((0,0),999.)),
        ]
        for name,mutation in mutations:
            with self.subTest(name=name,mutation=repr(mutation)):
                f=fixture();update_mat(f.resultdir/name,mutation)
                with self.assertRaises(ValueError):build_extension(**{**f.kwargs,"output_dir":f.root/"bad"})
                with self.assertRaises(ValueError):proof_again(f)

    def test_capture_order_missing_pose_and_flags_rejected_even_rehashed(self):
        for mutation in ("order","missing","flag","reject"):
            with self.subTest(mutation=mutation):
                f=fixture();path=f.root/"capture/heatmap_pose_capture.jsonl"
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                if mutation=="order": rows[0],rows[1]=rows[1],rows[0]
                elif mutation=="missing":
                    rows[0]["q"]=None;rows[0]["q_numeric"]=False;rows[0]["numeric_pose_available"]=False
                    rows[0]["missing_reason"]["q"]="solver_returned_none"
                elif mutation=="flag":rows[1]["q_numeric"]=False
                else:rows[0]["reject"]=True
                path.write_text("".join(json.dumps(x)+"\n" for x in rows))
                manifest=f.root/"capture/capture_manifest.json";cap=json.loads(manifest.read_text());cap["records"]["sha256"]=sha(path);atomic(manifest,cap)
                with self.assertRaises((ValueError,TypeError)):build_extension(**{**f.kwargs,"output_dir":f.root/"bad"})

    def test_corrupt_report_counts_labels_context_and_hashes_rejected(self):
        for mutation in ("counts","all_samples","context","code","rawhash","inputs","heads"):
            with self.subTest(mutation=mutation):
                f=fixture();report=copy.deepcopy(f.report)
                if mutation=="counts":report["counts"]["total"]=999
                elif mutation=="all_samples":report["extended_primary"]["metrics"]["pose"]["denominator_scope"]="all_samples"
                elif mutation=="context":report["context"]["final_sha256"]="a"*64
                elif mutation=="code":report["observer_code_sha256"][str(HERE/"capture.py")]="0"*64
                elif mutation=="rawhash":report["per_sample_sha256"]="0"*64
                elif mutation=="inputs":report["inputs"]["cfg"]["sha256"]="0"*64
                else:report["heads"]=["efficientpose"]
                atomic(report["report_path"],report)
                with self.assertRaises(ValueError):proof_again(f)

    def test_corrupt_extended_rows_rejected_even_rehashed(self):
        f=fixture();path=Path(f.report["per_sample_path"]);rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[1]["extended_hybrid"]["eR"]=0.
        path.write_text("".join(json.dumps(x)+"\n" for x in rows))
        report=copy.deepcopy(f.report);report["per_sample_sha256"]=sha(path);atomic(report["report_path"],report)
        with self.assertRaises(ValueError):proof_again(f)

    def test_context_is_finite_json_and_bound_to_capture(self):
        f=fixture()
        with self.assertRaises(ValueError):build_extension(**f.kwargs,context={"different":True})
        with self.assertRaises(ValueError):observe_heatmap_poses(inference,csv_path=f.csv,output_dir=f.root/"bad",heads=["heatmap"],context={"nan":float("nan")})
        with self.assertRaises(ValueError):independent_cpu_verify(f.report["report_path"],output_path=f.csv)

    def test_duplicate_csv_ids_and_wrong_cfg_domain_rejected(self):
        root=Path(tempfile.mkdtemp(prefix="ids_",dir=ARTIFACTS));csvpath=root/"bad.csv";csvpath.write_text("same\nsame\n")
        with self.assertRaises(ValueError):observe_heatmap_poses(inference,csv_path=csvpath,output_dir=root/"cap",heads=["heatmap"])
        f=fixture();cfg=yaml.safe_load(f.cfg.read_text());cfg["TEST"]["TEST_CSV"]="sunlamp/other.csv";f.cfg.write_text(yaml.safe_dump(cfg))
        with self.assertRaises(ValueError):build_extension(**{**f.kwargs,"output_dir":f.root/"bad"})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--report",required=True);args=parser.parse_args()
    assert os.environ.get("STY") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    assert os.environ.get("PYTHONNOUSERSITE")=="1"
    ARTIFACTS.mkdir(exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    value=dict(schema="spnv2-extended-evaluation-cpu-regression-v1",
               status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
               failures=[(str(t),txt) for t,txt in result.failures+result.errors],
               code_sha256=implementation(),screen=os.environ["STY"],CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
               command=sys.argv,scope="CPU real heatmap postprocess and frozen inference with tiny deterministic model/PnP fixtures; no real source/ODR inference or GPU run")
    output=Path(args.report);output.parent.mkdir(parents=True,exist_ok=True);atomic(output,value)
    raise SystemExit(0 if result.wasSuccessful() else 1)
