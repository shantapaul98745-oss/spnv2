"""Local TinyEHS contracts; no SPNv2, real images, SSH or shared-host work."""
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

OLD = Path(__file__).resolve().parents[2]/"odr003_round2_20261023/implementation"
sys.path.insert(0, str(OLD))
import test_protected_engine as contracts
from protected_engine import validate_engine_state
from durable_state import DurableRunStore
from state_adapter import equal_tree, isolated_scoring

from pair_runner import (ForwardBudgetHook, ResourceYield, capture_candidate, derive_branch_snapshots,
                         evaluate_branches, science_state, tree_digest, trace_prefix_matches, _metrics_summary)


class FakeBudget:
    def __init__(self): self.reserved=[]; self.finished=[]; self.time_checks=[]
    def check_time(self, reason): self.time_checks.append(reason)
    def reserve_forward(self, label, count=1):
        token=len(self.reserved); self.reserved.append((label,count)); return token
    def finish_forward(self, token, count=None): self.finished.append((token,count))


def metrics(error=1., valid=True):
    return dict(valid=valid, error=error if valid else None, original_valid=valid,
                original_error=error if valid else None, rotation_deg=1. if valid else None,
                translation_m=.1 if valid else None, heat_rotation_HIL=.4 if valid else None,
                effi_translation_HIL=.6 if valid else None, heat_error=error if valid else None,
                effi_error=error if valid else None)


class PairRunnerContracts(unittest.TestCase):
    def setUp(self):
        self.fixture=contracts.EngineContracts(); self.fixture.setUp()
        self.engine=self.fixture.engine("pair", count=128, reject_windows=(1,))
        for _ in range(4): self.engine.step(stop_after_prepare=True)
        self.prepared=self.engine.snapshot()
        self.candidate=capture_candidate(self.engine)
        self.ids=[f"panel/{i:03d}" for i in range(128)]
        self.offline=[dict(sample_id=sid,valid=True,error=.5) for sid in self.ids]
        self.budget=FakeBudget()
    def tearDown(self): self.fixture.tearDown()

    def evaluate(self,sid):
        adapter=self.engine.learner.adapter
        with isolated_scoring(adapter.model,adapter.bn_features):
            adapter.model.backbone(torch.ones((1,3,8,8)))
        error=1.+abs(float(next(p for p in adapter.model.parameters() if p.requires_grad).detach().flatten()[0]))
        return dict(sample_id=sid,prediction=dict(h_available=True,e_available=True,original_h_rejected=False,
                                                diagnostic_error=error),metrics=metrics(error))

    def test_capture_resolves_actual_candidate_once_and_keeps_full_base(self):
        self.assertEqual(self.engine.counters["candidates"],1)
        self.assertEqual(self.engine.counters["rollbacks"],1)
        self.assertEqual(self.candidate["phase"],"candidate")
        self.assertEqual(self.candidate["window_size"],4)
        self.assertEqual(self.prepared["learner"]["window_size"],3)
        branches=derive_branch_snapshots(self.candidate)
        self.assertEqual(branches["rollback"]["learning"]["counters"]["updates"],0)
        self.assertEqual(branches["rollback"]["learning"]["counters"]["accepted"],4)
        self.assertTrue(all(value is None for value in branches["rollback"]["learning"]["gradients"].values()))
        self.assertFalse(equal_tree(branches["rollback"]["learning"],self.prepared["learner"]["learning"]))

    def test_both_branches_exact_derivation_and_evaluation_restores_actual_state(self):
        before=self.engine.snapshot()
        result=evaluate_branches(self.engine,self.candidate,self.ids,self.evaluate,self.offline,
                                 equal=equal_tree,budget=self.budget)
        self.assertEqual(len(result["accept"]["rows"]),128)
        self.assertEqual(len(result["rollback"]["rows"]),128)
        self.assertTrue(equal_tree(science_state(before),science_state(self.engine.snapshot())))
        self.assertEqual(result["accept"]["summary"]["tail_count"],7)

    def test_branch_failure_restores_full_original_learning_rng_and_provider(self):
        before=self.engine.snapshot()
        def broken(sid):
            if sid.endswith("004"): raise RuntimeError("synthetic raw nonfinite/failure")
            return self.evaluate(sid)
        with self.assertRaises(RuntimeError):
            evaluate_branches(self.engine,self.candidate,self.ids,broken,self.offline,equal=equal_tree,budget=self.budget)
        self.assertTrue(equal_tree(science_state(before),science_state(self.engine.snapshot())))

    def test_unisolated_evaluation_state_mutation_fails_closed(self):
        def polluted(sid):
            value=self.evaluate(sid)
            self.engine.learner.adapter.model.scratch.add_(1)
            return value
        with self.assertRaisesRegex(ValueError,"evaluation changed"):
            evaluate_branches(self.engine,self.candidate,self.ids,polluted,self.offline,equal=equal_tree,budget=self.budget)

    def test_saved_panel_prefix_reuses_completed_forward_measurements(self):
        full=evaluate_branches(self.engine,self.candidate,self.ids,self.evaluate,self.offline,equal=equal_tree,budget=self.budget)
        calls=[]; reconstructed=[]
        def measure(sid): calls.append(sid); return self.evaluate(sid)
        result=evaluate_branches(self.engine,self.candidate,self.ids,measure,self.offline,equal=equal_tree,budget=self.budget,
                 saved_progress=lambda name,digest: full[name]["rows"][:64], reconstruct_panel=reconstructed.append)
        self.assertEqual(len(calls),128)
        self.assertEqual(len(reconstructed),128)
        self.assertEqual(result["accept"]["rows"],full["accept"]["rows"])

    def test_shared_host_yield_restores_actual_state_without_observing_again(self):
        before=self.engine.snapshot()
        with self.assertRaises(ResourceYield):
            evaluate_branches(self.engine,self.candidate,self.ids,self.evaluate,self.offline,equal=equal_tree,budget=self.budget,
                              resource_check=lambda: (_ for _ in ()).throw(ResourceYield("foreign task")))
        self.assertTrue(equal_tree(science_state(before),science_state(self.engine.snapshot())))
        self.assertEqual(self.engine.counters["observed"],4)

    def test_backbone_hook_covers_direct_scorer_and_model_without_double_count(self):
        model=self.engine.learner.adapter.model
        hook=ForwardBudgetHook(model.backbone,self.budget,synchronize=lambda: None)
        try:
            hook.phase="panel_accept"; self.evaluate(self.ids[0])
            hook.phase="adaptation"; model(torch.ones((1,3,8,8)))
            self.assertEqual(hook.counts,dict(panel_accept=1,adaptation=1))
            self.assertEqual(len(self.budget.finished),2)
            with self.assertRaisesRegex(ValueError,"duplicate"):
                ForwardBudgetHook(model.backbone,self.budget,synchronize=lambda: None)
        finally: hook.close()

    def test_hook_reserves_before_dispatch_and_failure_does_not_refund(self):
        class Failing(torch.nn.Module):
            def forward(self,image): raise RuntimeError("dispatch failed")
        module=Failing(); hook=ForwardBudgetHook(module,self.budget,synchronize=lambda: None)
        try:
            with self.assertRaises(RuntimeError): module(torch.ones((1,3,8,8)))
            self.assertEqual(len(self.budget.reserved),1); self.assertEqual(self.budget.finished,[])
        finally: hook.close()

    def test_missingness_keeps_all128_and_seven_tail(self):
        data=[dict(sample_id=sid,metrics=metrics()) for sid in self.ids]
        data[-1]["metrics"]=metrics(valid=False)
        result=_metrics_summary(data,self.offline)
        self.assertEqual(result["panel_count"],128); self.assertEqual(result["tail_count"],7)
        self.assertIsNone(result["mean_error"]); self.assertIsNone(result["mean_D"]); self.assertIsNone(result["tail_D"])

    def test_final_rejoin_includes_optimizer_rng_provider_pools_not_only_weights(self):
        # Continue the actual branch; make a second identical complete run.
        self.engine.run(); expected=self.engine.snapshot()
        other=self.fixture.engine("other",count=128,reject_windows=(1,))
        other.run(); actual=other.snapshot(); actual["identity"]=expected["identity"]
        self.assertTrue(equal_tree(science_state(actual),science_state(expected)))
        self.assertEqual(tree_digest(science_state(actual)),tree_digest(science_state(expected)))
        actual["pools"]["completed_decisions"]+=1
        self.assertFalse(equal_tree(science_state(actual),science_state(expected)))

    def test_prediction_or_guard_trace_drift_stops_before_branch_acceptance(self):
        trace=copy.deepcopy(self.engine.scientific_trace)
        trace_prefix_matches(trace,trace,equal_tree)
        trace[-1]["decision"]="commit"
        with self.assertRaisesRegex(ValueError,"diverged"):
            trace_prefix_matches(trace,self.engine.scientific_trace,equal_tree)


if __name__=="__main__": unittest.main()
