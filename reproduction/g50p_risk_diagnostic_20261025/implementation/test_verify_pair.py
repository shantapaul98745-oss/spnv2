"""Synthetic saved-evidence verifier checks; no real SPNv2 or target images."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
from pair_runner import atomic_json, sha256, canonical_sha
from pair_runner import reference_trace
from pair_runner import configure_frozen_modules
from types import SimpleNamespace

from pair_runner import (OUTPUT_SCHEMA, derive_branch_snapshots, evaluate_branches,
                         science_state, tree_digest)
from verify_pair import verify_saved, verify_budget, verify_branch_rows, _receipt
import test_pair_runner as runner_tests


def budget_snapshot(n):
    return dict(schema="g50p-diagnostic-budget-snapshot-v1", physical_forwards_reserved_upper=n,
                physical_forwards_confirmed=n, physical_forwards_unknown_upper=0, unknown_forward_tokens=[])


class PairVerifierContracts(unittest.TestCase):
    def setUp(self):
        self.runner=runner_tests.PairRunnerContracts(); self.runner.setUp()
        r=self.runner
        self.branches=evaluate_branches(r.engine,r.candidate,r.ids,r.evaluate,r.offline,equal=runner_tests.equal_tree,budget=r.budget)
        self.task=dict(run_id="pair",selection=dict(candidate_number=1),producer=dict(commit="3ef"),panel=dict(manifest="frozen"))
        self.expected=r.engine.snapshot(); self.actual=science_state(self.expected)
        self.trace=copy.deepcopy(r.engine.scientific_trace)
        self.labels={sid:dict(quaternion_wxyz=[1,0,0,0],translation_m=[0,0,1]) for sid in r.ids}
        costs=dict(schema="g50p-risk-diagnostic-attempt-cost-v1",run_nonce="a"*32,status="pair_main_complete",
                   budget_before=budget_snapshot(0),budget=budget_snapshot(256),
                   forward_categories=dict(panel_accept=128,panel_rollback=128))
        self.report=dict(schema=OUTPUT_SCHEMA,status="pair_main_complete_pending_independent_verification",
                         run_id="pair",selection=self.task["selection"],producer=self.task["producer"],panel=self.task["panel"],
                         candidate_learner_sha256=tree_digest(r.candidate),
                         expected_branch_learning_sha256={name:tree_digest(value) for name,value in derive_branch_snapshots(r.candidate).items()},
                         matched_scientific_sha256=tree_digest(self.actual),expected_scientific_sha256=tree_digest(self.actual),
                         excluded_match_fields=["ledger_boundary"],original_producer_files_modified=False,
                         original_target=self.trace[-1],branch_results=self.branches,
                         budget_before=budget_snapshot(0),budget=budget_snapshot(256),
                         forward_categories=dict(panel_accept=128,panel_rollback=128),attempt_costs=[costs],
                         actual_attempt_elapsed_seconds=.1)
    def tearDown(self): self.runner.tearDown()
    def recompute(self,p,label,cfg): return runner_tests.metrics(p["diagnostic_error"])
    def verify(self,report=None,actual=None):
        r=self.runner
        return verify_saved(self.report if report is None else report,task=self.task,candidate=r.candidate,
                 actual_match=self.actual if actual is None else actual,expected_match=self.expected,
                 original_trace=self.trace,ids=r.ids,labels=self.labels,cfg=None,recompute=self.recompute,
                 offline=r.offline,equal=runner_tests.equal_tree)

    def test_saved_proofs_recompute_all_metrics_without_model_forward(self):
        count=[]
        def recompute(*args): count.append(1); return self.recompute(*args)
        r=self.runner
        result=verify_saved(self.report,task=self.task,candidate=r.candidate,actual_match=self.actual,
                   expected_match=self.expected,original_trace=self.trace,ids=r.ids,labels=self.labels,
                   cfg=None,recompute=recompute,offline=r.offline,equal=runner_tests.equal_tree)
        self.assertEqual(len(count),256)
        self.assertEqual(result["model_forwards_added_by_verification"],0)
        self.assertEqual(result["fixed_denominator"],128); self.assertEqual(result["fixed_tail"],7)
        self.assertEqual(result["budget"]["physical_model_forwards"],256)

    def test_main_pass_flag_does_not_replace_independent_evidence(self):
        bad=copy.deepcopy(self.report); bad["status"]="passed"
        with self.assertRaisesRegex(ValueError,"masquerades"): self.verify(bad)

    def test_altered_saved_metrics_fail_even_when_same_coverage(self):
        bad=copy.deepcopy(self.report); bad["branch_results"]["accept"]["rows"][3]["metrics"]["error"]+=1
        with self.assertRaisesRegex(ValueError,"error differs"): self.verify(bad)

    def test_swapped_or_omitted_panel_ids_fail(self):
        for mutation in ("reverse","omit"):
            bad=copy.deepcopy(self.report)
            rows=bad["branch_results"]["accept"]["rows"]
            if mutation=="reverse": rows.reverse()
            else: rows.pop()
            with self.assertRaisesRegex(ValueError,"fixed128"): self.verify(bad)

    def test_rollback_cannot_be_three_gradient_partial_state(self):
        bad=copy.deepcopy(self.report)
        bad["expected_branch_learning_sha256"]["rollback"]=tree_digest(self.runner.prepared["learner"])
        with self.assertRaisesRegex(ValueError,"complete-window rollback"): self.verify(bad)

    def test_exact_rejoin_checks_rng_optimizer_pools_not_only_model(self):
        for field in ("pools","provider_state"):
            bad=copy.deepcopy(self.actual)
            if field=="pools": bad[field]["completed_decisions"]+=1
            else: bad[field]["requests"].append("unseen")
            with self.assertRaisesRegex(ValueError,"exactly rejoin"): self.verify(actual=bad)

    def test_missing_cannot_be_removed_to_change_tail_or_denominator(self):
        bad=copy.deepcopy(self.report); bad["branch_results"]["rollback"]["summary"]["tail_count"]=6
        with self.assertRaisesRegex(ValueError,"fixed128/tail7"): self.verify(bad)

    def test_all_improvements_and_regressions_must_remain_in_changes(self):
        bad=copy.deepcopy(self.report); bad["branch_results"]["per_image_changes"].pop()
        with self.assertRaisesRegex(ValueError,"omitted or changed"): self.verify(bad)

    def test_each_actual_forward_requires_reservation_and_confirmation(self):
        for field in ("physical_forwards_reserved_upper","physical_forwards_confirmed"):
            bad=copy.deepcopy(self.report)
            bad["attempt_costs"][0]["budget"][field]-=1
            with self.assertRaisesRegex(ValueError,"every backbone"): verify_budget(bad)

    def test_unknown_dispatched_work_cannot_be_declared_complete(self):
        bad=copy.deepcopy(self.report); bad["attempt_costs"][0]["budget"]["unknown_forward_tokens"]=["inflight"]
        with self.assertRaises(ValueError): verify_budget(bad)

    def test_resumed_attempt_costs_all_charged_once(self):
        bad=copy.deepcopy(self.report)
        first=copy.deepcopy(bad["attempt_costs"][0]); first.update(run_nonce="b"*32,status="waiting_resources")
        first["forward_categories"]=dict(panel_accept=64); first["budget"]=budget_snapshot(64)
        second=copy.deepcopy(bad["attempt_costs"][0]); second["budget_before"]=budget_snapshot(64)
        second["forward_categories"]=dict(panel_accept=64,panel_rollback=128)
        bad["attempt_costs"]=[first,second]
        result=verify_budget(bad)
        self.assertEqual(result["actual_attempt_count"],2)
        self.assertEqual(result["physical_model_forwards"],256)


class ReceiptContracts(unittest.TestCase):
    """Pure file/metadata tests: no Tiny fixture and no model invocation."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.run_task=dict(run_id="pair",output_dir=str(self.root))
        self.task_path=self.root/"task.json";atomic_json(self.task_path,self.run_task)
        self.log=self.root/"main.log";self.log.write_text("fixture main log")
        self.exit_path=Path(str(self.log)+".exitcode");self.exit_path.write_text("0")
        self.launch=dict(run_id="pair",run_nonce="n"*32,pid=1,starttime=5,boot_id="fixture-boot",
                         argv=["python","--task-json",str(self.task_path)])
        atomic_json(self.root/"launch.json",self.launch)
        self.report=dict(task_sha256=sha256(self.task_path),launch_record_sha256=canonical_sha(self.launch),
                 attempt_costs=[dict(run_nonce="n"*32,status="pair_main_complete",launch_record_sha256=canonical_sha(self.launch))])
        atomic_json(self.root/"report.json",self.report)
        self.outcome=dict(status="pair_main_complete",exit_code=0,report=str(self.root/"report.json"),report_sha256=sha256(self.root/"report.json"))
        atomic_json(self.root/"outcome.json",self.outcome)
        attempt=dict(run_id="pair",task_sha256=sha256(self.task_path),task_path=str(self.task_path),nonce="n"*32,
                     log_path=str(self.log),launch_directory=str(self.root),exit_code=0,outcome=self.outcome)
        self.spec=dict(run_id="pair",task=dict(path=str(self.task_path),sha256=sha256(self.task_path)),
               report=dict(path=str(self.root/"report.json"),sha256=sha256(self.root/"report.json")),
               log_path=str(self.log),exit_code_path=str(self.exit_path),outcome_path=str(self.root/"outcome.json"),
               launch_record_path=str(self.root/"launch.json"),attempts=[attempt])
    def tearDown(self):self.temp.cleanup()
    def test_real_files_and_actual_exit_required(self):
        path,report=_receipt(self.spec,self.run_task,self.task_path)
        self.assertEqual(path,self.root/"report.json")
        self.exit_path.write_text("1")
        with self.assertRaisesRegex(ValueError,"exit is not0"):_receipt(self.spec,self.run_task,self.task_path)
    def test_same_run_name_is_insufficient_if_command_uses_other_task(self):
        self.launch["argv"][-1]=str(self.root/"another.json");atomic_json(self.root/"launch.json",self.launch)
        with self.assertRaisesRegex(ValueError,"command used another"):_receipt(self.spec,self.run_task,self.task_path)
    def test_prior_implementation_failure_cannot_be_relabelled_as_wait(self):
        self.spec["attempts"][0]["exit_code"]=1
        with self.assertRaisesRegex(ValueError,"silently retried"):_receipt(self.spec,self.run_task,self.task_path)


class TraceExtractionContracts(unittest.TestCase):
    """Pure JSON metadata tests; zero model construction or forward."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/"g50p_risk_diagnostic_20261025"
        self.inputs=self.root/"inputs";self.inputs.mkdir(parents=True)
        self.identity=dict(run_id="original_p",source_sha256="b"*64,config_sha256="c"*64,manifest_sha256="d"*64)
        self.trace=[dict(kind="observation",sample_id="lightbox/img000001.jpg")]
        self.document=dict(identity=self.identity,run_key="P:lightbox:2021",run_id="original_p",
              source_sha256="b"*64,protocol_sha256="e"*64,state_sha256="a"*64,
              scientific_trace=self.trace,pools={},stream_records=[],counters={})
        self.trace_path=self.inputs/"original_p.json";atomic_json(self.trace_path,self.document)
        self.index_binding=dict(path=str(self.root/"frozen_inputs.json"),sha256="f"*64)
        self.proof=dict(schema="g50p-risk-diagnostic-A-extraction-proof-v1",status="passed",model_forwards=0,
              frozen_inputs=self.index_binding,parent_final_state_sha256={"P:lightbox:2021":"a"*64},
              outputs=[dict(path=str(self.trace_path),sha256=sha256(self.trace_path))])
        self.proof_path=self.inputs/"extraction_proof.json";atomic_json(self.proof_path,self.proof)
        self.task=dict(output_dir=str(self.root/"pairs/pair"),reference_trace_proof_path=str(self.proof_path),
              index_binding=self.index_binding,selection=dict(domain="lightbox",stream_seed=2021,
              original_final_checkpoint=dict(state_sha256="a"*64)))
        self.budget=runner_tests.FakeBudget()
    def tearDown(self):self.temp.cleanup()
    def test_actual_extraction_output_parses_without_loading_original_final(self):
        self.assertEqual(reference_trace(self.task,self.identity,budget=self.budget),self.trace)
        missing=copy.deepcopy(self.task);missing.pop("reference_trace_proof_path")
        with self.assertRaises(KeyError):reference_trace(missing,self.identity,budget=self.budget)
    def test_changed_parent_identity_output_bytes_or_GT_fields_fail_closed(self):
        bad_proof=copy.deepcopy(self.proof);bad_proof["parent_final_state_sha256"]["P:lightbox:2021"]="0"*64
        atomic_json(self.proof_path,bad_proof)
        with self.assertRaisesRegex(ValueError,"matching zero-forward"):
            reference_trace(self.task,self.identity,budget=self.budget)
        atomic_json(self.proof_path,self.proof)
        changed=copy.deepcopy(self.document);changed["identity"]["run_id"]="different"
        atomic_json(self.trace_path,changed)
        with self.assertRaisesRegex(ValueError,"bound file changed"):
            reference_trace(self.task,self.identity,budget=self.budget)
        changed=copy.deepcopy(self.document);changed["evaluation"]={"rows":[]}
        atomic_json(self.trace_path,changed)
        proof=copy.deepcopy(self.proof);proof["outputs"][0]["sha256"]=sha256(self.trace_path)
        atomic_json(self.proof_path,proof)
        with self.assertRaisesRegex(ValueError,"GT-derived fields"):
            reference_trace(self.task,self.identity,budget=self.budget)


class BootstrapImportContracts(unittest.TestCase):
    """Mock original-core bootstrap only; zero models or model forwards."""
    def scenario(self, fail_config=False):
        calls=[];configured={"ready":False}
        original=dict(baseline_task=dict(output_dir="old/run",run_id="old",bindings={"developer_labels":{"path":"dev_labels.json"}}),
                      frozen_modules={name:{"path":name+".py","sha256":"f"*64}
                                      for name in ("prediction","state_adapter","run_exploration","data_boundary","gate_core")})
        def verify(task):calls.append("verify");return {"config":"bound-config"},{},"verified-original-core",{}
        def configure(task,paths,core):
            calls.append(("configure",task["output_dir"],task["run_id"],core))
            if fail_config:raise ValueError("configuration failed before prediction")
            configured["ready"]=True;return "fixedcfg",{}
        baseline=SimpleNamespace(GTBoundary=lambda path:SimpleNamespace(phase="preflight"),verify_bindings=verify,configure=configure)
        def importer(name,spec):
            calls.append(("import",name))
            if name=="prediction" and not configured["ready"]:raise ModuleNotFoundError("No module named utils")
            return baseline if name=="run_exploration" else SimpleNamespace(name=name)
        return original,SimpleNamespace(import_bound_module=importer),calls
    def test_prediction_waits_for_verified_original_core_path_bootstrap(self):
        original,backend,calls=self.scenario();saved=copy.deepcopy(original)
        result=configure_frozen_modules(backend,original,output_dir="new/pairs/run",run_id="diagnostic",phase="adaptation")
        position=next(i for i,row in enumerate(calls) if isinstance(row,tuple) and row[0]=="configure")
        self.assertGreater(calls.index(("import","prediction")),position)
        self.assertIn(("configure","new/pairs/run","diagnostic","verified-original-core"),calls)
        self.assertEqual(result[-1],"fixedcfg");self.assertEqual(result[1].phase,"adaptation")
        self.assertEqual(original,saved)
    def test_bootstrap_failure_does_not_import_prediction_or_construct_model(self):
        original,backend,calls=self.scenario(fail_config=True)
        with self.assertRaisesRegex(ValueError,"configuration failed"):
            configure_frozen_modules(backend,original,output_dir="new/pairs/run",run_id="diagnostic",phase="evaluation")
        self.assertNotIn(("import","prediction"),calls)


if __name__=="__main__": unittest.main()
