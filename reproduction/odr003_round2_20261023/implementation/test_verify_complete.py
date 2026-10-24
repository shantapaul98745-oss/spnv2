"""Synthetic saved-record checks only; no model construction or forward."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from round2_protocol import frozen_protocol, derive_controls
from verify_complete import (METRIC_FIELDS, checked_developer_rows,
                             evaluate_fixed_reference, derive_verified_controls, verify_original_baseline_state,
                             validate_offline_checkpoint, checked_runtime_receipt, PROOF_SCHEMA)
from verify_complete import checked_fixed_reference, verify_offline_source_state
from frozen_backend import canonical_sha, sha256


def rows(error=1., n=512):
    return [dict(sample_id=f"d/{i}", valid=True, error=error, original_valid=True,
                 original_error=error, rotation_deg=2., translation_m=.2,
                 heat_rotation_HIL=error*.4, effi_translation_HIL=error*.6,
                 heat_error=error, effi_error=error, original_rejected=False) for i in range(n)]


def runtime_receipt():
    policy = frozen_protocol()["deterministic_execution"]
    keys = ('cublas_workspace_config', 'torch_deterministic_algorithms', 'warn_only',
            'cudnn_benchmark', 'cudnn_deterministic', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32',
            'float32_matmul_precision')
    return dict(schema='odr003-round2-deterministic-runtime-v1', policy_version='strict-cuda-fp32-v1',
                policy_sha256=canonical_sha(policy), cudnn_enabled=True, torch_version='2.5.1+cu121',
                cuda_version='12.1', cudnn_version=90100, **{key: policy[key] for key in keys})


class VerifyCompleteTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"d/{i}" for i in range(512)]

    def test_tail_ceil_and_includes_zero_degradation(self):
        o, a, p = rows(), rows(), rows()
        p[0]["error"] = 2.
        p[1]["error"] = .5
        result = evaluate_fixed_reference(self.ids, o, a, {"P": p}, expected_reference_count=512, fixed_reference_ids=self.ids)
        self.assertEqual(result["tail_count"], 26)
        m = result["methods"]["P"]["metrics"]
        self.assertAlmostEqual(m["mean_D"], 1/512)
        self.assertAlmostEqual(m["tail_D"], 1/26)
        self.assertEqual(result["methods"]["P"]["per_sample"][1]["delta_vs_A"], -.5)

    def test_missing_reference_cannot_be_offset_by_gained_prediction(self):
        o, a, p = rows(), rows(), rows()
        for row in (o[-1], a[-1]): row.update(valid=False, error=None)
        p[0].update(valid=False, error=None)
        result = evaluate_fixed_reference(self.ids, o, a, {"P": p}, expected_reference_count=511, fixed_reference_ids=self.ids[:-1])
        report = result["methods"]["P"]
        self.assertEqual(report["coverage"], report["A_coverage"])
        self.assertEqual(report["reference_missing"], 1)
        self.assertFalse(report["coverage_pass"])
        self.assertIsNone(report["metrics"]["mean_D"])

    def test_reference_denominator_cannot_shrink_when_new_A_is_missing(self):
        o, a = rows(), rows()
        a[0].update(valid=False, error=None)
        result = evaluate_fixed_reference(self.ids, o, a, {"P": rows()}, expected_reference_count=512, fixed_reference_ids=self.ids)
        self.assertEqual(result['reference_count'], 512)
        self.assertEqual(result['methods']['A']['reference_missing'], 1)
        self.assertIsNone(result['methods']['A']['metrics']['mean_error'])
        self.assertEqual(result['reference_ids'], self.ids)

    def test_omitted_or_duplicate_rows_fail(self):
        for bad in (rows()[:-1], rows()[::-1]):
            with self.assertRaises(ValueError):
                evaluate_fixed_reference(self.ids, rows(), rows(), {"P": bad}, expected_reference_count=512, fixed_reference_ids=self.ids)

    def test_nonfinite_and_negative_fail(self):
        for value in (float("nan"), float("inf"), -1., True):
            bad = rows(); bad[0]["error"] = value
            with self.assertRaises(ValueError):
                evaluate_fixed_reference(self.ids, rows(), rows(), {"P": bad}, expected_reference_count=512, fixed_reference_ids=self.ids)

    def test_saved_pose_metrics_are_recomputed_not_claimed(self):
        labels = [dict(sample_id=sid, quaternion_wxyz=[1,0,0,0], translation_m=[0,0,1]) for sid in self.ids]
        metric = {key: rows()[0][key] for key in METRIC_FIELDS}
        prediction = dict(h_available=True, e_available=True, original_h_rejected=False,
                          heat_q=[1,0,0,0], heat_t=[0,0,1], effi_R=[[1,0,0],[0,1,0],[0,0,1]], effi_t=[0,0,1])
        doc = dict(next_index=512, rows=[dict(sample_id=sid, prediction=copy.deepcopy(prediction), metrics=copy.deepcopy(metric)) for sid in self.ids])
        calls = []
        def recompute(p, label, cfg): calls.append((p, label)); return copy.deepcopy(metric)
        result = checked_developer_rows(doc, labels, self.ids, recompute=recompute, cfg=None)
        self.assertEqual(len(calls), 512)
        self.assertEqual(len(result), 512)
        bad = copy.deepcopy(doc); bad["rows"][5]["metrics"]["error"] = .1
        with self.assertRaisesRegex(ValueError, "error differs"):
            checked_developer_rows(bad, labels, self.ids, recompute=recompute, cfg=None)

    def test_flat_old_prediction_and_explicit_none(self):
        labels = [dict(sample_id=sid, quaternion_wxyz=[1,0,0,0], translation_m=[0,0,1]) for sid in self.ids]
        metric = dict.fromkeys(METRIC_FIELDS); metric.update(valid=False, original_valid=False)
        flat = [dict(sample_id=sid, h_available=False, e_available=False, original_h_rejected=True, **metric) for sid in self.ids]
        result = checked_developer_rows(flat, labels, self.ids, recompute=lambda *a: metric, cfg=None, flat=True)
        self.assertEqual(result[0]["error"], None)
        self.assertTrue(result[0]["original_rejected"])

    def test_evaluation_cursor_cannot_be_a_log_commit(self):
        with self.assertRaises(ValueError):
            checked_developer_rows(dict(next_index=511, rows=[]), [], self.ids,
                                   recompute=lambda *a: {}, cfg=None)

    def test_controls_only_from_verified_P_counts_not_accuracy(self):
        report = dict(stage="complete", definition=dict(group="P", domain="sunlamp", stream_seed=2022),
                      source_sha256=frozen_protocol()["source_sha256"],
                      protocol_sha256=frozen_protocol()["protocol_sha256"], counts=dict(candidates=107, commits=71))
        proof = dict(status="passed", report_sha256="a"*64, counts=report["counts"])
        control = derive_verified_controls(report, proof, report_binding=dict(sha256="a"*64))
        self.assertEqual(control, derive_controls("sunlamp", 2022, 107, 71))
        self.assertFalse(control["RW"].get("exact_update_budget_guaranteed", False))
        bad = copy.deepcopy(proof); bad["counts"] = dict(candidates=108, commits=71)
        with self.assertRaises(ValueError): derive_verified_controls(report, bad, report_binding=dict(sha256="a"*64))

    def test_Q_zero_keeps_RW_undefined_RP_zero(self):
        report = dict(stage="complete", definition=dict(group="P", domain="lightbox", stream_seed=2021),
                      source_sha256=frozen_protocol()["source_sha256"],
                      protocol_sha256=frozen_protocol()["protocol_sha256"], counts=dict(candidates=0, commits=0))
        proof = dict(status="passed", report_sha256="x", counts=report["counts"])
        c = derive_verified_controls(report, proof, report_binding=dict(sha256="x"))
        self.assertIsNone(c["RW"]["commit_probability"])
        self.assertFalse(c["RW"]["enabled"])
        self.assertEqual(c["RP"]["selected_positions"], [])

    def test_risk_formulas_match_existing_frozen_first_round_function(self):
        path = Path(__file__).parents[2]/"odr003_exploratory_20261023/implementation/risk_metrics.py"
        spec = importlib.util.spec_from_file_location("old_risk_fixture", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        o, a, p = rows(.9), rows(.6), rows(.7)
        for i in range(0, 512, 7): p[i].update(error=1.8, rotation_deg=13.)
        actual = evaluate_fixed_reference(self.ids, o, a, {"P": p}, expected_reference_count=512, fixed_reference_ids=self.ids)["methods"]["P"]
        normalized = [{row["sample_id"]: module._normalize(row) for row in data} for data in (o,a,p)]
        old = module._one_method(self.ids, self.ids, *normalized, .01)
        for old_name, new_name in (("mean_error", "mean_error"), ("mean_positive_degradation", "mean_D"),
                                   ("worst_5pct_degradation", "tail_D"), ("negative_transfer_rate", "negative_transfer_rate")):
            self.assertEqual(old["metrics"][old_name], actual["metrics"][new_name])

    def test_reference_same_count_different_ID_or_order_is_rejected(self):
        for ref in (self.ids[::-1], self.ids[:-1]+['unknown'], self.ids[:-1]+[self.ids[0]]):
            with self.assertRaisesRegex(ValueError, 'reference IDs'):
                evaluate_fixed_reference(self.ids, rows(), rows(), {'P': rows()},
                                         expected_reference_count=512, fixed_reference_ids=ref)

    def test_missing_new_O_does_not_shrink_reference_or_zero_fill_D(self):
        o = rows(); o[4].update(valid=False, error=None)
        result = evaluate_fixed_reference(self.ids, o, rows(), {'P': rows()},
                                         expected_reference_count=512, fixed_reference_ids=self.ids)
        self.assertEqual(result['reference_count'], 512)
        self.assertEqual(result['methods']['P']['missing_reference_ids'], [self.ids[4]])
        self.assertIsNone(result['methods']['P']['metrics']['mean_D'])
        self.assertIsNone(result['methods']['P']['per_sample'][4]['positive_degradation'])

    def test_saved_backend_flags_cannot_be_just_a_success_claim(self):
        from deterministic_runtime import validate_saved_receipt
        actual = runtime_receipt()
        self.assertEqual(validate_saved_receipt(actual), actual)
        for key, value in [('cudnn_benchmark', True), ('cuda_matmul_allow_tf32', True), ('warn_only', True),
                           ('torch_deterministic_algorithms', False), ('cuda_version', None)]:
            bad = copy.deepcopy(actual); bad[key] = value
            with self.assertRaises(ValueError): validate_saved_receipt(bad)

    def baseline_state(self, method):
        import torch
        n = 1024
        records = [dict(sample_id=str(i)) for i in range(n)]
        accepted = n if method == 'A' else n//2
        learning = dict(model={'weight': torch.ones(2)}, buffers={}, boundary_buffers={},
            gradients={'weight': None}, requires_grad={'weight': True}, modes={'': False},
            optimizer={'state': {0: {'step': torch.tensor(accepted//4)}}, 'param_groups': [{'params':[0]}]},
            optimizer_type='torch.optim.adamw.AdamW', optimizer_parameter_names=[['weight']], scaler=None,
            bn_features={'last_feature': torch.ones(1)}, bn_mean={}, bn_mean_sq={}, rng={},
            counters=dict(observed=n, accepted=accepted, updates=accepted//4, tail=0))
        trace = []
        for i, record in enumerate(records):
            keep = method == 'A' or i%2 == 0
            trace.append(dict(id=f'observation:{i:04d}', kind='observation', observation_index=i,
                sample_id=record['sample_id'], accepted=keep, gate_reason='injected_gate' if method=='A' else 'fixed',
                gate_score=None if method=='A' else float(keep), arrival={'keep':keep}))
        state = dict(schema='odr003-round2-original-g50-reference-v1', finished=True, next_index=n,
            identity={'run':'bound'}, stream_records=records, learning=learning, observation_trace=trace,
            finish_counts=dict(observed=n,accepted=accepted,rejected=n-accepted,updates=accepted//4,used=accepted,tail=0))
        return state, records

    def test_original_A_all_inputs_and_G50_gate_recomputed(self):
        for method in ('A', 'G50'):
            state, records = self.baseline_state(method)
            result = verify_original_baseline_state(state, expected_stream=records, identity={'run':'bound'},
                method=method, calibration={}, gate_decider=lambda name,a,c:(a['keep'],'fixed',float(a['keep'])))
            self.assertEqual(result['counters']['observed'], 1024)
            self.assertEqual(result['counters']['commits'], 256 if method=='A' else 128)
            bad = copy.deepcopy(state); bad['observation_trace'][0]['accepted'] = False
            with self.assertRaises(ValueError): verify_original_baseline_state(bad, expected_stream=records,
                identity={'run':'bound'}, method=method, calibration={},
                gate_decider=lambda name,a,c:(a['keep'],'fixed',float(a['keep'])))

    def test_original_baseline_cannot_claim_updates_with_stale_optimizer(self):
        state, records = self.baseline_state('A'); state['learning']['optimizer']['state'] = {}
        with self.assertRaisesRegex(ValueError, 'optimizer state'):
            verify_original_baseline_state(state,expected_stream=records,identity={'run':'bound'},method='A',
                                           calibration={},gate_decider=lambda *args:True)

    def test_O_cannot_contain_observations_optimizer_steps_or_pending_cache(self):
        state, _ = self.baseline_state('A')
        learning = state['learning']; learning.update(bn_features={}, optimizer={**learning['optimizer'],'state':{}},
            counters=dict(observed=0,accepted=0,updates=0,tail=0))
        zero = dict(observed=0,accepted=0,candidates=0,commits=0,rollbacks=0,tail_discarded=0,
                    consecutive_rollbacks=0,max_consecutive_rollbacks=0)
        offline=dict(schema='odr003-round2-offline-source-v1',finished_source_checkpoint=True,next_index=0,
                     identity={'run':'bound'},learning=learning,counters=zero)
        validate_offline_checkpoint(offline)
        for path in ('counters', 'optimizer', 'cache'):
            bad=copy.deepcopy(offline)
            if path=='counters':bad['counters']['observed']=1
            elif path=='optimizer':bad['learning']['optimizer']['state']={0:{'step':1}}
            else:bad['learning']['bn_features']={'pending':1}
            with self.assertRaises(ValueError):validate_offline_checkpoint(bad)

    def test_frozen_reference_rechecked_against_actual_original_summary_and_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            def save(name,value):
                path=directory/name;path.write_text(json.dumps(value),encoding='utf8')
                return dict(path=str(path),sha256=sha256(path))
            manifest=save('developer_images.json',dict(items=[{'sample_id':sid} for sid in self.ids]))
            origin=save('summary.json',dict(domains=dict(lightbox=dict(streams={'2021':dict(reference_ids=self.ids)}))))
            proof=save('proof.json',dict(status='passed',summary_sha256=origin['sha256']))
            doc=dict(schema='odr003-round2-fixed-reference-v1',domain='lightbox',stream_seed=2021,
                reference_ids=self.ids,reference_count=512,developer_manifest=manifest,
                developer_ids_sha256=canonical_sha(self.ids),origin_summary=origin,origin_proof=proof,
                origin_reference_path=['domains','lightbox','streams','2021','reference_ids'])
            binding=save('reference.json',doc)
            self.assertEqual(checked_fixed_reference(binding,domain='lightbox',seed=2021,
                developer_ids=self.ids,developer_manifest=manifest,cache={}),self.ids)
            wrong=copy.deepcopy(doc);wrong['reference_ids']=self.ids[::-1]
            with self.assertRaises(ValueError):checked_fixed_reference(save('bad.json',wrong),domain='lightbox',seed=2021,
                developer_ids=self.ids,developer_manifest=manifest,cache={})
            wrong=copy.deepcopy(doc);wrong['origin_reference_path']=['domains','lightbox','streams','2022','reference_ids']
            with self.assertRaises(ValueError):checked_fixed_reference(save('bad2.json',wrong),domain='lightbox',seed=2021,
                developer_ids=self.ids,developer_manifest=manifest,cache={})
            save('summary.json',{'tampered':True})
            with self.assertRaises(ValueError):checked_fixed_reference(binding,domain='lightbox',seed=2021,
                developer_ids=self.ids,developer_manifest=manifest,cache={})

    def test_actual_O_source_tensor_bytes_must_match_not_only_source_flag(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);source=directory/'source.pt'
            state,_=self.baseline_state('A');learning=state['learning']
            learning.update(bn_features={},optimizer={**learning['optimizer'],'state':{}},
                counters=dict(observed=0,accepted=0,updates=0,tail=0))
            torch.save(learning['model'],source);source_binding=dict(path=str(source),sha256=sha256(source))
            receipt=directory/'receipt.json'
            receipt.write_text(json.dumps(dict(status='passed',strict_model_load_performed=True,
                source_offline_identity_verified=True,source_sha256=source_binding['sha256'],state_dict_key=None)))
            zero=dict(observed=0,accepted=0,candidates=0,commits=0,rollbacks=0,tail_discarded=0,
                      consecutive_rollbacks=0,max_consecutive_rollbacks=0)
            offline=dict(schema='odr003-round2-offline-source-v1',finished_source_checkpoint=True,next_index=0,
                         identity={'bound':'source'},learning=learning,counters=zero)
            receipt_binding=dict(path=str(receipt),sha256=sha256(receipt))
            with patch('verify_complete.frozen_protocol',return_value={'source_sha256':source_binding['sha256']}):
                result=verify_offline_source_state(offline,identity={'bound':'source'},source_binding=source_binding,
                    official_receipt=receipt_binding,cache={})
                self.assertTrue(result['all_model_tensors_equal_official_source'])
                bad=copy.deepcopy(offline);bad['learning']['model']['weight'][0]+=1
                with self.assertRaisesRegex(ValueError,'source_tensors'):
                    verify_offline_source_state(bad,identity={'bound':'source'},source_binding=source_binding,
                        official_receipt=receipt_binding,cache={})


if __name__ == "__main__": unittest.main()
