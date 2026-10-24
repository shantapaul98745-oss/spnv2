"""Metadata-only deterministic baseline and finite continuation contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import complete_phase_tools as tools
from round2_protocol import build_baseline_matrix, build_complete_matrix, frozen_protocol


class CompletePreparationContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.plan = Path(self.tmp.name)
        self.patcher = patch.object(tools, 'atomic_json', side_effect=self.write); self.patcher.start()
        self.protocol = frozen_protocol()
        self.write(self.plan/'confirmation.json', {'status': 'confirmed'})
        self.write(self.plan/'tasks/eng.json', {'frozen_modules': {'unchanged': {'path': 'frozen.py', 'sha256': 'f'*64}}})
        originals = {f"{d['domain']}_{d['stream_seed']}": dict(method='G50', domain=d['domain'], seed=d['stream_seed'],
                     bindings={'stream_manifest': {'path': 'frozen-stream', 'sha256': 'a'*64}})
                     for d in build_complete_matrix()[:6]}
        self.write(self.plan/'baseline_bindings.json', {'tasks': originals})
        self.index = dict(implementation_sha256={}, tasks=[{'task_path': str(self.plan/'tasks/eng.json')}],
                          run_prefix='unique_det', fixed_reference_bindings={f"{d['domain']}:{d['stream_seed']}":
                            {'path': 'fixed-'+d['domain'], 'sha256': 'b'*64} for d in build_complete_matrix()[:6]})
        self.write(self.plan/'preflight/engineering_proof.json', dict(status='passed', allow_complete_matrix=True,
                   implementation_sha256={}, protocol_sha256=self.protocol['protocol_sha256']))
        self.baseline_path, self.baselines = tools.prepare_baselines(self.plan, self.index)
        self.baseline_accepted = {row['run_id']: self.accept(row) for row in self.baselines['tasks']}

    def tearDown(self):
        self.patcher.stop(); self.tmp.cleanup()

    @staticmethod
    def write(path, data):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf8')

    def accept(self, row, *, q=9, k=7):
        task = tools.read(row['task_path']); output = Path(task['output_dir'])
        self.write(output/'predictions.jsonl', {'saved_prediction': True})
        report = dict(definition=row['definition'], counts={'candidates': q, 'commits': k},
                      protocol_sha256=self.protocol['protocol_sha256'], source_sha256=self.protocol['source_sha256'],
                      outputs={'predictions': tools.binding(output/'predictions.jsonl')})
        self.write(output/'report.json', report)
        self.write(output/'proof.json', dict(status='passed', report_sha256=tools.sha256(output/'report.json'),
                   definition=row['definition'], task_binding={'path': row['task_path'], 'sha256': row['task_sha256']}))
        return dict(report=tools.binding(output/'report.json'), proof=tools.binding(output/'proof.json'),
                    verification_attempt={'run_id': row['run_id']+'_verify', 'exit_code': 0})

    def prepare_P(self):
        return tools.prepare_P(self.plan, self.index, self.baselines, self.baseline_accepted)

    def accepted_P(self, P, *, zero_first=False):
        accepted = copy.deepcopy(self.baseline_accepted)
        rows = [r for r in P['tasks'] if r['definition']['group'] == 'P']
        for i, row in enumerate(rows):
            accepted[row['run_id']] = self.accept(row, q=0 if zero_first and i == 0 else 9,
                                                  k=0 if zero_first and i == 0 else 7)
        return accepted

    def test_exact_fourteen_new_baselines_and_two_offline_actions(self):
        self.assertEqual([r['definition'] for r in self.baselines['tasks']], build_baseline_matrix())
        tasks = [tools.read(r['task_path']) for r in self.baselines['tasks']]
        self.assertEqual(sum(t['action'] == 'run_offline' and t['stage'] == 'offline' for t in tasks), 2)
        self.assertEqual(sum(t['stage'] == 'complete' for t in tasks), 12)
        self.assertTrue(all(t['baseline_task']['method'] == 'G50' for t in tasks))
        self.assertEqual(tools.prepare_baselines(self.plan, self.index), (self.baseline_path, self.baselines))

    def test_exact_six_P_only_after_fourteen_accepted_and_engineering(self):
        with self.assertRaises(ValueError): tools.prepare_P(self.plan, self.index)
        missing = dict(self.baseline_accepted); missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError): tools.prepare_P(self.plan, self.index, self.baselines, missing)
        path, P = self.prepare_P()
        self.assertEqual(len(P['tasks']), 20)
        self.assertEqual([r['definition'] for r in P['tasks'][14:]], build_complete_matrix()[:6])
        self.assertEqual(self.prepare_P(), (path, P))
        proof = tools.read(self.plan/'preflight/engineering_proof.json'); proof['allow_complete_matrix'] = False
        self.write(self.plan/'preflight/engineering_proof.json', proof)
        with self.assertRaises(ValueError): self.prepare_P()

    def test_reuse_contains_current_fourteen_only_and_no_historical_substitution(self):
        _, P = self.prepare_P(); doc = tools.checked(P['reusable_baselines'])
        self.assertEqual(doc['schema'], 'odr003-round2-reused-baselines-v2')
        self.assertEqual(len(doc['baseline_proofs']), 14)
        self.assertFalse(doc['historical_results_reused_as_new_backend'])
        for pair in doc['pairs'].values():
            self.assertEqual(set(pair), {'O', 'A', 'G50'})
            self.assertTrue(all(Path(v['task']['path']).name.startswith('unique_det_') for v in pair.values()))

    def test_no_controls_before_six_independent_P_acceptances(self):
        _, P = self.prepare_P(); accepted = self.accepted_P(P)
        accepted.pop(P['tasks'][-1]['run_id'])
        with self.assertRaises(ValueError): tools.prepare_controls(self.plan, self.index, P, accepted)
        self.assertFalse((self.plan/'complete/all_index.json').exists())

    def test_full_matrix_thirty_odr_plus_two_O_and_real_P_budgets(self):
        _, P = self.prepare_P(); accepted = self.accepted_P(P)
        _, all_runs = tools.prepare_controls(self.plan, self.index, P, accepted)
        self.assertEqual(len(all_runs['tasks']), 32)
        self.assertEqual(sum(r['definition']['group'] != 'O' for r in all_runs['tasks']), 30)
        for row in all_runs['tasks'][20:]:
            task = tools.read(row['task_path'])
            self.assertEqual(task['control']['RW']['commit_probability'], 7/9)
            self.assertEqual(len(task['control']['RP']['selected_positions']), 28)
            self.assertFalse(task['control']['RW']['quota_filling'])

    def test_Q0_skips_only_paired_RW_retaining_zero_RP(self):
        _, P = self.prepare_P(); accepted = self.accepted_P(P, zero_first=True)
        _, all_runs = tools.prepare_controls(self.plan, self.index, P, accepted)
        self.assertEqual(len(all_runs['tasks']), 31)
        self.assertEqual(len(all_runs['skipped_controls']), 1)
        self.assertEqual(all_runs['skipped_controls'][0]['definition']['group'], 'RW')
        zero = [r for r in all_runs['tasks'] if r['definition']['group'] == 'RP'
                and r['definition']['domain'] == 'lightbox' and r['definition']['stream_seed'] == 2021][0]
        self.assertEqual(tools.read(zero['task_path'])['control']['RP']['selected_positions'], [])

    def test_changed_current_baseline_or_P_report_blocks_compilation(self):
        row = self.baselines['tasks'][0]
        self.write(self.baseline_accepted[row['run_id']]['report']['path'], {'tampered': True})
        with self.assertRaises(ValueError): self.prepare_P()

    def test_verifier_baselines_need_no_reuse_and_P_binds_new_reuse_and_reference(self):
        row = self.baselines['tasks'][0]; receipt = dict(run_id=row['run_id'], attempts=[{'exit_code': 0}])
        path, name = tools.verify_task(self.plan, self.baselines, row, receipt, queue_wait_seconds=77)
        self.assertIsNone(tools.read(path)['reusable_baselines'])
        _, P = self.prepare_P(); row = P['tasks'][14]
        path, name = tools.verify_task(self.plan, P, row, receipt, queue_wait_seconds=77)
        task = tools.read(path)
        self.assertEqual(set(task['reusable_baselines']), {'O', 'A', 'G50'})
        self.assertEqual(task['fixed_reference_binding'], self.index['fixed_reference_bindings']['lightbox:2021'])
        self.assertEqual(tools.verify_task(self.plan, P, row, receipt, queue_wait_seconds=999), (path, name))
        self.assertEqual(tools.read(path)['queue_wait_seconds'], 77)

    def test_summary_requires_all32_proofs_and_explicit_baseline14(self):
        _, P = self.prepare_P(); accepted = self.accepted_P(P)
        path, all_runs = tools.prepare_controls(self.plan, self.index, P, accepted)
        for row in all_runs['tasks'][20:]: accepted[row['run_id']] = self.accept(row)
        state = dict(complete_accepted=accepted, complete_main={k: {'run_id': k} for k in accepted},
                     attempts=[], queue_wait_seconds=42.)
        task_path, _ = tools.summary_task(self.plan, path, all_runs, state)
        task = tools.read(task_path)
        self.assertEqual(len(task['run_proofs']), 32); self.assertEqual(len(task['baseline_proofs']), 14)
        self.assertEqual(task['baseline_reuse'], all_runs['reusable_baselines'])
        state['queue_wait_seconds'] = 999
        self.assertEqual(tools.summary_task(self.plan, path, all_runs, state)[0], task_path)
        state['complete_accepted'].pop(next(iter(accepted)))
        with self.assertRaises(ValueError): tools.summary_task(self.plan, path, all_runs, state)


if __name__ == '__main__': unittest.main(verbosity=2)
