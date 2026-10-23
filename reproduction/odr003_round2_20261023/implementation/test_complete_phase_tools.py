"""Metadata-only autonomous continuation tests; never execute a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import complete_phase_tools as tools
from round2_protocol import build_complete_matrix, frozen_protocol


class CompletePreparationContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan = Path(self.tmp.name)
        self.patcher = patch.object(tools, 'atomic_json', side_effect=self.write)
        self.patcher.start()
        self.protocol = frozen_protocol()
        self.write(self.plan/'confirmation.json', {'status':'confirmed_implementation_authorized'})
        self.write(self.plan/'tasks/eng.json', {'frozen_modules':{'unchanged':{'path':'frozen.py','sha256':'f'*64}}})
        old = self.plan/'old'
        tasks, runs, inputs = {}, [], []
        for definition in build_complete_matrix()[:6]:
            domain, seed = definition['domain'], definition['stream_seed']
            baseline = dict(output_dir=str(old/f'G50_{domain}_{seed}'), method='G50',
                            domain=domain, seed=seed, bindings={'stream_manifest':{'path':'frozen-stream','sha256':'a'*64}})
            tasks[f'{domain}_{seed}'] = baseline
            for group in ('O', 'A', 'G50'):
                if group == 'O' and seed != 2021: continue
                stream = 'offline' if group == 'O' else str(seed)
                directory = old/f'{group}_{domain}_{stream}'
                self.write(directory/'task.json', dict(method=group, domain=domain, seed=seed))
                self.write(directory/'report.json', dict(status='main_complete', method=group))
                self.write(directory/'proof.json', dict(status='passed', report_sha256=tools.sha256(directory/'report.json')))
                self.write(directory/'predictions.jsonl', dict(saved_prediction=True))
                entry = dict(run_id=directory.name, method=group, domain=domain, stream=stream)
                entry.update({key:tools.binding(directory/name) for key,name in
                              [('task','task.json'),('report','report.json'),('proof','proof.json'),('predictions','predictions.jsonl')]})
                runs.append(entry)
                inputs.append(dict(path=str(directory), report_sha256=entry['report']['sha256'], proof_sha256=entry['proof']['sha256']))
        self.write(self.plan/'baseline_bindings.json', {'tasks':tasks})
        self.write(self.plan/'baseline_result_bindings.json', dict(source_sha256=self.protocol['source_sha256'], runs=runs, measurement_repeated=False))
        self.write(old/'summary.json', dict(status='accepted', inputs=inputs))
        self.write(old/'proof.json', dict(status='passed', summary_sha256=tools.sha256(old/'summary.json')))
        self.index = dict(implementation_sha256={}, tasks=[{'task_path':str(self.plan/'tasks/eng.json')}],
                          first_round_acceptance=dict(summary=tools.binding(old/'summary.json'), proof=tools.binding(old/'proof.json')))
        self.write(self.plan/'preflight/engineering_proof.json',
                   dict(status='passed', allow_complete_matrix=True, implementation_sha256={}, protocol_sha256=self.protocol['protocol_sha256']))

    def tearDown(self):
        self.patcher.stop(); self.tmp.cleanup()

    @staticmethod
    def write(path, data):
        path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf8')

    def accept(self, row, *, q=9, k=7):
        task = tools.read(row['task_path'])
        output = Path(task['output_dir'])
        report = dict(definition=row['definition'], counts={'candidates':q, 'commits':k},
                      protocol_sha256=self.protocol['protocol_sha256'], source_sha256=self.protocol['source_sha256'])
        self.write(output/'report.json', report)
        self.write(output/'proof.json', dict(status='passed', report_sha256=tools.sha256(output/'report.json')))
        return dict(report=tools.binding(output/'report.json'), proof=tools.binding(output/'proof.json'),
                    verification_attempt={'run_id':row['run_id']+'_verify', 'exit_code':0})

    def test_exact_six_P_frozen_and_engineering_required(self):
        path, compiled = tools.prepare_P(self.plan, self.index)
        self.assertEqual([r['definition'] for r in compiled['tasks']], build_complete_matrix()[:6])
        self.assertTrue(all(tools.read(row['task_path'])['stage']=='complete' for row in compiled['tasks']))
        self.assertEqual(tools.prepare_P(self.plan, self.index), (path, compiled))
        proof = tools.read(self.plan/'preflight/engineering_proof.json'); proof['allow_complete_matrix']=False
        self.write(self.plan/'preflight/engineering_proof.json', proof)
        with self.assertRaises(ValueError): tools.prepare_P(self.plan, self.index)

    def test_no_controls_before_six_independent_P_acceptances(self):
        _, compiled = tools.prepare_P(self.plan, self.index)
        accepted = {row['run_id']:self.accept(row) for row in compiled['tasks'][:5]}
        with self.assertRaises(ValueError): tools.prepare_controls(self.plan, self.index, compiled, accepted)
        self.assertFalse((self.plan/'complete/all_index.json').exists())

    def test_all_controls_exactly_eighteen_and_match_real_P_counts(self):
        _, P = tools.prepare_P(self.plan, self.index)
        accepted = {row['run_id']:self.accept(row) for row in P['tasks']}
        _, compiled = tools.prepare_controls(self.plan, self.index, P, accepted)
        self.assertEqual(len(compiled['tasks']), 18)
        for row in compiled['tasks'][6:]:
            task = tools.read(row['task_path'])
            self.assertEqual(task['control']['RW']['commit_probability'], 7/9)
            self.assertEqual(len(task['control']['RP']['selected_positions']), 28)
            self.assertFalse(task['control']['RW']['quota_filling'])
        self.assertEqual(compiled['skipped_controls'], [])

    def test_Q0_skips_RW_but_retains_real_zero_update_RP(self):
        _, P = tools.prepare_P(self.plan, self.index)
        accepted = {row['run_id']:self.accept(row, q=0 if i==0 else 9, k=0 if i==0 else 7)
                    for i,row in enumerate(P['tasks'])}
        _, compiled = tools.prepare_controls(self.plan, self.index, P, accepted)
        self.assertEqual(len(compiled['tasks']), 17)
        self.assertEqual(len(compiled['skipped_controls']), 1)
        self.assertEqual(compiled['skipped_controls'][0]['definition']['group'], 'RW')
        zero = [row for row in compiled['tasks'] if row['definition']['group']=='RP'
                and row['definition']['domain']=='lightbox' and row['definition']['stream_seed']==2021][0]
        self.assertEqual(tools.read(zero['task_path'])['control']['RP']['selected_positions'], [])

    def test_changed_P_proof_or_report_blocks_compilation(self):
        _, P = tools.prepare_P(self.plan, self.index)
        accepted = {row['run_id']:self.accept(row) for row in P['tasks']}
        self.write(accepted[P['tasks'][0]['run_id']]['report']['path'], {'tampered':True})
        with self.assertRaises(ValueError): tools.prepare_controls(self.plan, self.index, P, accepted)

    def test_independent_verifier_task_reuses_baselines_no_rerun(self):
        _, P = tools.prepare_P(self.plan, self.index)
        row = P['tasks'][0]
        receipt = dict(run_id=row['run_id'], attempts=[{'exit_code':75},{'exit_code':0}])
        path, name = tools.verify_task(self.plan, P, row, receipt, queue_wait_seconds=77.)
        task = tools.read(path)
        self.assertEqual(task['all_attempts'], receipt['attempts'])
        self.assertEqual(set(task['reusable_baselines']), {'O','A','G50'})
        self.assertEqual(task['action'], 'verify_complete')
        self.assertEqual(tools.verify_task(self.plan, P, row, receipt, queue_wait_seconds=999.), (path,name))
        self.assertEqual(tools.read(path)['queue_wait_seconds'], 77.)

    def test_summary_requires_all_proofs_and_stays_frozen_across_wait_attempts(self):
        _, P = tools.prepare_P(self.plan, self.index)
        accepted = {row['run_id']:self.accept(row) for row in P['tasks']}
        path, compiled = tools.prepare_controls(self.plan, self.index, P, accepted)
        for row in compiled['tasks'][6:]: accepted[row['run_id']] = self.accept(row)
        state=dict(complete_accepted=accepted, complete_main={key:{'run_id':key} for key in accepted},
                   attempts=[], queue_wait_seconds=42.)
        result = tools.summary_task(self.plan, path, compiled, state)
        state['queue_wait_seconds']=999.; state['attempts'].append({'status':'waiting_resources'})
        self.assertEqual(tools.summary_task(self.plan, path, compiled, state),result)
        state['complete_accepted'].pop(next(iter(accepted)))
        with self.assertRaises(ValueError): tools.summary_task(self.plan, path, compiled, state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
