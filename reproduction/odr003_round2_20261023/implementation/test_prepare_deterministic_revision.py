"""Metadata preparation contracts: old runs stay untouched, scope fails closed."""
import json
from pathlib import Path
import tempfile
import unittest

from frozen_backend import sha256
from prepare_deterministic_revision import prepare, REVISION
from prepare_round2 import FROZEN_NAMES


class PrepareDeterministicTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        (self.root/'implementation').mkdir(); (self.root/'local_validation').mkdir()
        names=('run_round2.py','offline_baseline.py','deterministic_runtime.py','guarded_entry.py',
               'queue_round2.py','verify_engineering_phase.py','verify_complete.py','summarize_round2.py')
        for name in names:(self.root/'implementation'/name).write_text('# bound synthetic metadata\n')
        code={p.name:sha256(p) for p in (self.root/'implementation').glob('*.py')}
        self.write('local_validation/contract_report.json',dict(status='passed',implementation_sha256=code))
        self.write('confirmation.json',dict(status='confirmed_implementation_authorized',maximum_complete_odr=30))
        tasks={f'{d}_{s}':dict(output_dir=f'/original/runs/G50_{d}_{s}',
                              code_sha256={name+'.py':'a'*64 for name in FROZEN_NAMES})
               for d in ('lightbox','sunlamp') for s in (2021,2022,2023)}
        self.write('baseline_bindings.json',dict(tasks=tasks))
        self.inputs=dict(confirmation={'path':'/confirmed','sha256':'b'*64},
                         fixed_reference_bindings={f'{d}:{s}':{'path':f'/fixed/{d}/{s}','sha256':'c'*64}
                           for d in ('lightbox','sunlamp') for s in (2021,2022,2023)},
                         prior_execution_history={'path':'/old/history','sha256':'d'*64},
                         execution_policy={'path':'/policy','sha256':'e'*64},synthetic_contracts_sha256='f'*64)
        self.candidate=self.root/'candidate'

    def tearDown(self):self.temp.cleanup()
    def write(self,name,value): (self.root/name).write_text(json.dumps(value),encoding='utf8')
    def test_unique_new_engineering_preserves_source_bindings_and_old_runs(self):
        old=self.root/'runs/old';old.mkdir(parents=True);(old/'report.json').write_text('preserved')
        result=prepare(self.root,self.candidate,self.inputs)
        self.assertEqual(result['tasks'],8);self.assertTrue(result['no_production_mutation'])
        self.assertEqual((old/'report.json').read_text(),'preserved')
        self.assertFalse((self.root/'engineering_index.json').exists())
        for path in (self.candidate/'tasks'/REVISION).glob('*.json'):
            task=json.loads(path.read_text());self.assertEqual(task['definition']['observations'],128)
            self.assertNotIn('initial_checkpoint_import_required',task)
            self.assertIn('/deterministic_20261024/',task['output_dir'])
        self.assertEqual(prepare(self.root,self.candidate,self.inputs),result)
    def test_changed_code_needs_new_contracts(self):
        (self.root/'implementation/run_round2.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'local contracts'):prepare(self.root,self.candidate,self.inputs)
    def test_old_18_run_confirmation_cannot_authorize_expansion(self):
        self.write('confirmation.json',dict(status='confirmed_implementation_authorized',maximum_complete_odr=18))
        with self.assertRaisesRegex(ValueError,'explicit confirmation'):prepare(self.root,self.candidate,self.inputs)
    def test_missing_fixed_reference_pair_is_not_recreated(self):
        self.inputs['fixed_reference_bindings'].pop('sunlamp:2023')
        with self.assertRaisesRegex(ValueError,'six unchanged'):prepare(self.root,self.candidate,self.inputs)
    def test_existing_different_candidate_is_not_replaced(self):
        prepare(self.root,self.candidate,self.inputs)
        (self.candidate/'engineering_index.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Refusing to replace'):prepare(self.root,self.candidate,self.inputs)


if __name__=='__main__':unittest.main()
