"""Offline source/evaluation recovery on synthetic saved records only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from durable_state import DurableRunStore
from frozen_backend import sha256
import offline_baseline as offline
import run_round2
from test_run_round2 import FakeDeveloperDataset, FakePrediction


class FakeLearner:
    def __init__(self):
        self.adapter = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: {'fixed': 7}))
        self.restores = []

    def _capture_learning(self):
        return dict(model={'fixed': 7}, buffers={}, optimizer={'state': {}, 'param_groups': []},
                    optimizer_type='synthetic-no-optimizer-work', optimizer_parameter_names=[],
                    modes={}, requires_grad={}, gradients={}, bn_features={}, bn_mean={},
                    bn_mean_sq={}, boundary_buffers={}, rng={'synthetic': 1}, scaler=None,
                    counters=dict(observed=0, accepted=0, updates=0, tail=0))

    def _restore_learning(self, state):
        self.restores.append(copy.deepcopy(state))


class OfflineBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = dict(run_id='synthetic-offline', source_sha256='a'*64,
                             config_sha256='b'*64, manifest_sha256='c'*64)
        self.store = DurableRunStore(self.root/'durable', self.identity,
                                    state_validator=offline.validate_offline_state)
        self.learner = FakeLearner()
        self.dataset = FakeDeveloperDataset()
        self.labels = self.root/'synthetic_dev_labels.json'
        self.write_json(self.labels, dict(image_manifest_sha256='d'*64,
            items=[dict(sample_id=row['sample_id'], quaternion_wxyz=[1,0,0,0], translation_m=[0,0,10])
                   for row in self.dataset.manifest['items']]))
        self.task = dict(baseline_task=dict(bindings=dict(
            developer_labels=dict(path=str(self.labels), sha256=sha256(self.labels)),
            developer_manifest=dict(path='synthetic-unused-manifest', sha256='d'*64))))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding='utf8')

    def evaluate(self, prediction):
        saved = offline.source_checkpoint(self.store, self.learner, self.identity)
        with patch.object(run_round2, 'atomic_json', side_effect=self.write_json), \
             patch.object(run_round2, '_progress'):
            return run_round2.evaluate_developer(self.task,
                dict(developer_labels=self.labels, developer_manifest='synthetic-unused-manifest'),
                None, None, None, prediction,
                SimpleNamespace(ImageManifestDataset=lambda *a, **k: self.dataset), None,
                self.learner.adapter.model, None, SimpleNamespace(phase='source'), self.store,
                saved.state, self.root, lambda: None, lambda: None)

    def test_initial_complete_source_checkpoint_has_zero_observations_and_updates(self):
        saved = offline.source_checkpoint(self.store, self.learner, self.identity)
        self.assertEqual(saved.scientific_cursors, offline.ZERO_CURSORS)
        self.assertEqual(saved.state['counters'], offline.ZERO_COUNTS)
        self.assertTrue(saved.state['finished_source_checkpoint'])
        self.assertEqual(saved.info.generation, 1)
        restored = offline.source_checkpoint(self.store, self.learner, self.identity)
        self.assertEqual(restored.info.generation, 1)
        self.assertEqual(len(self.learner.restores), 1)

    def test_offline_evaluation_recovery_charges_replays_but_no_odr_observations(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            self.evaluate(FakePrediction(fail_call=71))
        saved = self.store.recover()
        self.assertEqual(saved.state['evaluation']['next_index'], 64)
        self.assertEqual(saved.scientific_cursors, offline.ZERO_CURSORS)
        self.dataset.requested.clear()
        resumed = FakePrediction()
        report = self.evaluate(resumed)
        self.assertEqual(self.dataset.requested[0], 64)
        self.assertEqual(resumed.calls, 448)
        self.assertEqual(report['count'], 512)
        final = self.store.recover()
        self.assertEqual(final.scientific_cursors, offline.ZERO_CURSORS)
        self.assertEqual(final.state['counters'], offline.ZERO_COUNTS)
        self.assertEqual(final.state['learning']['model'], {'fixed': 7})
        self.assertEqual(self.store.work_summary()['forwards_confirmed_completed'], {'developer_evaluation': 518})
        self.assertEqual(self.store.work_summary()['replay_forwards_confirmed_completed'], {'developer_evaluation': 7})
        repeated = FakePrediction()
        self.evaluate(repeated)
        self.assertEqual(repeated.calls, 0)

    def test_saved_source_mutation_is_rejected_before_evaluation(self):
        saved = offline.source_checkpoint(self.store, self.learner, self.identity)
        bad = copy.deepcopy(saved.state); bad['learning']['model']['fixed'] = 8
        self.store.save(bad, offline.ZERO_CURSORS)
        with self.assertRaisesRegex(ValueError, 'freshly strict-loaded source'):
            offline.source_checkpoint(self.store, self.learner, self.identity)

    def test_offline_state_rejects_any_adaptation(self):
        saved = offline.source_checkpoint(self.store, self.learner, self.identity).state
        for mutate in (
            lambda state: state['counters'].update(observed=1),
            lambda state: state['counters'].update(observed=False),
            lambda state: state.update(next_index=1),
            lambda state: state.update(next_index=False),
            lambda state: state['learning']['counters'].update(updates=1),
            lambda state: state['learning']['counters'].update(updates=False),
            lambda state: state['learning']['optimizer']['state'].update(parameter={'step': 1}),
            lambda state: state['learning']['gradients'].update(parameter=1),
        ):
            state = copy.deepcopy(saved); mutate(state)
            with self.assertRaises(ValueError): offline.validate_offline_state(state)

    def test_admission_failure_prevents_even_backend_preparation(self):
        with patch.object(offline, 'prepare_environment', side_effect=AssertionError('not admitted')) as prepare:
            with self.assertRaisesRegex(RuntimeError, 'admission'):
                offline.execute('unused-task', admission={}, launch_record={}, conflict_check=lambda: None)
            prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
