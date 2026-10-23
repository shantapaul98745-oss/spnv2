"""Synthetic record/checkpoint tests; no SPNv2 model or server use."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from durable_state import DurableRunStore
import preforward_repair as repair
from frozen_backend import canonical_sha, sha256
from round2_protocol import frozen_protocol


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2), encoding='utf8')
    return dict(path=str(path.resolve()), sha256=sha256(path))


class Fixture:
    def __init__(self, root):
        self.plan = Path(root).resolve() / 'plan'; self.plan.mkdir()
        self.output = self.plan / 'runs' / repair.RUN_ID; self.output.mkdir(parents=True)
        self.source = self.plan / 'assets/source.pt'; self.source.parent.mkdir()
        model = {'weight': torch.tensor([1., 2.]), 'running_mean': torch.tensor([0.])}
        torch.save(model, self.source)
        self.source_sha = sha256(self.source)
        receipt = write(self.plan / 'receipt.json', dict(status='passed', strict_model_load_performed=True,
             source_offline_identity_verified=True, source_sha256=self.source_sha, source_path=str(self.source), state_dict_key=None))
        implementation = {}
        files = {}
        for name in ('frozen_backend.py', 'run_round2.py', 'reference_driver.py'):
            file = self.plan / 'old_implementation' / name; file.parent.mkdir(exist_ok=True)
            file.write_text('# original preserved ' + name, encoding='utf8')
            files[name] = dict(path=str(file), sha256=sha256(file)); implementation[name] = sha256(file)
        self.launch_task = self.plan / 'tasks' / (repair.RUN_ID + '.json')
        self.task = dict(run_id=repair.RUN_ID, stage='engineering', output_dir=str(self.output),
            definition=dict(trajectory='G50_reference', domain='lightbox', stream_seed=2021, observations=128,
                            protection_enabled=False),
            protocol=dict(source_sha256=self.source_sha, fixed_rules=True),
            baseline_task=dict(bindings=dict(source=dict(path=str(self.source), sha256=self.source_sha),
                official_receipt=receipt, stream_manifest=dict(path=str(self.plan / 'stream.json'), sha256='a' * 64))),
            frozen_modules={'state_adapter': {'path': '/unchanged', 'sha256': 'b' * 64}},
            confirmation_sha256='c' * 64, implementation_sha256=implementation)
        old_task = write(self.output / 'task.json', self.task)
        self.identity = repair._identity(self.task, old_task)
        self.store_path = self.output / 'durable'
        self.store = DurableRunStore(self.store_path, self.identity, state_validator=repair._zero_state)
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=[])
        learning = dict(model=model, buffers={'running_mean': model['running_mean'].clone()},
            boundary_buffers={'running_mean': model['running_mean'].clone()},
            optimizer={'state': {}, 'param_groups': [{'params': [0], 'lr': .0001}]},
            optimizer_type='torch.optim.adamw.AdamW', optimizer_parameter_names=[['weight']],
            modes={'': False}, requires_grad={'weight': True}, gradients={'weight': None},
            bn_features={}, bn_mean={}, bn_mean_sq={}, rng=rng, scaler=None,
            counters=dict(accepted=0, observed=0, updates=0, tail=0))
        provider = dict(schema='odr003-round2-image-provider-v1', manifest_sha256='a' * 64,
            manifest_semantic_sha256='d' * 64, preprocess_sha256='e' * 64, tensor_sha256={},
            data_rng_policy='deterministic_preprocessing_no_RNG_consumption')
        self.state = dict(schema='odr003-round2-original-g50-reference-v1', identity=self.identity,
            stream_records=[dict(sample_id='image%04d' % i, image_sha256='f' * 64, preprocess_sha256='e' * 64) for i in range(128)],
            learning=learning, provider_state=provider, next_index=0, observation_trace=[], finished=False,
            finish_counts=None, ledger_boundary=self.store.journal_boundary())
        info = self.store.save(self.state, repair.ZERO_CURSORS)
        self.store.append_event('checkpoint_published', dict(generation=1, reason='initial', checkpoint_seconds=.01,
            scientific_cursors=repair.ZERO_CURSORS.copy(), timed_scope='state_capture_and_complete_generation_save_with_fsync'))
        self.read_attempt = self.store.begin_work('image_read', {}, logical_key='observation:0000:read')
        log = self.plan / 'logs' / 'failed.log'; log.parent.mkdir()
        log.write_text('Recorded first read failed before arrival scoring\n', encoding='utf8')
        exitcode = Path(str(log) + '.exitcode'); exitcode.write_text('1\n', encoding='ascii')
        launch = dict(run_id=repair.RUN_ID, run_nonce=repair.FAILED_NONCE,
            project_id='odr003_round2_20261023', role='worker', log_path=str(log),
            argv=['python', 'guarded_entry.py', '--task-json', str(self.launch_task)], pid=123, starttime=456,
            boot_id='synthetic-boot', session='synthetic-screen')
        launch_dir = self.plan / 'launches' / repair.FAILED_NONCE
        failure = dict(status='failed', type='RuntimeError', message=repair.FAILURE_MESSAGE,
            traceback='run_round2.py -> reference_driver.py -> frozen_backend.py: ' + repair.FAILURE_MESSAGE,
            execution_attempt_elapsed_seconds=3.3701783306896687,
            peak_gpu_allocated_bytes=63475712, peak_gpu_reserved_bytes=77594624)
        outcome = dict(status='failed', exit_code=1, execution_attempt_elapsed_seconds=3.983125071041286,
                       peak_gpu_allocated_bytes=63475712, peak_gpu_reserved_bytes=77594624)
        artifacts = dict(launch=write(launch_dir / 'launch.json', launch),
            outcome=write(launch_dir / 'outcome.json', outcome), failure=write(self.output / 'failure.json', failure),
            log=dict(path=str(log), sha256=sha256(log)), exit_code=dict(path=str(exitcode), sha256=sha256(exitcode)))
        self.spec = dict(schema=repair.SPEC_SCHEMA, failed_nonce=repair.FAILED_NONCE, run_id=repair.RUN_ID,
            plan_root=str(self.plan), old_task=old_task, old_output_dir=str(self.output), generation=1,
            old_store_files=repair._tree_files(self.store_path), old_implementation_files=files,
            artifacts=artifacts, original_task_launch_path=str(self.launch_task))
        self.patches = [patch.object(repair, 'SOURCE_SHA256', self.source_sha),
            patch.object(repair, 'OLD_TASK_SHA256', old_task['sha256']),
            patch.object(repair, 'OLD_INSTALLED_TASK_SHA256', old_task['sha256']),
            patch.object(repair, 'OLD_STATE_SHA256', sha256(info.path / 'state.bin')),
            patch.object(repair, 'OLD_METADATA_SHA256', sha256(info.path / 'metadata.json'))]
        for p in self.patches: p.start()
        self.spec_binding = write(self.plan / 'revision/spec.json', self.spec)
        self.new_task = copy.deepcopy(self.task)
        self.new_task.update(output_dir=str(self.plan / 'runs/preprocess_rng_fix_20261024' / repair.RUN_ID),
             implementation_sha256={**implementation, 'frozen_backend.py': '0' * 64},
             execution_revision='preprocess_rng_fix_20261024', initial_checkpoint_import_required=True,
             initial_checkpoint_origin=self.spec_binding)
        self.new_binding = write(self.launch_task, self.new_task)
        self.newstore = Path(self.new_task['output_dir']) / 'durable'

    def close(self):
        for p in reversed(self.patches): p.stop()


class PreforwardRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.fx = Fixture(self.tmp.name)

    def tearDown(self):
        self.fx.close(); self.tmp.cleanup()

    def test_preserved_failure_zero_forward_but_unknown_read_cost(self):
        before = repair._tree_files(self.fx.store_path)
        result = repair.verify_zero_progress_failure(self.fx.spec)
        self.assertEqual(result['actual_exit_code'], 1)
        self.assertEqual(result['actual_outcome']['status'], 'failed')
        self.assertEqual(result['physical_work']['attempts_started'], 1)
        self.assertEqual(result['physical_work']['attempts_finished'], 0)
        self.assertTrue(result['physical_work']['compute_time_incomplete'])
        self.assertEqual(result['zero_forward_proof']['model_forward_upper_bound'], 0)
        self.assertEqual(result['zero_forward_proof']['unconfirmed_non_model_work'][0]['attempt_id'], self.fx.read_attempt)
        self.assertEqual(result['retained_costs']['execution_attempt_elapsed_seconds']['failure'], 3.3701783306896687)
        self.assertEqual(result['retained_costs']['execution_attempt_elapsed_seconds']['entry_outcome'], 3.983125071041286)
        self.assertEqual(repair._tree_files(self.fx.store_path), before)

    def test_import_and_final_reverification_preserve_complete_learning(self):
        old = repair._tree_files(self.fx.store_path)
        proof = repair.import_zero_checkpoint(self.fx.spec_binding, self.fx.new_binding, self.fx.newstore)
        binding = write(self.fx.plan / 'revision/proof.json', proof)
        index = dict(tasks=[dict(run_id=repair.RUN_ID, task_path=self.fx.new_binding['path'], task_sha256=self.fx.new_binding['sha256'])]
                       + [dict(run_id='other' + str(i)) for i in range(7)])
        audit = repair.verify_preforward_repair(binding, expected_index=index, plan=self.fx.plan)
        self.assertEqual(audit['scientific_trajectory_count_increment'], 0)
        self.assertEqual(audit['prior_failed_engineering_attempts'][0]['actual_exit_code'], 1)
        self.assertEqual(repair._tree_files(self.fx.store_path), old)
        migrated = DurableRunStore(self.fx.newstore, repair._identity(self.fx.new_task, self.fx.new_binding),
                                  state_validator=repair._zero_state).recover().state
        self.assertTrue(repair._same(migrated['learning'], self.fx.state['learning']))
        self.assertEqual(migrated['provider_state']['schema'], 'odr003-round2-image-provider-v2')
        self.assertTrue(repair._same(migrated['provider_state']['data_rng'], migrated['learning']['rng']))

    def test_no_second_import_even_of_identical_state(self):
        repair.import_zero_checkpoint(self.fx.spec, self.fx.new_binding, self.fx.newstore)
        with self.assertRaisesRegex(ValueError, 'fresh exact replacement'):
            repair.import_zero_checkpoint(self.fx.spec, self.fx.new_binding, self.fx.newstore)

    def test_final_audit_checks_import_generation_even_after_later_progress(self):
        proof = repair.import_zero_checkpoint(self.fx.spec, self.fx.new_binding, self.fx.newstore)
        identity = repair._identity(self.fx.new_task, self.fx.new_binding)
        store = DurableRunStore(self.fx.newstore, identity, state_validator=lambda state: None)
        current = copy.deepcopy(store.recover().state)
        current['next_index'] = 1
        current['learning']['counters']['observed'] = 1
        store.save(current, {**repair.ZERO_CURSORS, 'observed': 1, 'next_index': 1})
        binding = write(self.fx.plan / 'revision/proof.json', proof)
        audit = repair.verify_preforward_repair(binding)
        self.assertEqual(audit['status'], 'passed')
        self.assertEqual(store.recover().info.generation, 2)

    def test_other_nonce_is_never_exempt(self):
        spec = copy.deepcopy(self.fx.spec); spec['failed_nonce'] = 'another_failure'
        with self.assertRaisesRegex(ValueError, 'exact recorded'):
            repair.verify_zero_progress_failure(spec)

    def test_changed_failure_evidence_fails_closed(self):
        Path(self.fx.spec['artifacts']['failure']['path']).write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Evidence hash differs'):
            repair.verify_zero_progress_failure(self.fx.spec)

    def test_archived_task_and_compact_installed_copy_must_agree(self):
        archived = write(self.fx.plan / 'revisions/repair/before/original_task.json', self.fx.task)
        spec = copy.deepcopy(self.fx.spec); spec['old_task'] = archived
        installed = self.fx.output / 'task.json'
        installed.write_text(json.dumps(self.fx.task, separators=(',', ':')), encoding='utf8')
        with patch.object(repair, 'OLD_INSTALLED_TASK_SHA256', sha256(installed)):
            result = repair.verify_zero_progress_failure(spec)
        self.assertEqual(result['installed_task_binding']['sha256'], sha256(installed))

    def test_unbound_new_journal_event_blocks(self):
        self.fx.store.begin_work('arrival_scoring', {'arrival_scoring': 1}, logical_key='unexpected')
        with self.assertRaisesRegex(ValueError, 'tree changed'):
            repair.verify_zero_progress_failure(self.fx.spec)

    def test_rebound_non_model_work_is_not_general_retry(self):
        self.fx.store.begin_work('arrival_scoring', {'arrival_scoring': 1}, logical_key='unexpected')
        spec = copy.deepcopy(self.fx.spec); spec['old_store_files'] = repair._tree_files(self.fx.store_path)
        with self.assertRaisesRegex(ValueError, 'Unexpected old work'):
            repair.verify_zero_progress_failure(spec)

    def test_parameter_or_seed_change_does_not_get_migration(self):
        task = copy.deepcopy(self.fx.new_task); task['definition']['stream_seed'] = 2022
        binding = write(self.fx.launch_task, task)
        with self.assertRaisesRegex(ValueError, 'scientific task'):
            repair.import_zero_checkpoint(self.fx.spec, binding, self.fx.newstore)
        self.assertFalse(self.fx.newstore.exists())

    def test_missing_import_requirement_does_not_get_migration(self):
        task = copy.deepcopy(self.fx.new_task); task.pop('initial_checkpoint_import_required')
        binding = write(self.fx.launch_task, task)
        with self.assertRaisesRegex(ValueError, 'explicit zero-state'):
            repair.import_zero_checkpoint(self.fx.spec, binding, self.fx.newstore)

    def test_old_learning_nonzero_or_partial_state_is_rejected(self):
        state = copy.deepcopy(self.fx.state); state['learning']['counters']['observed'] = 1
        with self.assertRaisesRegex(ValueError, 'nonzero'): repair._zero_state(state)
        state = copy.deepcopy(self.fx.state); state['learning']['gradients']['weight'] = torch.ones(2)
        with self.assertRaisesRegex(ValueError, 'Partial gradients'): repair._zero_state(state)
        state = copy.deepcopy(self.fx.state); state['learning']['optimizer']['state'] = {0: {'step': torch.tensor(1)}}
        with self.assertRaisesRegex(ValueError, 'empty original'): repair._zero_state(state)

    def test_new_store_must_not_reuse_failed_output(self):
        with self.assertRaisesRegex(ValueError, 'fresh exact replacement'):
            repair.import_zero_checkpoint(self.fx.spec, self.fx.new_binding, self.fx.store_path)

    def test_imported_tensor_tamper_fails_even_after_proof_rebinding(self):
        proof = repair.import_zero_checkpoint(self.fx.spec, self.fx.new_binding, self.fx.newstore)
        statepath = Path(proof['imported_generation_state']['path'])
        statepath.write_bytes(b'corrupted tensor state')
        proof['imported_generation_state']['sha256'] = sha256(statepath)
        binding = write(self.fx.plan / 'revision/proof.json', proof)
        with self.assertRaises(Exception): repair.verify_preforward_repair(binding)

    def test_unknown_cost_is_not_zero_filled(self):
        outcome = json.loads(Path(self.fx.spec['artifacts']['outcome']['path']).read_text())
        outcome.pop('peak_gpu_reserved_bytes')
        self.fx.spec['artifacts']['outcome'] = write(self.fx.spec['artifacts']['outcome']['path'], outcome)
        result = repair.verify_zero_progress_failure(self.fx.spec)
        self.assertIsNone(result['retained_costs']['peak_gpu_reserved_bytes']['entry_outcome'])

    def test_verifier_rejects_other_task_in_same_index(self):
        proof = repair.import_zero_checkpoint(self.fx.spec, self.fx.new_binding, self.fx.newstore)
        binding = write(self.fx.plan / 'revision/proof.json', proof)
        index = dict(tasks=[dict(run_id=repair.RUN_ID, task_path=self.fx.new_binding['path'], task_sha256='wrong')]
                       + [dict(run_id='other' + str(i)) for i in range(7)])
        with self.assertRaisesRegex(ValueError, 'indexed logical'): repair.verify_preforward_repair(binding, expected_index=index)

    def importer(self):
        directory = Path(repair.__file__).resolve().parent
        task = dict(schema='odr003-round2-task-v1', action='import_zero_checkpoint', stage='verification',
            run_id='g50p2_preforward_import', protocol=frozen_protocol(),
            implementation_sha256={p.name: sha256(p) for p in sorted(directory.glob('*.py'))},
            origin_spec=self.fx.spec_binding, new_task=self.fx.new_binding,
            proof_path=str(self.fx.plan / 'revisions/preprocessing_rng_repair_20261024/import_proof.json'))
        taskpath = self.fx.plan / 'tasks/g50p2_preforward_import.json'; write(taskpath, task)
        launch = dict(run_id=task['run_id'], run_nonce='synthetic_importer_nonce_20261024',
                      project_id='odr003_round2_20261023', role='worker', argv=['python', str(taskpath)])
        write(self.fx.plan / 'launches' / launch['run_nonce'] / 'launch.json', launch)
        return taskpath, task, launch

    def test_guarded_execute_imports_once_then_only_verifies(self):
        taskpath, task, launch = self.importer()
        with patch('guarded_entry.PLAN', self.fx.plan), patch('guarded_entry.atomic_json', side_effect=write):
            first = repair.execute(taskpath, admission=dict(allow_start=True, boundary_verified=True),
                                   launch_record=launch, conflict_check=lambda: None)
            original_proof = Path(task['proof_path']).read_bytes()
            second = repair.execute(taskpath, admission=dict(allow_start=True, boundary_verified=True),
                                    launch_record=launch, conflict_check=lambda: None)
        self.assertEqual(first['status'], 'preforward_repair_passed')
        self.assertTrue(first['checkpoint_import_performed']); self.assertFalse(second['checkpoint_import_performed'])
        self.assertEqual(first['actual_source_CPU_loads_this_attempt'], 4)
        self.assertEqual(second['actual_source_CPU_loads_this_attempt'], 2)
        self.assertEqual(Path(task['proof_path']).read_bytes(), original_proof)
        receipt = json.loads((self.fx.newstore.parent / 'zero_checkpoint_import.json').read_text())
        self.assertEqual(receipt['new_task_canonical_sha256'], canonical_sha(self.fx.new_task))
        self.assertEqual(receipt['origin_spec'], self.fx.spec_binding)

    def test_guarded_execute_busy_does_not_load_source_or_import(self):
        taskpath, _, launch = self.importer()
        with patch('guarded_entry.PLAN', self.fx.plan), patch.object(repair, 'import_zero_checkpoint') as importer:
            result = repair.execute(taskpath, admission=dict(allow_start=True, boundary_verified=True),
                                    launch_record=launch, conflict_check=lambda: 'foreign compute')
        self.assertEqual(result['exit_code'], 75); importer.assert_not_called()
        self.assertFalse(self.fx.newstore.exists())

    def test_guarded_execute_requires_real_boundary_admission(self):
        taskpath, _, launch = self.importer()
        with patch('guarded_entry.PLAN', self.fx.plan):
            with self.assertRaisesRegex(ValueError, 'Fresh shared-host admission'):
                repair.execute(taskpath, admission=dict(allow_start=True, boundary_verified=False),
                               launch_record=launch, conflict_check=lambda: None)

    def test_guarded_execute_preserves_orphan_import_without_guessing(self):
        taskpath, _, launch = self.importer()
        self.fx.newstore.mkdir(parents=True)
        with patch('guarded_entry.PLAN', self.fx.plan):
            with self.assertRaisesRegex(ValueError, 'without its immutable proof'):
                repair.execute(taskpath, admission=dict(allow_start=True, boundary_verified=True),
                               launch_record=launch, conflict_check=lambda: None)


if __name__ == '__main__':
    unittest.main()
