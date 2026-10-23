"""Tamper checks for metadata-only reuse of completed engineering producers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_revision import SCHEMA, SCOPE, validate_revision


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def codes(directory):
    return {path.name: binding(path)['sha256'] for path in sorted(directory.glob('*.py'))}


class AuditorRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.plan = Path(self.temporary.name)
        self.revision = self.plan/'revisions/audit_binding_fix'
        self.snapshot = self.revision/'before/implementation'
        self.current = self.plan/'implementation'
        self.snapshot.mkdir(parents=True)
        self.current.mkdir()
        for name, body in {'verify_engineering_phase.py': '# original auditor\n',
                           'protected_engine.py': '# unchanged scientific producer\n'}.items():
            (self.snapshot/name).write_text(body, encoding='utf8')
            (self.current/name).write_text(body, encoding='utf8')
        (self.current/'verify_engineering_phase.py').write_text('# repaired metadata auditor\n', encoding='utf8')
        (self.current/'audit_revision.py').write_text('# metadata repair admission\n', encoding='utf8')
        self.producer = codes(self.snapshot)
        tasks = []
        for number in range(8):
            run_id = 'engineering_' + str(number)
            definition = dict(run_key=run_id, domain='lightbox' if number < 4 else 'sunlamp')
            task = dict(run_id=run_id, definition=definition, stage='engineering', action='run',
                        implementation_sha256=self.producer,
                        baseline_task=dict(developer_labels='DO_NOT_READ_GT'))
            item = write(self.plan/'tasks'/(run_id+'.json'), task)
            tasks.append(dict(run_id=run_id, definition=definition,
                              task_path=item['path'], task_sha256=item['sha256']))
        self.original = dict(schema='odr003-round2-engineering-index-v1', tasks=tasks,
                             implementation_sha256=self.producer,
                             synthetic_contracts=dict(path='original_contracts', sha256='a'*64),
                             protocol_sha256='b'*64, pre_forward_repair=dict(path='retained_zero_import', sha256='c'*64))
        self.original_binding = write(self.revision/'before/engineering_index.json', self.original)
        completed = [dict(run_id=row['run_id'], report_sha256='d'*64) for row in tasks]
        receipt_binding = write(self.plan/'engineering_receipts.json', dict(runs=completed))
        failed_task = dict(action='verify_engineering', stage='verification',
                           run_id='g50p2_engineering_verification', implementation_sha256=self.producer,
                           engineering_index=self.original_binding, execution_receipts=receipt_binding)
        self.failed_task_binding = write(self.revision/'before/failed_task.json', failed_task)
        self.outcome = dict(status='failed', exit_code=1, error_type='ValueError',
                            error='frozen data boundary differs: lightbox/stream_2021')
        outcome_binding = write(self.revision/'before/outcome.json', self.outcome)
        log = self.revision/'before/failed.log'
        log.write_text('ValueError: '+self.outcome['error']+'\n', encoding='utf8')
        exitcode = self.revision/'before/failed.log.exitcode'
        exitcode.write_text('1\n', encoding='ascii')
        active = dict(run_id=failed_task['run_id'], task_sha256=self.failed_task_binding['sha256'],
                      exit_code=1, outcome=self.outcome, nonce='b18ee82116f4412cb56bccf4c16f0f59')
        self.status = dict(status='blocked', engineering_verified=False, completed=completed,
                           complete_main={}, complete_accepted={}, complete_index=None,
                           index_sha256=self.original_binding['sha256'], active=active, attempts=[active])
        status_binding = write(self.revision/'before/status.json', self.status)
        self.manifest = dict(schema=SCHEMA, scope=SCOPE, original_index=self.original_binding,
                             original_implementation_root=str(self.snapshot),
                             failed_verification=dict(task=self.failed_task_binding, outcome=outcome_binding,
                                                      log=binding(log), exitcode=binding(exitcode), status=status_binding),
                             allowed_changed_files=['verify_engineering_phase.py'],
                             allowed_added_files=['audit_revision.py'])
        self.index = {**self.original, 'implementation_sha256': codes(self.current),
                      'synthetic_contracts': dict(path='updated_contracts', sha256='e'*64)}
        self.save_manifest()

    def save_manifest(self):
        self.index['auditor_repair'] = write(self.revision/'manifest.json', self.manifest)

    def validate(self):
        return validate_revision(self.index, self.current, plan=self.plan)

    def test_reuses_exact_successful_producer_map_with_explicit_new_auditor(self):
        self.assertEqual(self.producer, self.validate())
        self.assertNotEqual(self.producer, self.index['implementation_sha256'])

    def test_no_revision_requires_and_returns_current_map(self):
        index = dict(implementation_sha256=codes(self.current))
        self.assertEqual(index['implementation_sha256'], validate_revision(index, self.current, plan=self.plan))

    def test_manifest_checksum_tamper_is_rejected(self):
        Path(self.index['auditor_repair']['path']).write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.validate()

    def test_scientific_index_field_change_is_rejected(self):
        self.index['protocol_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'scientific/index'):
            self.validate()

    def test_original_producer_snapshot_tamper_is_rejected(self):
        (self.snapshot/'protected_engine.py').write_text('# changed\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'producer code snapshot'):
            self.validate()

    def test_current_unapproved_scientific_code_change_is_rejected(self):
        (self.current/'protected_engine.py').write_text('# changed\n', encoding='utf8')
        self.index['implementation_sha256'] = codes(self.current)
        self.manifest['allowed_changed_files'].append('protected_engine.py')
        self.manifest['allowed_changed_files'].sort()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'unapproved scientific'):
            self.validate()

    def test_current_code_drift_not_rebound_is_rejected(self):
        (self.current/'verify_engineering_phase.py').write_text('# drift\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'Current auditor implementation'):
            self.validate()

    def test_added_unapproved_module_is_rejected(self):
        (self.current/'new_scientific_algorithm.py').write_text('# forbidden\n', encoding='utf8')
        self.index['implementation_sha256'] = codes(self.current)
        self.manifest['allowed_added_files'].append('new_scientific_algorithm.py')
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'unapproved scientific'):
            self.validate()

    def test_manifest_cannot_overstate_allowed_patch_files(self):
        self.manifest['allowed_changed_files'].append('verify_complete.py')
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'allowlist differs'):
            self.validate()

    def test_original_completed_task_hash_must_remain_unchanged(self):
        Path(self.original['tasks'][0]['task_path']).write_text('{}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            self.validate()

    def test_rebound_different_failure_still_cannot_be_excused(self):
        self.outcome['error'] = 'NaN in scientific model'
        self.manifest['failed_verification']['outcome'] = write(
            Path(self.manifest['failed_verification']['outcome']['path']), self.outcome)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'exact metadata-verification failure'):
            self.validate()

    def test_all_eight_complete_and_no_complete_odr_required(self):
        self.status['complete_main'] = dict(unapproved={})
        self.manifest['failed_verification']['status'] = write(
            Path(self.manifest['failed_verification']['status']['path']), self.status)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'all eight completed'):
            self.validate()

    def test_failed_audit_attempt_is_not_deleted_from_history(self):
        self.status['attempts'] = []
        self.manifest['failed_verification']['status'] = write(
            Path(self.manifest['failed_verification']['status']['path']), self.status)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
