"""Metadata-only admission of one verifier repair over completed producers.

The original eight producer tasks and their complete checkpoints retain their
identities. A separately bound auditor revision may fix audit/controller code;
it cannot authorize another scientific trajectory or change scientific code.
Live process, shared-resource and project-lock checks remain the launcher's job.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


SCHEMA = 'odr003-round2-auditor-repair-v1'
SCOPE = 'verification_only_no_scientific_reexecution'
ALLOWED_CHANGED = frozenset({
    'verify_engineering_phase.py', 'verify_complete.py', 'queue_round2.py',
    'test_engineering_repair_audit.py', 'test_verify_complete.py',
    'test_queue_round2.py',
})
ALLOWED_ADDED = frozenset({
    'audit_revision.py', 'test_audit_revision.py',
    'artifact_binding.py', 'test_artifact_binding.py',
})
_INDEX_VARIABLES = frozenset({'implementation_sha256', 'synthetic_contracts', 'auditor_repair'})
_FAILURE = 'frozen data boundary differs: lightbox/stream_2021'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _bound(value):
    _require(isinstance(value, dict) and set(value) == {'path', 'sha256'},
             'Invalid auditor repair file binding')
    _require(isinstance(value['sha256'], str)
             and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None,
             'Invalid auditor repair checksum')
    path = Path(value['path']).resolve(strict=True)
    _require(path.is_file() and _sha(path) == value['sha256'],
             'Auditor repair evidence changed: ' + str(path))
    return path


def _code_map(directory):
    directory = Path(directory).resolve(strict=True)
    _require(directory.is_dir(), 'Auditor implementation root is not a directory')
    return {path.name: _sha(path) for path in sorted(directory.glob('*.py'))}


def _map(value, message):
    _require(isinstance(value, dict) and bool(value), message)
    _require(all(isinstance(name, str) and Path(name).name == name and name.endswith('.py')
                 and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                 for name, digest in value.items()), message)
    return value


def _failure_evidence(manifest, original, original_binding):
    evidence = manifest.get('failed_verification')
    _require(isinstance(evidence, dict)
             and set(evidence) == {'task', 'outcome', 'log', 'exitcode', 'status'},
             'Auditor repair requires complete failed-verification evidence')
    paths = {name: _bound(binding) for name, binding in evidence.items()}
    task, outcome, status = (_read(paths[name]) for name in ('task', 'outcome', 'status'))
    _require(task.get('action') == 'verify_engineering'
             and task.get('stage') == 'verification'
             and task.get('run_id') == 'g50p2_engineering_verification'
             and task.get('implementation_sha256') == original['implementation_sha256']
             and task.get('engineering_index', {}).get('sha256') == original_binding['sha256'],
             'Failed audit task does not bind the original producer index')
    _require(outcome.get('status') == 'failed' and outcome.get('exit_code') == 1
             and outcome.get('error_type') == 'ValueError' and outcome.get('error') == _FAILURE
             and paths['exitcode'].read_text(encoding='ascii').strip() == '1'
             and _FAILURE in paths['log'].read_text(encoding='utf8'),
             'Only the preserved exact metadata-verification failure may be repaired')
    entries = original['tasks']
    run_ids = {row['run_id'] for row in entries}
    completed = status.get('completed', [])
    completed_ids = [row.get('run_id') for row in completed]
    _require(status.get('status') == 'blocked' and status.get('engineering_verified') is False
             and len(completed_ids) == len(set(completed_ids)) == 8
             and set(completed_ids) == run_ids
             and not status.get('complete_main') and not status.get('complete_accepted')
             and status.get('complete_index') is None
             and status.get('index_sha256') == original_binding['sha256'],
             'Auditor-only recovery requires all eight completed producers and no complete ODR')
    active = status.get('active') or {}
    _require(active.get('run_id') == task['run_id']
             and active.get('task_sha256') == evidence['task']['sha256']
             and active.get('exit_code') == 1 and active.get('outcome') == outcome,
             'Blocked state does not preserve the exact failed audit attempt')
    attempts = [row for row in status.get('attempts', []) if row.get('nonce') == active.get('nonce')]
    _require(isinstance(active.get('nonce'), str) and len(active['nonce']) >= 16
             and len(attempts) == 1 and attempts[0] == active,
             'Failed audit attempt identity is missing or duplicated')
    receipts = _read(_bound(task['execution_receipts']))
    _require(receipts.get('runs') == completed,
             'Original immutable engineering receipts differ from completed producers')


def validate_revision(index, implementation_dir, *, plan):
    """Return the immutable producer map after strict auditor-only checks.

    With no revision, producer and execution maps must be identical. With a
    revision, every original task/report/checkpoint remains bound to its old map;
    the new map covers only an explicitly allowed metadata-auditor repair.
    """
    current = _map(index.get('implementation_sha256'), 'Invalid current implementation map')
    _require(_code_map(implementation_dir) == current,
             'Current auditor implementation binding differs')
    if 'auditor_repair' not in index:
        return dict(current)
    plan = Path(plan).resolve(strict=True)
    manifest_path = _bound(index['auditor_repair'])
    _require(manifest_path.is_relative_to(plan / 'revisions'),
             'Auditor repair manifest escapes this plan revision history')
    manifest = _read(manifest_path)
    _require(manifest.get('schema') == SCHEMA and manifest.get('scope') == SCOPE,
             'Unapproved auditor repair scope')
    original_binding = manifest['original_index']
    original_path = _bound(original_binding)
    _require(original_path.is_relative_to(plan / 'revisions'),
             'Original producer index snapshot escapes revision history')
    original = _read(original_path)
    _require('auditor_repair' not in original,
             'This admission supports only the single preserved original auditor failure')
    producer = _map(original.get('implementation_sha256'), 'Invalid original producer map')
    _require({key: value for key, value in original.items() if key not in _INDEX_VARIABLES}
             == {key: value for key, value in index.items() if key not in _INDEX_VARIABLES},
             'Auditor repair changed scientific/index fields')
    snapshot = Path(manifest['original_implementation_root']).resolve(strict=True)
    _require(snapshot.is_relative_to(plan / 'revisions') and _code_map(snapshot) == producer,
             'Original producer code snapshot differs')
    removed = set(producer) - set(current)
    changed = {name for name in set(producer) & set(current) if producer[name] != current[name]}
    added = set(current) - set(producer)
    _require(not removed and changed <= ALLOWED_CHANGED and added <= ALLOWED_ADDED,
             'Auditor repair changed an unapproved scientific/implementation file')
    _require(manifest.get('allowed_changed_files') == sorted(changed)
             and manifest.get('allowed_added_files') == sorted(added),
             'Explicit auditor repair file allowlist differs')
    rows = original.get('tasks', [])
    _require(len(rows) == 8 and len({row['run_id'] for row in rows}) == 8,
             'Auditor repair requires the exact eight unique producer tasks')
    for row in rows:
        task = _read(_bound({'path': row['task_path'], 'sha256': row['task_sha256']}))
        _require(task.get('run_id') == row['run_id'] and task.get('definition') == row['definition']
                 and task.get('stage') == 'engineering' and task.get('action', 'run') == 'run'
                 and task.get('implementation_sha256') == producer,
                 'Original successful producer task changed')
    _failure_evidence(manifest, original, original_binding)
    return dict(producer)
