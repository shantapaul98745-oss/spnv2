"""Bounded, zero-forward validation of versioned audit dispatch metadata."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    from budget import atomic_json
    task = json.loads(Path(task_path).read_text())
    def checked(binding):
        p = Path(binding['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != binding['sha256']:
            raise ValueError('Changed validation binding: '+str(p))
        return p
    if not admission['allow_start'] or not admission['boundary_verified'] or conflict_check():
        raise ValueError('Fresh admission required')
    source = checked(task['candidate_manager'])
    spec = importlib.util.spec_from_file_location('_admission_candidate_manager', source)
    manager = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = manager
    spec.loader.exec_module(manager)
    idx = json.loads(checked(task['candidate_index']).read_text())
    state = json.loads(checked(task['original_status']).read_text())
    manager.validate_index(idx)
    tamper = copy.deepcopy(idx)
    tamper['summary_template']['implementation_sha256'] = {}
    try:
        manager.validate_index(tamper)
    except ValueError:
        stale_rejected = True
    else:
        raise ValueError('Stale audit template escaped validation')
    root = Path(task['test_output'])
    root.mkdir(parents=True, exist_ok=False)
    (root/'tasks').mkdir()
    old_path = root/'tasks/verify_pilots.json'
    old_bytes = b'{"preserved_old_failed_task":true}\n'
    old_path.write_bytes(old_bytes)
    manager.PLAN = root
    test_index = copy.deepcopy(idx)
    test_index['audit_task_paths'] = {k: str(root/'tasks'/Path(p).name)
                                      for k,p in idx['audit_task_paths'].items()}
    state['index'] = task['candidate_index']
    checks = []
    for kind, action in [('pilot_admission','verify_pilots'),('summary','summarize_diagnostic')]:
        path, run_id = manager.audit_task(test_index, state, kind)
        produced = json.loads(path.read_text())
        if produced['implementation_sha256'] != idx['implementation_sha256'] or produced['action'] != action:
            raise ValueError('Actual generated audit task has stale bindings')
        if produced['execution_receipts']['path'].endswith('pilot_receipts.json'):
            raise ValueError('Audit dispatch reused historical receipt filename')
        if manager.audit_task(test_index, state, kind) != (path, run_id):
            raise ValueError('Versioned audit dispatch is not idempotent')
        checks.append(dict(kind=kind, task=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if old_path.read_bytes() != old_bytes:
        raise ValueError('Old failed task was overwritten')
    proof = dict(schema='g50p-audit-template-targeted-validation-v1', status='passed',
                 model_forwards=0, candidate_index=task['candidate_index'],
                 candidate_manager=task['candidate_manager'], original_status=task['original_status'],
                 stale_template_rejected=stale_rejected, old_failed_task_preserved=True,
                 actual_generated_tasks=checks, both_accepted_pairs_preserved=True,
                 science_or_budget_limit_changed=False)
    output = Path(task['proof_path'])
    if output.exists():
        raise ValueError('Validation proof already exists')
    atomic_json(output, proof)
    return dict(status='audit_template_fix_validated', exit_code=0, model_forwards=0,
                proof=str(output), proof_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
