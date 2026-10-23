"""One recorded, zero-progress preprocessing repair; never a retry policy.

This module reads the preserved failed attempt and its complete checkpoint.
Only a caller that already holds the shared-host admission and project locks
may run its tensor/source checks. It performs no model construction or forward.
The old store is opened under a read-only shared lock and is never rewritten.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from durable_state import DurableRunStore
from frozen_backend import canonical_sha, convert_legacy_empty_state, sha256


FAILED_NONCE = 'be01859de69743f181c337a9a17fcf01'
RUN_ID = 'g50p2_eng_lightbox_g50_reference'
SOURCE_SHA256 = 'd8f9c281d23adaeb77d03023aee8d40ae1cb9e18dd503539c4719bbe719514cc'
OLD_TASK_SHA256 = 'dee8eb701b32a7e16eb3d31f3170f197ea74562081c1dbc858ca667930e22739'
OLD_INSTALLED_TASK_SHA256 = '59cba68f90ffa626b57a0dea3119ede87abecf64fec4fba3ddc43427f919f1fb'
OLD_STATE_SHA256 = 'e226ddf5726398187aeaca56e657fc11acde362d610895f91ae242fc1bb982fa'
OLD_METADATA_SHA256 = '51610578e372d9cb9e5375d46aab7b8f1f8d40548b922422ffe272568ef0831b'
FAILURE_MESSAGE = 'Frozen inference preprocessing unexpectedly consumed RNG'
ZERO_CURSORS = dict(observed=0, candidates=0, commits=0, rollbacks=0, next_index=0)
SPEC_SCHEMA = 'odr003-round2-preforward-repair-spec-v1'
PROOF_SCHEMA = 'odr003-round2-preforward-repair-proof-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _binding(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'Incomplete artifact binding')
    path = Path(binding['path']).resolve(strict=True)
    require(path.is_file() and not Path(binding['path']).is_symlink(), 'Bound evidence must be an actual file')
    require(sha256(path) == binding['sha256'], 'Evidence hash differs: ' + str(path))
    return path


def _file_binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path))


def _tree_files(root):
    root = Path(root).resolve(strict=True)
    files = {}
    for child in sorted(root.rglob('*')):
        require(not child.is_symlink(), 'A preserved store must not contain symlinks')
        if child.is_file():
            files[child.relative_to(root).as_posix()] = sha256(child)
    return files


class _ReadOnlyStore(DurableRunStore):
    @contextmanager
    def _lock(self):
        # Do not create/truncate even the old lock inode. Existing writer
        # exclusion remains mandatory; shared readers cannot overlap a writer.
        import os
        handle = (self.root / '.writer.lock').open('rb')
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBRLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            yield
        finally:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _same(a, b):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _tree_digest(value):
    """Value digest independent of pickle storage IDs; includes every RNG byte."""
    import numpy as np
    import torch
    h = hashlib.sha256()
    def walk(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(('tensor:' + str(t.dtype) + ':' + repr(tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(('array:' + str(v.dtype) + ':' + repr(v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            h.update(b'dict[')
            for k in sorted(v, key=lambda key: (type(key).__name__, repr(key))):
                walk(k); walk(v[k])
            h.update(b']')
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode() + b'[')
            for item in v: walk(item)
            h.update(b']')
        elif isinstance(v, float): h.update(b'float:' + struct.pack('!d', v))
        elif isinstance(v, (str, int, bool)) or v is None:
            h.update((type(v).__name__ + ':' + repr(v)).encode() + b'\0')
        else: raise ValueError('Unsupported complete-state value: ' + type(v).__name__)
    walk(value)
    return h.hexdigest()


def _zero_state(state):
    """Strictly the initial reference state; protected/nonempty runs excluded."""
    required = {'schema', 'identity', 'stream_records', 'learning', 'provider_state', 'next_index',
                'observation_trace', 'finished', 'finish_counts', 'ledger_boundary'}
    require(isinstance(state, dict) and set(state) == required, 'Unexpected initial checkpoint schema')
    require(state['schema'] == 'odr003-round2-original-g50-reference-v1', 'Only the recorded original G50 reference can be repaired')
    require(state['next_index'] == 0 and state['observation_trace'] == [] and state['finished'] is False
            and state['finish_counts'] is None, 'Checkpoint has scientific progress or a completed tail')
    require(len(state['stream_records']) == 128 and len({r['sample_id'] for r in state['stream_records']}) == 128,
            'Initial frozen engineering prefix differs')
    learning = state['learning']
    keys = {'model', 'buffers', 'optimizer', 'optimizer_type', 'optimizer_parameter_names', 'modes',
            'requires_grad', 'gradients', 'bn_features', 'bn_mean', 'bn_mean_sq', 'boundary_buffers',
            'rng', 'scaler', 'counters'}
    require(isinstance(learning, dict) and set(learning) == keys, 'Incomplete learning state')
    require(learning['counters'] == dict(accepted=0, observed=0, updates=0, tail=0), 'Learning progress is nonzero')
    require(learning['optimizer_type'] == 'torch.optim.adamw.AdamW'
            and learning['optimizer']['state'] == {} and bool(learning['optimizer']['param_groups']),
            'Optimizer is not the empty original AdamW state')
    require(learning['gradients'] and all(v is None for v in learning['gradients'].values()), 'Partial gradients exist')
    require(all(learning[k] == {} for k in ('bn_features', 'bn_mean', 'bn_mean_sq')), 'An unfinished learning window exists')
    require(learning['scaler'] is None and _same(learning['buffers'], learning['boundary_buffers']),
            'Initial buffers/scaler differ')
    require(set(learning['rng']) == {'python', 'numpy', 'cpu', 'cuda'}, 'Complete compute RNG is missing')
    require(set(learning['gradients']) == set(learning['requires_grad']), 'Parameter identities differ')
    require(learning['model'] and learning['modes'] and all(type(x) is bool for x in learning['modes'].values()),
            'Full model/mode state is missing')
    require(state['ledger_boundary']['sequence'] == 0, 'Initial checkpoint contains journal progress')


def _identity(task, task_binding):
    return dict(run_id=task['run_id'], source_sha256=task['protocol']['source_sha256'],
                config_sha256=task_binding['sha256'],
                manifest_sha256=task['baseline_task']['bindings']['stream_manifest']['sha256'])


def _spec_read(spec):
    if isinstance(spec, dict) and set(spec) == {'path', 'sha256'}:
        return _json(_binding(spec))
    require(isinstance(spec, dict), 'A bound spec or complete spec mapping is required')
    return copy.deepcopy(spec)


def _old_read(spec):
    require(spec.get('schema') == SPEC_SCHEMA and spec.get('failed_nonce') == FAILED_NONCE
            and spec.get('run_id') == RUN_ID and spec.get('generation') == 1,
            'Only the exact recorded first-attempt fault may use this repair')
    plan = Path(spec['plan_root']).resolve(strict=True)
    output = Path(spec['old_output_dir']).resolve(strict=True)
    require(output == plan / 'runs' / RUN_ID, 'Preserved failed output path differs')
    task_path = _binding(spec['old_task'])
    require((task_path == output / 'task.json'
             or (task_path.is_relative_to(plan / 'revisions') and task_path.name == 'original_task.json'))
            and spec['old_task']['sha256'] == OLD_TASK_SHA256,
            'Preserved original task differs')
    task = _json(task_path)
    # The queue bound the original task bytes while execute installed an atomic
    # compact JSON copy. Their JSON contents must agree despite different bytes.
    installed_task = output / 'task.json'
    require(sha256(installed_task) == OLD_INSTALLED_TASK_SHA256 and _json(installed_task) == task,
            'Preserved installed task differs from the originally launched task')
    require(task['run_id'] == RUN_ID and task['stage'] == 'engineering'
            and task['definition']['trajectory'] == 'G50_reference'
            and task['definition']['domain'] == 'lightbox' and task['definition']['stream_seed'] == 2021
            and task['definition']['observations'] == 128 and task['protocol']['source_sha256'] == SOURCE_SHA256,
            'Fault did not belong to the authorized first engineering trajectory')
    require(Path(task['output_dir']).resolve() == output, 'Old task output differs')
    expected = spec['old_store_files']
    store_path = output / 'durable'
    require(isinstance(expected, dict) and expected and _tree_files(store_path) == expected,
            'Preserved checkpoint/journal tree changed')
    require(expected.get('checkpoints/generation-00000000000000000001/state.bin') == OLD_STATE_SHA256
            and expected.get('checkpoints/generation-00000000000000000001/metadata.json') == OLD_METADATA_SHA256,
            'Original complete generation is not the recorded one')
    for name in ('frozen_backend.py', 'run_round2.py', 'reference_driver.py'):
        binding = spec['old_implementation_files'][name]
        _binding(binding)
        require(binding['sha256'] == task['implementation_sha256'][name], 'Old fault implementation differs: ' + name)
    identity = _identity(task, spec['old_task'])
    require(_json(store_path / 'identity.json') == identity, 'Old store identity differs')
    store = _ReadOnlyStore(store_path, identity, state_validator=_zero_state)
    recovered = store.recover()
    require(recovered is not None and recovered.info.generation == 1 and not recovered.rejected_generations
            and recovered.scientific_cursors == ZERO_CURSORS, 'Only the original complete zero checkpoint is eligible')
    state = recovered.state
    require(state['identity'] == identity, 'State identity differs')
    # Explicitly validate legacy v1-empty before any import is attempted.
    convert_legacy_empty_state(state['provider_state'], initial_data_rng=state['learning']['rng'])
    events, torn = store.read_journal()
    require(not torn and len(events) == 2 and events[0]['kind'] == 'checkpoint_published'
            and events[1]['kind'] == 'work_started', 'Unexpected old work/journal history')
    require(events[0]['payload']['reason'] == 'initial' and events[0]['payload']['generation'] == 1
            and events[0]['payload']['scientific_cursors'] == ZERO_CURSORS, 'Initial publication differs')
    work = events[1]['payload']
    require(work['work_kind'] == 'image_read' and work['logical_key'] == 'observation:0000:read'
            and work['forwards_reserved'] == {} and work['replay_of'] is None,
            'The preserved failure is not a single first image read without model work')
    physical = store.work_summary()
    require(physical['attempts_started'] == 1 and physical['attempts_finished'] == 0
            and physical['forwards_reserved_attempt_upper_bound'] == {}
            and physical['forwards_confirmed_completed'] == {}
            and physical['forwards_execution_upper_bound'] == {}
            and physical['inflight_unknown_attempt_ids'] == [work['attempt_id']], 'Any model forward or other work prevents this repair')
    artifacts = {name: _binding(spec['artifacts'][name]) for name in ('launch', 'outcome', 'failure', 'log', 'exit_code')}
    launch, outcome, failure = (_json(artifacts[name]) for name in ('launch', 'outcome', 'failure'))
    require(artifacts['launch'] == plan / 'launches' / FAILED_NONCE / 'launch.json'
            and artifacts['outcome'] == plan / 'launches' / FAILED_NONCE / 'outcome.json'
            and artifacts['failure'] == output / 'failure.json', 'Failure evidence paths differ')
    require(launch['run_id'] == RUN_ID and launch['run_nonce'] == FAILED_NONCE
            and launch['project_id'] == 'odr003_round2_20261023' and launch['role'] == 'worker',
            'Failed launch ownership differs')
    require(Path(launch['log_path']).resolve() == artifacts['log']
            and artifacts['exit_code'] == Path(str(artifacts['log']) + '.exitcode'), 'Failed log/exit binding differs')
    argv = launch['argv']
    require('--task-json' in argv and argv[argv.index('--task-json') + 1] == spec['original_task_launch_path'],
            'Failed launch task command differs')
    require(int(artifacts['exit_code'].read_text('ascii').strip()) == 1
            and outcome.get('exit_code') == 1 and outcome.get('status') == 'failed'
            and failure.get('status') == 'failed' and failure.get('type') == 'RuntimeError'
            and failure.get('message') == FAILURE_MESSAGE,
            'The actual failure is not the exact retained preprocessing assertion')
    trace = failure.get('traceback', '')
    require(all(name in trace for name in ('run_round2.py', 'reference_driver.py', 'frozen_backend.py', FAILURE_MESSAGE)),
            'Actual traceback does not establish the first-image provider failure')
    # Source equality is checked from actual tensor bytes, never a passed flag.
    import torch
    bindings = task['baseline_task']['bindings']
    source_path = _binding(bindings['source']); receipt_path = _binding(bindings['official_receipt'])
    require(bindings['source']['sha256'] == SOURCE_SHA256, 'Official source identity differs')
    receipt = _json(receipt_path)
    require(receipt.get('status') == 'passed' and receipt.get('strict_model_load_performed') is True
            and receipt.get('source_offline_identity_verified') is True
            and receipt['source_sha256'] == SOURCE_SHA256 and Path(receipt['source_path']).resolve() == source_path,
            'Actual official source receipt differs')
    raw_source = torch.load(source_path, map_location='cpu', weights_only=True)
    source_model = raw_source if receipt.get('state_dict_key') is None else raw_source[receipt['state_dict_key']]
    require(_same(source_model, state['learning']['model']), 'Initial full model tensors differ from official source')
    require(_tree_files(store_path) == expected, 'Reading the old store changed preserved evidence')
    return spec, task, state, recovered, physical, artifacts, outcome, failure


def verify_zero_progress_failure(spec):
    spec = _spec_read(spec)
    spec, task, state, recovered, physical, artifacts, outcome, failure = _old_read(spec)
    costs = {name: {'failure': failure.get(name), 'entry_outcome': outcome.get(name)}
             for name in ('execution_attempt_elapsed_seconds', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')}
    for sources in costs.values():
        for value in sources.values():
            require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
                    'Recorded attempt cost is invalid')
    return dict(status='passed', failed_nonce=FAILED_NONCE, run_id=RUN_ID, actual_exit_code=1,
                actual_outcome=outcome, failure_binding=spec['artifacts']['failure'],
                scientific_cursors=ZERO_CURSORS.copy(), physical_work=physical,
                zero_forward_proof=dict(model_forward_upper_bound=0,
                    unconfirmed_non_model_work=[dict(attempt_id=physical['inflight_unknown_attempt_ids'][0],
                                                   work_kind='image_read', forwards_reserved={})]),
                initial_generation=recovered.info.generation, initial_state_sha256=OLD_STATE_SHA256,
                learning_value_sha256=_tree_digest(state['learning']), source_tensors_equal=True,
                retained_costs=costs, unknown_image_read_compute_time=True,
                old_store_content_unchanged=True, original_task=spec['old_task'],
                installed_task_binding=_file_binding(Path(spec['old_output_dir']) / 'task.json'))


def _new_task_binding(new_task):
    if isinstance(new_task, (str, Path)):
        return _file_binding(new_task)
    _binding(new_task)
    return copy.deepcopy(new_task)


def _validate_new_task(spec, old_task, binding):
    path = _binding(binding); new_task = _json(path)
    before, after = copy.deepcopy(old_task), copy.deepcopy(new_task)
    for key in ('implementation_sha256', 'output_dir', 'execution_revision',
                'initial_checkpoint_import_required', 'initial_checkpoint_origin'):
        before.pop(key, None); after.pop(key, None)
    require(before == after, 'The repair changed a scientific task/definition/protocol field')
    require(new_task.get('execution_revision') == 'preprocess_rng_fix_20261024'
            and new_task.get('initial_checkpoint_import_required') is True,
            'The replacement lacks its explicit zero-state import requirement')
    origin = new_task.get('initial_checkpoint_origin')
    require(isinstance(origin, dict) and _json(_binding(origin)) == spec,
            'The replacement does not bind this exact repair spec')
    plan = Path(spec['plan_root']).resolve(strict=True)
    output = Path(new_task['output_dir']).resolve()
    require(output == plan / 'runs' / 'preprocess_rng_fix_20261024' / RUN_ID,
            'New execution must use the one explicit replacement output')
    require(path.is_relative_to(plan / 'tasks') and path.name == RUN_ID + '.json', 'New bound task path differs')
    require(new_task['implementation_sha256'] != old_task['implementation_sha256'], 'No implementation correction is bound')
    return new_task, output


def import_zero_checkpoint(spec, new_task, newstore_path):
    """Import the original complete initial state once; no source cold restart."""
    spec = _spec_read(spec)
    evidence = verify_zero_progress_failure(spec)
    _, old_task, old_state, _, _, _, _, _ = _old_read(spec)
    binding = _new_task_binding(new_task)
    task, output = _validate_new_task(spec, old_task, binding)
    newstore_path = Path(newstore_path).resolve()
    require(newstore_path == output / 'durable' and not newstore_path.exists(), 'Import target must be a fresh exact replacement store')
    require(not (output / 'failure.json').exists() and not (output / 'report.json').exists(), 'Replacement output already ran')
    state = copy.deepcopy(old_state)
    state['identity'] = _identity(task, binding)
    state['provider_state'] = convert_legacy_empty_state(old_state['provider_state'], initial_data_rng=old_state['learning']['rng'])
    # No pending window exists; this only validates the complete reference schema.
    store = DurableRunStore(newstore_path, state['identity'], state_validator=_zero_state)
    require(store.journal_boundary()['sequence'] == 0, 'Replacement store has prior physical work')
    require(state['ledger_boundary']['sequence'] == 0, 'Imported boundary was not initial')
    info = store.save(state, ZERO_CURSORS)
    recovered = store.recover()
    require(recovered is not None and recovered.info.generation == 1 and not recovered.rejected_generations
            and recovered.scientific_cursors == ZERO_CURSORS and _same(state, recovered.state)
            and _same(old_state['learning'], recovered.state['learning']), 'Imported complete state did not round-trip exactly')
    proof = dict(schema=PROOF_SCHEMA, status='passed', repair_scope='exact_once_zero_progress_preprocessing_implementation_fault',
                 spec=spec, spec_sha256=canonical_sha(spec), new_task=binding, new_store_path=str(newstore_path),
                 imported_generation=1, imported_generation_metadata=_file_binding(info.path / 'metadata.json'),
                 imported_generation_state=_file_binding(info.path / 'state.bin'),
                 unchanged_learning_required=True, learning_value_sha256=_tree_digest(old_state['learning']),
                 changed_state_fields=['identity', 'provider_state'], old_failure_evidence=evidence,
                 scientific_trajectory_count_increment=0, old_store_content_unchanged=True)
    require(_tree_files(Path(spec['old_output_dir']) / 'durable') == spec['old_store_files'], 'Old checkpoint tree changed during import')
    return proof


def verify_preforward_repair(binding, *, expected_index=None, plan=None):
    """Recheck saved repair evidence for the final independent engineering phase."""
    proof_path = _binding(binding); proof = _json(proof_path)
    require(proof.get('schema') == PROOF_SCHEMA and proof.get('status') == 'passed'
            and proof.get('repair_scope') == 'exact_once_zero_progress_preprocessing_implementation_fault'
            and proof.get('unchanged_learning_required') is True
            and proof.get('scientific_trajectory_count_increment') == 0
            and proof.get('changed_state_fields') == ['identity', 'provider_state'], 'Repair proof scope differs')
    spec = proof['spec']
    require(canonical_sha(spec) == proof['spec_sha256'], 'Saved repair spec identity differs')
    if plan is not None: require(Path(plan).resolve() == Path(spec['plan_root']).resolve(), 'Repair proof belongs to another plan')
    evidence = verify_zero_progress_failure(spec)
    require(evidence == proof['old_failure_evidence'], 'Actual prior failed attempt evidence differs')
    _, old_task, old_state, _, _, _, _, _ = _old_read(spec)
    new_task, output = _validate_new_task(spec, old_task, proof['new_task'])
    if expected_index is not None:
        entries = [row for row in expected_index['tasks'] if row['run_id'] == RUN_ID]
        require(len(expected_index['tasks']) == 8 and len(entries) == 1
                and entries[0]['task_path'] == proof['new_task']['path']
                and entries[0]['task_sha256'] == proof['new_task']['sha256'], 'Repair is not the one indexed logical engineering trajectory')
    store_path = Path(proof['new_store_path']).resolve(strict=True)
    require(store_path == output / 'durable' and proof['imported_generation'] == 1, 'Imported store/generation differs')
    meta = _binding(proof['imported_generation_metadata']); statepath = _binding(proof['imported_generation_state'])
    generation_dir = store_path / 'checkpoints/generation-00000000000000000001'
    require(meta == generation_dir / 'metadata.json' and statepath == generation_dir / 'state.bin', 'Imported generation evidence escaped its store')
    identity = _identity(new_task, proof['new_task'])
    # Later complete generations may exist after a successful run; inspect the
    # immutable imported generation, never require final progress to remain zero.
    store = _ReadOnlyStore(store_path, identity, state_validator=lambda value: None)
    imported, info = store._read_generation(1, generation_dir, __import__('pickle').loads, deserialize=True)
    _zero_state(imported)
    expected = copy.deepcopy(old_state); expected['identity'] = identity
    expected['provider_state'] = convert_legacy_empty_state(old_state['provider_state'], initial_data_rng=old_state['learning']['rng'])
    require(info.scientific_cursors == ZERO_CURSORS and _same(imported, expected)
            and _same(imported['learning'], old_state['learning'])
            and _tree_digest(imported['learning']) == proof['learning_value_sha256'], 'Imported learning/RNG or provider migration differs')
    prior = dict(nonce=FAILED_NONCE, run_id=RUN_ID, actual_exit_code=1, actual_outcome=evidence['actual_outcome'],
                 outcome_path=spec['artifacts']['outcome']['path'], outcome_sha256=spec['artifacts']['outcome']['sha256'],
                 scientific_cursors=ZERO_CURSORS.copy(), physical_work=evidence['physical_work'],
                 zero_forward_proof=evidence['zero_forward_proof'], retained_costs=evidence['retained_costs'],
                 unknown_image_read_compute_time=True)
    return dict(status='passed', proof_binding=binding, spec_sha256=proof['spec_sha256'],
                unchanged_learning_required=True, learning_value_sha256=proof['learning_value_sha256'],
                scientific_trajectory_count_increment=0, imported_generation=1,
                prior_failed_engineering_attempts=[prior], old_store_content_unchanged=True,
                model_forward_performed=False, label_files_opened=False)


def execute(task_path, *, admission, launch_record, conflict_check):
    """Guarded CPU-only import action; an existing import is never executed twice.

    The shared guarded entry owns all project locks and live admission. A failure
    after creating a store but before publishing proof is deliberately held for
    review, rather than guessed into a successful import or retried in place.
    """
    from guarded_entry import PLAN, atomic_json
    from round2_protocol import frozen_protocol
    started = time.perf_counter()
    require(admission.get('allow_start') is True and admission.get('boundary_verified') is True,
            'Fresh shared-host admission is required before CPU checkpoint import')
    task_path = Path(task_path).resolve(strict=True)
    task = _json(task_path)
    plan = Path(PLAN).resolve(strict=True)
    require(task.get('schema') == 'odr003-round2-task-v1'
            and task.get('action') == 'import_zero_checkpoint' and task.get('stage') == 'verification'
            and task.get('run_id') == 'g50p2_preforward_import'
            and task.get('protocol') == frozen_protocol()
            and task_path == plan / 'tasks/g50p2_preforward_import.json'
            and launch_record.get('run_id') == task['run_id'], 'Wrong guarded zero-import task/launch')
    code = {path.name: sha256(path) for path in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    require(task.get('implementation_sha256') == code, 'Checkpoint importer code binding differs')
    spec_path = _binding(task['origin_spec']); spec = _json(spec_path)
    require(Path(spec['plan_root']).resolve() == plan, 'Import spec belongs to another plan')
    new_task_binding = task['new_task']; new_task_path = _binding(new_task_binding)
    new_task = _json(new_task_path)
    require(new_task.get('initial_checkpoint_origin') == task['origin_spec'], 'Replacement task origin differs')
    output = Path(new_task['output_dir']).resolve()
    proof_path = Path(task['proof_path']).resolve()
    require(proof_path == plan / 'revisions/preprocessing_rng_repair_20261024/import_proof.json',
            'Checkpoint import proof escaped its one fixed destination')
    receipt_path = output / 'zero_checkpoint_import.json'
    conflict = conflict_check()
    if conflict:
        return dict(exit_code=75, status='waiting_resources', reason=conflict,
                    model_forwards=0, checkpoint_import_performed=False,
                    execution_attempt_elapsed_seconds=time.perf_counter()-started)
    launch_path = plan / 'launches' / str(launch_record.get('run_nonce', '')) / 'launch.json'
    require(launch_record.get('run_nonce') and launch_path.is_file()
            and canonical_sha(_json(launch_path)) == canonical_sha(launch_record),
            'Actual guarded importer launch record differs')
    reused = proof_path.exists()
    if reused:
        proof_binding = _file_binding(proof_path)
        proof = _json(proof_path)
        require(proof['spec'] == spec and proof['new_task'] == new_task_binding,
                'Existing checkpoint import proof belongs to another task')
        execution = proof.get('import_execution', {})
        require(execution.get('task_binding') == _file_binding(task_path),
                'Existing import proof lacks its actual guarded importer task')
        original_launch = _json(_binding(execution['launch_record_binding']))
        require(original_launch.get('run_id') == task['run_id']
                and execution.get('launch_record_sha256') == canonical_sha(original_launch),
                'Existing import proof guarded launch differs')
    else:
        require(not (output / 'durable').exists() and not receipt_path.exists(),
                'An imported store without its immutable proof must be preserved for review')
        imported_at = time.perf_counter()
        proof = import_zero_checkpoint(task['origin_spec'], new_task_binding, output / 'durable')
        proof['import_execution'] = dict(
            task_binding=_file_binding(task_path), launch_record_binding=_file_binding(launch_path),
            launch_record_sha256=canonical_sha(launch_record),
            checkpoint_import_elapsed_seconds=time.perf_counter()-imported_at,
            actual_source_CPU_loads_during_checkpoint_import=2,
            actual_model_forwards=0, scientific_trajectory_count_increment=0,
            resumed_learning_from='original complete generation; no new source initialization',
            additional_independent_verification_cost='recorded in actual entry outcome after proof publication',
            earlier_failed_costs='retained in old_failure_evidence; not replaced by this import')
        require(not proof_path.exists(), 'A successful import proof may not be replaced')
        atomic_json(proof_path, proof)
        proof_binding = _file_binding(proof_path)
    verify_started = time.perf_counter()
    audit = verify_preforward_repair(proof_binding, plan=plan)
    require(audit['status'] == 'passed', 'Independent import readback did not pass')
    verification_elapsed = time.perf_counter()-verify_started
    receipt = dict(status='passed', new_task_sha256=new_task_binding['sha256'],
                   new_task_canonical_sha256=canonical_sha(new_task), origin_spec=task['origin_spec'],
                   proof_binding=proof_binding)
    if receipt_path.exists():
        require(_json(receipt_path) == receipt, 'Existing zero-import receipt differs; never replace it')
    else:
        # A proof left by a crash can reconstruct only this deterministic receipt;
        # the complete state import above is never executed again.
        atomic_json(receipt_path, receipt)
    conflict = conflict_check()
    result = dict(exit_code=75 if conflict else 0,
                  status='waiting_resources' if conflict else 'preforward_repair_passed',
                  reason=conflict, proof=str(proof_path), proof_sha256=proof_binding['sha256'],
                  zero_checkpoint_receipt=str(receipt_path), checkpoint_import_performed=not reused,
                  actual_source_CPU_loads_this_attempt=2 if reused else 4,
                  independent_import_verification_elapsed_seconds=verification_elapsed,
                  execution_attempt_elapsed_seconds=time.perf_counter()-started,
                  model_forwards=0, scientific_trajectory_count_increment=0)
    return result
