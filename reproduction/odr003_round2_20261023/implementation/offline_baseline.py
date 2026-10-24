"""Two deterministic official-source developer evaluations, without ODR.

This entry is reachable only through guarded_entry's actual shared-host
admission.  It creates a complete source checkpoint before any developer pass;
evaluation recovery reuses the existing 64-row durable transaction mechanism.
No observation-stream image is loaded, selected, scored or adapted here.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
import traceback

from frozen_backend import sha256, canonical_sha, import_bound_module
from guarded_entry import PLAN, atomic_json
from round2_protocol import frozen_protocol, build_baseline_matrix
from deterministic_runtime import prepare_environment, apply, validate_live
from run_round2 import (load_json, _identity, _progress, attempt_statistics,
                        bind_deterministic_runtime, evaluate_developer)


SCHEMA = 'odr003-round2-offline-source-v1'
ZERO_COUNTS = dict(observed=0, accepted=0, candidates=0, commits=0, rollbacks=0,
                   tail_discarded=0, consecutive_rollbacks=0, max_consecutive_rollbacks=0)
ZERO_CURSORS = dict(observed=0, candidates=0, commits=0, rollbacks=0, next_index=0)


def validate_offline_state(state):
    required = {'schema', 'identity', 'learning', 'finished_source_checkpoint',
                'counters', 'next_index', 'ledger_boundary'}
    if (not isinstance(state, dict) or not required <= state.keys()
            or state['schema'] != SCHEMA or state['finished_source_checkpoint'] is not True
            or state['counters'] != ZERO_COUNTS or type(state['next_index']) is not int or state['next_index'] != 0
            or not isinstance(state['counters'], dict)
            or any(type(value) is not int for value in state['counters'].values())):
        raise ValueError('Offline source state cannot contain observations or updates')
    needed = {'model', 'buffers', 'optimizer', 'optimizer_type', 'optimizer_parameter_names',
              'modes', 'requires_grad', 'gradients', 'bn_features', 'bn_mean', 'bn_mean_sq',
              'boundary_buffers', 'rng', 'scaler', 'counters'}
    learning = state['learning']
    if (not isinstance(learning, dict) or not needed <= learning.keys()
            or learning['counters'] != dict(observed=0, accepted=0, updates=0, tail=0)
            or any(type(value) is not int for value in learning['counters'].values())
            or any(gradient is not None for gradient in learning['gradients'].values())
            or learning['optimizer'].get('state') or learning['scaler'] is not None):
        raise ValueError('Offline checkpoint contains adaptation state')


def verify_offline_task(task):
    if (task.get('schema') != 'odr003-round2-task-v1' or task.get('action') != 'run_offline'
            or task.get('stage') != 'offline' or task.get('protocol') != frozen_protocol()):
        raise ValueError('Wrong fixed offline task/protocol')
    matches = [row for row in build_baseline_matrix() if row['group'] == 'O'
               and row['run_key'] == task['definition']['run_key']]
    if len(matches) != 1 or matches[0] != task['definition']:
        raise ValueError('Offline evaluation is outside the two authorized definitions')
    output = Path(task['output_dir']).resolve()
    if not output.is_relative_to(PLAN/'runs') or output.name != task['run_id']:
        raise ValueError('Offline output escapes this experiment')
    baseline = task['baseline_task']
    if (baseline['method'] != 'G50' or baseline['domain'] != task['definition']['domain']
            or baseline['seed'] != 2021
            or baseline['bindings']['source']['sha256'] != task['protocol']['source_sha256']):
        raise ValueError('Offline official source/domain identity differs')
    for name, binding in task['frozen_modules'].items():
        if Path(binding['path']).name != name+'.py' or sha256(binding['path']) != binding['sha256']:
            raise ValueError('Offline frozen module changed: '+name)
    binding = task['engineering_admission']
    if sha256(binding['path']) != binding['sha256']:
        raise ValueError('Offline engineering admission changed')
    proof = load_json(binding['path'])
    if (proof.get('status') != 'passed' or proof.get('allow_complete_matrix') is not True
            or proof.get('implementation_sha256') != task['implementation_sha256']
            or proof.get('protocol_sha256') != task['protocol']['protocol_sha256']):
        raise ValueError('Offline evaluation lacks independent engineering admission')
    return output


def source_checkpoint(store, learner, identity):
    """Reuse a complete source state or publish exactly one initial state."""
    recovered = store.recover()
    if recovered is not None:
        state = recovered.state
        validate_offline_state(state)
        if state['identity'] != identity or recovered.scientific_cursors != ZERO_CURSORS:
            raise ValueError('Offline checkpoint identity/cursors differ')
        from state_adapter import equal_tree
        if not equal_tree(state['learning']['model'], learner.adapter.model.state_dict()):
            raise ValueError('Offline saved model differs from the freshly strict-loaded source')
        learner._restore_learning(state['learning'])
        return recovered
    state = dict(schema=SCHEMA, identity=copy.deepcopy(identity),
                 learning=learner._capture_learning(), finished_source_checkpoint=True,
                 counters=dict(ZERO_COUNTS), next_index=0, ledger_boundary=store.journal_boundary())
    validate_offline_state(state)
    store.save(state, dict(ZERO_CURSORS))
    return store.recover()


def execute(task_path, *, admission, launch_record, conflict_check):
    if not admission.get('allow_start') or not admission.get('boundary_verified'):
        raise RuntimeError('Shared-host admission is required before offline heavy imports')
    prepare_environment()
    task_path = Path(task_path)
    task = load_json(task_path)
    output = verify_offline_task(task)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        raise RuntimeError('Completed offline evaluation must be verified, never rerun')
    if (output/'failure.json').exists():
        raise RuntimeError('Preserved offline failure blocks automatic retry')
    installed = output/'task.json'
    if installed.exists() and load_json(installed) != task:
        raise ValueError('Offline output belongs to another immutable task')
    if not installed.exists(): atomic_json(installed, task)
    started = time.monotonic()
    adapter = None
    _progress(output, task, phase='loading_official_source', counts=ZERO_COUNTS)
    frozen = {name: import_bound_module(name, binding) for name, binding in task['frozen_modules'].items()
              if name != 'prediction'}
    baseline = frozen['run_exploration']
    boundary = baseline.GTBoundary(task['baseline_task']['bindings']['developer_labels']['path'])
    try:
        paths, receipt, core, prepared = baseline.verify_bindings(task['baseline_task'])
        cfg, overrides = baseline.configure({**task['baseline_task'], 'output_dir': str(output),
                                            'run_id': task['run_id']}, paths, core)
        from nets.build import SPNv2
        from dataset.transforms.build import build_transforms
        from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints, set_seeds_cudnn
        import torch
        import numpy as np
        from state_adapter import OriginalAAdapter, equal_tree
        from protected_learning import WindowLearner
        from durable_state import DurableRunStore
        prediction = import_bound_module('prediction', task['frozen_modules']['prediction'])
        if not torch.cuda.is_available():
            raise RuntimeError('Offline real-model evaluation requires the admitted GPU')
        reason = conflict_check()
        if reason:
            _progress(output, task, phase='resource_wait', counts=ZERO_COUNTS,
                      status='waiting_resources', reason=reason)
            return dict(status='waiting_resources', exit_code=75, reason=reason, **attempt_statistics(started))
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        set_seeds_cudnn(cfg, seed=2021)
        deterministic = bind_deterministic_runtime(output, apply(torch))
        torch.cuda.reset_peak_memory_stats(device)
        model = SPNv2(cfg)
        source = torch.load(paths['source'], map_location='cpu', weights_only=True)
        source_state = source if receipt.get('state_dict_key') is None else source[receipt['state_dict_key']]
        loaded = model.load_state_dict(source_state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError('Offline official source strict load failed')
        del source, source_state
        model = model.float().to(device)
        adapter = OriginalAAdapter.configure(model, device, batch=4)
        learner = WindowLearner(adapter)
        operator_warmup = prediction.warmup_bbox_decoder(model)
        baseline.finite_state(model, adapter.optimizer)
        camera = load_camera_intrinsics(str(paths['camera']))
        points = load_tango_3d_keypoints(str(paths['keypoints']))
        transforms = build_transforms(cfg, is_train=False, load_labels=False)
        identity = _identity(task, task_path)
        store = DurableRunStore(output/'durable', identity, state_validator=validate_offline_state)
        saved = source_checkpoint(store, learner, identity)
        recovered_generation = saved.info.generation
        validate_live(deterministic, torch)
        model.load_state_dict(saved.state['learning']['model'], strict=True)
        if not equal_tree(saved.state['learning']['model'], model.state_dict()):
            raise ValueError('Offline strict source reload differs')
        atomic_json(output/'effective_config_overrides.json', overrides)
        evaluation = evaluate_developer(task, paths, cfg, camera, points, prediction,
                     frozen['data_boundary'], transforms, model, adapter, boundary, store,
                     saved.state, output, conflict_check, lambda: torch.cuda.synchronize(device),
                     deterministic_receipt=deterministic)
        if evaluation.get('waiting_resources'):
            return dict(status='waiting_resources', exit_code=75, reason=evaluation['reason'], **attempt_statistics(started))
        validate_live(deterministic, torch)
        if boundary.denied:
            raise ValueError('Offline GT boundary violation was attempted')
        saved = store.recover()
        validate_offline_state(saved.state)
        if not equal_tree(saved.state['learning']['model'], model.state_dict()):
            raise ValueError('Offline developer evaluation changed source weights')
        report = dict(schema='odr003-round2-run-report-v1', status='main_completed_pending_independent_verification',
                     run_id=task['run_id'], stage=task['stage'], definition=task['definition'],
                     task_sha256=sha256(task_path), implementation_sha256=task['implementation_sha256'],
                     protocol_sha256=task['protocol']['protocol_sha256'], source_sha256=task['protocol']['source_sha256'],
                     stream_binding=task['baseline_task']['bindings']['stream_manifest'], counts=dict(ZERO_COUNTS),
                     final_generation=saved.info.generation, final_state_sha256=saved.info.metadata['state_sha256'],
                     strict_source_load=True, strict_final_reload=True, operator_warmup=operator_warmup,
                     deterministic_runtime=deterministic,
                     deterministic_runtime_binding=dict(path=str(output/'deterministic_runtime.json'),
                                                        sha256=sha256(output/'deterministic_runtime.json')),
                     physical_work=store.work_summary(), evaluation=evaluation,
                     outputs=dict(predictions=evaluation['flat_predictions']),
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     attempt_elapsed_seconds=time.monotonic()-started, recovered_generation=recovered_generation,
                     launch_record_sha256=canonical_sha(launch_record), torch_version=torch.__version__,
                     numpy_version=np.__version__, gt_denied_attempts=boundary.denied,
                     adaptation_model_forwards=0, observation_stream_images_read=0,
                     scientific_scope='exploration_not_formal_efficacy')
        atomic_json(output/'report.json', report)
        _progress(output, task, phase='awaiting_independent_verification', counts=ZERO_COUNTS, status='main_complete')
        return dict(status='main_complete', exit_code=0, report=str(output/'report.json'), **attempt_statistics(started))
    except Exception as error:
        atomic_json(output/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
                    traceback=traceback.format_exc(), action='preserve_and_block_downstream', **attempt_statistics(started)))
        raise
    finally:
        if adapter is not None: adapter.close()
