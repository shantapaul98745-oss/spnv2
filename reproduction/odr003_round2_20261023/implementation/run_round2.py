"""Real-model entry called only from guarded_entry after shared-host admission.

Engineering trajectories never evaluate developer GT. Complete exploration also
requires a bound independent engineering admission. Reports remain unaccepted
until a separate verifier validates their actual checkpoint, trace and counts.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

from frozen_backend import (sha256, canonical_sha, import_bound_module, FrozenScorers,
                            DeterministicImageProvider, stream_records)
from guarded_entry import atomic_json, PLAN
from round2_protocol import frozen_protocol, build_engineering_matrix, build_complete_matrix, derive_controls


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_runtime_task(task):
    if task.get('schema') != 'odr003-round2-task-v1':
        raise ValueError('Wrong task schema')
    if task['protocol'] != frozen_protocol():
        raise ValueError('Confirmed fixed rules or seeds changed')
    rows = build_engineering_matrix() if task['stage'] == 'engineering' else build_complete_matrix()
    expected = [r for r in rows if r['run_key'] == task['definition']['run_key']]
    if len(expected) != 1 or expected[0] != task['definition']:
        raise ValueError('Run does not belong to the fixed authorized matrix')
    output = Path(task['output_dir']).resolve()
    if not output.is_relative_to(PLAN/'runs') or output.name != task['run_id']:
        raise ValueError('Run output escapes the independent round-two plan')
    baseline = task['baseline_task']
    if (baseline['method'] != 'G50' or baseline['domain'] != task['definition']['domain']
            or baseline['seed'] != task['definition']['stream_seed']
            or baseline['bindings']['source']['sha256'] != frozen_protocol()['source_sha256']):
        raise ValueError('Frozen official-source G50 identity mismatch')
    for name, binding in task['frozen_modules'].items():
        if Path(binding['path']).name != name+'.py' or sha256(binding['path']) != binding['sha256']:
            raise ValueError('Frozen module changed: '+name)
    if task['stage'] == 'complete':
        binding = task['engineering_admission']
        if sha256(binding['path']) != binding['sha256']:
            raise ValueError('Engineering admission hash changed')
        proof = load_json(binding['path'])
        if (proof.get('status') != 'passed' or proof.get('allow_complete_matrix') is not True
                or proof.get('implementation_sha256') != task['implementation_sha256']
                or proof.get('protocol_sha256') != task['protocol']['protocol_sha256']):
            raise ValueError('Complete exploration lacks independent engineering admission')
    elif task['stage'] != 'engineering':
        raise ValueError('Only engineering and complete stages are authorized')
    return output


def controls(task):
    if task['stage'] != 'complete' or task['definition']['group'] == 'P':
        return {}
    binding = task['paired_P_report']
    proof_binding = task['paired_P_proof']
    if sha256(binding['path']) != binding['sha256'] or sha256(proof_binding['path']) != proof_binding['sha256']:
        raise ValueError('Paired P report/proof changed')
    report, proof = load_json(binding['path']), load_json(proof_binding['path'])
    if (proof.get('status') != 'passed' or proof.get('report_sha256') != binding['sha256']
            or report['definition']['group'] != 'P'
            or report['definition']['domain'] != task['definition']['domain']
            or report['definition']['stream_seed'] != task['definition']['stream_seed']
            or report['protocol_sha256'] != task['protocol']['protocol_sha256']
            or report['source_sha256'] != task['protocol']['source_sha256']
            or report['stream_binding'] != task['baseline_task']['bindings']['stream_manifest']):
        raise ValueError('Matched control identity/protocol differs from accepted P')
    result = derive_controls(task['definition']['domain'], task['definition']['stream_seed'],
                             report['counts']['candidates'], report['counts']['commits'])
    if task['control'] != result:
        raise ValueError('Random probability/permutation is not reproducible from P counts')
    return result


def _identity(task, task_path):
    return dict(run_id=task['run_id'], source_sha256=task['protocol']['source_sha256'],
                config_sha256=sha256(task_path),
                manifest_sha256=task['baseline_task']['bindings']['stream_manifest']['sha256'])


def scientific_counts(engine):
    if hasattr(engine, 'counters'):
        return dict(engine.counters)
    a = engine.adapter
    return dict(observed=a.observed, accepted=a.accepted, candidates=a.accepted//4,
                commits=a.updates, rollbacks=0, tail_discarded=a.tail,
                consecutive_rollbacks=0, max_consecutive_rollbacks=0)


def attempt_statistics(started):
    result = dict(execution_attempt_elapsed_seconds=time.monotonic()-started,
                  peak_gpu_allocated_bytes=None, peak_gpu_reserved_bytes=None)
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        result['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
    return result


def _progress(output, task, *, phase, counts, status='running', **extra):
    atomic_json(output/'progress.json', dict(schema='odr003-round2-progress-v1',
                updated_at=datetime.now(timezone.utc).isoformat(), run_id=task['run_id'],
                phase=phase, counts=counts, status=status, **extra))


def execute(task_path, *, admission, launch_record, conflict_check):
    # Only guarded_entry supplies actual freshly verified live admission.
    if not admission.get('allow_start') or not admission.get('boundary_verified'):
        raise RuntimeError('Shared-host admission is required before heavy imports')
    task_path = Path(task_path)
    task = load_json(task_path)
    output = verify_runtime_task(task)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        raise RuntimeError('An existing completed run must be independently verified, never rerun')
    if (output/'failure.json').exists():
        raise RuntimeError('Preserved engineering failure blocks automatic retry')
    installed_task = output/'task.json'
    if installed_task.exists() and load_json(installed_task) != task:
        raise ValueError('Existing run belongs to different immutable settings')
    if not installed_task.exists(): atomic_json(installed_task, task)
    _progress(output, task, phase='loading_frozen_assets', counts={})
    adapter = None
    engine = None
    started = time.monotonic()
    from durable_state import DurableRunStore
    # Check permissions and protocols without opening developer labels.
    frozen = {name: import_bound_module(name, binding) for name, binding in task['frozen_modules'].items()
              if name not in {'prediction'}}
    baseline = frozen['run_exploration']
    boundary = baseline.GTBoundary(task['baseline_task']['bindings']['developer_labels']['path'])
    try:
        paths, receipt, core, prepared = baseline.verify_bindings(task['baseline_task'])
        configuration_task = {**task['baseline_task'], 'output_dir': str(output), 'run_id': task['run_id']}
        cfg, overrides = baseline.configure(configuration_task, paths, core)
        from nets.build import SPNv2
        from dataset.transforms.build import build_transforms
        from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints, set_seeds_cudnn
        import torch
        import numpy as np
        from state_adapter import OriginalAAdapter, rng_state, restore_rng, equal_tree
        from protected_learning import WindowLearner
        from protected_engine import ProtectedEngine, validate_engine_state, ControlledInterruption
        from guard_pools import GuardPools
        prediction = import_bound_module('prediction', task['frozen_modules']['prediction'])
        if not torch.cuda.is_available():
            raise RuntimeError('Actual GPU trajectory cannot be replaced with a CPU result')
        # Catch a foreign task that started during slow asset verification.
        conflict = conflict_check()
        if conflict:
            _progress(output, task, phase='resource_wait', counts={}, status='waiting_resources', reason=conflict)
            return dict(status='waiting_resources', exit_code=75, reason=conflict, **attempt_statistics(started))
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        set_seeds_cudnn(cfg, seed=task['definition']['stream_seed'])
        torch.cuda.reset_peak_memory_stats(device)
        model = SPNv2(cfg)
        source = torch.load(paths['source'], map_location='cpu', weights_only=True)
        key = receipt.get('state_dict_key')
        state = source if key is None else source[key]
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ValueError('Official source strict load failed')
        del source, state
        model = model.float().to(device)
        adapter = OriginalAAdapter.configure(model, device, batch=4)
        operator_warmup = prediction.warmup_bbox_decoder(model)
        baseline.finite_state(model, adapter.optimizer)
        camera = load_camera_intrinsics(str(paths['camera']))
        points = load_tango_3d_keypoints(str(paths['keypoints']))
        calibration = baseline.load_calibration(task['baseline_task'], paths, prepared)
        transforms = build_transforms(cfg, is_train=False, load_labels=False)
        dataset = frozen['data_boundary'].ImageManifestDataset(paths['stream_manifest'], transforms=transforms)
        if len(dataset) != 1024 or dataset.manifest['seed'] != task['definition']['stream_seed']:
            raise ValueError('Frozen observation order/size changed')
        preprocess = canonical_sha(dict(config=task['baseline_task']['bindings']['config'],
                         core=task['baseline_task']['bindings']['source_core_manifest'],
                         data_code=task['frozen_modules']['data_boundary']))
        provider = DeterministicImageProvider(dataset, preprocess_sha256=preprocess,
                         rng_state=rng_state, restore_rng=restore_rng, equal_tree=equal_tree,
                         manifest_file_sha256=task['baseline_task']['bindings']['stream_manifest']['sha256'])
        records = stream_records(dataset, preprocess, limit=task['definition']['observations'])
        scorers = FrozenScorers(model=model, adapter=adapter, cfg=cfg, camera=camera,
                         points=points, calibration=calibration, prediction_module=prediction,
                         gate_module=frozen['gate_core'])
        seeds = task['definition']['rng_seeds']
        pools = GuardPools(uniform_seed=seeds['uniform_pool'], fragile_seed=seeds['vulnerable_pool'],
                           heatmap_threshold=float(cfg.TEST.HEATMAP_THRESHOLD))
        identity = _identity(task, task_path)
        sync = lambda: torch.cuda.synchronize(device)
        trajectory = task['definition'].get('trajectory')
        if trajectory == 'G50_reference':
            from reference_driver import ReferenceDriver, validate_reference_state
            store = DurableRunStore(output/'durable', identity, state_validator=validate_reference_state)
            engine = ReferenceDriver(stream_records=records, adapter=adapter, store=store,
                         image_provider=provider, scorer=scorers.arrival, gate=scorers.gate, sync=sync)
        else:
            mode = ('disabled' if trajectory == 'P_disabled' else 'P') if task['stage'] == 'engineering' else task['definition']['group']
            control = controls(task)
            store = DurableRunStore(output/'durable', identity, state_validator=validate_engine_state)
            args = {}
            if mode == 'RW':
                rw = control['RW']
                if not rw['enabled']: raise ValueError('Q_P=0: paired RW is undefined and must be skipped')
                args = dict(rw_probability=rw['commit_probability'], rw_seed=seeds['rw_coin'])
            elif mode == 'RP':
                selected = set(control['RP']['selected_positions'])
                args = dict(rp_mask=[i in selected for i in range(1024)])
            engine = ProtectedEngine(mode=mode, stream_records=records, learner=WindowLearner(adapter),
                         pools=pools, store=store, image_provider=provider, arrival_scorer=scorers.arrival,
                         protection_scorer=scorers.protection, gate=scorers.gate, sync=sync,
                         fault_points=('candidate_2_tentative', 'observation_31') if trajectory == 'P_interrupted_resumed' else (),
                         verify_purity=task['stage'] == 'engineering', **args)
        recovered = engine.resume()
        atomic_json(output/'effective_config_overrides.json', overrides)
        boundary.phase = 'adaptation'
        last_progress = last_resource_check = time.monotonic()
        while engine.step():
            baseline.finite_state(model, adapter.optimizer)
            now = time.monotonic()
            if now-last_progress >= 60:
                _progress(output, task, phase='adaptation', counts=scientific_counts(engine))
                last_progress = now
            if now-last_resource_check >= 60:
                conflict = conflict_check()
                last_resource_check = now
                if conflict:
                    # Cooperative exit only: nobody else's process is signaled.
                    engine.checkpoint('shared_host_cooperative_yield')
                    _progress(output, task, phase='resource_wait', counts=scientific_counts(engine),
                              status='waiting_resources', reason=conflict)
                    return dict(status='waiting_resources', exit_code=75, reason=conflict, **attempt_statistics(started))
        saved = store.recover()
        final_state = saved.state
        learning = final_state['learning'] if trajectory == 'G50_reference' else final_state['learner']['learning']
        if not equal_tree(learning['model'], model.state_dict()):
            raise ValueError('Actual durable final model differs from live model')
        model.load_state_dict(learning['model'], strict=True)
        if not equal_tree(learning['model'], model.state_dict()):
            raise ValueError('Final strict reload did not reproduce all model tensors')
        evaluation = None
        if task['stage'] == 'complete':
            evaluation = evaluate_developer(task, paths, cfg, camera, points, prediction,
                         frozen['data_boundary'], transforms, model, adapter, boundary, store, final_state,
                         output, conflict_check, sync)
            if evaluation.get('waiting_resources'):
                return dict(status='waiting_resources', exit_code=75, reason=evaluation['reason'], **attempt_statistics(started))
        if boundary.denied:
            raise ValueError('GT boundary violation was attempted')
        sync()
        saved = store.recover()
        report = dict(schema='odr003-round2-run-report-v1', status='main_completed_pending_independent_verification',
                     run_id=task['run_id'], stage=task['stage'], definition=task['definition'],
                     task_sha256=sha256(task_path), implementation_sha256=task['implementation_sha256'],
                     protocol_sha256=task['protocol']['protocol_sha256'], source_sha256=task['protocol']['source_sha256'],
                     stream_binding=task['baseline_task']['bindings']['stream_manifest'],
                     counts=scientific_counts(engine), final_generation=saved.info.generation,
                     final_state_sha256=saved.info.metadata['state_sha256'],
                     strict_source_load=True, strict_final_reload=True, operator_warmup=operator_warmup,
                     physical_work=store.work_summary(), evaluation=evaluation,
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     attempt_elapsed_seconds=time.monotonic()-started, recovered_generation=None if recovered is None else recovered.generation,
                     launch_record_sha256=canonical_sha(launch_record), torch_version=torch.__version__,
                     numpy_version=np.__version__, gt_denied_attempts=boundary.denied,
                     scientific_scope='exploration_not_formal_efficacy')
        atomic_json(output/'report.json', report)
        _progress(output, task, phase='awaiting_independent_verification', counts=scientific_counts(engine), status='main_complete')
        return dict(status='main_complete', exit_code=0, report=str(output/'report.json'), **attempt_statistics(started))
    except Exception as error:
        # Planned engineering process termination preserves its previous complete
        # checkpoint. It is not permission to retry an arbitrary model failure.
        if type(error).__name__ == 'ControlledInterruption':
            _progress(output, task, phase='controlled_interruption', counts=scientific_counts(engine),
                      status='resume_required', point=str(error))
            return dict(status='controlled_interruption', exit_code=76, point=str(error), **attempt_statistics(started))
        atomic_json(output/'failure.json', dict(status='failed', type=type(error).__name__,
                    message=str(error), traceback=traceback.format_exc(),
                    action='preserve_and_block_downstream_no_automatic_parameter_changes', **attempt_statistics(started)))
        raise
    finally:
        if adapter is not None: adapter.close()


def evaluate_developer(task, paths, cfg, camera, points, prediction, data_module, transforms,
                       model, adapter, boundary, store, final_state, output, conflict_check, sync):
    """Developer-only evaluation resumes its own complete state; no reserve read."""
    from state_adapter import equal_tree
    boundary.phase = 'evaluation'
    if sha256(paths['developer_labels']) != task['baseline_task']['bindings']['developer_labels']['sha256']:
        raise ValueError('Developer labels changed')
    dev = data_module.ImageManifestDataset(paths['developer_manifest'], transforms=transforms)
    label_doc = load_json(paths['developer_labels'])
    labels = label_doc['items']
    if (len(labels) != 512 or [x['sample_id'] for x in labels] != [x['sample_id'] for x in dev.manifest['items']]
            or label_doc['image_manifest_sha256'] != task['baseline_task']['bindings']['developer_manifest']['sha256']):
        raise ValueError('Developer boundary changed')
    state = store.recover().state
    evaluation = copy.deepcopy(state.get('evaluation', dict(next_index=0, rows=[])))
    validate_evaluation_state(evaluation, [row['sample_id'] for row in labels])
    attempts = {}
    for event in store.read_journal()[0]:
        if event['kind'] == 'work_started':
            attempts[event['payload']['logical_key']] = event['payload']['attempt_id']
    frozen_model = final_state['learner']['learning']['model']
    for i in range(evaluation['next_index'], 512):
        if i % 64 == 0:
            reason = conflict_check()
            if reason:
                _progress(output, task, phase='resource_wait', counts=state['counters'],
                          status='waiting_resources', reason=reason, evaluated=evaluation['next_index'])
                return dict(waiting_resources=True, reason=reason)
        key = f'developer:{i:04d}'
        attempt = store.begin_work('developer_evaluation', {'developer_evaluation': 1},
                                    logical_key=key, replay_of=attempts.get(key))
        t = time.perf_counter()
        sample = dev[i]
        record = prediction.extract(model, adapter, sample['image'].unsqueeze(0), cfg, camera, points, profile='full')
        metrics = prediction.developer_metrics(record, dict(quaternion=labels[i]['quaternion_wxyz'],
                                                           translation=labels[i]['translation_m']), cfg)
        sync()
        store.finish_work(attempt, completed_forwards={'developer_evaluation': 1}, compute_seconds=time.perf_counter()-t)
        record.pop('timing_seconds', None)
        evaluation['rows'].append(dict(sample_id=sample['sample_id'], prediction=record, metrics=metrics))
        evaluation['next_index'] = i+1
        if (i+1) % 64 == 0:
            validate_evaluation_state(evaluation, [row['sample_id'] for row in labels])
            if not equal_tree(frozen_model, model.state_dict()):
                raise ValueError('Evaluation changed model before checkpoint publication')
            checkpoint_started = time.perf_counter()
            state['evaluation'] = copy.deepcopy(evaluation)
            info = store.save(state, dict(observed=state['counters']['observed'], candidates=state['counters']['candidates'],
                       commits=state['counters']['commits'], rollbacks=state['counters']['rollbacks'], next_index=state['next_index']))
            store.append_event('checkpoint_published', dict(generation=info.generation,
                               reason='developer_evaluation', checkpoint_seconds=time.perf_counter()-checkpoint_started,
                               evaluated=evaluation['next_index']))
            _progress(output, task, phase='developer_evaluation', counts=state['counters'], evaluated=i+1)
    if not equal_tree(frozen_model, model.state_dict()):
        raise ValueError('Evaluation changed the frozen adapted model')
    validate_evaluation_state(evaluation, [row['sample_id'] for row in labels], complete=True)
    atomic_json(output/'developer_predictions.json', evaluation)
    return dict(count=512, extended_valid=sum(x['metrics']['valid'] for x in evaluation['rows']),
                original_valid=sum(x['metrics']['original_valid'] for x in evaluation['rows']),
                prediction_path=str(output/'developer_predictions.json'),
                prediction_sha256=sha256(output/'developer_predictions.json'))


def validate_evaluation_state(evaluation, developer_ids, *, complete=False):
    import math
    if (type(evaluation.get('next_index')) is not int or not isinstance(evaluation.get('rows'), list)
            or not 0 <= evaluation['next_index'] <= 512 or len(developer_ids) != 512
            or len(set(developer_ids)) != 512 or len(evaluation['rows']) != evaluation['next_index']
            or (complete and evaluation['next_index'] != 512)):
        raise ValueError('Incomplete or inconsistent developer checkpoint cursor/rows')
    for i, row in enumerate(evaluation['rows']):
        if (set(row) != {'sample_id', 'prediction', 'metrics'} or row['sample_id'] != developer_ids[i]
                or not isinstance(row['prediction'], dict) or not isinstance(row['metrics'], dict)
                or not {'valid', 'error', 'original_valid', 'original_error', 'rotation_deg', 'translation_m'} <= row['metrics'].keys()
                or not {'h_available', 'e_available', 'original_h_rejected', 'heat_q', 'effi_t'} <= row['prediction'].keys()):
            raise ValueError('Developer checkpoint row schema/order changed')
        for flag, value in [('valid', 'error'), ('original_valid', 'original_error')]:
            if type(row['metrics'][flag]) is not bool or row['metrics'][flag] != (row['metrics'][value] is not None):
                raise ValueError('Developer finite denominator flag disagrees with value')
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite developer checkpoint value')
            if isinstance(value, dict):
                for item in value.values(): finite(item)
            elif isinstance(value, list):
                for item in value: finite(item)
        finite(row)
