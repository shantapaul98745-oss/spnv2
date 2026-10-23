"""Read-only CPU diagnostics of already completed engineering checkpoints.

This action cannot certify engineering acceptance, instantiate a model, read GT,
or execute a forward. Its caller supplies the unchanged shared-host admission.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import time


PLAN = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_round2_20261023')
torch = np = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def checked(binding):
    require(set(binding) == {'path', 'sha256'}, 'Invalid diagnostic metadata binding')
    path = Path(binding['path']).resolve(strict=True)
    require(sha(path.read_bytes()) == binding['sha256'], 'Diagnostic metadata changed: ' + str(path))
    return path


def numeric(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return dict(dtype_a=str(a.dtype), dtype_b=str(b.dtype), shape_a=list(a.shape), shape_b=list(b.shape))
    if torch.is_tensor(a):
        require(a.device.type == b.device.type == 'cpu', 'Diagnostic tensors must remain CPU-only')
        same = torch.equal(a, b)
        if same: return None
        mask = a != b
        aa, bb = a.to(torch.float64), b.to(torch.float64)
        delta = (aa-bb).abs()
        finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
        return dict(dtype=str(a.dtype), shape=list(a.shape), mismatched_elements=int(mask.sum()),
                    total_elements=a.numel(), all_finite=finite,
                    max_absolute=float(delta.max()) if finite and a.numel() else None,
                    max_relative=float((delta / torch.maximum(aa.abs(), bb.abs()).clamp_min(1e-30)).max())
                    if finite and a.numel() else None)
    if np.array_equal(a, b): return None
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    finite = bool(np.isfinite(aa).all() and np.isfinite(bb).all())
    delta = np.abs(aa-bb)
    return dict(dtype=str(a.dtype), shape=list(a.shape), mismatched_elements=int(np.count_nonzero(a != b)),
                total_elements=int(a.size), all_finite=finite,
                max_absolute=float(delta.max()) if finite and a.size else None,
                max_relative=float((delta / np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1e-30)).max())
                if finite and a.size else None)


def first_difference(a, b, path='root'):
    if torch.is_tensor(a) or isinstance(a, np.ndarray):
        if not (type(a) is type(b)):
            return dict(path=path, type_a=type(a).__name__, type_b=type(b).__name__)
        diff = numeric(a, b)
        return None if diff is None else dict(path=path, **diff)
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return dict(path=path, reason='mapping_keys_or_type_differ')
        for key in a:
            diff = first_difference(a[key], b[key], path+'.'+str(key))
            if diff is not None: return diff
        return None
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            return dict(path=path, reason='sequence_type_or_length_differ')
        for i, (x, y) in enumerate(zip(a, b)):
            diff = first_difference(x, y, path+'.'+str(i))
            if diff is not None: return diff
        return None
    if type(a) is not type(b) or a != b:
        diff = dict(path=path, value_a=str(a)[:100], value_b=str(b)[:100])
        if type(a) in (int, float) and type(b) in (int, float):
            diff['absolute_difference'] = abs(a-b) if math.isfinite(a) and math.isfinite(b) else None
        return diff
    return None


def learning(state):
    return state['learning'] if 'learning' in state else state['learner']['learning']


def observations(state):
    return state['observation_trace'] if 'observation_trace' in state else [
        row for row in state['scientific_trace'] if row['kind'] == 'observation']


def compare(a, b):
    aa, bb = learning(a), learning(b)
    changed = []
    for name in aa['model']:
        d = first_difference(aa['model'][name], bb['model'][name], name)
        if d is not None: changed.append(dict(name=name, **d))
    oa, ob = observations(a), observations(b)
    first_arrival = None
    for i, (x, y) in enumerate(zip(oa, ob)):
        d = first_difference(x['arrival'], y['arrival'], 'arrival')
        if d is not None:
            first_arrival = dict(index=i, sample_id_a=x['sample_id'], sample_id_b=y['sample_id'], difference=d)
            break
    pa, pb = a['provider_state'], b['provider_state']
    ids = sorted(set(pa['tensor_sha256']) | set(pb['tensor_sha256']))
    different_ids = [sid for sid in ids if pa['tensor_sha256'].get(sid) != pb['tensor_sha256'].get(sid)]
    return dict(learning_component_first_differences={key: first_difference(aa.get(key), bb.get(key), key)
                                                     for key in sorted(set(aa) | set(bb))},
                model_mismatched_tensor_count=len(changed), model_tensor_count=len(aa['model']),
                model_mismatched_names=[row['name'] for row in changed], model_first_ten_differences=changed[:10],
                model_max_absolute_difference=max((row.get('max_absolute') or 0. for row in changed), default=0.),
                model_largest_ten_differences=sorted(changed, key=lambda row: row.get('max_absolute') or 0., reverse=True)[:10],
                counters_a=a.get('counters', aa['counters']), counters_b=b.get('counters', bb['counters']),
                observations_a=len(oa), observations_b=len(ob), first_arrival_difference=first_arrival,
                accepted_ids_equal=[x['sample_id'] for x in oa if x['accepted']] == [x['sample_id'] for x in ob if x['accepted']],
                first_acceptance_difference=first_difference([(x['sample_id'], x['accepted']) for x in oa],
                                                            [(x['sample_id'], x['accepted']) for x in ob], 'acceptance'),
                first_window_difference=first_difference(
                    [x for x in a.get('scientific_trace', []) if x['kind']=='window'],
                    [x for x in b.get('scientific_trace', []) if x['kind']=='window'], 'windows'),
                pool_first_difference=first_difference(a.get('pools'), b.get('pools'), 'pools'),
                provider=dict(tensor_digests_identical=not different_ids, mismatched_digest_count=len(different_ids),
                              first_ten_mismatched_ids=different_ids[:10],
                              materialization_order_equal=pa['materialization_order'] == pb['materialization_order'],
                              rng_first_differences={key: first_difference(pa.get(key), pb.get(key), key)
                                                    for key in ('initial_data_rng', 'data_rng', 'rng_replay')}))


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda raw: torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load_generation(task, report, number):
    directory = Path(task['output_dir'])/'durable/checkpoints'/f'generation-{number:020d}'
    raw = (directory/'metadata.json').read_bytes()
    require(sha(raw) == (directory/'metadata.sha256').read_text(encoding='ascii').strip(), 'Saved metadata checksum differs')
    metadata = json.loads(raw)
    expected = dict(run_id=task['run_id'], source_sha256=task['protocol']['source_sha256'],
                    config_sha256=report['task_sha256'],
                    manifest_sha256=task['baseline_task']['bindings']['stream_manifest']['sha256'])
    require(metadata['generation'] == number and metadata['identity'] == expected, 'Saved generation identity differs')
    blob = (directory/'state.bin').read_bytes()
    require(len(blob) == metadata['state_bytes'] and sha(blob) == metadata['state_sha256'], 'Saved state checksum differs')
    if number == report['final_generation']:
        require(metadata['state_sha256'] == report['final_state_sha256'], 'Report final-state binding differs')
    state = CPUUnpickler(io.BytesIO(blob)).load()
    require(state['identity'] == expected, 'Deserialized saved state identity differs')
    return state, dict(generation=number, metadata_sha256=sha(raw), state_sha256=metadata['state_sha256'],
                       scientific_cursors=metadata['scientific_cursors'])


def execute(task_path, *, admission, launch_record, conflict_check):
    started = time.monotonic()
    require(admission.get('allow_start') is True and admission.get('boundary_verified') is True,
            'Fresh shared-host admission is required for saved-state CPU diagnosis')
    task = read(task_path)
    require(task.get('action') in ('diagnose_saved_state', 'saved_state_diagnostic')
            and task.get('stage') == 'verification' and launch_record['run_id'] == task['run_id'], 'Wrong diagnostic task/launch')
    def audit(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0]))
            if ('speedplus' in p.parts and ('labels' in p.parts or p.suffix.lower() == '.csv')) or p.name in ('dev_labels.json', 'test.json'):
                raise PermissionError('Saved-state diagnostic forbids target GT reads')
    import sys
    sys.addaudithook(audit)
    global torch, np
    import torch
    import numpy as np
    require(not torch.cuda.is_initialized(), 'CPU diagnostic must not initialize CUDA')
    index = read(checked(task['engineering_index']))
    require(len(index['tasks']) == 8, 'Diagnosis is restricted to the eight fixed saved trajectories')
    receipts = read(checked(task['execution_receipts']))
    by_id = {row['run_id']:row for row in receipts['runs']}
    require(len(by_id)==8 and set(by_id)=={row['run_id'] for row in index['tasks']},
            'Diagnostic receipt/index set differs')
    runs = {}
    for row in index['tasks']:
        rt = read(checked(dict(path=row['task_path'], sha256=row['task_sha256'])))
        report_path = Path(rt['output_dir'])/'report.json'
        receipt = by_id[rt['run_id']]
        require(str(report_path)==receipt['report_path'], 'Diagnostic report location differs')
        report = read(checked(dict(path=str(report_path), sha256=receipt['report_sha256'])))
        require(report['run_id'] == rt['run_id'] and report['task_sha256'] == row['task_sha256'], 'Saved main report/task differs')
        runs[(rt['definition']['domain'], rt['definition']['trajectory'])] = (rt, report)
    results = []
    for domain in ('lightbox', 'sunlamp'):
        for left, right in (('G50_reference', 'P_disabled'), ('P_continuous', 'P_interrupted_resumed')):
            ta, ra = runs[(domain, left)]; tb, rb = runs[(domain, right)]
            phases = [('initial', 1, 1), ('final', ra['final_generation'], rb['final_generation'])]
            def boundary64(t, r):
                for i in range(1, r['final_generation']+1):
                    p = Path(t['output_dir'])/'durable/checkpoints'/f'generation-{i:020d}'/'metadata.json'
                    if read(p)['scientific_cursors']['next_index'] == 64: return i
                return None
            ga, gb = boundary64(ta, ra), boundary64(tb, rb)
            if ga is not None and gb is not None: phases.insert(1, ('observation64', ga, gb))
            for phase, ga, gb in phases:
                if conflict_check(): return dict(exit_code=75, status='waiting_resources', model_forwards=0, proof_published=False)
                a, ma = load_generation(ta, ra, ga); b, mb = load_generation(tb, rb, gb)
                result = compare(a, b)
                results.append(dict(domain=domain, left=left, right=right, phase=phase, state_a=ma, state_b=mb, **result))
                del a, b, result
                gc.collect()
    require(not torch.cuda.is_initialized(), 'CPU diagnosis initialized CUDA unexpectedly')
    output = Path(task['diagnostic_output']).resolve()
    require(output.is_relative_to(PLAN/'revisions') and not output.exists(), 'Diagnostic report must be a fresh revision artifact')
    result = dict(schema='odr003-round2-saved-state-diagnostic-v1', status='diagnostic_only_not_engineering_acceptance',
                  task_sha256=sha(Path(task_path).read_bytes()), engineering_index=task['engineering_index'], comparisons=results,
                  models_constructed=0, model_forwards=0, GT_read=False, source_or_saved_state_changed=False)
    raw = (json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()
    with output.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return dict(exit_code=0, status='saved_state_diagnostic_complete', report=str(output), report_sha256=sha(raw),
                model_forwards=0, models_constructed=0, GT_read=False, engineering_accepted=False,
                execution_elapsed_seconds=time.monotonic()-started,
                cpu_peak_rss_kib=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss)
