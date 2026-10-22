"""Read-only CPU diagnosis of retained correctness-a evidence; no model forward."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def compare(a, b):
    result = {'equal': True, 'different_leaves': 0, 'max_abs_difference': 0.0, 'paths': []}

    def bad(path, reason, difference=None):
        result['equal'] = False
        result['different_leaves'] += 1
        if len(result['paths']) < 40:
            result['paths'].append({'path': path, 'reason': reason})
        if difference is not None:
            result['max_abs_difference'] = max(result['max_abs_difference'], difference)

    def visit(x, y, path):
        if isinstance(x, torch.Tensor):
            if not isinstance(y, torch.Tensor) or x.shape != y.shape or x.dtype != y.dtype:
                bad(path, 'tensor_metadata'); return
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
                bad(path, 'nonfinite_tensor'); return
            if not torch.equal(x, y):
                bad(path, 'tensor_value', float((x.double() - y.double()).abs().max()))
        elif isinstance(x, dict):
            if not isinstance(y, dict) or x.keys() != y.keys():
                bad(path, 'mapping_keys'); return
            for key in x:
                visit(x[key], y[key], path + '/' + str(key))
        elif isinstance(x, (tuple, list)):
            if type(x) is not type(y) or len(x) != len(y):
                bad(path, 'sequence_metadata'); return
            for i, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, path + '/' + str(i))
        elif type(x) is not type(y):
            bad(path, 'scalar_type')
        elif x != y:
            bad(path, 'scalar_value', abs(float(x) - float(y)) if isinstance(x, (float, int)) else None)

    visit(a, b, '')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--failure-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('STY'), 'CPU diagnosis must run in its own guarded screen'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only diagnosis required'
    assert not args.output.exists(), 'Do not overwrite diagnosis evidence'
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    rp = args.failure_dir / 'report.json'
    failure_sha = sha(rp)
    report = json.loads(rp.read_text())
    assert report['status'] == 'failed'
    artifacts = report['artifacts']
    loaded = {}
    for name in ('original_A_reference', 'all_accepted'):
        asset = artifacts[name]
        assert sha(asset['path']) == asset['sha256'], name
        loaded[name] = torch.load(asset['path'], map_location='cpu', weights_only=True)
    reference, candidate = loaded['original_A_reference'], loaded['all_accepted']
    losses = [{k: v for k, v in x.items() if k != 'index'} for x in candidate['losses']]
    candidate_records = [{k: v for k, v in x.items() if k != 'accepted'} for x in candidate['records']]
    results = {
        'losses': compare(reference['losses'], losses),
        'entire_final_state': compare(reference['state'], candidate['state']),
        'state_sections': {key: compare(value, candidate['state'][key]) for key, value in reference['state'].items()},
        'counts': compare(reference['counts'], {key: candidate['counts'][key] for key in reference['counts']}),
        'sampler_position': compare(reference['sampler_position'], candidate['sampler_position']),
        'all_raw_predictions': compare(reference['records'], candidate_records),
        'postprocessed_predictions': compare([r['prediction'] for r in reference['records']], [r['prediction'] for r in candidate_records]),
        'raw_predictions_images_1_through_15': compare(reference['records'][1:], candidate_records[1:]),
    }
    first_r = reference['records'][0]['raw_eh_outputs']['efficientpose']
    first_c = candidate['records'][0]['raw_eh_outputs']['efficientpose']
    bbox_r, bbox_c = first_r[1], first_c[1]
    selected_r = torch.argmax(first_r[0], dim=1)
    selected_c = torch.argmax(first_c[0], dim=1)
    mismatch = bbox_r != bbox_c
    locations = mismatch.nonzero(as_tuple=False)
    r = bbox_r.numpy(); c = bbox_c.numpy()
    # Postprocessed bbox coordinates are finite nonnegative float32, so their
    # IEEE bit-pattern distance directly counts positive floating-point ULPs.
    assert r.dtype == c.dtype == np.float32 and (r >= 0).all() and (c >= 0).all()
    distance = np.abs(r.view(np.int32).astype(np.int64) - c.view(np.int32).astype(np.int64))
    nonzero = distance[distance != 0]
    examples = []
    for location in locations[:16].tolist():
        index = tuple(location)
        examples.append({'index': location, 'reference': float(bbox_r[index]), 'candidate': float(bbox_c[index]),
                         'absolute_difference': abs(float(bbox_r[index]) - float(bbox_c[index])),
                         'ulp_distance': int(distance[index])})
    results['first_image_bbox_detail'] = {
        'shape': list(bbox_r.shape), 'dtype': str(bbox_r.dtype),
        'changed_coordinates': int(mismatch.sum()), 'total_coordinates': bbox_r.numel(),
        'changed_anchors': int(mismatch.any(dim=-1).sum()),
        'max_absolute_difference': float((bbox_r - bbox_c).abs().max()),
        'ulp_histogram': {str(k): v for k, v in sorted(Counter(nonzero.tolist()).items())},
        'selected_anchor_reference': selected_r.tolist(), 'selected_anchor_candidate': selected_c.tolist(),
        'selected_anchor_equal': compare(selected_r, selected_c),
        'selected_bbox_equal': compare(bbox_r[0, selected_r.squeeze()], bbox_c[0, selected_c.squeeze()]),
        'classification_equal': compare(first_r[0], first_c[0]),
        'rotation_raw_equal': compare(first_r[2], first_c[2]),
        'translation_equal': compare(first_r[3], first_c[3]),
        'examples': examples,
    }
    core = Path(report['source_core_manifest']['root'])
    postprocess = core / 'utils/postprocess.py'
    source_text = postprocess.read_text()
    start = source_text.index('@torch.jit.script\ndef raw_output_to_bbox')
    end = source_text.index('\n# -----------------------------------------------------------------------', start)
    function_text = source_text[start:end]
    assert sha(postprocess) == report['source_core_manifest']['files']['utils/postprocess.py']
    assert sha(rp) == failure_sha
    assert not torch.cuda.is_initialized()
    output = {
        'status': 'diagnosis_completed_original_correctness_still_failed',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'diagnostic_script_sha256': sha(__file__), 'input_failure_report_sha256': failure_sha,
        'artifact_hashes_verified': {name: artifacts[name]['sha256'] for name in loaded},
        'results': results,
        'postprocess_source': {'path': str(postprocess), 'sha256': sha(postprocess),
                               'raw_output_to_bbox_is_scripted': True, 'function': function_text},
        'observations': [
            'Reference executes the first E/H scoring call in this fresh process; ALL executes after all16 reference scoring calls.',
            'The only retained raw-output difference is in first-image bbox decoding, while later15 images, selected predictions, loss, and final state are separately checked above.',
            'Scripted bbox decoding contains multiply-add, exp, coordinate reconstruction and clamp; its profiling executor may optimize later calls differently.',
        ],
        'hypothesis': 'TorchScript cold/profiling versus warm optimized bbox decoding is consistent with the location and timing, but not yet proven.',
        'not_proven': ['Saved evidence omits predecode box_regression tensors and execution graphs.',
                       'CPU artifact analysis alone cannot distinguish JIT arithmetic fusion from a first-call bbox-subnet/CUDA kernel effect.'],
        'proposed_engineering_test': [
            'Preserve original correctness_a as failed; do not change equality tolerance or omit any anchor comparison.',
            'For a new bounded GPU diagnostic, capture one fixed raw box_regression plus anchors, then invoke the original scripted decoder repeatedly without model updates; retain first/subsequent output tensors and graphs.',
            'Before a fresh full correctness attempt, initialize the original graph and shape under the same frozen scientific settings, then restore official model, optimizer, buffers, modes, gradients, hook caches and RNG before either arm.',
            'Record all warmup forwards separately, verify initial state equality and compare complete raw E/H outputs including every bbox coordinate at zero tolerance.',
            'If mismatch persists, stop and diagnose further; never infer success from merely close or selected-only outputs.',
        ],
        'scientific_settings_changed': False, 'tolerance_changed': False, 'gpu_started': False,
        'cuda_initialized': torch.cuda.is_initialized(), 'model_forward_count': 0,
        'ground_truth_read': False, 'prior_evidence_modified': False,
    }
    with args.output.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False, sort_keys=True)
        f.write('\n')
    print(json.dumps({'status': output['status'], 'output': str(args.output), 'sha256': sha(args.output),
                      'losses_equal': results['losses']['equal'], 'state_equal': results['entire_final_state']['equal'],
                      'predictions_equal': results['postprocessed_predictions']['equal'],
                      'changed_bbox_coordinates': int(mismatch.sum()),
                      'ulp_histogram': results['first_image_bbox_detail']['ulp_histogram'],
                      'cuda_initialized': torch.cuda.is_initialized()}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
