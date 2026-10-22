"""Strict, label-free GPU engineering gate for the confirmed ODR003 adapter.

Run only in a unique guarded screen after official-source/data receipts pass.
The orchestrator owns GPU scheduling. This script never selects method scores,
reads pose labels, downloads assets, or changes a tolerance after observing a
difference. It executes 16-image original-A/ALL/mixed/resume comparisons, with
CPU evidence snapshots and transactional scoring checks. All timings include
engineering instrumentation and are not scientific throughput measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


PLAN = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
MIXED_MASK = [True, True, True, True, False, True, False, True,
              False, True, True, False, True, False, True, False]
READ_AUDIT = {'blocked_gt_reads': [], 'blocked_network': [], 'opened_annotation_files': 0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_boundary(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace('\\', '/').lower()
        filename = name.rsplit('/', 1)[-1]
        if '/labels/' in name or name.endswith('.csv') or filename in {
            'dev_labels.json', 'train.json', 'test.json', 'validation.json', 'labels.json'
        }:
            READ_AUDIT['blocked_gt_reads'].append(name)
            raise RuntimeError('Correctness test forbids annotation-file access: ' + name)
    if event in ('socket.connect', 'socket.getaddrinfo'):
        READ_AUDIT['blocked_network'].append(event)
        raise RuntimeError('Correctness test forbids network access')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def safe_pack(value):
    """Persist RNG arrays using only weights-only-loadable built-in types."""
    if isinstance(value, np.ndarray):
        return {'__numpy_array__': True, 'dtype': str(value.dtype),
                'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, dict):
        return {key: safe_pack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(safe_pack(item) for item in value)
    return value


def safe_unpack(value):
    if isinstance(value, dict) and value.get('__numpy_array__') is True:
        return np.asarray(value['values'], dtype=value['dtype']).reshape(value['shape'])
    if isinstance(value, dict):
        return {key: safe_unpack(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(safe_unpack(item) for item in value)
    return value


def compare_tree(left, right):
    result = {'equal': True, 'mismatch_count': 0, 'max_abs_numeric_difference': 0.0,
              'first_mismatches': [], 'nonfinite': []}

    def mismatch(path, reason, difference=None):
        result['equal'] = False
        result['mismatch_count'] += 1
        if len(result['first_mismatches']) < 24:
            result['first_mismatches'].append({'path': path, 'reason': reason})
        if difference is not None and math.isfinite(difference):
            result['max_abs_numeric_difference'] = max(result['max_abs_numeric_difference'], difference)

    def walk(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                mismatch(path, 'tensor_metadata')
                return
            a, b = a.detach().cpu(), b.detach().cpu()
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                result['nonfinite'].append(path)
                mismatch(path, 'nonfinite_tensor')
            elif not torch.equal(a, b):
                difference = float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.0
                mismatch(path, 'tensor_value', difference)
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
                mismatch(path, 'array_metadata')
            elif not np.isfinite(a).all() or not np.isfinite(b).all():
                result['nonfinite'].append(path)
                mismatch(path, 'nonfinite_array')
            elif not np.array_equal(a, b):
                mismatch(path, 'array_value', float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                mismatch(path, 'mapping_keys')
                return
            for key in a:
                walk(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                mismatch(path, 'sequence_shape')
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + '/' + str(index))
        elif type(a) is not type(b):
            mismatch(path, 'scalar_type')
        elif isinstance(a, float) and (not math.isfinite(a) or not math.isfinite(b)):
            result['nonfinite'].append(path)
            mismatch(path, 'nonfinite_scalar')
        elif a != b:
            diff = abs(float(a) - float(b)) if isinstance(a, (int, float)) and not isinstance(a, bool) else None
            mismatch(path, 'scalar_value', diff)

    walk(left, right, '')
    return result


def snapshot(adapter, *, caches=True):
    model = adapter.model
    value = {
        'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(adapter.optimizer.state_dict()),
        'gradients': {name: cpu_tree(p.grad) for name, p in model.named_parameters()},
        'requires_grad': {name: p.requires_grad for name, p in model.named_parameters()},
        'modes': {name: module.training for name, module in model.named_modules()},
        'rng': cpu_tree(rng_state()),
    }
    if caches:
        value['bn_features'] = cpu_tree(adapter.bn_features)
        value['bn_mean'] = cpu_tree(getattr(adapter, 'bn_mean', {}))
        value['bn_mean_sq'] = cpu_tree(getattr(adapter, 'bn_mean_sq', {}))
    return value


def tail_mutable_state(adapter):
    value = snapshot(adapter, caches=False)
    value.pop('rng')
    return value


class TestContext:
    def __init__(self, args, report, cfg, source_state, images, camera, points):
        self.args, self.report, self.cfg = args, report, cfg
        self.source_state, self.images = source_state, images
        self.camera, self.points = camera, points
        self.device = torch.device('cuda:0')
        self.initial_rng = None
        self.initial_state = None
        self.current = None
        self.phase = 'initialization'
        self.forward_kind = 'unknown'
        self.counts = Counter()
        self.phase_start = None
        self.scoring_seconds = 0.0

    def check(self, label, left, right):
        comparison = compare_tree(left, right)
        comparison['label'] = label
        comparison['phase'] = self.phase
        self.report['checks'].append(comparison)
        require(comparison['equal'], 'Strict equality failed: ' + label + '; ' + json.dumps(comparison))

    @contextmanager
    def phase_timer(self, name):
        self.phase = name
        self.counts = Counter()
        self.scoring_seconds = 0.0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.report['phases'].append({
                'phase': name, 'seconds': time.perf_counter() - start,
                'scoring_seconds_including_instrumentation': self.scoring_seconds,
                'backbone_forward_counts': dict(self.counts),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'engineering_instrumentation_included': True,
            })

    def new_model(self, reference=False):
        model = build_spnv2(self.cfg).cpu()
        result = model.load_state_dict(self.source_state, strict=True)
        require(not result.missing_keys and not result.unexpected_keys, 'Official source strict load failed')
        self.check('strict_official_reset_before_cuda', cpu_tree(model.state_dict()), self.source_state)
        model = model.to(self.device)
        if reference:
            # Literal setup semantics from frozen tools/odr.py, independent of
            # OriginalAAdapter.configure and with its own hook dictionary.
            model.eval()
            model.requires_grad_(False)
            features, handles = {}, []
            for name, module in model.backbone.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.requires_grad_(True)
                    module.eval()
                    def hook(_m, inputs, _out, key=name):
                        features[key] = inputs[0].detach()
                    handles.append(module.register_forward_hook(hook))
            optimizer = get_optimizer(self.cfg, model)
            adapter = SimpleNamespace(model=model, optimizer=optimizer, device=self.device,
                                      bn_features=features, handles=handles, bn_mean={}, bn_mean_sq={})
        else:
            adapter = OriginalAAdapter.configure(model, self.device, batch=4)
        def count_forward(_m, _inputs, _outputs):
            self.counts[self.forward_kind] += 1
        adapter.handles.append(model.backbone.register_forward_hook(count_forward))
        if self.initial_rng is None:
            self.initial_rng = cpu_tree(rng_state())
        else:
            restore_rng(self.initial_rng)
        state = snapshot(adapter)
        if self.initial_state is None:
            self.initial_state = state
        else:
            self.check('complete_initial_state_reset', state, self.initial_state)
        self.current = adapter
        return adapter

    def release(self, adapter):
        for handle in adapter.handles:
            handle.remove()
        adapter.bn_features.clear()
        if hasattr(adapter, 'bn_mean'):
            adapter.bn_mean.clear()
            adapter.bn_mean_sq.clear()
        self.current = None

    def score(self, adapter, image, index):
        started = time.perf_counter()
        before = snapshot(adapter)
        raw, handles = {}, []
        for name, head in zip(adapter.model.head_names, adapter.model.heads):
            if name in ('heatmap', 'efficientpose'):
                def save_head(_m, _inputs, outputs, key=name):
                    raw[key] = cpu_tree(outputs)
                handles.append(head.register_forward_hook(save_head))
        self.forward_kind = 'isolated_score'
        try:
            prediction = extract(adapter.model, adapter, image, self.cfg, self.camera, self.points)
        finally:
            for handle in handles:
                handle.remove()
        require(set(raw) == {'heatmap', 'efficientpose'}, 'Missing raw E/H scoring outputs')
        # Runtime observations are recorded separately; they are not numeric
        # predictions and cannot satisfy bitwise equality between executions.
        timings = prediction.pop('timing_seconds', None)
        if timings is not None:
            self.report.setdefault('prediction_timings', []).append(
                {'phase': self.phase, 'index': index, 'timing_seconds': timings})
        self.check('score_isolation_image_' + str(index), before, snapshot(adapter))
        self.scoring_seconds += time.perf_counter() - started
        return {'index': index, 'sample_id': self.images[index]['sample_id'],
                'raw_eh_outputs': raw, 'prediction': prediction}

    def artifact(self, name, value):
        path = self.args.run_dir / (name + '.pt')
        require(not path.exists(), 'Refusing evidence overwrite')
        torch.save(safe_pack(cpu_tree(value)), path)
        record = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
        self.report['artifacts'][name] = record
        return record


def original_reference(ctx):
    with ctx.phase_timer('frozen_original_A'):
        adapter = ctx.new_model(reference=True)
        model, records, losses = adapter.model, [], []
        updates = [0]
        real_step = adapter.optimizer.step
        def counted_step(*args, **kwargs):
            result = real_step(*args, **kwargs)
            updates[0] += 1
            return result
        adapter.optimizer.step = counted_step
        def before_forward(_model, args, kwargs):
            index = len(records)
            require(index < 16, 'Reference repeated input stream')
            records.append(ctx.score(adapter, args[0], index))
            ctx.forward_kind = 'adaptation'
        def after_forward(_model, args, kwargs, outputs):
            loss, items = outputs
            require(bool(torch.isfinite(loss).all()), 'Nonfinite reference loss')
            losses.append({'loss': float(loss.detach().cpu()), 'entropy': float(items['ent'].cpu())})
        pre = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        post = model.register_forward_hook(after_forward, with_kwargs=True)
        try:
            # A fixed list iterator avoids DataLoader RNG/prefetch differences.
            # Input transforms and order have already been frozen and recorded.
            do_adapt(0, ctx.cfg, model, adapter.bn_features,
                     [item['image'] for item in ctx.images], adapter.optimizer,
                     log_dir=None, device=ctx.device, scaler=None)
        finally:
            pre.remove()
            post.remove()
        require(len(records) == len(losses) == 16 and updates[0] == 4, 'Reference count mismatch')
        result = {'records': records, 'losses': losses, 'state': snapshot(adapter),
                  'counts': {'observed': 16, 'accepted': 16, 'updates': updates[0], 'used': 16, 'tail': 0},
                  'sampler_position': 16}
        ctx.artifact('original_A_reference', result)
        ctx.release(adapter)
        del adapter, model
        gc.collect()
        torch.cuda.empty_cache()
        return result


def adapted_loop(ctx, mask, *, name, resume=None, stop_at=None):
    with ctx.phase_timer(name):
        adapter = ctx.new_model()
        records, losses = [], []
        start_index = 0
        if resume is not None:
            saved = safe_unpack(torch.load(resume, map_location='cpu', weights_only=True))
            require(saved['source_sha256'] == ctx.report['source_sha256'], 'Resume source drift')
            require(saved['manifest_sha256'] == ctx.report['manifest_sha256'], 'Resume input stream drift')
            require(saved['mask'] == mask and saved['sampler_position'] == 4, 'Resume mask/cursor drift')
            adapter.restore(saved['adapter'])
            ctx.check('restored_complete_checkpoint_state', cpu_tree(adapter.checkpoint()), saved['adapter'])
            records, losses = saved['records'], saved['losses']
            start_index = saved['sampler_position']
        last_complete = tail_mutable_state(adapter)
        checkpoint_record = None
        until = stop_at if stop_at is not None else len(ctx.images)
        for index in range(start_index, until):
            item = ctx.images[index]
            prediction = ctx.score(adapter, item['image'], index)
            prediction['accepted'] = mask[index]
            records.append(prediction)
            if mask[index]:
                seen_loss = []
                def capture_loss(_model, _inputs, outputs):
                    loss, parts = outputs
                    seen_loss.append({'index': index, 'loss': float(loss.detach().cpu()), 'entropy': float(parts['ent'].cpu())})
                handle = adapter.model.register_forward_hook(capture_loss)
                ctx.forward_kind = 'adaptation'
                try:
                    returned = adapter.accept(item['image'])
                finally:
                    handle.remove()
                require(len(seen_loss) == 1 and seen_loss[0]['entropy'] == returned, 'Loss capture mismatch')
                losses.append(seen_loss[0])
            adapter.observed += 1
            require(adapter.observed == index + 1, 'Sampler/observed cursor mismatch')
            if mask[index] and adapter.accepted % 4 == 0:
                last_complete = tail_mutable_state(adapter)
            if stop_at is not None and index + 1 == stop_at:
                checkpoint = {'adapter': cpu_tree(adapter.checkpoint()), 'sampler_position': index + 1,
                              'source_sha256': ctx.report['source_sha256'], 'manifest_sha256': ctx.report['manifest_sha256'],
                              'mask': mask, 'records': records, 'losses': losses}
                checkpoint_record = ctx.artifact('mixed_boundary_checkpoint', checkpoint)
        if stop_at is None:
            counts = adapter.finish()
            if counts['tail']:
                ctx.check('discard_tail_restores_last_complete_update_state', tail_mutable_state(adapter), last_complete)
                require(not adapter.bn_features and not adapter.bn_mean and not adapter.bn_mean_sq,
                        'Discarded tail retains temporary BN caches')
            result = {'records': records, 'losses': losses, 'state': snapshot(adapter),
                      'counts': counts, 'sampler_position': adapter.observed}
            ctx.artifact(name, result)
        else:
            result = {'checkpoint': checkpoint_record, 'sampler_position': adapter.observed}
        ctx.release(adapter)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        return result


def preflight(args, report):
    require(os.environ.get('STY'), 'Run inside a new guarded screen')
    receipt = read_json(args.receipt)
    require(receipt.get('status') == 'passed' and receipt.get('source_offline_identity_verified') is True
            and receipt.get('strict_model_load_performed') is True, 'Official offline identity receipt is not passed')
    for key in ('source', 'config', 'source_core_manifest', 'strict_load_report'):
        require(sha(receipt[key + '_path']) == receipt[key + '_sha256'], 'Receipt asset drift: ' + key)
    manifest = read_json(receipt['source_core_manifest_path'])
    core = Path(manifest['root'])
    for relative, expected in manifest['files'].items():
        require(sha(core / relative) == expected, 'Frozen A code drift: ' + relative)
    for key in ('camera', 'keypoint'):
        require(sha(manifest[key + '_path']) == manifest[key + '_sha256'], 'Geometry asset drift: ' + key)
    prepared, verified = read_json(args.prepared_data_proof), read_json(args.verified_data_proof)
    require(verified.get('status') == 'passed', 'Data boundary verification not passed')
    require(verified.get('prepared_proof_sha256') == sha(args.prepared_data_proof), 'Data receipt drift')
    source_stream = [asset for name, asset in prepared['artifacts'].items()
                     if '/stream_' in name and Path(asset['path']).resolve() == args.manifest.resolve()]
    require(len(source_stream) == 1 and sha(args.manifest) == source_stream[0]['sha256'], 'Unverified paired stream')
    os.environ['TORCH_HOME'] = str(args.torch_home)
    cached = args.torch_home / 'hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth'
    require(cached.is_file(), 'Cached ImageNet constructor asset missing; network downloads prohibited')
    strict = read_json(receipt['strict_load_report_path'])
    require(sha(cached) == strict['cached_imagenet_build_dependency']['sha256'], 'ImageNet constructor cache changed')
    report.update({'receipt_path': str(args.receipt), 'receipt_sha256': sha(args.receipt),
                   'source_path': receipt['source_path'], 'source_sha256': receipt['source_sha256'],
                   'config_path': receipt['config_path'], 'config_sha256': receipt['config_sha256'],
                   'source_core_manifest': manifest, 'manifest_path': str(args.manifest), 'manifest_sha256': sha(args.manifest),
                   'data_prepared_sha256': sha(args.prepared_data_proof), 'data_verified_sha256': sha(args.verified_data_proof),
                   'cached_imagenet': {'path': str(cached), 'sha256': sha(cached)},
                   'implementation_files': {name: sha(Path(__file__).parent / name)
                                            for name in ('test_real_model.py', 'state_adapter.py', 'prediction.py', 'data_boundary.py')}})
    return receipt, manifest, core


def run(args, report):
    global torch, np, rng_state, restore_rng, OriginalAAdapter, extract
    global build_spnv2, get_optimizer, do_adapt
    receipt, manifest, core = preflight(args, report)
    sys.path.insert(0, str(core))
    import torch
    import numpy as np
    from state_adapter import rng_state, restore_rng, OriginalAAdapter
    from prediction import extract
    from data_boundary import ImageManifestDataset
    from config import cfg, update_config
    from nets import build_spnv2
    from solver import get_optimizer
    from engine.adapter import do_adapt
    from dataset.transforms import build_transforms
    from utils.utils import set_seeds_cudnn, load_camera_intrinsics, load_tango_3d_keypoints

    require(torch.cuda.is_available(), 'Real CUDA device required; CPU simulation is insufficient')
    gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    others = []
    for line in gpu_processes.splitlines():
        if not line.strip():continue
        pid,name,mb = [x.strip() for x in line.split(',')]
        if int(pid)==os.getpid():continue
        desktop = name == '/usr/libexec/gnome-remote-desktop-daemon' and float(mb)<=256
        if desktop:
            desktop = str(Path('/proc',pid,'exe').resolve()) == name
        if not desktop:others.append(line)
    require(not others, 'Other GPU compute processes present: ' + repr(others))
    torch.set_num_threads(2)
    update_config(cfg, SimpleNamespace(cfg=receipt['config_path'], opts=['MODEL.PRETRAIN_FILE', '', 'ODR.NUM_TRAIN_SAMPLES', '16']))
    require(cfg.CUDA and not cfg.FP16 and cfg.MODEL.EFFICIENTDET_PHI == 3 and not cfg.MODEL.USE_GROUPNORM_BACKBONE, 'Unexpected source architecture/precision')
    require(list(cfg.MODEL.HEAD.NAMES) == ['heatmap', 'efficientpose', 'segmentation'] and not cfg.MODEL.HEAD.LOSS_HEADS,
            'Full EHS entropy-only adaptation required')
    require(list(cfg.DATASET.INPUT_SIZE) == [768, 512] and cfg.ODR.IMAGES_PER_BATCH == 4 and cfg.TRAIN.IMAGES_PER_GPU == 1, 'Input/batch protocol drift')
    require(cfg.TRAIN.OPTIMIZER == 'AdamW' and cfg.TRAIN.LR == 1e-4 and cfg.TRAIN.WD == 0
            and cfg.TRAIN.GAMMA1 == .9 and cfg.TRAIN.GAMMA2 == .999 and cfg.TRAIN.ACCUM_STEPS == 1, 'Optimizer protocol drift')
    require(cfg.CUDNN.BENCHMARK is True and cfg.CUDNN.DETERMINISTIC is False and cfg.CUDNN.ENABLED is True, 'cuDNN scientific settings changed')
    require(cfg.ODR.MIN_ENTROPY and cfg.SEED == 2021, 'Original A seed/loss mode drift')
    set_seeds_cudnn(cfg, seed=2021)
    report['runtime_config'] = cfg.dump()
    report['config_overrides'] = {'MODEL.PRETRAIN_FILE': '', 'ODR.NUM_TRAIN_SAMPLES': 16}
    report['device'] = {'name': torch.cuda.get_device_name(0), 'properties': str(torch.cuda.get_device_properties(0)),
                        'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'cudnn_enabled': torch.backends.cudnn.enabled, 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'compute_processes_before': gpu_processes}
    source_state = torch.load(receipt['source_path'], map_location='cpu', weights_only=True)
    require(isinstance(source_state, dict) and all(isinstance(value, torch.Tensor) for value in source_state.values()), 'Official source must be direct state_dict')
    source_state = cpu_tree(source_state)
    require(all(bool(torch.isfinite(value).all()) for value in source_state.values()), 'Nonfinite official source')
    dataset = ImageManifestDataset(args.manifest, transforms=build_transforms(cfg, is_train=False, load_labels=False), verify_image_sha=True)
    require(len(dataset) == 1024 and dataset.manifest['purpose'] == 'paired_adaptation_stream', 'Expected frozen N1024 image-only stream')
    require(dataset.manifest['seed'] == 2021, 'Correctness input stream must use predeclared seed2021')
    images = []
    for index in range(16):
        item = dataset[index]
        item['image'] = item['image'].unsqueeze(0)
        require(item['image'].shape == (1, 3, 512, 768) and item['image'].dtype == torch.float32, 'Unexpected transformed image')
        images.append(item)
    report['input_ids'] = [item['sample_id'] for item in images]
    report['input_images'] = dataset.manifest['items'][:16]
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    points = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)
    ctx = TestContext(args, report, cfg, source_state, images, camera, points)

    # The original scripted bbox decoder changes its arithmetic execution graph
    # after the first call. Warm only that pure operator on synthetic inputs,
    # before constructing either measured model; no observation or update.
    from nets.heads.anchors import create_anchors
    from utils.postprocess import raw_output_to_bbox
    initial_runtime_rng = cpu_tree(rng_state())
    anchors,_ = create_anchors(image_size=torch.as_tensor(cfg.DATASET.INPUT_SIZE),
        scales=eval(cfg.MODEL.HEAD.ANCHOR_SCALE),ratios=cfg.MODEL.HEAD.ANCHOR_RATIO)
    anchors=anchors.to(ctx.device)
    dummy=torch.zeros((1,anchors.shape[0],4),dtype=torch.float32,device=ctx.device)
    warmed=[]
    for _ in range(6):
        warmed.append(raw_output_to_bbox(dummy,torch.as_tensor(cfg.DATASET.INPUT_SIZE),anchors).cpu())
    require(all(torch.equal(x,warmed[-1]) for x in warmed[1:]),'Scripted bbox warmup did not stabilize')
    restore_rng(initial_runtime_rng)
    report['operator_warmup']={'operator':'original raw_output_to_bbox','calls':6,
        'synthetic_inputs':True,'model_forward_calls':0,'source_state_modified':False,
        'reason':'fixed-input cold/hot decoder diagnostic; preserve zero-tolerance comparisons'}
    del anchors,dummy,warmed

    reference = original_reference(ctx)
    all_result = adapted_loop(ctx, [True] * 16, name='all_accepted')
    comparable_records = [{key: value for key, value in record.items() if key != 'accepted'} for record in all_result['records']]
    ctx.check('all_accepted_per_image_EH_predictions_validity', reference['records'], comparable_records)
    comparable_losses = [{key: value for key, value in record.items() if key != 'index'} for record in all_result['losses']]
    ctx.check('all_accepted_per_image_unscaled_loss', reference['losses'], comparable_losses)
    ctx.check('all_accepted_complete_model_optimizer_gradients_caches_modes_rng', reference['state'], all_result['state'])
    ctx.check('all_accepted_counts', reference['counts'], {key: all_result['counts'][key] for key in reference['counts']})
    del reference, all_result
    gc.collect()

    continuous = adapted_loop(ctx, MIXED_MASK, name='mixed_continuous')
    expected = {'observed': 16, 'accepted': 10, 'rejected': 6, 'updates': 2, 'used': 8, 'tail': 2}
    ctx.check('mixed_expected_counts', continuous['counts'], expected)
    prefix = adapted_loop(ctx, MIXED_MASK, name='mixed_prefix', stop_at=4)
    require(prefix['sampler_position'] == 4, 'Checkpoint after first complete update required')
    resumed = adapted_loop(ctx, MIXED_MASK, name='mixed_resumed', resume=Path(prefix['checkpoint']['path']))
    ctx.check('resumed_records_losses_and_full_state', continuous, resumed)
    report['total_backbone_forward_counts'] = dict(sum((Counter(p['backbone_forward_counts']) for p in report['phases']), Counter()))
    require(report['total_backbone_forward_counts'] == {'isolated_score': 64, 'adaptation': 52}, 'Unexpected actual forward count')
    require(not READ_AUDIT['blocked_gt_reads'] and not READ_AUDIT['blocked_network'], 'Forbidden access was attempted')
    require(sha(receipt['source_path']) == report['source_sha256'] and sha(receipt['config_path']) == report['config_sha256'], 'Frozen source/config mutated')
    for relative, expected_sha in manifest['files'].items():
        require(sha(core / relative) == expected_sha, 'Frozen A mutated during correctness test: ' + relative)
    report['status'] = 'passed'
    report['strict_equality_required'] = True
    report['comparison_tolerance'] = 0.0
    report['scientific_accuracy_computed'] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, default=PLAN/'preflight/official_source_receipt.json')
    parser.add_argument('--manifest', type=Path, default=PLAN/'data/lightbox/stream_2021.json')
    parser.add_argument('--prepared-data-proof', type=Path, default=PLAN/'preflight/data_boundary_prepared.json')
    parser.add_argument('--verified-data-proof', type=Path, default=PLAN/'preflight/data_boundary_verified.json')
    parser.add_argument('--torch-home', type=Path, default=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/env/cache/torch'))
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    require(not args.run_dir.exists(), 'Use a new run directory; prior evidence must remain intact')
    args.run_dir.mkdir(parents=True)
    report = {'schema': 'odr003.real_correctness.v1', 'status': 'running',
              'created_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'screen': os.environ.get('STY'), 'checks': [], 'phases': [], 'artifacts': {},
              'mixed_mask': MIXED_MASK, 'purpose': 'engineering_correctness_only_no_accuracy_GT',
              'strict_equality_required': True, 'comparison_tolerance': 0.0}
    sys.addaudithook(audit_boundary)
    started = time.perf_counter()
    try:
        run(args, report)
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        report['total_seconds'] = time.perf_counter() - started
        report['access_audit'] = READ_AUDIT
        write_json(args.run_dir/'report.json', report)
        print(json.dumps({'status': report['status'], 'report': str(args.run_dir/'report.json'),
                          'report_sha256': sha(args.run_dir/'report.json'), 'checks': len(report['checks']),
                          'phases': len(report['phases'])}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
