"""Real SPNv2 phi3 GPU acceptance; launch only in a separately guarded screen.

smoke: 512 original samples, physical4 x accumulation4, checkpoint at attempt16,
finish32, delete CUDA objects, rebuild, resume16->32 and require bitwise equality.
batch16: one isolated full-resolution physical16 window, never downsize/retry.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--mode', choices=('smoke', 'batch16'), required=True)
    return parser.parse_args()


def write_json(path, value):
    temporary = path.with_name('.'+path.name+'.tmp')
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def exact_tree(left, right, name='root'):
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f'Exact tensor mismatch: {name}')
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError(f'Exact dictionary keys mismatch: {name}')
        for key in left:
            exact_tree(left[key], right[key], f'{name}.{key}')
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f'Exact sequence mismatch: {name}')
        for index, (a, b) in enumerate(zip(left, right)):
            exact_tree(a, b, f'{name}[{index}]')
    elif left != right:
        raise AssertionError(f'Exact scalar mismatch: {name}: {left!r} != {right!r}')


def run(args, report):
    if os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('PYTHONNOUSERSITE=1 is required')
    if not os.environ.get('TORCH_HOME'):
        raise RuntimeError('A fixed project-owned TORCH_HOME must be provided')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.utils.data import DataLoader
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source/'core'))
    from config import cfg as defaults, update_config
    from dataset.build import build_dataset, _seed_worker
    from nets import build_spnv2
    import nets.backbones.efficientnet as efficientnet
    from solver import get_optimizer, get_scaler, adjust_learning_rate
    import engine.trainer as trainer
    from utils.utils import set_seeds_cudnn
    from utils.training_state import (TrainingState, EpochBatchSampler, SeededDataset,
        build_manifest, save_training_checkpoint, load_training_checkpoint,
        atomic_torch_save, capture_rng_state, sha256_file, emit_event, assert_finite_weights)

    cfg = defaults.clone()
    update_config(cfg, argparse.Namespace(cfg=args.cfg, opts=[]))
    if args.mode == 'batch16':
        cfg.defrost()
        cfg.TRAIN.IMAGES_PER_GPU = 16
        cfg.TRAIN.ACCUM_STEPS = 1
        cfg.freeze()
    conditions = dict(phi3=cfg.MODEL.EFFICIENTDET_PHI == 3,
        resolution=list(cfg.DATASET.INPUT_SIZE) == [768, 512], seed=cfg.SEED == 2021,
        cuda=bool(cfg.CUDA), amp=bool(cfg.FP16), adamw=cfg.TRAIN.OPTIMIZER == 'AdamW',
        learning_rate=cfg.TRAIN.LR == .001, weight_decay=cfg.TRAIN.WD == .0001,
        reproducible=bool(cfg.TRAIN.REPRODUCIBLE),
        three_heads=set(cfg.MODEL.HEAD.NAMES) == {'heatmap', 'efficientpose', 'segmentation'},
        three_losses=set(cfg.MODEL.HEAD.LOSS_HEADS) == {'heatmap', 'efficientpose', 'segmentation'},
        physical_batch=cfg.TRAIN.IMAGES_PER_GPU == (4 if args.mode == 'smoke' else 16),
        accumulation=cfg.TRAIN.ACCUM_STEPS == (4 if args.mode == 'smoke' else 1),
        single_process=not cfg.DIST.MULTIPROCESSING_DISTRIBUTED)
    for name in ('ADJUST_BRIGHTNESS_CONTRAST', 'APPLY_BLUR', 'APPLY_SOLAR_FLARE',
                 'APPLY_NOISE', 'APPLY_RANDOM_ERASE', 'APPLY_TEXTURE_RANDOMIZATION'):
        conditions['augmentation_'+name] = bool(getattr(cfg.AUGMENT, name))
    report['protocol_checks'] = conditions
    if not all(conditions.values()):
        raise AssertionError('Protocol mismatch: '+', '.join(key for key, passed in conditions.items() if not passed))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this explicitly GPU-only acceptance')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    props = torch.cuda.get_device_properties(device)
    report.update(source=str(source), cfg=str(Path(args.cfg).resolve()),
        cfg_file_sha256=sha256_file(args.cfg), effective_config=cfg.dump(),
        effective_config_sha256=hashlib.sha256(cfg.dump().encode()).hexdigest(), seed=int(cfg.SEED),
        test_sha256=sha256_file(__file__), device=dict(name=props.name, total_memory=props.total_memory,
        capability=list(torch.cuda.get_device_capability(device))),
        runtime=dict(torch=str(torch.__version__), cuda=str(torch.version.cuda),
                     cudnn=torch.backends.cudnn.version(), torch_home=os.environ['TORCH_HOME']))
    # Forbid a fallback network download. The production network receives exactly
    # the already cached official state, loaded locally with no architecture edits.
    def cached_only(url, *unused, **kwargs):
        cached = Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/', 1)[-1]
        if not cached.is_file():
            raise FileNotFoundError(f'Official weights are not cached; download forbidden: {cached}')
        return torch.load(cached, map_location='cpu', weights_only=True)
    efficientnet.load_state_dict_from_url = cached_only

    artifacts = Path(args.artifacts).resolve()
    if artifacts.exists() and any(artifacts.iterdir()):
        raise FileExistsError(f'Use a new empty artifact directory: {artifacts}')
    artifacts.mkdir(parents=True, exist_ok=True)
    count = 512 if args.mode == 'smoke' else 16

    def construct():
        set_seeds_cudnn(cfg, seed=cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model = build_spnv2(cfg).to(device)
        def verify_actual_input(module, inputs):
            shape = list(inputs[0].shape)
            expected_shape = [cfg.TRAIN.IMAGES_PER_GPU, 3, 512, 768]
            if shape != expected_shape:
                raise AssertionError(f'Actual input is not full-resolution RGB physical batch: {shape} != {expected_shape}')
            observed = report.setdefault('observed_training_shapes', [])
            if shape not in observed:
                observed.append(shape)
        model.register_forward_pre_hook(verify_actual_input)
        raw = build_dataset(cfg, split='train', load_labels=True)
        original_count = len(raw)
        if original_count < count:
            raise ValueError(f'Insufficient original training rows: {original_count}')
        raw.csv = raw.csv.iloc[:count].copy().reset_index(drop=True)
        sampler = EpochBatchSampler(len(raw), cfg.TRAIN.IMAGES_PER_GPU,
                    cfg.TRAIN.ACCUM_STEPS, cfg.SEED, cfg.TRAIN.SHUFFLE)
        data = SeededDataset(raw, cfg.SEED)
        loader = DataLoader(data, batch_sampler=sampler,
                    num_workers=min(cfg.TRAIN.IMAGES_PER_GPU, cfg.TRAIN.WORKERS),
                    pin_memory=cfg.PIN_MEMORY, worker_init_fn=_seed_worker,
                    generator=torch.Generator().manual_seed(int(cfg.SEED)+1))
        val = build_dataset(cfg, split='val', load_labels=True)
        optimizer = get_optimizer(cfg, model)
        scaler = get_scaler(cfg)
        if scaler is None:
            raise AssertionError('The actual production CUDA GradScaler must be active')
        adjust_learning_rate(optimizer, 0, cfg)
        manifest = build_manifest(cfg, model, datasets={'train': data, 'val': val})
        return model, optimizer, scaler, sampler, loader, manifest, original_count

    emit_event('gpu_acceptance_start', phase='gpu_acceptance', mode=args.mode,
               epochs=1, epoch=1, samples=count, physical_batch=cfg.TRAIN.IMAGES_PER_GPU,
               accumulation_steps=cfg.TRAIN.ACCUM_STEPS)
    report['stage'] = 'construct'
    model, optimizer, scaler, sampler, loader, manifest, original_count = construct()
    report['manifest'] = manifest
    report['code_sha256'] = manifest['code']
    report['original_training_rows'] = original_count
    report['subset_rows'] = count
    report['subset_selection'] = 'first rows of the original CSV, then epoch-keyed permutation'
    write_json(artifacts/'manifest.json', manifest)
    initial_params = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                      if parameter.requires_grad}
    initial_path = artifacts/'model_initial.pth.tar'
    atomic_torch_save(model.state_dict(), initial_path)
    initial_checkpoint = artifacts/'initial_checkpoint.pth.tar'
    save_training_checkpoint(initial_checkpoint, model, optimizer, scaler, TrainingState(), sampler, manifest)
    continuous_ids = []
    original_emit = trainer.emit_event
    def collect_continuous(event, state=None, **fields):
        if event == 'update':
            continuous_ids.extend(str(value) for value in fields.get('sample_ids', []))
        return original_emit(event, state, **fields)
    trainer.emit_event = collect_continuous
    boundary_path = artifacts/'boundary_attempt_00016.pth.tar'
    checkpoint_times = []
    def boundary(current):
        if args.mode == 'smoke' and current.update_attempts == 16:
            before = time.monotonic()
            save_training_checkpoint(boundary_path, model, optimizer, scaler, current, sampler, manifest)
            checkpoint_times.append(time.monotonic()-before)
            emit_event('checkpoint', current, phase='gpu_acceptance', path=str(boundary_path),
                       epochs=1, epoch=1, reason='gpu_resume_acceptance_attempt16')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    report['stage'] = 'continuous_training'
    try:
        whole = trainer.do_train(0, cfg, model, loader, optimizer, device=device,
                                 scaler=scaler, state=TrainingState(), on_boundary=boundary)
    finally:
        trainer.emit_event = original_emit
    torch.cuda.synchronize()
    elapsed = time.monotonic()-began
    report['continuous'] = dict(seconds=elapsed, images_per_second=count/elapsed,
        checkpoint_seconds=sum(checkpoint_times), state=whole.state_dict(),
        memory_allocated_peak=torch.cuda.max_memory_allocated(device),
        memory_reserved_peak=torch.cuda.max_memory_reserved(device), sample_ids=continuous_ids)
    expected_attempts = 32 if args.mode == 'smoke' else 1
    if whole.update_attempts != expected_attempts or whole.samples_seen != count:
        raise AssertionError(f'Unexpected complete-window counters: {whole.state_dict()}')
    assert_finite_weights(model)
    if args.mode == 'batch16':
        report['artifacts'] = {path.name: dict(path=str(path), sha256=sha256_file(path), size=path.stat().st_size)
            for path in (initial_path, initial_checkpoint)}
        report.update(status='passed', feasible=True,
                      interpretation='One actual physical16 full-resolution forward/backward window completed; no downsizing or retry.')
        return
    if whole.optimizer_updates < 3:
        raise AssertionError(f'Fewer than three actual successful optimizer updates: {whole.optimizer_updates}')
    components = {'backbone': dict(prefix='backbone.', changed_tensors=0, changed_elements=0, max_abs_change=0.)}
    for index, head in enumerate(cfg.MODEL.HEAD.NAMES):
        components[head] = dict(prefix=f'heads.{index}.', changed_tensors=0, changed_elements=0, max_abs_change=0.)
    for name, parameter in model.named_parameters():
        if name not in initial_params:
            continue
        final = parameter.detach().cpu()
        difference = (final-initial_params[name]).abs()
        for item in components.values():
            if name.startswith(item['prefix']):
                changed = int(torch.count_nonzero(difference))
                item['changed_tensors'] += int(changed > 0)
                item['changed_elements'] += changed
                item['max_abs_change'] = max(item['max_abs_change'], float(difference.max()))
    if not all(item['changed_elements'] > 0 for item in components.values()):
        raise AssertionError(f'At least one component did not update trainable parameters: {components}')
    report['component_parameter_changes'] = components
    final_path = artifacts/'model_final.pth.tar'
    atomic_torch_save(model.state_dict(), final_path)
    final_checkpoint = artifacts/'continuous_final_checkpoint.pth.tar'
    save_training_checkpoint(final_checkpoint, model, optimizer, scaler, whole, sampler, manifest)
    whole_state = whole.state_dict()
    whole_rng = capture_rng_state()
    # No CUDA object from the continuous model survives into reconstruction.
    del model, optimizer, scaler, sampler, loader, initial_params, parameter, final, difference
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report['allocated_after_continuous_delete'] = torch.cuda.memory_allocated(device)
    report['stage'] = 'reconstruct_and_restore'
    model, optimizer, scaler, sampler, loader, rebuilt_manifest, _ = construct()
    resumed, saved_manifest = load_training_checkpoint(boundary_path, model, optimizer, scaler,
                                                       sampler, rebuilt_manifest, device=device)
    if resumed.update_attempts != 16 or resumed.next_batch != 64:
        raise AssertionError(f'Unexpected restored boundary: {resumed.state_dict()}')
    prefix_ids = continuous_ids[:resumed.samples_seen]
    resumed_ids = []
    def collect_resumed(event, state=None, **fields):
        if event == 'update':
            resumed_ids.extend(str(value) for value in fields.get('sample_ids', []))
        return original_emit(event, state, **fields)
    emit_event('resume', resumed, phase='gpu_acceptance', path=str(boundary_path),
               epochs=1, epoch=1, batches=sampler.total_batches)
    trainer.emit_event = collect_resumed
    torch.cuda.reset_peak_memory_stats(device)
    resumed_began = time.monotonic()
    report['stage'] = 'resumed_training_and_verification'
    try:
        trainer.do_train(0, cfg, model, loader, optimizer, device=device, scaler=scaler, state=resumed)
    finally:
        trainer.emit_event = original_emit
    torch.cuda.synchronize()
    resumed_rng = capture_rng_state()
    expected = torch.load(final_checkpoint, map_location='cpu', weights_only=True)
    exact_tree(expected['state_dict'], model.state_dict(), 'model')
    exact_tree(expected['optimizer'], optimizer.state_dict(), 'optimizer')
    exact_tree(expected['scaler'], scaler.state_dict(), 'scaler')
    exact_tree(whole_state, resumed.state_dict(), 'counters')
    exact_tree(whole_rng, resumed_rng, 'all_rng_states')
    if prefix_ids+resumed_ids != continuous_ids:
        raise AssertionError('The resumed sample-ID stream differs from continuous training')
    assert_finite_weights(model)
    resumed_final = artifacts/'model_resumed_final.pth.tar'
    atomic_torch_save(model.state_dict(), resumed_final)
    report['resume'] = dict(exact=True, model_equal=True, optimizer_equal=True, scaler_equal=True,
        counters_equal=True, rng_equal=True, sample_ids_equal=True, checkpoint_attempt=16,
        seconds=time.monotonic()-resumed_began, state=resumed.state_dict(), sample_ids=resumed_ids,
        memory_allocated_peak=torch.cuda.max_memory_allocated(device),
        memory_reserved_peak=torch.cuda.max_memory_reserved(device))
    report['artifacts'] = {path.name: dict(path=str(path), sha256=sha256_file(path), size=path.stat().st_size)
        for path in (initial_path, initial_checkpoint, boundary_path, final_path, final_checkpoint, resumed_final)}
    current_code = {str(path.relative_to(source)): sha256_file(path)
        for folder in ('core', 'tools') for path in sorted((source/folder).rglob('*.py'))}
    if current_code != manifest['code']:
        raise AssertionError('Source changed during GPU acceptance')
    report['code_sha256'] = current_code
    report['status'] = 'passed'
    emit_event('gpu_acceptance_complete', resumed, phase='gpu_acceptance', mode=args.mode,
               epochs=1, epoch=1, exact_resume=True, report=str(Path(args.report).resolve()))


def main():
    args = arguments()
    report_path = Path(args.report).resolve()
    if report_path.exists():
        raise FileExistsError(f'Report already exists: {report_path}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='failed', mode=args.mode, started=time.time(),
                  command=sys.argv, artifacts_directory=str(Path(args.artifacts).resolve()))
    exitcode = 1
    try:
        run(args, report)
        exitcode = 0
    except Exception as error:
        import torch
        oom = isinstance(error, torch.cuda.OutOfMemoryError)
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if args.mode == 'batch16' and oom:
            report.update(status='limited', feasible=False, expected_cuda_oom=True,
                          interpretation='Actual physical16 CUDA allocation failed; no downsizing or retry.')
            exitcode = 0
        else:
            report['status'] = 'failed'
        print(report['traceback'], file=sys.stderr, flush=True)
        if torch.cuda.is_initialized():
            report['failure_memory'] = dict(allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(), peak=torch.cuda.max_memory_allocated())
    finally:
        report['finished'] = time.time()
        report['exitcode'] = exitcode
        write_json(report_path, report)
    return exitcode


if __name__ == '__main__':
    sys.exit(main())
