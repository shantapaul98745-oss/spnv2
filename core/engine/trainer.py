"""SPNv2 training with local-loss gradient accumulation and boundary events."""
import itertools
import logging
import math
import time

import torch
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

from utils.utils import AverageMeter, ProgressMeter
from utils.training_state import TrainingState, emit_event, assert_finite_weights, tensors_are_finite

logger = logging.getLogger("Training")


def do_train(epoch, cfg, model, data_loader, optimizer, log_dir=None,
             device=torch.device('cpu'), scaler=None, rank=0, state=None, on_boundary=None):
    """Train one epoch; on_boundary(state) runs after zeroing completed-window grads.

    state.next_batch is the next absolute microbatch index, never a partial window.
    An exact-resume sampler already omits consumed batches; no augmentation replay.
    """
    accumulation = int(cfg.TRAIN.ACCUM_STEPS)
    if accumulation < 1:
        raise ValueError('TRAIN.ACCUM_STEPS must be at least one')
    state = TrainingState() if state is None else state
    state.epoch = epoch
    sampler = data_loader.batch_sampler
    total_batches = getattr(sampler, 'total_batches', len(data_loader))
    total_batches = total_batches // accumulation * accumulation
    start_batch = state.next_batch
    if start_batch % accumulation or not 0 <= start_batch <= total_batches:
        raise ValueError('Training must start at a complete update boundary')
    if total_batches == 0:
        raise ValueError('Dataset contains no complete logical training batch')
    if start_batch and getattr(sampler, 'start_batch', None) != start_batch:
        raise ValueError('Resume requires a position-aware batch sampler')

    batch_time = AverageMeter('', 'ms', ':6.1f')
    data_time = AverageMeter('', 'ms', ':6.1f')
    names = []
    if 'heatmap' in cfg.MODEL.HEAD.LOSS_HEADS:
        names += ['hmap']
    if 'efficientpose' in cfg.MODEL.HEAD.LOSS_HEADS:
        names += ['cls', 'box', 'pose']
    if 'segmentation' in cfg.MODEL.HEAD.LOSS_HEADS:
        names += ['seg']
    meters = {name: AverageMeter(name, '', ':.2e') for name in names}
    progress = ProgressMeter(total_batches, batch_time, list(meters.values()),
                             prefix="Training {:03d} ".format(epoch+1))
    model.train()
    assert_finite_weights(model)
    optimizer.zero_grad(set_to_none=True)
    last_report = end = time.monotonic()
    window_ids = []
    if rank == 0:
        physical_batch = getattr(sampler, 'batch_size', getattr(data_loader, 'batch_size', None))
        original_samples = getattr(sampler, 'size', len(data_loader.dataset))
        consumed_samples = total_batches * physical_batch if physical_batch is not None else None
        emit_event('train_start', state, epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH,
                   batches=total_batches, lr=optimizer.param_groups[0]['lr'],
                   accumulation_steps=accumulation, original_samples=original_samples,
                   consumed_samples_per_epoch=consumed_samples,
                   dropped_samples=original_samples-consumed_samples if consumed_samples is not None else None,
                   logical_windows=total_batches//accumulation,
                   dropped_microbatches=original_samples//physical_batch-total_batches
                       if physical_batch is not None else None)

    def failure(message):
        if rank == 0:
            emit_event('failure', state, epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH,
                       batches=total_batches, error=message)
        raise FloatingPointError(message)

    # islice excludes incomplete logical windows before model forward (and loading).
    iterator = itertools.islice(data_loader, total_batches-start_batch)
    for idx, (images, targets) in enumerate(iterator, start=start_batch):
        start = time.monotonic()
        data_time.update((start - end)*1000)
        targets = dict(targets)
        indices = targets.pop('_sample_index', None)
        ids = targets.get('imgpath', indices)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids is not None:
            window_ids.extend(list(ids) if isinstance(ids, (list, tuple)) else [str(ids)])

        with autocast(enabled=scaler is not None):
            loss, loss_items = model(images, is_train=True, gpu=device, **targets)
        if not bool(torch.isfinite(loss.detach()).all()):
            failure(f'Nonfinite loss at epoch {epoch+1}, batch {idx+1}')
        original_loss = float(loss.detach())
        if not all(math.isfinite(float(value)) for value in loss_items.values()):
            failure(f'Nonfinite loss component at epoch {epoch+1}, batch {idx+1}')
        if scaler is not None:
            scaler.scale(loss / accumulation).backward()
        else:
            (loss / accumulation).backward()

        state.last_loss = original_loss
        state.samples_seen += int(images.shape[0])
        state.microbatches_seen += 1
        if (idx + 1) % accumulation == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            finite_grads = tensors_are_finite(p.grad for p in model.parameters())
            if finite_grads:
                norm = clip_grad_norm_(model.parameters(), 1.0)
                if not math.isfinite(float(norm)):
                    failure(f'Nonfinite gradient norm at epoch {epoch+1}, batch {idx+1}')
            elif scaler is None:
                failure(f'Nonfinite gradients at epoch {epoch+1}, batch {idx+1}')

            state.update_attempts += 1
            if scaler is not None:
                before_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                # GradScaler decreases scale if and only if found_inf skipped step.
                skipped = float(scaler.get_scale()) < before_scale
                if not finite_grads and not skipped:
                    failure('AMP failed to skip an update containing nonfinite gradients')
            else:
                optimizer.step()
                skipped = False
            if skipped:
                state.amp_skips += 1
                if state.update_attempts > cfg.TRAIN.AMP_SKIP_WARMUP:
                    state.consecutive_amp_skips += 1
            else:
                state.optimizer_updates += 1
                state.consecutive_amp_skips = 0
            state.next_batch = idx + 1
            optimizer.zero_grad(set_to_none=True)
            assert_finite_weights(model)
            if rank == 0:
                emit_event('update', state, epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH,
                           batches=total_batches, lr=optimizer.param_groups[0]['lr'],
                           sample_ids=window_ids, amp_skipped=skipped)
            window_ids = []
            if state.consecutive_amp_skips >= cfg.TRAIN.AMP_MAX_CONSECUTIVE_SKIPS:
                failure('AMP exceeded consecutive skipped-update limit after warmup')
            if on_boundary is not None:
                on_boundary(state)

        for key, value in loss_items.items():
            if key not in meters:
                meters[key] = AverageMeter(key, '', ':.2e')
            meters[key].update(float(value), images.shape[0])
        batch_time.update((time.monotonic() - start)*1000)
        end = time.monotonic()
        if rank == 0:
            if cfg.VERBOSE:
                progress.display(idx+1)
            if idx == start_batch or (idx+1) % 200 == 0 or idx+1 == total_batches or end-last_report >= 60:
                logger.info('Epoch %d/%d | TRAIN | batch %d/%d (%.1f%%) | loss=%.4g',
                            epoch+1, cfg.TRAIN.END_EPOCH, idx+1, total_batches,
                            100.0*(idx+1)/total_batches, original_loss)
                emit_event('progress', state, epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH,
                           batch=idx+1, batches=total_batches, loss=original_loss,
                           lr=optimizer.param_groups[0]['lr'])
                last_report = end
    if rank == 0:
        progress.display_summary()
        emit_event('train_epoch_end', state, epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH,
                   batches=total_batches, lr=optimizer.param_groups[0]['lr'])
    return state
