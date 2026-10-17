'''
Copyright (c) 2022 SLAB Group
Licensed under MIT License (see LICENSE.md)
Author: Tae Ha Park (tpark94@stanford.edu)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import argparse
import warnings
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import _init_paths

from config  import cfg, update_config
from nets    import build_spnv2
from dataset import get_dataloader
from solver  import get_optimizer, adjust_learning_rate, get_scaler
from engine.trainer    import do_train
from engine.inference  import do_valid
from utils.checkpoints import load_checkpoint, save_checkpoint
from utils.training_state import (TrainingState, emit_event, build_manifest,
    save_training_checkpoint, load_training_checkpoint, atomic_torch_save,
    prepare_output_directory, sha256_file, validate_reproducible_inputs)
from utils.utils import set_seeds_cudnn, setup_logger, create_logger_directories, \
                        write_model_info, load_camera_intrinsics, load_tango_3d_keypoints

def parse_args():
    parser = argparse.ArgumentParser(description='Train SPNv2')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training (default: 1)')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training (default: 0)')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args

def main(cfg):
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.DIST.RANK = args.rank
    cfg.freeze()

    if cfg.TRAIN.ACCUM_STEPS < 1:
        raise ValueError('TRAIN.ACCUM_STEPS must be positive')
    if cfg.TRAIN.REPRODUCIBLE:
        if args.world_size != 1 or cfg.DIST.MULTIPROCESSING_DISTRIBUTED:
            raise ValueError('Reproducible boundary resume supports one process')
        if cfg.SEED is None:
            raise ValueError('Reproducible training requires explicit SEED')
        if cfg.TRAIN.CHECKPOINT_UPDATES < 1:
            raise ValueError('TRAIN.CHECKPOINT_UPDATES must be positive')
        if cfg.TRAIN.VALID_FRACTION != 0.1:
            raise ValueError('New source training uses the fixed validation fraction 0.1')
        validate_reproducible_inputs(cfg)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        prepare_output_directory(cfg)

    _, output_dir, log_dir = \
        create_logger_directories(cfg, phase='train', write_cfg_to_file=True)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.multiprocessing_distributed = cfg.DIST.MULTIPROCESSING_DISTRIBUTED
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Adjust world_size
        args.world_size = ngpus_per_node * args.world_size

        # Spawn distributed processes
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, output_dir, log_dir)
        )
    else:
        main_worker(
            args.gpu,
            ngpus_per_node,
            args,
            output_dir,
            log_dir
        )

def main_worker(gpu, ngpus_per_node, args, output_dir, log_dir):

    # Set all seeds & cudNN
    set_seeds_cudnn(cfg, seed=cfg.SEED)
    if cfg.TRAIN.REPRODUCIBLE:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    # GPU?
    args.gpu = gpu
    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    # Distributed?
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))

        dist.init_process_group(
            backend=cfg.DIST.BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config(cfg, args)

    # setup logger
    logger = setup_logger(log_dir, args.rank, 'train', to_console=args.distributed)

    # build network
    model = build_spnv2(cfg)

    # GPU device
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

    if args.distributed:
        # SyncBN
        if not cfg.MODEL.USE_GROUPNORM_BACKBONE or \
              not cfg.MODEL.USE_GROUPNORM_HEADS:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('   - SyncBN activated for distributed training')

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            model = torch.nn.parallel.DistributedDataParallel(
                model.to(device), device_ids=[args.gpu],
                find_unused_parameters=cfg.MODEL.FIND_UNUSED_PARAM
            )
            logger.info('   - Model wrapped to nn.parallel.DistributedDataParallel')
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model.to(device))
    else:
        model = model.to(device)

    # write model summary to file
    if args.rank == 0:
        write_model_info(model.module if args.distributed else model, log_dir)

    # Dataloaders
    train_loader = get_dataloader(cfg,
                                  split='train',
                                  distributed=args.distributed,
                                  load_labels=True)
    val_loader   = get_dataloader(cfg,
                                  split='val',
                                  distributed=args.distributed,
                                  load_labels=True)

    # Optimizer & scaler for mixed-precision training
    optimizer = get_optimizer(cfg, model)
    scaler    = get_scaler(cfg) # None if cfg.FP16 = False, cfg.CUDA = False

    if cfg.TRAIN.REPRODUCIBLE:
        train_reproducible(cfg, model, train_loader, val_loader, optimizer, scaler,
                           output_dir, log_dir, device, logger)
        return

    # Load checkpoints
    checkpoint_file = osp.join(output_dir, f'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and osp.exists(checkpoint_file):
        last_epoch, best_score = load_checkpoint(
                        checkpoint_file,
                        model,
                        optimizer,
                        scaler,
                        device)
        begin_epoch = last_epoch
    else:
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        last_epoch  = -1
        best_score  = 1e20

    # For validation
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)

    # ---------------------------------------
    # Main loop
    # ---------------------------------------
    training_state = TrainingState()
    score   = best_score
    is_best = False
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        print('')

        # Learning rate adjustment
        adjust_learning_rate(optimizer, epoch, cfg)

        if args.rank == 0:
            logger.info(
                '========== Epoch %d/%d | TRAIN START | batches=%d | lr=%s ==========',
                epoch + 1, cfg.TRAIN.END_EPOCH, len(train_loader),
                ', '.join(f"{group['lr']:.2e}" for group in optimizer.param_groups))

        # Single epoch training
        training_state.next_batch = 0
        training_state = do_train(epoch,
                 cfg,
                 model,
                 train_loader,
                 optimizer,
                 log_dir=log_dir,
                 device=device,
                 scaler=scaler,
                 rank=args.rank,
                 state=training_state)

        if not args.distributed or (args.distributed and args.rank == 0):
            # Validate on the fraction of dataset
            if (epoch+1) % cfg.TRAIN.VALID_FREQ == 0:
                logger.info('Epoch %d/%d | VALIDATION START', epoch + 1, cfg.TRAIN.END_EPOCH)
                emit_event('validation_start', training_state, phase='validation', epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH)
                score = do_valid(epoch,
                                 cfg,
                                 model,
                                 val_loader,
                                 camera,
                                 keypts_true_3D,
                                 valid_fraction=cfg.TRAIN.VALID_FRACTION,
                                 log_dir=None,
                                 device=device)

                emit_event('validation_end', training_state, phase='validation', epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH, score=float(score))

                if score < best_score:
                    best_score = score
                    is_best = True
                else:
                    is_best = False

            # Save
            save_checkpoint({
                'epoch': epoch+1,
                'backbone': cfg.MODEL.BACKBONE.NAME,
                'heads': cfg.MODEL.HEAD.NAMES,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None
            }, is_best, epoch+1==cfg.TRAIN.END_EPOCH, output_dir)

            emit_event('checkpoint', training_state, epoch=epoch+1, epochs=cfg.TRAIN.END_EPOCH, path=checkpoint_file)
            logger.info('========== Epoch %d/%d | COMPLETE | checkpoint saved ==========',
                        epoch + 1, cfg.TRAIN.END_EPOCH)

    if args.rank == 0:
        emit_event('training_complete', training_state, epoch=cfg.TRAIN.END_EPOCH, epochs=cfg.TRAIN.END_EPOCH)
        logger.info('Training finished: %d/%d epochs complete', cfg.TRAIN.END_EPOCH, cfg.TRAIN.END_EPOCH)

def train_reproducible(cfg, model, train_loader, val_loader, optimizer, scaler,
                       output_dir, log_dir, device, logger):
    """Run fixed source validation and persist only complete update boundaries."""
    manifest = build_manifest(cfg, model, datasets={'train': train_loader.dataset,
                                                  'val': val_loader.dataset})
    sampler = train_loader.batch_sampler
    checkpoint_file = osp.join(output_dir, 'checkpoint.pth.tar')
    state = TrainingState(epoch=cfg.TRAIN.BEGIN_EPOCH)
    if cfg.TRAIN.RESUME_FILE:
        state, manifest = load_training_checkpoint(
            cfg.TRAIN.RESUME_FILE, model, optimizer, scaler, sampler, manifest, device)
        emit_event('resume', state, epochs=cfg.TRAIN.END_EPOCH,
                   path=str(cfg.TRAIN.RESUME_FILE), batches=sampler.total_batches,
                   lr=optimizer.param_groups[0]['lr'])
    else:
        sampler.set_epoch(state.epoch, 0)
        initial_path = osp.join(output_dir, 'initial_state.pth.tar')
        save_training_checkpoint(initial_path, model, optimizer, scaler, state, sampler, manifest)
        Path(output_dir, 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
        emit_event('initial_state', state, epochs=cfg.TRAIN.END_EPOCH, path=initial_path,
                   initial_state_sha256=manifest['initial_state_sha256'], batches=sampler.total_batches,
                   lr=optimizer.param_groups[0]['lr'])

    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypoints = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)

    def checkpoint(current, reason, completed_epoch=None):
        save_training_checkpoint(checkpoint_file, model, optimizer, scaler, current, sampler, manifest)
        fields = dict(epochs=cfg.TRAIN.END_EPOCH, path=checkpoint_file, reason=reason,
                      batches=sampler.total_batches, lr=optimizer.param_groups[0]['lr'],
                      checkpoint_sha256=sha256_file(checkpoint_file))
        if completed_epoch is not None:
            fields['epoch'] = completed_epoch
        emit_event('checkpoint', current, **fields)

    def boundary(current):
        if current.update_attempts % cfg.TRAIN.CHECKPOINT_UPDATES == 0:
            checkpoint(current, 'update_interval')

    try:
        for epoch in range(state.epoch, cfg.TRAIN.END_EPOCH):
            state.epoch = epoch
            sampler.set_epoch(epoch, state.next_batch)
            adjust_learning_rate(optimizer, epoch, cfg)
            do_train(epoch, cfg, model, train_loader, optimizer, log_dir=log_dir,
                     device=device, scaler=scaler, state=state, on_boundary=boundary)
            # Fixed source fraction. No target-domain score enters model selection.
            if (epoch+1) % cfg.TRAIN.VALID_FREQ == 0 or epoch+1 == cfg.TRAIN.END_EPOCH:
                emit_event('validation_start', state, phase='validation', epoch=epoch+1,
                           epochs=cfg.TRAIN.END_EPOCH, fraction=0.1,
                           batches=int(len(val_loader)*0.1))
                def validation_progress(progress):
                    progress = dict(progress)
                    for key in ('event', 'phase', 'epoch', 'epochs'):
                        progress.pop(key, None)
                    emit_event('validation_progress', state, phase='validation', epoch=epoch+1,
                               epochs=cfg.TRAIN.END_EPOCH, **progress)

                summary = do_valid(epoch, cfg, model, val_loader, camera, keypoints,
                                   valid_fraction=0.1, log_dir=None, device=device,
                                   return_summary=True, event_callback=validation_progress)
                score = summary['score']
                if score is not None and not math.isfinite(float(score)):
                    raise FloatingPointError('Nonfinite source validation score')
                emit_event('validation_end', state, phase='validation', epoch=epoch+1,
                           epochs=cfg.TRAIN.END_EPOCH, fraction=0.1, score=score,
                           batch=int(len(val_loader)*0.1), batches=int(len(val_loader)*0.1),
                           validation_samples=summary['n_samples'], n_samples=summary['n_samples'],
                           heatmap_rejections=summary['heatmap_rejections'],
                           heatmap_valid=summary['heatmap_valid'], metrics=summary['metrics'])
            state.epoch, state.next_batch = epoch+1, 0
            checkpoint(state, 'epoch_boundary', completed_epoch=epoch+1)
        final_path = osp.join(output_dir, 'model_final.pth.tar')
        atomic_torch_save(model.state_dict(), final_path)
        emit_event('training_complete', state, epoch=cfg.TRAIN.END_EPOCH,
                   epochs=cfg.TRAIN.END_EPOCH, path=final_path,
                   final_sha256=sha256_file(final_path), batches=sampler.total_batches,
                   lr=optimizer.param_groups[0]['lr'])
        logger.info('Training finished: %d/%d epochs complete', cfg.TRAIN.END_EPOCH, cfg.TRAIN.END_EPOCH)
    except Exception as exc:
        emit_event('failure', state, epochs=cfg.TRAIN.END_EPOCH,
                   error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main(cfg)
