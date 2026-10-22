"""One explicitly bound ODR003 task; no queue, retries, downloads or implicit resume.

CLI: python run_exploration.py --task-json /absolute/path/task.json
Execution must be inside the project's guarded screen. A task JSON is immutable
and binds all code, source weights, preflight receipts and image manifests by SHA.
The calling manager must independently verify the report before advancing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

PLAN = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
DATA_ROOT = Path('/home/DATA/Downloads/DataSet/speedplus')
METHODS = {'O', 'A', 'V', 'E50', 'E75', 'G50', 'G75', 'R'}
REQUIRED_CODE = {'run_exploration.py', 'data_boundary.py', 'prediction.py', 'state_adapter.py', 'gate_core.py', 'risk_metrics.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json_text(obj)+'\n'); f.flush(); os.fsync(f.fileno())


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('x', encoding='utf-8') as f:
        f.write(json_text(obj)+'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def file_binding(value, *, verify=True):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('A file binding must contain exactly path and sha256')
    path = Path(value['path'])
    if not path.is_absolute() or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
        raise ValueError('Invalid absolute file binding')
    if verify and sha(path) != value['sha256']:
        raise ValueError(f'Bound file changed: {path}')
    return path.resolve()


class GTBoundary:
    """Deny full target label sources always, developer labels until evaluation."""
    def __init__(self, allowed_developer):
        self.phase = 'preflight'
        self.allowed = Path(allowed_developer).resolve()
        self.developer_reads = 0
        self.denied = []
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        # No source label file is needed by this runner, even in evaluation.
        if path.name == 'test.json' or (path.is_relative_to(DATA_ROOT) and ('labels' in path.parts or path.suffix.lower() == '.csv')):
            self.denied.append(str(path))
            raise PermissionError('Full target GT source forbidden by ODR003 boundary')
        if path.name == 'dev_labels.json':
            if self.phase != 'evaluation' or path.resolve() != self.allowed:
                self.denied.append(str(path))
                raise PermissionError('Developer GT forbidden outside its evaluation boundary')
            self.developer_reads += 1


class Progress:
    def __init__(self, directory, task):
        self.directory = directory
        self.task = task
        self.last = 0.0
        self.started = time.monotonic()
        self.phase = 'preflight'
        self.observed = self.evaluated = self.updates = 0

    def emit(self, *, force=False, status='running', **extra):
        now = time.monotonic()
        if not force and now-self.last < 60:
            return
        self.last = now
        proc = Path(f'/proc/{os.getpid()}/stat').read_text()
        stat_tail = proc[proc.rfind(')')+2:].split()
        obj = dict(schema='odr003-progress-v1', status=status, phase=self.phase,
            run_id=self.task['run_id'], method=self.task['method'], domain=self.task['domain'], stream=self.task['stream'],
            updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), starttime=int(stat_tail[19]),
            observed=self.observed, evaluated=self.evaluated, updates=self.updates,
            elapsed_seconds=now-self.started, **extra)
        atomic_json(self.directory/'progress.json', obj)
        print('SPNV2_EVENT '+json_text({**obj, 'event':'odr003_progress',
            'source_sha256':self.task['bindings']['source']['sha256'],
            'config_sha256':self.task['bindings']['config']['sha256'],
            'output_dir':str(self.directory)}),flush=True)


def verify_task(task):
    required = {'schema', 'run_id', 'method', 'domain', 'stream', 'seed', 'output_dir', 'bindings', 'code_sha256'}
    if not required <= task.keys() or task['schema'] != 'odr003-task-v1':
        raise ValueError('Wrong task schema')
    if task['method'] not in METHODS or task['domain'] not in {'lightbox','sunlamp'}:
        raise ValueError('Unknown method/domain')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task['run_id']):
        raise ValueError('Unsafe run ID')
    if task['method'] == 'O':
        if task['stream'] != 'offline' or task['seed'] != 2021:
            raise ValueError('Offline O is one seed2021 source evaluation per domain')
    elif task['stream'] not in {'2021','2022','2023'} or task['seed'] != int(task['stream']):
        raise ValueError('Only the three confirmed paired streams are allowed')
    output = Path(task['output_dir']).resolve()
    if not output.is_relative_to(PLAN/'runs') or output.name != task['run_id']:
        raise ValueError('Unique run output must stay under this plan/runs/run_id')
    if not os.environ.get('STY'):
        raise RuntimeError('Run this task through run_in_screen_guarded.sh')
    if task.get('checkpoint_every_updates', 64) != 64:
        raise ValueError('Boundary checkpoint interval is fixed to64 for this runner')
    code = task['code_sha256']
    if set(code) != REQUIRED_CODE:
        raise ValueError('All six implementation modules require explicit hashes')
    implementation = Path(__file__).resolve().parent
    for name, expected in code.items():
        if sha(implementation/name) != expected:
            raise ValueError(f'Implementation hash mismatch: {name}')
    return output


def verify_bindings(task):
    b = task['bindings']
    required = {'official_receipt','source','config','source_core_manifest','data_prepared','data_verified',
                'developer_manifest','developer_labels','camera','keypoints'}
    if not required <= b.keys():
        raise ValueError('Missing task asset binding')
    # Developer labels are hashed only after the adaptation phase ends.
    paths = {key:file_binding(value, verify=(key != 'developer_labels')) for key,value in b.items()}
    receipt = read_json(paths['official_receipt'])
    if receipt.get('status') != 'passed' or receipt.get('source_offline_identity_verified') is not True or receipt.get('strict_model_load_performed') is not True:
        raise ValueError('Official source identity and strict-load receipt must pass first')
    for key, rprefix in [('source','source'), ('config','config'), ('source_core_manifest','source_core_manifest')]:
        if Path(receipt[rprefix+'_path']).resolve() != paths[key] or receipt[rprefix+'_sha256'] != b[key]['sha256']:
            raise ValueError('Official receipt/task binding disagreement: '+key)
    if not paths['source'].is_relative_to(PLAN/'assets'):
        raise ValueError('Use the explicitly authorized, copied source asset in this plan')
    strict_path=Path(receipt['strict_load_report_path'])
    if sha(strict_path)!=receipt['strict_load_report_sha256']:
        raise ValueError('Strict-load proof changed')
    strict=read_json(strict_path)
    if strict.get('status')!='passed' or strict.get('strict_load') is not True or strict.get('all_loaded_tensors_equal_source') is not True:
        raise ValueError('Actual strict-load proof does not pass')
    dependency=strict.get('cached_imagenet_build_dependency')
    if dependency and (not Path(dependency['path']).is_file() or sha(dependency['path'])!=dependency['sha256']):
        raise ValueError('Verified local architecture-build cache missing/changed; no implicit download allowed')
    core_manifest = read_json(paths['source_core_manifest'])
    core = Path(core_manifest['root']).resolve()
    files = core_manifest['files']
    if not isinstance(files,dict) or not files:
        raise ValueError('Frozen core manifest files must map relative paths to SHA256')
    actual_py = {str(p.relative_to(core)) for p in core.rglob('*.py')}
    if set(files) != actual_py:
        raise ValueError('Frozen core Python file coverage differs')
    for rel, expected in files.items():
        path = (core/rel).resolve()
        if not path.is_relative_to(core) or path.suffix != '.py' or sha(path) != expected:
            raise ValueError('Frozen original A source drift: '+rel)
    for binding,prefix in [('camera','camera'),('keypoints','keypoint')]:
        if paths[binding]!=Path(core_manifest[prefix+'_path']).resolve() or b[binding]['sha256']!=core_manifest[prefix+'_sha256']:
            raise ValueError('Camera/keypoint geometry assets differ from strict-load source manifest')
    prepared = read_json(paths['data_prepared'])
    verified = read_json(paths['data_verified'])
    if prepared.get('status') != 'prepared' or verified.get('status') != 'passed' or verified['prepared_proof_sha256'] != b['data_prepared']['sha256']:
        raise ValueError('Data preparation/verification evidence mismatch')
    domain = task['domain']
    for binding, artifact in [('developer_manifest',domain+'/developer'),('developer_labels',domain+'/developer_labels')]:
        expected = prepared['artifacts'][artifact]
        if paths[binding] != Path(expected['path']).resolve() or b[binding]['sha256'] != expected['sha256']:
            raise ValueError('Developer boundary differs from frozen data proof')
    if task['method'] != 'O':
        if 'stream_manifest' not in paths:
            raise ValueError('Adaptation needs its exact frozen stream')
        expected = prepared['artifacts'][domain+'/stream_'+task['stream']]
        if paths['stream_manifest'] != Path(expected['path']).resolve() or b['stream_manifest']['sha256'] != expected['sha256']:
            raise ValueError('Adaptation stream differs from frozen data proof')
    if verified['counts'][domain]['developer'] != 512:
        raise ValueError('Developer boundary denominator changed')
    return paths, receipt, core, prepared


def configure(task, paths, core):
    sys.path.insert(0, str(core))
    from config import cfg as defaults
    from config.efficientdet import EFFICIENTDET_EXTRAS
    cfg = defaults.clone(); cfg.defrost(); cfg.merge_from_file(str(paths['config']))
    cfg.MODEL.BACKBONE.EXTRA = EFFICIENTDET_EXTRAS[cfg.MODEL.BACKBONE.NAME]
    expected = [cfg.MODEL.EFFICIENTDET_PHI == 3, cfg.MODEL.BACKBONE.NAME == 'efficientdet_d3',
        not cfg.MODEL.USE_GROUPNORM_BACKBONE, bool(cfg.MODEL.USE_GROUPNORM_HEADS),
        list(cfg.MODEL.HEAD.NAMES) == ['heatmap','efficientpose','segmentation'],
        list(cfg.MODEL.HEAD.LOSS_HEADS) == [], bool(cfg.ODR.MIN_ENTROPY),
        list(cfg.DATASET.INPUT_SIZE) == [768,512], not cfg.FP16, cfg.ODR.IMAGES_PER_BATCH == 4,
        cfg.ODR.NUM_TRAIN_SAMPLES == 1024, cfg.TRAIN.IMAGES_PER_GPU == 1, cfg.TEST.IMAGES_PER_GPU == 1,
        cfg.TRAIN.ACCUM_STEPS == 1, cfg.TRAIN.OPTIMIZER == 'AdamW', cfg.TRAIN.LR == 1e-4,
        cfg.TRAIN.WD == 0, cfg.TRAIN.GAMMA1 == .9, cfg.TRAIN.GAMMA2 == .999]
    if not all(expected):
        raise ValueError('Frozen original A settings differ from the confirmed experiment')
    changes = {'ROOT': str(core.parent), 'OUTPUT_DIR': task['output_dir'], 'LOG_DIR': task['output_dir'],
        'EXP_NAME': task['run_id'], 'SEED': task['seed'], 'AUTO_RESUME': False, 'CUDA': True,
        'DATASET.ROOT': str(DATA_ROOT.parent), 'DATASET.CAMERA': str(paths['camera']),
        'DATASET.KEYPOINTS': str(paths['keypoints']), 'MODEL.PRETRAIN_FILE': str(paths['source']),
        'MODEL.BACKBONE.PRETRAINED': None, 'TEST.MODEL_FILE': str(paths['source']),
        'TRAIN.TRAIN_CSV': task['domain']+'/labels/forbidden.csv',
        'TRAIN.VAL_CSV': task['domain']+'/labels/forbidden.csv', 'TEST.TEST_CSV': task['domain']+'/labels/forbidden.csv'}
    for dotted,value in changes.items():
        keys=dotted.split('.'); node=cfg
        for key in keys[:-1]: node=node[key]
        node[keys[-1]]=value
    cfg.freeze()
    return cfg, changes


def cpu_tree(value):
    import copy
    import torch
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def finite_state(model, optimizer=None):
    import torch
    def check(value):
        if isinstance(value,torch.Tensor) and (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite persistent model/optimizer state')
        if isinstance(value,dict):
            for v in value.values(): check(v)
        elif isinstance(value,(tuple,list)):
            for v in value: check(v)
    check(model.state_dict())
    if optimizer is not None: check(optimizer.state_dict())


def save_checkpoint(path, adapter, task, task_sha, sampler_index, counters, *, final=False):
    import torch
    from state_adapter import rng_state
    finite_state(adapter.model,adapter.optimizer)
    if not final and adapter.accepted % 4:
        raise ValueError('Mid-window checkpoints forbidden')
    complete_accepted = adapter.updates*4
    obj = dict(schema='odr003-complete-window-v1', task_sha256=task_sha, source_sha256=task['bindings']['source']['sha256'],
        bindings=task['bindings'], code_sha256=task['code_sha256'], run_id=task['run_id'],
        method=task['method'], domain=task['domain'], stream=task['stream'],
        model=cpu_tree(adapter.model.state_dict()), optimizer=cpu_tree(adapter.optimizer.state_dict()),
        scaler=None, trainable_parameter_names=[n for n,p in adapter.model.named_parameters() if p.requires_grad],
        rng=cpu_tree(rng_state()), sampler=dict(next_index=sampler_index, stream_binding=task['bindings'].get('stream_manifest')),
        adapter=dict(accepted=complete_accepted, updates=adapter.updates, observed=adapter.observed, batch=4,
            modes={n:m.training for n,m in adapter.model.named_modules()}, bn_features={},bn_mean={},bn_mean_sq={}),
        counters=counters, tail_discarded=bool(final), final=final, checkpoint_boundary='complete_window_or_final_after_tail_rollback')
    with Path(path).open('xb') as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size,observed=adapter.observed,updates=adapter.updates)


def load_calibration(task, paths, prepared):
    if task['method'] not in {'E50','E75','G50','G75'}:
        return None
    if not {'calibration','calibration_proof'} <= paths.keys(): raise ValueError('Independently verified pooled source calibration required')
    proof=read_json(paths['calibration_proof'])
    if proof.get('status')!='passed' or proof.get('bundle_sha256')!=task['bindings']['calibration']['sha256'] or proof.get('recomputed') is not True:
        raise ValueError('Independent calibration verification mismatch')
    bundle=read_json(paths['calibration'])
    if bundle.get('status') != 'passed' or bundle.get('source_sha256') != task['bindings']['source']['sha256']:
        raise ValueError('Calibration source provenance missing')
    expected={d:prepared['artifacts'][d+'/developer']['sha256'] for d in ('lightbox','sunlamp')}
    if bundle.get('developer_manifest_sha256') != expected:
        raise ValueError('Calibration developer boundaries differ')
    calibration=bundle['calibration']
    if calibration.get('schema') != 'odr003-frozen-shared-calibration-v1' or calibration.get('ecdf') != 'midrank' or calibration.get('interpolation') != 'linear':
        raise ValueError('Unsupported gate calibration')
    return calibration


def load_mask(task, paths):
    if task['method'] != 'R': return None
    if not {'random_mask','selection_report','matched_g_report'} <= paths.keys():
        raise ValueError('R needs frozen mask, selected-G evidence and exploration selection')
    mask=read_json(paths['random_mask']); selection=read_json(paths['selection_report'])['selection']; g=read_json(paths['matched_g_report'])
    selected=selection.get('selected_method')
    if selection.get('run_random_controls') is not True or selected not in {'G50','G75'}:
        raise ValueError('No predeclared eligible G trend for random controls')
    if g.get('status') != 'passed' or g.get('method') != selected or g.get('domain') != task['domain'] or g.get('stream') != task['stream']:
        raise ValueError('Matched G report does not match selected method/domain/stream')
    if g['source_sha256'] != task['bindings']['source']['sha256'] or g['bindings']['stream_manifest'] != task['bindings']['stream_manifest']:
        raise ValueError('Matched G source/stream identity differs')
    values=mask['mask']; target=g['counts']['used']
    if mask.get('mask_id') not in {'0','1','2'} or mask.get('source_sha256') != task['bindings']['source']['sha256'] or mask.get('stream_manifest_sha256') != task['bindings']['stream_manifest']['sha256']:
        raise ValueError('Random mask identity differs')
    if len(values)!=1024 or any(type(x) is not bool for x in values) or sum(values)!=target or target%4:
        raise ValueError('Random mask fails exact full-window update-budget match')
    if mask.get('matched_g_report_sha256') != task['bindings']['matched_g_report']['sha256']:
        raise ValueError('Random mask not bound to the matched G result')
    from gate_core import random_budget_mask
    mask_seed=mask.get('mask_seed')
    if type(mask_seed) is not int or mask_seed<0 or values!=random_budget_mask(1024,target,mask_seed,batch=4):
        raise ValueError('Random mask cannot be reproduced from its frozen seed')
    return mask


def run(task_path):
    task=read_json(task_path); task_sha=sha(task_path)
    output=verify_task(task)
    output.parent.mkdir(parents=True,exist_ok=True); output.mkdir(exist_ok=False)
    progress=Progress(output,task); progress.emit(force=True)
    write_new(output/'task.json',task)
    boundary=GTBoundary(task['bindings']['developer_labels']['path'])
    adapter=None
    try:
        paths,receipt,core,prepared=verify_bindings(task)
        cfg,changes=configure(task,paths,core)
        with (output/'effective_config.yaml').open('x') as f: f.write(cfg.dump())
        write_new(output/'config_path_overrides.json',changes)
        import numpy as np
        import torch
        from nets.build import SPNv2
        from dataset.transforms.build import build_transforms
        from utils.utils import load_camera_intrinsics,load_tango_3d_keypoints,set_seeds_cudnn
        from data_boundary import ImageManifestDataset,load_image_manifest
        from state_adapter import OriginalAAdapter,equal_tree
        import prediction
        from gate_core import decide
        if not torch.cuda.is_available(): raise RuntimeError('This is an actual GPU experiment; CPU substitution forbidden')
        device=torch.device('cuda:0')
        torch.cuda.set_device(device)
        set_seeds_cudnn(cfg,seed=task['seed'])
        torch.cuda.reset_peak_memory_stats(device)
        timings={'preflight_and_load_seconds':0.,'adaptation_seconds':0.,'image_read_seconds':0.,
            'scoring_seconds':0.,'scoring_forward_seconds':0.,'scoring_geometry_seconds':0.,
            'update_seconds':0.,'checkpoint_seconds':0.,'evaluation_seconds':0.,
            'evaluation_forward_seconds':0.,'evaluation_geometry_seconds':0.}
        def sync(): torch.cuda.synchronize(device)
        started=time.perf_counter()
        model=SPNv2(cfg)
        source=torch.load(paths['source'],map_location='cpu',weights_only=True)
        state_key=receipt.get('state_dict_key')
        state=source if state_key is None else source[state_key]
        if not isinstance(state,dict) or not state or not all(isinstance(x,torch.Tensor) for x in state.values()):
            raise ValueError('Official checkpoint format differs from strict-load receipt')
        loaded=model.load_state_dict(state,strict=True)
        if loaded.missing_keys or loaded.unexpected_keys: raise ValueError('Source strict load failed')
        finite_state(model)
        del source,state
        model=model.float().to(device)
        adapter=OriginalAAdapter.configure(model,device,batch=4)
        operator_warmup=prediction.warmup_bbox_decoder(model)
        if any(p.dtype != torch.float32 for p in model.parameters()): raise ValueError('FP32 contract failed')
        camera=load_camera_intrinsics(str(paths['camera'])); points=load_tango_3d_keypoints(str(paths['keypoints']))
        transforms=build_transforms(cfg,is_train=False,load_labels=False)
        dev=ImageManifestDataset(paths['developer_manifest'],transforms=transforms)
        dev_ids=[x['sample_id'] for x in dev.manifest['items']]
        if len(dev_ids)!=512 or any(not x.startswith(task['domain']+'/') for x in dev_ids): raise ValueError('Wrong developer membership')
        calibration=load_calibration(task,paths,prepared); mask=load_mask(task,paths)
        sync(); timings['preflight_and_load_seconds']=time.perf_counter()-started
        checkpoints=[]; forward_counts={'quality':0,'accepted_update':0,'developer_evaluation':0}
        profile={'E50':'entropy','E75':'entropy','V':'pose','G50':'full','G75':'full'}.get(task['method'])
        decision_path=output/'decisions.jsonl'
        progress.phase='adaptation'; boundary.phase='adaptation'; progress.emit(force=True)
        sync(); adaptation_started=time.perf_counter()
        with decision_path.open('x',encoding='utf-8') as decisions:
            if task['method'] != 'O':
                stream=ImageManifestDataset(paths['stream_manifest'],transforms=transforms)
                if len(stream)!=1024 or stream.manifest['seed']!=task['seed']: raise ValueError('Wrong paired observation stream')
                for index in range(1024):
                    t=time.perf_counter(); sample=stream[index]; image=sample['image'].unsqueeze(0)
                    timings['image_read_seconds']+=time.perf_counter()-t
                    quality=None
                    if profile is not None:
                        sync(); t=time.perf_counter()
                        quality=prediction.extract(model,adapter,image,cfg,camera,points,profile=profile)
                        sync(); timings['scoring_seconds']+=time.perf_counter()-t; forward_counts['quality']+=1
                        timings['scoring_forward_seconds']+=quality['timing_seconds']['forward']
                        timings['scoring_geometry_seconds']+=quality['timing_seconds']['geometry']
                    accepted,reason,score=decide(task['method'],quality,calibration,random_accept=mask['mask'][index] if mask is not None else None)
                    adapter.observed+=1
                    loss=None; previous_updates=adapter.updates
                    if accepted:
                        sync(); t=time.perf_counter(); loss=adapter.accept(image); sync()
                        timings['update_seconds']+=time.perf_counter()-t; forward_counts['accepted_update']+=1
                        if not math.isfinite(loss): raise FloatingPointError('Nonfinite recorded adaptation loss')
                    row=dict(index=index,sample_id=sample['sample_id'],domain=task['domain'],stream=task['stream'],method=task['method'],
                        accepted=accepted,reason=reason,score=score,loss=loss,quality=quality,
                        accepted_total=adapter.accepted,updates=adapter.updates,observed=adapter.observed)
                    decisions.write(json_text(row)+'\n'); decisions.flush()
                    if adapter.updates!=previous_updates and adapter.updates%64==0:
                        t=time.perf_counter()
                        counters=dict(observed=adapter.observed,accepted=adapter.accepted,rejected=adapter.observed-adapter.accepted,
                            updates=adapter.updates,used=4*adapter.updates,tail=0)
                        checkpoints.append(save_checkpoint(output/f'checkpoint_u{adapter.updates:04d}.pt',adapter,task,task_sha,index+1,counters))
                        timings['checkpoint_seconds']+=time.perf_counter()-t
                    progress.observed=adapter.observed; progress.updates=adapter.updates; progress.emit()
            decisions.flush(); os.fsync(decisions.fileno())
        counts=adapter.finish()
        sync(); timings['adaptation_seconds']=time.perf_counter()-adaptation_started
        if task['method']=='O' and any(counts.values()): raise ValueError('Offline state unexpectedly adapted')
        if task['method']!='O' and (counts['observed']!=1024 or counts['accepted']+counts['rejected']!=1024 or counts['used']+counts['tail']!=counts['accepted']):
            raise ValueError('Observation/update/tail counts disagree')
        if task['method']=='A' and (counts['used']!=1024 or counts['updates']!=256): raise ValueError('Original A update count differs')
        if mask is not None and (counts['used']!=sum(mask['mask']) or counts['tail']!=0): raise ValueError('Random budget mismatch')
        progress.phase='final_checkpoint'; progress.emit(force=True)
        t=time.perf_counter()
        final_checkpoint=save_checkpoint(output/'final_checkpoint.pt',adapter,task,task_sha,counts['observed'],counts,final=True)
        # Strictly reload actual serialized final state; do not accept save() alone.
        saved=torch.load(final_checkpoint['path'],map_location='cpu',weights_only=False)
        if not equal_tree(saved['model'],model.state_dict()) or not equal_tree(saved['optimizer'],cpu_tree(adapter.optimizer.state_dict())):
            raise ValueError('Serialized final state disagrees before reload')
        load_result=model.load_state_dict(saved['model'],strict=True)
        if load_result.missing_keys or load_result.unexpected_keys or not equal_tree(saved['model'],model.state_dict()):
            raise ValueError('Final strict reload failed')
        del saved
        timings['checkpoint_seconds']+=time.perf_counter()-t
        adapter.close()
        # Adapter remains as a label-free isolation-state holder; hooks removed.
        progress.phase='developer_evaluation'; boundary.phase='evaluation'; progress.emit(force=True)
        if sha(paths['developer_labels']) != task['bindings']['developer_labels']['sha256']:
            raise ValueError('Developer label artifact changed')
        label_doc=read_json(paths['developer_labels'])
        labels=label_doc['items']
        if label_doc.get('domain')!=task['domain'] or [r['sample_id'] for r in labels]!=dev_ids or label_doc['image_manifest_sha256']!=task['bindings']['developer_manifest']['sha256']:
            raise ValueError('Developer label membership or manifest binding mismatch')
        sync(); evaluation_started=time.perf_counter()
        predictions_path=output/'predictions.jsonl'; risk_path=output/'risk_rows.jsonl'
        valid=original_valid=original_rejected=0
        with predictions_path.open('x',encoding='utf-8') as predictions, risk_path.open('x',encoding='utf-8') as risks:
            for index,label in enumerate(labels):
                sample=dev[index]
                if sample['sample_id']!=label['sample_id']: raise ValueError('Developer image/label order mismatch')
                record=prediction.extract(model,adapter,sample['image'].unsqueeze(0),cfg,camera,points,profile='full')
                timings['evaluation_forward_seconds']+=record['timing_seconds']['forward']
                timings['evaluation_geometry_seconds']+=record['timing_seconds']['geometry']
                metric_label={'quaternion':label['quaternion_wxyz'],'translation':label['translation_m']}
                metrics=prediction.developer_metrics(record,metric_label,cfg)
                base=dict(sample_id=sample['sample_id'],domain=task['domain'],stream=task['stream'],method=task['method'])
                if mask is not None: base['mask']=mask['mask_id']
                row={**base,**record,**metrics,'original_rejected':record['original_h_rejected']}
                predictions.write(json_text(row)+'\n')
                # O is measured once; risk rows explicitly repeat the same measurement
                # for the three paired streams as required by risk_metrics' contract.
                for stream_id in ('2021','2022','2023') if task['method']=='O' else (task['stream'],):
                    risk={**base,**metrics,'stream':stream_id,'original_rejected':record['original_h_rejected']}
                    risks.write(json_text(risk)+'\n')
                valid+=int(metrics['valid']); original_valid+=int(metrics['original_valid'])
                original_rejected+=int(record['original_h_rejected'])
                forward_counts['developer_evaluation']+=1
                progress.evaluated=index+1; progress.emit()
            for f in (predictions,risks): f.flush(); os.fsync(f.fileno())
        sync(); timings['evaluation_seconds']=time.perf_counter()-evaluation_started
        finite_state(model,adapter.optimizer)
        evaluated_state=torch.load(final_checkpoint['path'],map_location='cpu',weights_only=False)
        if not equal_tree(evaluated_state['model'],model.state_dict()) or not equal_tree(evaluated_state['optimizer'],cpu_tree(adapter.optimizer.state_dict())):
            raise ValueError('Developer evaluation polluted frozen model/optimizer state')
        del evaluated_state
        if sha(paths['source']) != task['bindings']['source']['sha256']: raise ValueError('Source bytes changed during task')
        if boundary.denied: raise RuntimeError('A forbidden GT read was attempted')
        bindings=task['bindings']
        outputs={name:dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for name,path in
            [('decisions',decision_path),('predictions',predictions_path),('risk_rows',risk_path),('effective_config',output/'effective_config.yaml')]}
        report=dict(schema='odr003-single-task-report-v1',status='passed',independent_verification_required=True,
            run_id=task['run_id'],method=task['method'],domain=task['domain'],stream=task['stream'],seed=task['seed'],
            task_sha256=task_sha,source_sha256=bindings['source']['sha256'],bindings=bindings,code_sha256=task['code_sha256'],
            source_strict_load=True,final_strict_reload=True,operator_warmup=operator_warmup,counts=counts,developer_count=512,
            trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
            extended_valid=valid,original_valid=original_valid,original_rejected=original_rejected,
            evaluation_model_optimizer_unchanged=True,
            all_samples=valid==512,score_profile=profile,forward_counts=forward_counts,
            timings=timings,wall_seconds=time.monotonic()-progress.started,
            timing_note='Scoring total includes transactional isolation and forward/geometry; subcomponents are nested, not additive across totals. CUDA work synchronized.',
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            gpu_name=torch.cuda.get_device_name(device),torch_version=torch.__version__,numpy_version=np.__version__,
            outputs=outputs,checkpoints=checkpoints,final_checkpoint=final_checkpoint,
            gt_boundary=dict(source_labels_forbidden=True,adaptation_developer_labels_forbidden=True,
                developer_label_open_events=boundary.developer_reads,forbidden_attempts=boundary.denied),
            checkpoint_resume='explicit separate recovery review required; no automatic retry or resume',
            scope='developer_exploration_only_no_reserved_or_full_domain_scores')
        if mask is not None: report['mask']=mask['mask_id']
        write_new(output/'report.json',report)
        progress.phase='complete'; progress.emit(force=True,status='passed',independent_verification_required=True)
        print(json_text({'status':'passed','run_id':task['run_id'],'report':str(output/'report.json'),'counts':counts}),flush=True)
    except BaseException as error:
        failure=dict(status='failed',run_id=task['run_id'],phase=progress.phase,type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),created_at=datetime.now(timezone.utc).isoformat(),next_action='preserve_run_and_block_downstream_no_automatic_retry')
        write_new(output/'failure.json',failure)
        progress.emit(force=True,status='failed',error=str(error))
        raise
    finally:
        if adapter is not None: adapter.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-json',required=True)
    run(parser.parse_args().task_json)
