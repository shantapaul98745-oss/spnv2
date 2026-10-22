"""Independent CPU acceptance of one bound ODR003 O/A/V/E/G/R run.

Recomputes metrics from saved poses and developer-only labels. Never forwards a
model or reads reserved/source GT. The main report is a claim, not a proof.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

PLAN = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
DATA_ROOT = Path('/home/DATA/Downloads/DataSet/speedplus')
CORE = PLAN.parent/'rebaseline_20261017/odr/source_A/core'
SOURCE_SHA = 'd8f9c281d23adaeb77d03023aee8d40ae1cb9e18dd503539c4719bbe719514cc'
METHODS = {'O','A','V','E50','E75','G50','G75','R'}
CODE = {'run_exploration.py','data_boundary.py','prediction.py','state_adapter.py','gate_core.py','risk_metrics.py'}
METRICS = ('valid','error','original_valid','original_error','rotation_deg','translation_m',
           'heat_rotation_HIL','effi_translation_HIL','heat_error','effi_error')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20), b''):
            h.update(part)
    return h.hexdigest()


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: '+value)


def finite_json(obj):
    if isinstance(obj, float):
        require(math.isfinite(obj), 'Nonfinite JSON number')
    elif isinstance(obj, dict):
        for v in obj.values(): finite_json(v)
    elif isinstance(obj, list):
        for v in obj: finite_json(v)


def read(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)
    finite_json(obj)
    return obj


def lines(path):
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()), 'Blank JSONL row')
            obj=json.loads(line, parse_constant=reject_constant); finite_json(obj)
            require(isinstance(obj,dict), 'JSONL row must be object')
            result.append(obj)
    return result


def number(value, name, *, nonnegative=True):
    require(type(value) in (int,float) and math.isfinite(value), name+' must be finite numeric')
    require(not nonnegative or value>=0, name+' must be nonnegative')
    return value


def count(value, name):
    require(type(value) is int and value>=0, name+' must be nonnegative integer')
    return value


def close(actual, expected, name):
    if expected is None or type(expected) is bool:
        require(type(actual) is type(expected) and actual==expected, name+' mismatch')
    else:
        number(actual,name,nonnegative=False)
        require(math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-12), name+' numeric mismatch')


def binding(obj, *, exact_keys=True):
    require(isinstance(obj,dict) and {'path','sha256'}<=obj.keys(), 'Missing file binding')
    if exact_keys: require(set(obj)=={'path','sha256'}, 'Unexpected binding fields')
    path=Path(obj['path'])
    require(path.is_absolute(), 'Binding path must be absolute')
    require(sha(path)==obj['sha256'], 'Bound bytes changed: '+str(path))
    if 'bytes' in obj: require(path.stat().st_size==obj['bytes'], 'Bound size changed')
    return path.resolve()


def same_ids(rows, ids, label):
    require([r['sample_id'] for r in rows]==ids, label+' IDs/order/coverage mismatch')
    require(len(ids)==len(set(ids)), label+' duplicate IDs')


def accepted_inputs(specs):
    require(isinstance(specs,list) and specs,'No independently accepted upstream runs')
    result=[]
    for spec in specs:
        directory=Path(spec['path']).resolve()
        require(directory.is_relative_to(PLAN/'runs'),'Upstream run outside current plan')
        require(sha(directory/'report.json')==spec['report_sha256'] and sha(directory/'proof.json')==spec['proof_sha256'],'Upstream report/proof changed')
        report=read(directory/'report.json');proof=read(directory/'proof.json')
        require(report['status']=='passed' and proof['status']=='passed' and proof['report_sha256']==spec['report_sha256'],'Upstream result is not independently accepted')
        require(report['source_sha256']==SOURCE_SHA,'Upstream source differs')
        result.append((directory,report))
    require(len({str(p) for p,r in result})==len(result),'Duplicate upstream runs')
    return result


class DeveloperOnlyReads:
    def __init__(self, label_path):
        self.allowed=Path(label_path).resolve(); self.opens=0
        sys.addaudithook(self.audit)

    def audit(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.name=='test.json' or (p.is_relative_to(DATA_ROOT) and ('labels' in p.parts or p.suffix.lower()=='.csv')):
            raise PermissionError('Verifier forbids full target label sources')
        if p.name=='dev_labels.json':
            require(p.resolve()==self.allowed, 'Verifier attempted another developer GT file')
            self.opens+=1


def inspect_pose(row, *, full=False):
    import numpy as np
    for k in ('h_available','e_available','original_h_rejected','pose_legal','geometry_scoreable'):
        require(type(row[k]) is bool, 'Pose availability must be explicit bool: '+k)
    arrays={'heat_q':(4,), 'heat_t':(3,), 'effi_R':(3,3), 'effi_t':(3,)}
    for k,shape in arrays.items():
        available=row['h_available'] if k.startswith('heat_') else row['e_available']
        value=row[k]
        if available:
            a=np.asarray(value);require(a.shape==shape and np.isfinite(a).all(), 'Invalid pose '+k)
        else: require(value is None, 'Unavailable pose must be explicit None: '+k)
    if row['h_available']:
        require(abs(float(np.linalg.norm(row['heat_q']))-1)<1e-6, 'H quaternion not unit')
    if row['e_available']:
        r=np.asarray(row['effi_R'],dtype=np.float32)
        require(abs(np.linalg.det(r)-1)<1e-6 and np.max(np.abs(r@r.T-np.eye(3)))<=1e-6, 'E rotation fails frozen validity')
    legal=row['h_available'] and row['e_available'] and not row['original_h_rejected']
    require(row['pose_legal']==legal, 'Legality flag inconsistent with poses/rejection')
    reason='legal' if legal else ('pnp_unavailable' if not row['h_available'] else 'pnp_rejected' if row['original_h_rejected'] else 'E_pose_unavailable')
    require(row['legality_reason']==reason, 'Legality reason mismatch')
    if not row['h_available']:
        require(row['original_h_rejected'], 'Missing H cannot be original accepted')
    if row.get('bbox') is not None:
        require(np.asarray(row['bbox']).shape==(4,), 'Bounding-box shape differs')
    if full:
        number(row['entropy'],'binary entropy')
        require(row['entropy']<=math.log(2)+1e-6, 'Binary entropy out of range')
        require(np.asarray(row['heat_peaks']).shape==(11,) and np.asarray(row['all_heat_xy']).shape==(11,2), 'All 11 heatmap points required')
        close(row['confidence'],float(np.quantile(row['heat_peaks'],.25,method='linear')),'confidence')
    return legal


def recompute_metrics(prediction, label, cfg):
    """Independent assembly using only original frozen metric functions."""
    import numpy as np
    from utils.metrics import speed_score,error_orientation,error_translation
    from utils.postprocess import quat2dcm
    qgt=np.asarray(label['quaternion_wxyz'],dtype=np.float32)
    tgt=np.asarray(label['translation_m'],dtype=np.float32)
    require(qgt.shape==(4,) and tgt.shape==(3,) and np.isfinite(qgt).all() and np.isfinite(tgt).all(), 'Bad developer pose GT')
    out={k:None for k in METRICS};out.update(valid=False,original_valid=False)
    if prediction['h_available']:
        q=np.asarray(prediction['heat_q'],dtype=np.float64);t=np.asarray(prediction['heat_t'],dtype=np.float64)
        _,rq,hscore=speed_score(t,q,tgt,qgt,representation='quaternion',applyThreshold=True,
            theta_q=cfg.TEST.SPEED_THRESHOLD_Q,theta_t=cfg.TEST.SPEED_THRESHOLD_T)
        out.update(rotation_deg=float(error_orientation(q,qgt,'quaternion')),heat_rotation_HIL=float(rq),heat_error=float(hscore))
    if prediction['e_available']:
        r=np.asarray(prediction['effi_R'],dtype=np.float32);t=np.asarray(prediction['effi_t'],dtype=np.float32)
        tr,_,escore=speed_score(t,r,tgt,quat2dcm(qgt),representation='rotationmatrix',applyThreshold=True,
            theta_q=cfg.TEST.SPEED_THRESHOLD_Q,theta_t=cfg.TEST.SPEED_THRESHOLD_T)
        out.update(translation_m=float(error_translation(t,tgt)),effi_translation_HIL=float(tr),effi_error=float(escore))
    if prediction['h_available'] and prediction['e_available']:
        mixed=out['heat_rotation_HIL']+out['effi_translation_HIL']
        accepted=not prediction['original_h_rejected']
        out.update(valid=True,error=mixed,original_valid=accepted,original_error=mixed if accepted else None)
    finite_json(out)
    return out


def geometry_scores(row,camera,points):
    import numpy as np
    from utils.postprocess import project_keypoints,quat2dcm
    legal=inspect_pose(row,full=True)
    if legal:
        q=np.asarray(row['heat_q']);t=np.asarray(row['heat_t']);r=np.asarray(row['effi_R'],dtype=np.float32);et=np.asarray(row['effi_t'],dtype=np.float32)
        projected=project_keypoints(q,t,camera['cameraMatrix'],camera['distCoeffs'],points).T
        if np.isfinite(projected).all():
            xy=np.asarray(row['all_heat_xy'])
            scores={'reprojection':float(np.linalg.norm(projected-xy,axis=1).mean()/math.hypot(1920,1200)),
                'rotation_disagreement':float(np.rad2deg(np.arccos(np.clip((np.trace(r@quat2dcm(q).T)-1)/2,-1,1)))),
                'translation_disagreement':float(np.linalg.norm(et-t)/max((np.linalg.norm(et)+np.linalg.norm(t))/2,1e-8))}
            for k,v in scores.items():close(row[k],v,k)
            require(row['geometry_scoreable'] is True and row['geometry_reason']=='scoreable', 'Finite legal geometry omitted')
        else:
            require(row['geometry_scoreable'] is False and row['geometry_reason']=='projection_unavailable', 'Projection failure unrecorded')
    else:
        require(row['geometry_scoreable'] is False and row['geometry_reason']==row['legality_reason'], 'Illegal pose marked geometry-scoreable')
    if not row['geometry_scoreable']:
        for k in ('reprojection','rotation_disagreement','translation_disagreement'):
            require(row[k] is None, 'Unavailable geometric signal must remain None')


def decision_check(row, method, calibration, mask_value, camera, points):
    import numpy as np
    quality=row['quality']; score=None
    if method=='A':
        require(quality is None,'A must not execute scoring'); accepted=True;reason='all'
    elif method=='R':
        require(quality is None and type(mask_value) is bool,'R scoring/mask invalid');accepted=mask_value;reason='random_budget_mask'
    elif method=='V':
        require(quality['score_profile']=='pose','V scoring profile changed');inspect_pose(quality)
        accepted=quality['pose_legal'];reason=quality['legality_reason']
    elif method.startswith('E'):
        require(set(quality)=={'entropy','score_profile','timing_seconds'} and quality['score_profile']=='entropy','E scoring schema changed')
        require(set(quality['timing_seconds'])=={'forward','geometry'},'E timing schema changed')
        for k,v in quality['timing_seconds'].items():number(v,'E timing '+k)
        require(quality['timing_seconds']['geometry']==0,'Entropy-only scorer used geometry')
        number(quality['entropy'],'entropy');require(quality['entropy']<=math.log(2)+1e-6,'Entropy range')
        score=-quality['entropy'];accepted=score>=calibration['thresholds'][method];reason='threshold_pass' if accepted else 'threshold_reject'
    else:
        require(quality['score_profile']=='full','G scoring profile changed');geometry_scores(quality,camera,points)
        if not quality['geometry_scoreable']:
            accepted=False;reason=quality['geometry_reason']
        else:
            ranks=[]
            for i,k in enumerate(('confidence','reprojection','rotation_disagreement','translation_disagreement')):
                ref=np.asarray(calibration['references'][i]);x=quality[k]*(1 if i==0 else -1)
                require(ref.ndim==1 and len(ref)>0 and np.isfinite(ref).all() and np.all(ref[1:]>=ref[:-1]),'Malformed frozen reference')
                ranks.append(float(((ref<x).sum()+(ref<=x).sum())/(2*len(ref))))
            score=min(ranks);accepted=score>=calibration['thresholds'][method];reason='threshold_pass' if accepted else 'threshold_reject'
    require(type(row['accepted']) is bool and row['accepted']==bool(accepted) and row['reason']==reason,'Decision differs from frozen method')
    close(row['score'],score,'gate score')
    if accepted:number(row['loss'],'accepted original loss')
    else:require(row['loss'] is None,'Rejected image contributed adaptation loss')


def inspect_checkpoint(path, task, task_sha, counters, source, model, whitelist, *, final):
    import torch
    # This is our own immutable local runner output; RNG includes NumPy objects.
    ck=torch.load(path,map_location='cpu',weights_only=False)
    require(ck['schema']=='odr003-complete-window-v1' and ck['task_sha256']==task_sha,'Checkpoint task identity')
    for k in ('bindings','code_sha256','run_id','method','domain','stream'):
        require(ck[k]==task[k],'Checkpoint binding mismatch: '+k)
    require(ck['source_sha256']==SOURCE_SHA and ck['final'] is final,'Checkpoint source/finality mismatch')
    require(ck['tail_discarded'] is final,'Final tail rollback declaration missing')
    require(ck['counters']==counters,'Checkpoint counters mismatch')
    require(ck['sampler']=={'next_index':counters['observed'],'stream_binding':task['bindings'].get('stream_manifest')},'Sampler continuation mismatch')
    a=ck['adapter'];require(a['accepted']==counters['used'] and a['updates']==counters['updates'] and a['observed']==counters['observed'] and a['batch']==4,'Adapter checkpoint boundary/count mismatch')
    for name in ('bn_features','bn_mean','bn_mean_sq'):require(a[name]=={},'Uncommitted statistics/cache in checkpoint')
    require(a['modes']=={n:False for n,m in model.named_modules()},'Frozen original A model modes changed')
    require(set(ck['model'])==set(source),'Final state key coverage differs')
    changed=[]
    for k,v in ck['model'].items():
        require(isinstance(v,torch.Tensor) and v.shape==source[k].shape and v.dtype==source[k].dtype,'Checkpoint tensor format mismatch: '+k)
        require(bool(torch.isfinite(v).all()),'Nonfinite checkpoint tensor: '+k)
        if not torch.equal(v,source[k]):
            require(k in whitelist,'Forbidden parameter/buffer update: '+k);changed.append(k)
    loaded=model.load_state_dict(ck['model'],strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys,'Checkpoint strict reload failed')
    require(all(torch.equal(v,ck['model'][k]) for k,v in model.state_dict().items()),'Checkpoint reload state differs')
    require(ck['trainable_parameter_names']==list(whitelist) and ck['scaler'] is None,'Checkpoint trainable whitelist/FP32 scaler differs')
    opt=ck['optimizer'];require(len(opt['param_groups'])==1,'Optimizer group count differs')
    group=opt['param_groups'][0];ids=group['params']
    require(ids==list(range(len(whitelist))),'Optimizer parameter whitelist/order differs')
    require(group['lr']==1e-4 and tuple(group['betas'])==(.9,.999) and group['weight_decay']==0 and group['eps']==1e-8,'Original optimizer hyperparameters changed')
    require(not group.get('amsgrad',False) and not group.get('maximize',False),'Optimizer variant changed')
    require(set(opt['state'])<=set(ids),'Optimizer state contains a forbidden parameter')
    require(bool(opt['state'])==bool(counters['updates']),'Optimizer state/update count mismatch')
    ordered=list(whitelist)
    for i,name in enumerate(ordered):
        if i not in opt['state']:require(torch.equal(ck['model'][name],source[name]),'Parameter without AdamW state changed')
    for i,state in opt['state'].items():
        require(set(state)=={'step','exp_avg','exp_avg_sq'},'Unexpected AdamW state schema')
        require(float(state['step'])==counters['updates'],'AdamW step count mismatch')
        for k in ('exp_avg','exp_avg_sq'):
            v=state[k];require(v.shape==source[ordered[i]].shape and bool(torch.isfinite(v).all()),'Optimizer tensor invalid')
    require(set(ck['rng'])=={'python','numpy','cpu','cuda'} and len(ck['rng']['cuda'])>=1,'Full RNG state missing')
    require(ck['rng']['cpu'].dtype==torch.uint8,'CPU RNG state malformed')
    if counters['updates']==0:require(not changed,'Zero-update source changed')
    return {'strict_reload':True,'changed_parameter_count':len(changed),'changed_parameter_names':changed,'optimizer_steps':counters['updates'],'state_keys':len(ck['model'])}


def verify(task_path, main_exitcode, proof_path):
    require(os.environ.get('STY'),'Use a unique guarded screen for CPU acceptance')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Verifier must be CPU only')
    require(Path(main_exitcode).read_text().strip()=='0','Main command did not successfully exit')
    task=read(task_path);task_sha=sha(task_path);method=task['method'];domain=task['domain']
    require(task['schema']=='odr003-task-v1' and method in METHODS and domain in ('lightbox','sunlamp'),'Wrong task schema/method/domain')
    output=Path(task['output_dir']).resolve()
    require(output.is_relative_to(PLAN/'runs') and output.name==task['run_id'],'Output location mismatch')
    require(not (output/'failure.json').exists(),'Failure artifact exists')
    require(read(output/'task.json')==task,'Run task copy differs')
    require((method=='O' and task['stream']=='offline' and task['seed']==2021) or (method!='O' and task['stream'] in ('2021','2022','2023') and task['seed']==int(task['stream'])),'Unconfirmed stream/seed')
    guard=DeveloperOnlyReads(task['bindings']['developer_labels']['path'])
    paths={k:binding(v) for k,v in task['bindings'].items()}
    require(task['bindings']['source']['sha256']==SOURCE_SHA,'Official source identity differs')
    require(set(task['code_sha256'])==CODE,'Implementation hash coverage')
    for name,expected in task['code_sha256'].items():require(sha(Path(__file__).parent/name)==expected,'Implementation drift: '+name)
    receipt=read(paths['official_receipt']);require(receipt['status']=='passed' and receipt['source_offline_identity_verified'] is True and receipt['strict_model_load_performed'] is True,'Official identity receipt invalid')
    for k in ('source','config','source_core_manifest'):
        require(Path(receipt[k+'_path']).resolve()==paths[k] and receipt[k+'_sha256']==task['bindings'][k]['sha256'],'Source receipt binding mismatch')
    require(sha(receipt['strict_load_report_path'])==receipt['strict_load_report_sha256'],'Official strict proof changed')
    strict_source=read(receipt['strict_load_report_path'])
    require(strict_source['status']=='passed' and strict_source['strict_load'] is True and strict_source['all_loaded_tensors_equal_source'] is True,'Official strict compatibility proof failed')
    core_manifest=read(paths['source_core_manifest']);require(Path(core_manifest['root']).resolve()==CORE,'Frozen original core root differs')
    require(set(core_manifest['files'])=={str(p.relative_to(CORE)) for p in CORE.rglob('*.py')},'Core Python coverage changed')
    for rel,expected in core_manifest['files'].items():require(sha(CORE/rel)==expected,'Original core drift: '+rel)
    for k in ('camera','keypoint'):
        bind='keypoints' if k=='keypoint' else k
        require(Path(core_manifest[k+'_path']).resolve()==paths[bind] and core_manifest[k+'_sha256']==task['bindings'][bind]['sha256'],'Geometry asset binding changed')
    prepared=read(paths['data_prepared']);verified=read(paths['data_verified'])
    require(prepared['status']=='prepared' and verified['status']=='passed' and verified['prepared_proof_sha256']==task['bindings']['data_prepared']['sha256'],'Data boundary proof mismatch')
    from data_boundary import load_image_manifest
    for key,artifact in [('developer_manifest',domain+'/developer'),('developer_labels',domain+'/developer_labels')]+([] if method=='O' else [('stream_manifest',domain+'/stream_'+task['stream'])]):
        expected=prepared['artifacts'][artifact]
        require(paths[key]==Path(expected['path']).resolve() and task['bindings'][key]['sha256']==expected['sha256'],'Frozen data identity mismatch')
    dev=load_image_manifest(paths['developer_manifest']);dev_ids=[x['sample_id'] for x in dev['items']]
    require(len(dev_ids)==512 and all(x.startswith(domain+'/') for x in dev_ids),'Developer denominator/domain mismatch')
    labels=read(paths['developer_labels']);same_ids(labels['items'],dev_ids,'Developer labels')
    require(labels['image_manifest_sha256']==task['bindings']['developer_manifest']['sha256'] and labels['domain']==domain,'Developer label boundary differs')
    image_manifests=[dev]
    stream_ids=[]
    if method!='O':
        stream=load_image_manifest(paths['stream_manifest']);image_manifests.append(stream)
        stream_ids=[x['sample_id'] for x in stream['items']]
        require(len(stream_ids)==1024 and len(set(stream_ids))==1024 and stream['seed']==task['seed'] and all(x.startswith(domain+'/') for x in stream_ids),'Adaptation observations not exactly1024 distinct paired IDs')
    checked_images=set()
    for manifest in image_manifests:
        require(Path(manifest['image_root']).resolve()==DATA_ROOT,'Unexpected image root')
        for item in manifest['items']:
            if item['sample_id'] in checked_images:continue
            p=DATA_ROOT/item['domain']/manifest['image_folder']/item['filename']
            require(sha(p)==item['image_sha256'],'Actual image bytes changed: '+item['sample_id']);checked_images.add(item['sample_id'])
    report=read(output/'report.json')
    require(report['schema']=='odr003-single-task-report-v1' and report['status']=='passed' and report['independent_verification_required'] is True,'Main report schema differs')
    for key in ('run_id','method','domain','stream','seed','bindings','code_sha256'):require(report[key]==task[key],'Main task identity mismatch: '+key)
    require(report['task_sha256']==task_sha and report['source_sha256']==SOURCE_SHA,'Task/source report hashes differ')
    outputs={k:binding(v,exact_keys=False) for k,v in report['outputs'].items()}
    require(set(outputs)=={'decisions','predictions','risk_rows','effective_config'},'Output manifest coverage differs')
    for k,filename in [('decisions','decisions.jsonl'),('predictions','predictions.jsonl'),('risk_rows','risk_rows.jsonl'),('effective_config','effective_config.yaml')]:require(outputs[k]==output/filename,'Unexpected output path')
    sys.path.insert(0,str(CORE))
    import numpy as np
    import torch
    import yaml
    from config import cfg as defaults
    from config.efficientdet import EFFICIENTDET_EXTRAS
    from nets.build import SPNv2
    from utils.utils import load_camera_intrinsics,load_tango_3d_keypoints
    torch.set_num_threads(2)
    cfg=defaults.clone();cfg.defrost();cfg.merge_from_file(str(paths['config']))
    cfg.MODEL.BACKBONE.EXTRA=EFFICIENTDET_EXTRAS[cfg.MODEL.BACKBONE.NAME]
    overrides={'ROOT':str(CORE.parent),'OUTPUT_DIR':str(output),'LOG_DIR':str(output),'EXP_NAME':task['run_id'],'SEED':task['seed'],'AUTO_RESUME':False,'CUDA':True,
        'DATASET.ROOT':str(DATA_ROOT.parent),'DATASET.CAMERA':str(paths['camera']),'DATASET.KEYPOINTS':str(paths['keypoints']),
        'MODEL.PRETRAIN_FILE':str(paths['source']),'MODEL.BACKBONE.PRETRAINED':None,'TEST.MODEL_FILE':str(paths['source']),
        'TRAIN.TRAIN_CSV':domain+'/labels/forbidden.csv','TRAIN.VAL_CSV':domain+'/labels/forbidden.csv','TEST.TEST_CSV':domain+'/labels/forbidden.csv'}
    require(read(output/'config_path_overrides.json')==overrides,'Config overrides not restricted to protocol paths/seed')
    for dotted,v in overrides.items():
        keys=dotted.split('.');node=cfg
        for key in keys[:-1]:node=node[key]
        node[keys[-1]]=v
    require(yaml.safe_load(cfg.dump())==yaml.safe_load(outputs['effective_config'].read_text()),'Effective configuration changed beyond allowed overrides')
    cfg.CUDA=False;cfg.MODEL.PRETRAIN_FILE='';cfg.freeze()
    camera=load_camera_intrinsics(str(paths['camera']));points=load_tango_3d_keypoints(str(paths['keypoints']))
    calibration=None;mask=None
    if method in ('E50','E75','G50','G75'):
        bundle=read(paths['calibration']);require(bundle['status']=='passed' and bundle['source_sha256']==SOURCE_SHA,'Calibration identity')
        require('calibration_proof' in paths,'Independent calibration proof binding missing')
        cp=read(paths['calibration_proof'])
        require(cp['status']=='passed' and cp['bundle_sha256']==task['bindings']['calibration']['sha256'] and cp['source_sha256']==SOURCE_SHA and cp['recomputed'] is True and cp['developer_records']==1024,'Independent calibration proof differs')
        upstream=accepted_inputs(bundle['inputs'])
        require(len(upstream)==2 and {r['domain'] for p,r in upstream}=={'lightbox','sunlamp'} and all(r['method']=='O' for p,r in upstream),'Calibration requires exactly two accepted Offline runs')
        require(bundle['developer_manifest_sha256']=={d:prepared['artifacts'][d+'/developer']['sha256'] for d in ('lightbox','sunlamp')},'Calibration developer domains differ')
        calibration=bundle['calibration'];require(calibration['schema']=='odr003-frozen-shared-calibration-v1' and calibration['ecdf']=='midrank' and calibration['interpolation']=='linear','Calibration algorithm differs')
        require(calibration['signals']==['confidence','reprojection','rotation_disagreement','translation_disagreement'] and calibration['direction']==[1,-1,-1,-1],'Geometry signal/direction changed')
    if method=='R':
        mask=read(paths['random_mask']);g=read(paths['matched_g_report']);selection_report=read(paths['selection_report']);selection=selection_report['selection']
        upstream=accepted_inputs(selection_report['inputs'])
        matching=[r for p,r in upstream if p/'report.json'==paths['matched_g_report']]
        require(len(matching)==1 and matching[0]==g,'Matched G is not one of the independently accepted selection inputs')
        require(selection['run_random_controls'] is True and selection['selected_method'] in ('G50','G75'),'R not selected by exploration')
        require(g['method']==selection['selected_method'] and g['domain']==domain and g['stream']==task['stream'] and g['source_sha256']==SOURCE_SHA,'R matched G differs')
        require(g['bindings']['stream_manifest']==task['bindings']['stream_manifest'],'R matched G input stream changed')
        require(mask['mask_id'] in ('0','1','2') and report['mask']==mask['mask_id'],'Random mask identity')
        require(mask['source_sha256']==SOURCE_SHA and mask['stream_manifest_sha256']==task['bindings']['stream_manifest']['sha256'] and mask['matched_g_report_sha256']==task['bindings']['matched_g_report']['sha256'],'Random mask bindings')
        require(len(mask['mask'])==1024 and all(type(x) is bool for x in mask['mask']) and sum(mask['mask'])==g['counts']['used'] and sum(mask['mask'])%4==0,'R update budget differs')
        require(type(mask.get('mask_seed')) is int and mask['mask_seed']>=0,'Frozen random mask generation seed missing')
        random_mask=np.zeros(1024,dtype=bool);random_mask[np.random.default_rng(mask['mask_seed']).choice(1024,g['counts']['used'],replace=False)]=True
        require(mask['mask']==random_mask.tolist(),'R mask not reproducible from frozen seed')
    decisions=lines(outputs['decisions']);same_ids(decisions,stream_ids,'Decisions')
    accepted=0
    for i,row in enumerate(decisions):
        for k in ('index','observed','accepted_total','updates'):count(row[k],'decision '+k)
        require(row['index']==i and row['domain']==domain and row['stream']==task['stream'] and row['method']==method,'Decision identity/order mismatch')
        decision_check(row,method,calibration,None if mask is None else mask['mask'][i],camera,points)
        accepted+=int(row['accepted'])
        require(row['observed']==i+1 and row['accepted_total']==accepted and row['updates']==accepted//4,'Decision cumulative counters inconsistent')
    counters=dict(observed=len(decisions),accepted=accepted,rejected=len(decisions)-accepted,updates=accepted//4,used=4*(accepted//4),tail=accepted%4)
    for k,v in report['counts'].items():count(v,'reported '+k)
    require(report['counts']==counters,'Reported counts differ from actual decisions')
    if method=='A':require(counters['used']==1024 and counters['tail']==0,'Original A not all accepted')
    if method=='O':require(not any(counters.values()),'Offline O was adapted')
    if method=='R':require(counters['used']==sum(mask['mask']) and counters['tail']==0,'Random tail/budget differs')
    predictions=lines(outputs['predictions']);same_ids(predictions,dev_ids,'Predictions')
    recomputed=[]
    for row,label in zip(predictions,labels['items']):
        require(row['domain']==domain and row['method']==method and row['stream']==task['stream'],'Developer prediction identity')
        require(row['score_profile']=='full','Developer prediction profile')
        geometry_scores(row,camera,points)
        metric=recompute_metrics(row,label,cfg)
        for k in METRICS:close(row[k],metric[k],label['sample_id']+'/'+k)
        require(row['original_rejected'] is row['original_h_rejected'],'Original H rejection alias mismatch')
        base=dict(sample_id=row['sample_id'],domain=domain,stream=task['stream'],method=method)
        if method=='R':require(row['mask']==mask['mask_id'],'Prediction mask differs');base['mask']=mask['mask_id']
        for stream in ('2021','2022','2023') if method=='O' else (task['stream'],):
            recomputed.append({**base,**metric,'stream':stream,'original_rejected':row['original_h_rejected']})
    risk=lines(outputs['risk_rows']);require(risk==recomputed,'Risk rows differ from independently reconstructed developer metrics')
    valid=sum(r['valid'] for r in predictions);original_valid=sum(r['original_valid'] for r in predictions);rejected=sum(r['original_h_rejected'] for r in predictions)
    require(report['developer_count']==512 and report['extended_valid']==valid and report['original_valid']==original_valid and report['original_rejected']==rejected and report['all_samples']==(valid==512),'Report valid/rejected denominators differ')
    profile={'E50':'entropy','E75':'entropy','V':'pose','G50':'full','G75':'full'}.get(method)
    require(report['score_profile']==profile,'Report scoring profile')
    require(report['forward_counts']=={'quality':1024 if profile else 0,'accepted_update':accepted,'developer_evaluation':512},'Forward count contract')
    gt=report['gt_boundary'];require(gt['source_labels_forbidden'] is True and gt['adaptation_developer_labels_forbidden'] is True and gt['forbidden_attempts']==[] and gt['developer_label_open_events']>=2,'GT boundary audit did not pass')
    for key in ('wall_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes'):number(report[key],key)
    require(report['wall_seconds']>0 and report['peak_gpu_reserved_bytes']>=report['peak_gpu_allocated_bytes']>0 and bool(report['gpu_name']),'Missing actual GPU resource record')
    required_timings={'preflight_and_load_seconds','adaptation_seconds','image_read_seconds','scoring_seconds','scoring_forward_seconds','scoring_geometry_seconds','update_seconds','checkpoint_seconds','evaluation_seconds','evaluation_forward_seconds','evaluation_geometry_seconds'}
    require(set(report['timings'])==required_timings,'Timing coverage incomplete')
    for k,v in report['timings'].items():number(v,k)
    require(report['timings']['evaluation_seconds']>0,'Evaluation timing absent')
    require(report['scope']=='developer_exploration_only_no_reserved_or_full_domain_scores','Unconfirmed evaluation scope')
    require(report['source_strict_load'] is True and report['final_strict_reload'] is True,'Strict-load main claims missing')
    require(report['evaluation_model_optimizer_unchanged'] is True,'Evaluation state preservation audit missing')
    cached=strict_source['cached_imagenet_build_dependency']
    require(sha(cached['path'])==cached['sha256'],'Cached architecture-build dependency changed')
    model=SPNv2(cfg).cpu().eval();source=torch.load(paths['source'],map_location='cpu',weights_only=True)
    model.load_state_dict(source,strict=True)
    allowed=set()
    for name,module in model.backbone.named_modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            allowed.update('backbone.'+name+'.'+k for k,_ in module.named_parameters(recurse=False))
    whitelist=[k for k,_ in model.named_parameters() if k in allowed]
    require(report['trainable_parameter_names']==whitelist,'Main trainable whitelist differs from actual original-A model')
    boundary_updates=list(range(64,counters['updates']+1,64))
    require([x['updates'] for x in report['checkpoints']]==boundary_updates,'Periodic complete-window checkpoint coverage differs')
    for item in report['checkpoints']:
        p=binding(item,exact_keys=False);require(p==output/f"checkpoint_u{item['updates']:04d}.pt",'Boundary checkpoint location')
        matching=next(r for r in decisions if r['accepted_total']==4*item['updates'] and r['accepted'])
        c=dict(observed=matching['observed'],accepted=matching['accepted_total'],rejected=matching['observed']-matching['accepted_total'],updates=item['updates'],used=4*item['updates'],tail=0)
        require(item['observed']==c['observed'],'Boundary checkpoint sampler index')
        inspect_checkpoint(p,task,task_sha,c,source,model,whitelist,final=False)
    final_path=binding(report['final_checkpoint'],exact_keys=False);require(final_path==output/'final_checkpoint.pt','Final checkpoint location')
    require(report['final_checkpoint']['observed']==counters['observed'] and report['final_checkpoint']['updates']==counters['updates'],'Final checkpoint manifest counters')
    final=inspect_checkpoint(final_path,task,task_sha,counters,source,model,whitelist,final=True)
    require(not torch.cuda.is_initialized(),'CPU verifier unexpectedly initialized CUDA')
    progress=read(output/'progress.json');require(progress['status']=='passed' and progress['phase']=='complete' and progress['observed']==counters['observed'] and progress['evaluated']==512 and progress['updates']==counters['updates'],'Final progress evidence inconsistent')
    # Detect mutable source/config/code/evidence changing during independent readback.
    for b in task['bindings'].values():binding(b)
    for name,expected in task['code_sha256'].items():require(sha(Path(__file__).parent/name)==expected,'Implementation changed during acceptance')
    result=dict(schema='odr003-independent-single-task-proof-v1',status='passed',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task_path=str(Path(task_path).resolve()),task_sha256=task_sha,run_id=task['run_id'],method=method,domain=domain,stream=task['stream'],
        report_path=str(output/'report.json'),report_sha256=sha(output/'report.json'),main_exitcode_path=str(Path(main_exitcode).resolve()),main_exitcode=0,
        bindings=task['bindings'],code_sha256=task['code_sha256'],outputs=report['outputs'],final_checkpoint=report['final_checkpoint'],
        counts=counters,developer_count=512,extended_valid=valid,original_valid=original_valid,original_rejected=rejected,
        independently_recomputed_metric_rows=len(predictions),risk_rows=len(risk),actual_image_files_hashed=len(checked_images),
        checkpoint_verification=final,only_original_A_backbone_BN_affine_may_change=True,all_buffers_equal_source=True,
        developer_only_labels_read=True,reserved_GT_read=False,full_source_GT_read=False,verifier_forward_passes=0,cuda_initialized=False,
        source_offline_identity_verified=True,source_sha256=SOURCE_SHA,
        verifier_path=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),
        limitations=['Saved predictions and serialized state are independently checked; this does not replay model inference or prove causation.',
                     'Formal efficacy and reserved/full-domain evaluation remain outside this exploratory proof.'])
    require(not Path(proof_path).exists(),'Do not overwrite previous independent proof')
    with Path(proof_path).open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'passed','run_id':task['run_id'],'proof':str(proof_path),'counts':counters},sort_keys=True))


def contract_self_test():
    """Small synthetic fail-closed contracts; never a scientific result."""
    import tempfile
    import unittest
    class Contracts(unittest.TestCase):
        def test_reject_nonfinite_json(self):
            for value in ('NaN','Infinity','-Infinity'):
                with self.assertRaises(ValueError):json.loads('{"x":'+value+'}',parse_constant=reject_constant)
            with self.assertRaises(ValueError):finite_json({'nested':[float('nan')]})
        def test_bound_file_mutation_detected(self):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'saved.json';p.write_text('{"x":1}')
                original={'path':str(p.resolve()),'sha256':sha(p)}
                binding(original);p.write_text('{"x":2}')
                with self.assertRaises(ValueError):binding(original)
        def test_missing_duplicate_and_reordered_ids_rejected(self):
            for rows in ([{'sample_id':'a'}],[{'sample_id':'a'},{'sample_id':'a'}],[{'sample_id':'b'},{'sample_id':'a'}]):
                with self.assertRaises(ValueError):same_ids(rows,['a','b'],'developer')
        def test_gate_tampering_and_rejected_gradient_rejected(self):
            good=dict(quality=None,accepted=True,reason='all',score=None,loss=.1)
            decision_check(good,'A',None,None,None,None)
            with self.assertRaises(ValueError):decision_check(dict(good,accepted=False),'A',None,None,None,None)
            rejected=dict(quality=None,accepted=False,reason='random_budget_mask',score=None,loss=None)
            decision_check(rejected,'R',None,False,None,None)
            with self.assertRaises(ValueError):decision_check(dict(rejected,loss=.1),'R',None,False,None,None)
        def test_missing_H_cannot_be_replaced_by_E(self):
            row=dict(h_available=False,e_available=False,original_h_rejected=True,pose_legal=False,geometry_scoreable=False,
                heat_q=None,heat_t=None,effi_R=None,effi_t=None,legality_reason='pnp_unavailable')
            inspect_pose(row)
            with self.assertRaises(ValueError):inspect_pose(dict(row,heat_q=[1.,0.,0.,0.]))
        def test_bool_is_not_count(self):
            with self.assertRaises(ValueError):count(True,'observed')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    require(result.wasSuccessful(),'Synthetic verifier contract failed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-json');parser.add_argument('--main-exitcode');parser.add_argument('--proof');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        require(os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use a unique guarded CPU screen')
        require(not any((args.task_json,args.main_exitcode,args.proof)),'Self-test is separate from actual acceptance')
        contract_self_test();return
    require(all((args.task_json,args.main_exitcode,args.proof)),'--task-json, --main-exitcode and --proof are required')
    try:verify(args.task_json,args.main_exitcode,args.proof)
    except BaseException as error:
        failure=Path(args.proof).with_name(Path(args.proof).name+'.failure.json')
        if not failure.exists():
            with failure.open('x',encoding='utf-8') as f:json.dump({'status':'failed','error':repr(error),'traceback':traceback.format_exc()},f,indent=2)
        raise


if __name__=='__main__':main()
