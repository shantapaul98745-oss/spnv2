"""Label-free original postprocessing plus explicit pre-rejection diagnostics.

Import only after adding the frozen original-A core to sys.path. GT-dependent
developer evaluation is intentionally a different function and call boundary.
"""
import math
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.postprocess import solve_pose_from_heatmaps, rot_6d_to_matrix, quat2dcm, project_keypoints
from utils.metrics import error_orientation, error_translation, speed_score
from state_adapter import isolated_scoring


def warmup_bbox_decoder(model):
    """Prime the original pure scripted operator without observing any image."""
    from utils.postprocess import raw_output_to_bbox
    head = model.heads[model.head_names.index('efficientpose')]
    dummy = torch.zeros((1,head.anchors.shape[0],4),device=head.anchors.device,dtype=torch.float32)
    last = None
    for i in range(6):
        current = raw_output_to_bbox(dummy,head.input_size,head.anchors)
        if i >= 2 and not torch.equal(current,last):
            raise RuntimeError('Original bbox decoder warmup did not stabilize')
        last = current
    return dict(operator='original raw_output_to_bbox',calls=6,synthetic_inputs=True,
                model_forward_calls=0,observations=0,optimizer_updates=0)


def finite_array(x, shape):
    return x is not None and np.asarray(x).shape == shape and np.isfinite(x).all()


def rotation_valid(x):
    return finite_array(x, (3,3)) and abs(np.linalg.det(x)-1) < 1e-6 and np.max(np.abs(x @ x.T-np.eye(3))) <= 1e-6


def require_finite_heads(outputs):
    for name, values in outputs.items():
        values = values if isinstance(values, (tuple, list)) else [values]
        for i, value in enumerate(values):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Nonfinite raw head output {name}[{i}]')


def extract(model, adapter, image, cfg, camera, points, profile='full'):
    """One isolated backbone pass and all EHS heads, with no GT input."""
    with isolated_scoring(model, adapter.bn_features):
        if torch.cuda.is_available() and str(adapter.device).startswith('cuda'):
            torch.cuda.synchronize(adapter.device)
        started = time.perf_counter()
        features = model.backbone(image.to(adapter.device, non_blocking=True))
        if profile not in ('full', 'pose', 'entropy'):
            raise ValueError('Unknown scoring profile')
        selected = {'segmentation'} if profile == 'entropy' else {'heatmap','efficientpose'} if profile == 'pose' else set(model.head_names)
        out = {name:head(features) for name,head in zip(model.head_names, model.heads) if name in selected}
        require_finite_heads(out)
        if torch.cuda.is_available() and str(adapter.device).startswith('cuda'):
            torch.cuda.synchronize(adapter.device)
        forward_seconds = time.perf_counter() - started
        entropy = None
        if 'segmentation' in out:
            logits = out['segmentation']
            # Full binary entropy only selects; adapter still uses original loss.
            ent = -(torch.sigmoid(logits)*F.logsigmoid(logits) + torch.sigmoid(-logits)*F.logsigmoid(-logits)).mean()
            if not bool(torch.isfinite(ent)):
                raise FloatingPointError('Nonfinite binary entropy')
            entropy = float(ent.cpu())
        if profile == 'entropy':
            return dict(entropy=entropy, score_profile=profile,
                        timing_seconds=dict(forward=forward_seconds, geometry=0.))
        geometry_started = time.perf_counter()
        hm = out['heatmap'].squeeze(0).cpu()
        cls, bbox, rot, trans = out['efficientpose']
        idx = torch.argmax(cls, dim=1)
        er = rot_6d_to_matrix(rot[0,idx,:]).squeeze().cpu().numpy()
        et = trans[0,idx].squeeze().cpu().numpy()
        eb = bbox[0,idx].squeeze().cpu().numpy()
        q = ht = None; rejected = True; solve_error = None
        if bool(torch.isfinite(hm).all()):
            try:
                _, q, ht, rejected = solve_pose_from_heatmaps(hm, cfg.DATASET.IMAGE_SIZE,
                    cfg.TEST.HEATMAP_THRESHOLD, camera, points)
            except cv2.error as error:
                solve_error = type(error).__name__
        h_finite = finite_array(q,(4,)) and finite_array(ht,(3,)) and np.linalg.norm(q) > 1e-12
        e_finite = rotation_valid(er) and finite_array(et,(3,))
        legal = bool(h_finite and e_finite and not rejected)
        reason = 'legal' if legal else ('pnp_unavailable' if not h_finite else 'pnp_rejected' if rejected else 'E_pose_unavailable')
        record = dict(entropy=entropy, pose_legal=legal, legality_reason=reason,
                      original_h_rejected=bool(rejected), h_available=bool(h_finite), e_available=bool(e_finite),
                      geometry_scoreable=False, geometry_reason=reason,
                      heat_q=q.tolist() if h_finite else None, heat_t=ht.tolist() if h_finite else None,
                      effi_R=er.tolist() if e_finite else None, effi_t=et.tolist() if e_finite else None,
                      bbox=eb.tolist() if np.isfinite(eb).all() else None, solve_error=solve_error,
                      confidence=None, reprojection=None, rotation_disagreement=None, translation_disagreement=None)
        record['score_profile'] = profile
        if profile == 'pose':
            record['timing_seconds'] = dict(forward=forward_seconds, geometry=time.perf_counter()-geometry_started)
            return record
        peaks, argmax = hm.reshape(hm.shape[0], -1).max(-1)
        if bool(torch.isfinite(peaks).all()):
            record['heat_peaks'] = peaks.tolist()
            record['confidence'] = float(np.quantile(peaks.numpy(), .25, method='linear'))
            width, height = cfg.DATASET.IMAGE_SIZE
            h, w = hm.shape[-2:]
            xy = np.stack([(argmax.numpy() % w) / w * width, (argmax.numpy() // w) / h * height], axis=-1)
            record['all_heat_xy'] = xy.tolist()
        else:
            xy = None; record['heat_peaks'] = None; record['all_heat_xy'] = None
        if legal and xy is not None:
            # All eleven predicted keypoints, not only PnP-selected points.
            projected = project_keypoints(q, ht, camera['cameraMatrix'], camera['distCoeffs'], points).T
            if np.isfinite(projected).all():
                record['reprojection'] = float(np.linalg.norm(projected-xy, axis=1).mean() / math.hypot(width,height))
                hr = quat2dcm(q)
                record['rotation_disagreement'] = float(np.rad2deg(np.arccos(np.clip((np.trace(er @ hr.T)-1)/2,-1,1))))
                record['translation_disagreement'] = float(np.linalg.norm(et-ht) / max((np.linalg.norm(et)+np.linalg.norm(ht))/2,1e-8))
                record['geometry_scoreable'] = all(math.isfinite(record[k]) for k in ('confidence','reprojection','rotation_disagreement','translation_disagreement'))
                record['geometry_reason'] = 'scoreable' if record['geometry_scoreable'] else 'nonfinite_geometry'
            else:
                record['geometry_reason'] = 'projection_unavailable'
        record['timing_seconds'] = dict(forward=forward_seconds, geometry=time.perf_counter()-geometry_started)
        return record


def developer_metrics(prediction, developer_label, cfg):
    """Called only after adaptation; the caller must enforce developer ID membership.

    Float32 GT and unchanged original speed_score operations preserve HIL threshold
    behavior. Extension uses finite H before geometric rejection, never E for H.
    """
    q_gt = np.asarray(developer_label['quaternion'], dtype=np.float32)
    t_gt = np.asarray(developer_label['translation'], dtype=np.float32)
    r_gt = quat2dcm(q_gt)
    out = dict(valid=False, error=None, original_valid=False, original_error=None,
               rotation_deg=None, translation_m=None, heat_rotation_HIL=None,
               effi_translation_HIL=None, heat_error=None, effi_error=None)
    if prediction['h_available']:
        q = np.asarray(prediction['heat_q']); t = np.asarray(prediction['heat_t'])
        st, sr, s = speed_score(t, q, t_gt, q_gt, representation='quaternion', applyThreshold=True,
                              theta_q=cfg.TEST.SPEED_THRESHOLD_Q, theta_t=cfg.TEST.SPEED_THRESHOLD_T)
        out.update(rotation_deg=float(error_orientation(q,q_gt,'quaternion')), heat_rotation_HIL=float(sr), heat_error=float(s))
    if prediction['e_available']:
        r = np.asarray(prediction['effi_R'], dtype=np.float32); t = np.asarray(prediction['effi_t'], dtype=np.float32)
        st, sr, s = speed_score(t, r, t_gt, r_gt, representation='rotationmatrix', applyThreshold=True,
                              theta_q=cfg.TEST.SPEED_THRESHOLD_Q, theta_t=cfg.TEST.SPEED_THRESHOLD_T)
        out.update(translation_m=float(error_translation(t,t_gt)), effi_translation_HIL=float(st), effi_error=float(s))
    if out['heat_rotation_HIL'] is not None and out['effi_translation_HIL'] is not None:
        out['error'] = out['heat_rotation_HIL'] + out['effi_translation_HIL']
        out['valid'] = True
        out['original_valid'] = not prediction['original_h_rejected']
        out['original_error'] = out['error'] if out['original_valid'] else None
    if any(isinstance(v,float) and not math.isfinite(v) for v in out.values()):
        raise FloatingPointError('Nonfinite developer metric')
    return out
