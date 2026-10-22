"""Frozen-source empirical gates; no labels, future-stream ranks or fitting online."""
from __future__ import annotations
import numpy as np

SIGNALS = ('confidence', 'reprojection', 'rotation_disagreement', 'translation_disagreement')
DIRECTIONS = np.asarray([1., -1., -1., -1.])


def rank(values, reference):
    """Midrank ECDF: ties share (count(<x)+count(<=x))/(2*n)."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all():
        raise ValueError('Invalid reference ECDF')
    ref = np.sort(ref)
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite scoring input')
    return (np.searchsorted(ref, x, side='left') + np.searchsorted(ref, x, side='right')) / (2. * len(ref))


def fit_calibration(records):
    """Fit one pooled two-domain mapping, on source developer predictions only."""
    if {r['domain'] for r in records} != {'lightbox', 'sunlamp'}:
        raise ValueError('Both domains required for shared calibration')
    if len({(r['domain'], r['sample_id']) for r in records}) != len(records):
        raise ValueError('Duplicate calibration IDs')
    legal = [r for r in records if r['geometry_scoreable']]
    if not legal:
        raise ValueError('No scoreable source developer predictions')
    raw = np.asarray([[r[k] for k in SIGNALS] for r in legal], dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('Invalid legal geometry')
    refs = np.sort(raw * DIRECTIONS, axis=0)
    combined = np.stack([rank(raw[:, i] * DIRECTIONS[i], refs[:, i]) for i in range(4)], axis=1).min(axis=1)
    entropy = np.asarray([r['entropy'] for r in records if r['entropy'] is not None], dtype=np.float64)
    if not len(entropy) or not np.isfinite(entropy).all():
        raise ValueError('No finite entropy reference')
    result = dict(schema='odr003-frozen-shared-calibration-v1', signals=list(SIGNALS),
                  direction=DIRECTIONS.tolist(), references=refs.T.tolist(),
                  ecdf='midrank', interpolation='linear', acceptance='score >= threshold',
                  geometry_n=len(legal), entropy_n=len(entropy), thresholds={})
    for acceptance in (50, 75):
        # Quantile of the combined score, not a threshold on each ECDF directly.
        result['thresholds']['G'+str(acceptance)] = float(np.quantile(combined, 1-acceptance/100, method='linear'))
        result['thresholds']['E'+str(acceptance)] = float(np.quantile(-entropy, 1-acceptance/100, method='linear'))
    return result


def decide(method, record, calibration=None, random_accept=None):
    if method in ('A', 'ALL'):
        return True, 'all', None
    if method == 'R':
        if type(random_accept) is not bool:
            raise ValueError('R requires a frozen boolean mask')
        return random_accept, 'random_budget_mask', None
    if method == 'V':
        return bool(record['pose_legal']), record['legality_reason'], None
    if method in ('E50', 'E75'):
        if record['entropy'] is None:
            return False, 'entropy_unavailable', None
        score = -float(record['entropy'])
    elif method in ('G50', 'G75'):
        if not record['geometry_scoreable']:
            return False, record['geometry_reason'], None
        values = [float(rank(float(record[k])*DIRECTIONS[i], calibration['references'][i])) for i,k in enumerate(SIGNALS)]
        score = min(values)
    else:
        raise ValueError('Unknown method '+method)
    if not np.isfinite(score):
        raise FloatingPointError('Nonfinite gate score')
    ok = score >= calibration['thresholds'][method]
    return bool(ok), 'threshold_pass' if ok else 'threshold_reject', score


def random_budget_mask(n, accepted_count, seed, batch=4):
    if not 0 <= accepted_count <= n:
        raise ValueError('Invalid accepted count')
    used = batch * (accepted_count // batch)
    mask = np.zeros(n, dtype=bool)
    mask[np.random.default_rng(seed).choice(n, used, replace=False)] = True
    return mask.tolist()
