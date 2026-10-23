"""Round-two wiring around frozen A/G50 functions; no CLI or launch side effects.

Imports of the original prediction/core are deliberately delayed. Callers must
verify their SHA bindings and pass the shared-host gate before constructing the
real model. This module neither changes PnP nor imports developer labels.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def import_bound_module(name, binding):
    """Fail if a previously loaded module shadows the immutable source file."""
    path = Path(binding['path']).resolve(strict=True)
    if path.name != name + '.py' or sha256(path) != binding['sha256']:
        raise ValueError('Frozen module identity changed: ' + name)
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError('A different module is already loaded: ' + name)
        return module
    sys.path.insert(0, str(path.parent))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != path:
        raise ValueError('Frozen import resolved to a different file: ' + name)
    return module


class HEGeometryView:
    """Use the original full geometry decoder, evaluating only existing H/E.

    `prediction.extract(..., profile='pose')` returns before the geometry risk
    signals. A read-only view selects H/E for the original full extractor while
    leaving the real model's EHS heads, loss and head_names unchanged.
    """
    def __init__(self, model):
        names = list(model.head_names)
        if names != ['heatmap', 'efficientpose', 'segmentation']:
            raise ValueError('Protection requires the frozen complete EHS model')
        self._model = model
        self.head_names = ('heatmap', 'efficientpose')
        self.heads = tuple(model.heads[names.index(name)] for name in self.head_names)

    def __getattr__(self, name):
        return getattr(self._model, name)


class FrozenScorers:
    def __init__(self, *, model, adapter, cfg, camera, points, calibration,
                 prediction_module, gate_module):
        self.model, self.adapter = model, adapter
        self.cfg, self.camera, self.points = cfg, camera, points
        self.calibration = copy.deepcopy(calibration)
        self.prediction, self.gates = prediction_module, gate_module
        self.protection_view = HEGeometryView(model)

    def arrival(self, image):
        return self.prediction.extract(self.model, self.adapter, image, self.cfg,
                                       self.camera, self.points, profile='full')

    def protection(self, image):
        from guard_pools import decode_arrival_fragility
        result = self.prediction.extract(self.protection_view, self.adapter, image,
                                        self.cfg, self.camera, self.points, profile='full')
        result['score_profile'] = 'protection_HE_geometry_original_decoder'
        decoded = decode_arrival_fragility(result.get('heat_peaks'),
                                           threshold=float(self.cfg.TEST.HEATMAP_THRESHOLD))
        result['pnp_selection_diagnostic'] = decoded
        result['used_indices'] = decoded['used_indices']
        result['used_count'] = decoded['used_count']
        return result

    def gate(self, arrival):
        return self.gates.decide('G50', arrival, self.calibration)


class DeterministicImageProvider:
    """Rebuild observed images without changing any learning RNG or GT boundary.

    The baseline inference preprocessing is deterministic. Unexpected randomness
    is rejected, not silently seeded into a new preprocessing protocol. The LRU
    is a performance cache only: every returned tensor is cloned, and scientific
    checkpoints bind the reconstructed tensor digests rather than cache contents.
    """
    SCHEMA = 'odr003-round2-image-provider-v1'

    def __init__(self, dataset, *, preprocess_sha256, rng_state, restore_rng,
                 equal_tree, cache_capacity=16, manifest_file_sha256=None):
        if len(preprocess_sha256) != 64 or cache_capacity < 1:
            raise ValueError('Invalid preprocessing identity/cache bound')
        items = dataset.manifest['items']
        ids = [r['sample_id'] for r in items]
        if len(ids) != len(set(ids)):
            raise ValueError('Observation manifest contains duplicate images')
        self.dataset, self.indices = dataset, {sid: i for i, sid in enumerate(ids)}
        self.preprocess_sha256 = preprocess_sha256
        self.manifest_semantic_sha256 = canonical_sha(dataset.manifest)
        self.manifest_sha256 = manifest_file_sha256 or self.manifest_semantic_sha256
        if len(self.manifest_sha256) != 64:
            raise ValueError('Invalid manifest file identity')
        self.rng_state, self.restore_rng, self.equal_tree = rng_state, restore_rng, equal_tree
        self.capacity = cache_capacity
        self.cache, self.tensor_sha256 = OrderedDict(), {}
        self.materializations = 0  # Physical work; deliberately not rewound.

    @staticmethod
    def tensor_digest(tensor):
        a = tensor.detach().cpu().contiguous()
        description = json.dumps({'shape': list(a.shape), 'dtype': str(a.dtype)}, sort_keys=True)
        return hashlib.sha256(description.encode() + b'\0' + a.numpy().tobytes()).hexdigest()

    def __call__(self, sample_id):
        if sample_id not in self.indices:
            raise ValueError('Image is outside the bound observation stream')
        if sample_id in self.cache:
            image = self.cache.pop(sample_id)
            self.cache[sample_id] = image
            return image.clone()
        before = self.rng_state()
        try:
            sample = self.dataset[self.indices[sample_id]]
            after = self.rng_state()
            if not self.equal_tree(before, after):
                raise RuntimeError('Frozen inference preprocessing unexpectedly consumed RNG')
            if sample['sample_id'] != sample_id or set(sample) - {'image', 'sample_id', 'domain', 'filename'}:
                raise ValueError('Input identity/label-free schema changed')
            image = sample['image'].unsqueeze(0).detach().cpu().clone()
        finally:
            self.restore_rng(before)
        digest = self.tensor_digest(image)
        if sample_id in self.tensor_sha256 and self.tensor_sha256[sample_id] != digest:
            raise ValueError('Reconstructed protection image tensor changed')
        self.tensor_sha256[sample_id] = digest
        self.materializations += 1
        self.cache[sample_id] = image
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return image.clone()

    def state_dict(self):
        return dict(schema=self.SCHEMA, manifest_sha256=self.manifest_sha256,
                    manifest_semantic_sha256=self.manifest_semantic_sha256,
                    preprocess_sha256=self.preprocess_sha256,
                    tensor_sha256=dict(self.tensor_sha256),
                    data_rng_policy='deterministic_preprocessing_no_RNG_consumption')

    def load_state_dict(self, state):
        if (state.get('schema') != self.SCHEMA or state.get('manifest_sha256') != self.manifest_sha256
                or state.get('manifest_semantic_sha256') != self.manifest_semantic_sha256
                or state.get('preprocess_sha256') != self.preprocess_sha256
                or state.get('data_rng_policy') != 'deterministic_preprocessing_no_RNG_consumption'
                or not set(state['tensor_sha256']) <= self.indices.keys()
                or any(len(x) != 64 for x in state['tensor_sha256'].values())):
            raise ValueError('Image reconstruction checkpoint identity changed')
        self.tensor_sha256 = dict(state['tensor_sha256'])
        self.cache.clear()


def stream_records(dataset, preprocess_sha256, *, limit):
    items = dataset.manifest['items']
    if len(items) != 1024 or limit not in (128, 1024):
        raise ValueError('Only the frozen 128-prefix or complete N1024 stream is authorized')
    return [dict(sample_id=r['sample_id'], image_sha256=r['image_sha256'],
                 preprocess_sha256=preprocess_sha256) for r in items[:limit]]
