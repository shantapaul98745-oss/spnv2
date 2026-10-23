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

    The original inference transform may draw from global RNGs even when a
    probability-one operation leaves the pixels deterministic. Its arithmetic
    and draws are preserved inside an independent data RNG stream. Every first
    materialization advances that stream once; protection/cache reconstruction
    replays that image's saved pre-transform RNG instead. Global model-compute
    RNGs are restored in every case, including failures. Complete checkpoints
    retain both the stream position and each observed image's replay states.

    The LRU is only a performance cache: returned tensors are cloned, and replay
    must reproduce both its tensor digest and post-transform RNG. This adapter
    handles the frozen transform's global Python/NumPy/Torch CPU/CUDA RNGs; a
    transform with a separate private generator needs an explicit state adapter.
    """
    SCHEMA = 'odr003-round2-image-provider-v2'
    RNG_POLICY = 'isolated_data_rng_first_materialization_and_per_image_replay_v1'

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
        self.initial_data_rng = copy.deepcopy(self.rng_state())
        self.data_rng = copy.deepcopy(self.initial_data_rng)
        self.rng_replay = {}
        self.materialization_order = []
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
        before = copy.deepcopy(self.rng_state())
        replay = self.rng_replay.get(sample_id)
        data_before = copy.deepcopy(replay['before'] if replay is not None else self.data_rng)
        try:
            self.restore_rng(data_before)
            sample = self.dataset[self.indices[sample_id]]
            data_after = copy.deepcopy(self.rng_state())
            if sample['sample_id'] != sample_id or set(sample) - {'image', 'sample_id', 'domain', 'filename'}:
                raise ValueError('Input identity/label-free schema changed')
            image = sample['image'].unsqueeze(0).detach().cpu().clone()
        finally:
            self.restore_rng(before)
        digest = self.tensor_digest(image)
        if replay is not None and not self.equal_tree(replay['after'], data_after):
            raise ValueError('Reconstructed preprocessing RNG consumption changed')
        if sample_id in self.tensor_sha256 and self.tensor_sha256[sample_id] != digest:
            raise ValueError('Reconstructed protection image tensor changed')
        if replay is None:
            self.rng_replay[sample_id] = dict(before=data_before, after=data_after)
            self.materialization_order.append(sample_id)
            self.data_rng = copy.deepcopy(data_after)
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
                    data_rng_policy=self.RNG_POLICY,
                    initial_data_rng=copy.deepcopy(self.initial_data_rng),
                    data_rng=copy.deepcopy(self.data_rng),
                    rng_replay=copy.deepcopy(self.rng_replay),
                    materialization_order=list(self.materialization_order))

    def load_state_dict(self, state):
        if (state.get('schema') != self.SCHEMA or state.get('manifest_sha256') != self.manifest_sha256
                or state.get('manifest_semantic_sha256') != self.manifest_semantic_sha256
                or state.get('preprocess_sha256') != self.preprocess_sha256
                or state.get('data_rng_policy') != self.RNG_POLICY
                or not isinstance(state.get('tensor_sha256'), dict)
                or not set(state['tensor_sha256']) <= self.indices.keys()
                or any(not isinstance(x, str) or len(x) != 64
                       or any(c not in '0123456789abcdef' for c in x)
                       for x in state['tensor_sha256'].values())
                or not isinstance(state.get('rng_replay'), dict)
                or set(state['rng_replay']) != set(state['tensor_sha256'])
                or not isinstance(state.get('materialization_order'), list)
                or len(state['materialization_order']) != len(state['tensor_sha256'])
                or set(state['materialization_order']) != set(state['tensor_sha256'])):
            raise ValueError('Image reconstruction checkpoint identity changed')
        # Validate the chronological first-read RNG chain, not dict key order.
        # restore_rng validates actual Python/NumPy/Torch state structures while
        # the finally clause prevents a malformed checkpoint changing globals.
        before = copy.deepcopy(self.rng_state())
        try:
            current = copy.deepcopy(state['initial_data_rng'])
            self.restore_rng(current)
            if not self.equal_tree(current, self.rng_state()):
                raise ValueError('Initial data RNG state did not round-trip')
            for sample_id in state['materialization_order']:
                replay = state['rng_replay'][sample_id]
                if (not isinstance(replay, dict) or set(replay) != {'before', 'after'}
                        or not self.equal_tree(current, replay['before'])):
                    raise ValueError('Image preprocessing RNG replay chain changed')
                self.restore_rng(replay['after'])
                if not self.equal_tree(replay['after'], self.rng_state()):
                    raise ValueError('Replayed data RNG state did not round-trip')
                current = replay['after']
            if not self.equal_tree(current, state['data_rng']):
                raise ValueError('Current data RNG position changed')
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError('Invalid preprocessing RNG checkpoint state') from exc
        finally:
            self.restore_rng(before)
        self.tensor_sha256 = dict(state['tensor_sha256'])
        self.initial_data_rng = copy.deepcopy(state['initial_data_rng'])
        self.data_rng = copy.deepcopy(state['data_rng'])
        self.rng_replay = copy.deepcopy(state['rng_replay'])
        self.materialization_order = list(state['materialization_order'])
        self.cache.clear()


def convert_legacy_empty_state(old_state, *, initial_data_rng):
    """Explicit audited migration for a proven zero-progress v1 checkpoint.

    This pure helper never loads a checkpoint or claims the whole run is empty.
    Its caller must independently prove zero observations/candidates/updates and
    zero model forwards, preserve the failed version, and take initial_data_rng
    from that checkpoint's unchanged learning RNG. Normal load_state_dict keeps
    rejecting v1. The v2 provider performs full RNG-structure validation on load.
    """
    expected = {'schema', 'manifest_sha256', 'manifest_semantic_sha256',
                'preprocess_sha256', 'tensor_sha256', 'data_rng_policy'}
    if (not isinstance(old_state, dict) or set(old_state) != expected
            or old_state['schema'] != 'odr003-round2-image-provider-v1'
            or old_state['data_rng_policy'] != 'deterministic_preprocessing_no_RNG_consumption'
            or not isinstance(old_state['tensor_sha256'], dict) or old_state['tensor_sha256']
            or not isinstance(initial_data_rng, dict)
            or not {'python', 'numpy', 'cpu', 'cuda'} <= initial_data_rng.keys()):
        raise ValueError('Only a proven empty legacy provider can be explicitly migrated')
    for key in ('manifest_sha256', 'manifest_semantic_sha256', 'preprocess_sha256'):
        value = old_state[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Legacy image provider identity is malformed')
    state = copy.deepcopy(old_state)
    state.update(schema=DeterministicImageProvider.SCHEMA,
                 data_rng_policy=DeterministicImageProvider.RNG_POLICY,
                 initial_data_rng=copy.deepcopy(initial_data_rng),
                 data_rng=copy.deepcopy(initial_data_rng),
                 rng_replay={}, materialization_order=[])
    return state


def stream_records(dataset, preprocess_sha256, *, limit):
    items = dataset.manifest['items']
    if len(items) != 1024 or limit not in (128, 1024):
        raise ValueError('Only the frozen 128-prefix or complete N1024 stream is authorized')
    return [dict(sample_id=r['sample_id'], image_sha256=r['image_sha256'],
                 preprocess_sha256=preprocess_sha256) for r in items[:limit]]
