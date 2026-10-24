"""Independent finite local-CPU science contract for a saved-store candidate.

Runs four 36-image *synthetic* transactions: old/candidate, continuous/recovered.
This does not import SPNv2, real images, labels, SSH or GPU execution, and does
not modify either store source file or any active experiment implementation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
PLAN = HERE.parents[1]
IMPL = PLAN/'implementation'
BASELINE = PLAN.parent/'odr003_exploratory_20261023/implementation'
SOURCES = PLAN/'performance_candidates/stream_validation_20261024'
sys.path.insert(0, str(IMPL))
sys.path.insert(1, str(BASELINE))

import numpy as np
import torch
from guard_pools import GuardPools
from protected_engine import ProtectedEngine, ControlledInterruption, validate_engine_state
from protected_learning import WindowLearner
from state_adapter import OriginalAAdapter, isolated_scoring
from test_protected_engine import Provider, legal_signal
from test_protected_learning import TinyEHS


EXPECTED = {'durable_state_original.py': '62d27f792bd13f9bf42146434e21aae1191c34d01953b70debb92cab1c32e2d3',
            'durable_state_candidate.py': '543dee24ba21ec209170082f6c983d9bba8e97b93ead5263661ff80ca8b94997'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_store(filename, name):
    path = SOURCES/filename
    if sha(path) != EXPECTED[filename]:
        raise ValueError('Independent source binding differs: '+filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same(actual, expected, path='science'):
    """Check all saved science recursively, including scalar types and RNGs."""
    if isinstance(actual, torch.Tensor):
        valid = (isinstance(expected, torch.Tensor) and actual.dtype == expected.dtype
                 and actual.shape == expected.shape and actual.device.type == expected.device.type == 'cpu'
                 and torch.equal(actual, expected))
        if not valid: raise AssertionError(path+' tensor differs')
        return
    if isinstance(actual, np.ndarray):
        if not (isinstance(expected, np.ndarray) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and np.array_equal(actual, expected)):
            raise AssertionError(path+' array differs')
        return
    if type(actual) is not type(expected):
        raise AssertionError(path+' type differs')
    if isinstance(actual, dict):
        if actual.keys() != expected.keys(): raise AssertionError(path+' keys differ')
        for key in actual: same(actual[key], expected[key], path+'.'+str(key))
        if hasattr(actual, '_metadata'):
            same(actual._metadata, expected._metadata, path+'._metadata')
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected): raise AssertionError(path+' length differs')
        for index, (a, b) in enumerate(zip(actual, expected)): same(a, b, path+f'[{index}]')
    elif actual != expected:
        raise AssertionError(path+' value differs')


def science(engine):
    state = engine.snapshot()
    # Physical journal UUIDs/timestamps/SHAs are intentionally different across
    # executions. Their science cursors and charged forward counts are checked
    # separately; all other complete recovery state is compared exactly.
    state.pop('ledger_boundary')
    return state


def factory(module, root, counts, *, faults):
    torch.manual_seed(17); np.random.seed(18); random.seed(19)
    adapter = OriginalAAdapter.configure(TinyEHS(burn_random=True, pollute=True), torch.device('cpu'))
    learner = WindowLearner(adapter)
    provider = Provider(36)
    by_identity = {id(image): (sid, index) for index, (sid, image) in enumerate(provider.images.items())}
    records = [dict(sample_id=sid, image_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                    preprocess_sha256='f'*64) for sid, image in provider.images.items()]
    identity = dict(run_id='independent-local-synthetic-science', source_sha256='a'*64,
                    config_sha256='b'*64, manifest_sha256='c'*64)

    class CountedStore(module.DurableRunStore):
        def _read_generation(self, number, path, deserializer, *, deserialize):
            counts['deserialize_true' if deserialize else 'deserialize_false'] += 1
            return super()._read_generation(number, path, deserializer, deserialize=deserialize)

    store = CountedStore(root, identity, state_validator=validate_engine_state)
    holder = {}

    def prediction(image):
        with isolated_scoring(adapter.model, adapter.bn_features):
            loss, _ = adapter.model(image, is_train=False, gpu=torch.device('cpu'))
            value = float(loss)
        if not np.isfinite(value): raise AssertionError('Synthetic prediction is nonfinite')
        return value

    def arrival(image):
        sid, index = by_identity[id(image)]
        row = legal_signal()
        row['synthetic_prediction'] = prediction(image)
        row['synthetic_accept'] = sid not in ('s0001', 's0019') and row['synthetic_prediction'] > 0
        if index % 3 == 1:
            row['heat_peaks'] = [.9]*6 + [.49]*5
        return row

    def protection(image):
        row = legal_signal()
        row['synthetic_prediction'] = prediction(image)
        engine = holder['engine']
        if learner.phase == 'candidate' and engine.counters['candidates'] in (2, 5):
            row.update(pose_legal=False, h_available=False, original_h_rejected=True,
                       heat_q=None, rotation_disagreement=None, reprojection=None)
        return row

    engine = ProtectedEngine(mode='P', stream_records=records, learner=learner,
        pools=GuardPools(91, 92), store=store, image_provider=provider,
        arrival_scorer=arrival, protection_scorer=protection,
        gate=lambda row: (row['synthetic_accept'], 'synthetic_prediction_and_frozen_mask', .4),
        fault_points=('candidate_2_tentative', 'observation_31') if faults else (),
        verify_purity=True)
    holder['engine'] = engine
    return engine, adapter


def run_case(module, root, *, faults):
    counts = dict(deserialize_true=0, deserialize_false=0, fsync_calls=0)
    original_fsync = os.fsync
    def fsync(descriptor):
        counts['fsync_calls'] += 1
        return original_fsync(descriptor)  # Actual flush, never a no-op mock.
    points = []
    adapters = []
    started = time.perf_counter()
    try:
        with patch.object(os, 'fsync', fsync):
            engine, adapter = factory(module, root, counts, faults=faults)
            adapters.append(adapter)
            for _ in range(3 if faults else 1):
                try:
                    engine.run()
                    break
                except ControlledInterruption as error:
                    points.append(str(error))
                    engine, adapter = factory(module, root, counts, faults=faults)
                    adapters.append(adapter)
            if not engine.finished: raise AssertionError('Finite recovery contract did not finish')
            final = science(engine)
            saved = engine.store.recover()
            cursors = saved.scientific_cursors
            work = engine.store.work_summary()
            publications = [row['payload'] for row in engine.store.read_journal()[0]
                            if row['kind'] == 'checkpoint_published']
            if points != (['candidate_2_tentative', 'observation_31'] if faults else []):
                raise AssertionError('Predeclared interruptions were not exercised exactly once')
            if not counts['deserialize_false'] or not counts['deserialize_true'] or not counts['fsync_calls']:
                raise AssertionError('Both store read branches and actual fsync must be exercised')
            if engine.counters['commits'] <= 0 or engine.counters['rollbacks'] != 2:
                raise AssertionError('Synthetic contract did not exercise commit and rollback')
            if engine.pools.fragile.candidate_count <= 0:
                raise AssertionError('Synthetic historical fragile reservoir was not exercised')
            if cursors['observed'] != 36 or len([r for r in engine.scientific_trace if r['kind']=='observation']) != 36:
                raise AssertionError('Recovery double-counted or lost a science observation')
        facts = dict(counters=copy.deepcopy(engine.counters),
                     saved_scientific_cursors=cursors,
                     interruptions=points, store_calls=counts,
                     checkpoint_publications=len(publications),
                     forward_counts=work['forwards_confirmed_completed'],
                     replay_forward_counts=work['replay_forwards_confirmed_completed'],
                     unconfirmed_work=work['inflight_unknown_attempt_ids'],
                     elapsed_seconds=time.perf_counter()-started)
        return final, facts
    finally:
        for adapter in adapters: adapter.close()


def main():
    if platform.system() != 'Windows' or torch.cuda.is_available() or '+cpu' not in torch.__version__:
        raise RuntimeError('Use only the isolated local Windows CPU Torch environment')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    original = load_store('durable_state_original.py', 'independent_original_durable_store')
    candidate = load_store('durable_state_candidate.py', 'independent_candidate_durable_store')
    facts, states = {}, {}
    with tempfile.TemporaryDirectory(prefix='g50p-independent-store-science-') as temporary:
        for implementation, module in (('original', original), ('candidate', candidate)):
            for faults in (False, True):
                key = implementation+('_resumed' if faults else '_continuous')
                states[key], facts[key] = run_case(module, Path(temporary)/key, faults=faults)
        comparisons = [('original_continuous', 'candidate_continuous'),
                       ('original_resumed', 'candidate_resumed'),
                       ('original_continuous', 'original_resumed'),
                       ('candidate_continuous', 'candidate_resumed')]
        for left, right in comparisons:
            same(states[left], states[right], left+' versus '+right)
        for suffix in ('continuous', 'resumed'):
            for field in ('counters', 'saved_scientific_cursors', 'interruptions',
                          'checkpoint_publications', 'forward_counts', 'replay_forward_counts', 'unconfirmed_work'):
                same(facts['original_'+suffix][field], facts['candidate_'+suffix][field], field)
    report = dict(schema='odr003-independent-local-stream-store-science-v1', status='passed',
                  script_sha256=sha(__file__), source_bindings=EXPECTED,
                  platform=dict(system=platform.system(), python=sys.version, torch=torch.__version__,
                                numpy=np.__version__, gpu_used=False),
                  comparisons=[dict(left=a, right=b, exact_science_equal=True) for a,b in comparisons],
                  checked_fields=['synthetic model predictions', 'gates', 'protection lists and signals',
                                  'commit/rollback decisions', 'full science trace', 'model and optimizer',
                                  'gradients and buffers', 'all RNG states', 'both pools and pending records',
                                  'science cursors and counters', 'provider recovery state',
                                  'fixed interrupted/replayed observation and update accounting'],
                  cases=facts, logical_observations_per_case=36, synthetic_cases=4,
                  fsync=dict(actual_file_fsync_executed=True, writes_disabled=False,
                             directory_fsync_supported_on_this_platform=False),
                  limits=['Local synthetic CPU contract only; not real SPNv2 or real GPU equivalence',
                          'No end-to-end experiment speed claim; timing is descriptive only',
                          'No production code/task/checkpoint/worker was changed',
                          'Physical journal UUIDs, timestamps and checkpoint byte SHAs are not science identity'])
    destination = HERE/'independent_science_report.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status='passed', report=str(destination), comparisons=len(comparisons),
                         synthetic_cases=4, observations_each=36, GPU=False, real_SPNv2=False)))


if __name__ == '__main__':
    main()
