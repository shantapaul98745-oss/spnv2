"""LOCAL ONLY: bounded correctness and full durable-save performance checks.

No model, Torch, GPU, GT, SSH, production writes or scientific runs. Scratch
directories live below this isolated candidate and are verified before cleanup.
The benchmark includes serialization, prior-generation validation and all save
fsync barriers; a separate complete recover is measured too. It does not measure
SPNv2 end-to-end speed, GPU work, or the protected learner's state-capture cost.
"""
from __future__ import annotations
from array import array
from contextlib import contextmanager
import argparse
import copy
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
IDENTITY = dict(run_id='isolated-local-validation', source_sha256='a'*64,
                config_sha256='b'*64, manifest_sha256='c'*64)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, sort_keys=True, indent=2)+'\n', encoding='utf8')
def module(name):
    path = ROOT/('durable_state_'+name+'.py')
    spec = importlib.util.spec_from_file_location('isolated_durable_'+name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value; spec.loader.exec_module(value)
    return value


@contextmanager
def scratch():
    parent = ROOT/'scratch'; parent.mkdir(exist_ok=True)
    normal_path = parent/uuid.uuid4().hex; normal_path.mkdir()
    require = normal_path.resolve().is_relative_to(ROOT) and normal_path.resolve() != ROOT
    if not require: raise RuntimeError('Scratch path escapes isolated candidate')
    # The workspace itself is long; keep all fixtures inside this candidate
    # while using Windows' documented extended-path spelling for deep files.
    path = Path('\\\\?\\'+str(normal_path.resolve())) if os.name == 'nt' else normal_path
    try: yield path
    finally:
        if not normal_path.resolve().is_relative_to(parent.resolve()) or normal_path.resolve() == parent.resolve():
            raise RuntimeError('Refusing unsafe scratch cleanup')
        shutil.rmtree(path)


def state(n=0, payload=b'bounded-local-payload'):
    return dict(identity=dict(IDENTITY), learning=dict(model={'weight': array('f', [1., -2., n/64])},
                optimizer={'step': n//4, 'moment': array('d', [0.25, 0.5])}, buffers={'BN': array('f', [0., 1.])},
                gradients={'weight': None if n % 4 == 0 else array('f', [0.1, 0.2, 0.3])}),
                pools={'uniform': ['image-%04d'%i for i in range(min(4, n))], 'candidate_count': n,
                       'independent_rng': random.Random(20261024+n).getstate()},
                rng=random.Random(2021+n).getstate(), pending_window=list(range(n % 4)),
                payload=payload, cursors=dict(observed=n, next_index=n, candidates=n//4, commits=n//4, rollbacks=0))


def validate(value):
    assert value['identity'] == IDENTITY
    c = value['cursors']; assert c['observed'] == c['next_index'] and c['commits']+c['rollbacks'] <= c['candidates']
    assert isinstance(value['learning']['model']['weight'], array) and isinstance(value['payload'], bytes)


def save(store, value): return store.save(value, value['cursors'])
def create(mod, root, failure=None): return mod.DurableRunStore(root, IDENTITY, state_validator=validate, failure_injector=failure)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.variants = [module('original'), module('candidate')]

    def test_origin_and_candidate_manifest_bindings(self):
        doc = json.loads((ROOT/'manifest.json').read_text(encoding='utf8'))
        self.assertEqual(sha(Path(doc['origin_path']).read_bytes()), doc['origin_sha256'])
        self.assertEqual(sha((ROOT/'durable_state_original.py').read_bytes()), doc['original_sha256'])
        self.assertEqual(sha((ROOT/'durable_state_candidate.py').read_bytes()), doc['candidate_sha256'])
        self.assertEqual(self.variants[0].SAVE_FAILURE_POINTS, self.variants[1].SAVE_FAILURE_POINTS)

    def test_valid_complete_state_and_same_serialized_scientific_bytes(self):
        for length in (0, 1, 1048575, 1048576, 1048577):
            source = state(64, b'x'*length); outcomes=[]
            for mod in self.variants:
                with scratch() as root:
                    store=create(mod, root); info=save(store, source); recovered=store.recover()
                    self.assertEqual(recovered.state, source); self.assertEqual(recovered.scientific_cursors, source['cursors'])
                    data, checked=store._read_generation(1, info.path, lambda _: self.fail('metadata route deserialized'), deserialize=False)
                    self.assertIsNone(data); self.assertEqual(checked.metadata, info.metadata)
                    outcomes.append((info.metadata['state_sha256'], (info.path/'state.bin').read_bytes()))
            self.assertEqual(outcomes[0], outcomes[1])

    def test_every_corruption_truncation_and_extra_length_rejected_equally(self):
        for kind in ('flip_first', 'flip_middle', 'flip_last', 'truncated', 'oversized', 'metadata_hash', 'metadata_identity'):
            results=[]
            for mod in self.variants:
                with scratch() as root:
                    store=create(mod, root); info=save(store, state(64)); path=info.path/'state.bin'; raw=path.read_bytes()
                    if kind.startswith('flip_'):
                        offset={'flip_first': 0, 'flip_middle': len(raw)//2, 'flip_last': len(raw)-1}[kind]
                        path.write_bytes(raw[:offset]+bytes([raw[offset]^1])+raw[offset+1:])
                    elif kind == 'truncated': path.write_bytes(raw[:-1])
                    elif kind == 'oversized': path.write_bytes(raw+b'EXTRA')
                    elif kind == 'metadata_hash': (info.path/'metadata.sha256').write_text('0'*64+'\n')
                    else:
                        meta=json.loads((info.path/'metadata.json').read_bytes()); meta['identity']['run_id']='wrong'
                        encoded=mod._json_bytes(meta); (info.path/'metadata.json').write_bytes(encoded)
                        (info.path/'metadata.sha256').write_text(sha(encoded)+'\n')
                    row=[]
                    for deserialize in (False, True):
                        try: store._read_generation(1, info.path, pickle.loads, deserialize=deserialize)
                        except Exception as error: row.append((type(error).__name__, str(error)))
                        else: self.fail('Corrupted saved generation accepted')
                    results.append(row)
            self.assertEqual(results[0], results[1], kind)

    def test_corrupt_latest_fallback_and_subsequent_save_preserve_science(self):
        results=[]
        for mod in self.variants:
            with scratch() as root:
                store=create(mod, root); save(store, state(0)); bad=save(store, state(64)); raw=(bad.path/'state.bin').read_bytes()
                (bad.path/'state.bin').write_bytes(raw+b'torn-extra')
                recovered=store.recover(); self.assertEqual(recovered.state, state(0)); self.assertEqual(len(recovered.rejected_generations), 1)
                info=save(store, state(32)); self.assertEqual(info.generation, 3)
                final=store.recover(); self.assertEqual(final.state, state(32))
                self.assertEqual((bad.path/'state.bin').read_bytes(), raw+b'torn-extra')
                results.append((final.state, final.info.metadata['state_sha256']))
        self.assertEqual(results[0], results[1])

    def test_all_invalid_generations_stop_and_cursor_cannot_go_backwards(self):
        for mod in self.variants:
            with scratch() as root:
                store=create(mod, root); info=save(store, state(64))
                with self.assertRaises(ValueError): save(store, state(63))
                self.assertEqual(store.recover().state, state(64))
                (info.path/'state.bin').write_bytes(b'bad')
                with self.assertRaises(mod.CheckpointCorruption): store.recover()
                with self.assertRaises(mod.CheckpointCorruption): save(store, state(128))

    def test_full_save_recover_every_atomic_failure_point_matches(self):
        for point in self.variants[0].SAVE_FAILURE_POINTS:
            outcomes=[]
            for mod in self.variants:
                with scratch() as root:
                    store=create(mod, root); save(store, state(0))
                    store.failure_injector=lambda actual: (_ for _ in ()).throw(RuntimeError(point)) if actual == point else None
                    with self.assertRaisesRegex(RuntimeError, point): save(store, state(64))
                    recovered=create(mod, root).recover()
                    outcomes.append((recovered.state, recovered.info.generation, recovered.scientific_cursors))
            self.assertEqual(outcomes[0], outcomes[1], point)

    def test_durable_journal_reservation_completion_and_torn_tail_semantics_unchanged(self):
        results=[]
        for mod in self.variants:
            with scratch() as root:
                store=create(mod, root); save(store, state(0))
                first=store.begin_work('saved-data-only', {'no_model_calls': 1}, logical_key='step1', attempt_id='a')
                store.finish_work(first, completed_forwards={'no_model_calls': 1}, compute_seconds=0.25)
                store.begin_work('saved-data-only', {'no_model_calls': 1}, logical_key='step2', attempt_id='b')
                summary=store.work_summary(); self.assertEqual(summary['forwards_execution_lower_bound'], {'no_model_calls': 1})
                self.assertEqual(summary['forwards_execution_upper_bound'], {'no_model_calls': 2})
                segment=next(store.journal.glob('segment-*')); segment.write_bytes(segment.read_bytes()+b'{torn')
                store.append_event('continued_without_truncation', {}, event_id='audit')
                final=store.work_summary(); self.assertEqual(len(final['torn_journal_tails']), 1)
                results.append({k: v for k,v in final.items() if k != 'torn_journal_tails'})
                self.assertEqual(store.recover().state, state(0))
        self.assertEqual(results[0], results[1])


def peak_rss():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in
                ('peak_working_set', 'working_set', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        value=Counters(); value.cb=ctypes.sizeof(value)
        kernel=ctypes.windll.kernel32; kernel.GetCurrentProcess.restype=wintypes.HANDLE
        measure=ctypes.windll.psapi.GetProcessMemoryInfo
        measure.argtypes=[wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        measure.restype=wintypes.BOOL
        if not measure(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise OSError('Cannot measure local process peak working set')
        return int(value.peak_working_set)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def benchmark_child(variant, size_mib):
    mod=module(variant); payload=b'\xa5'*(size_mib*1024*1024)
    with scratch() as root:
        store=create(mod, root); save(store, state(0, payload)); gc.collect(); baseline=peak_rss()
        source=state(64, payload); start=time.perf_counter(); info=save(store, source)
        save_seconds=time.perf_counter()-start; after_save=peak_rss()
        start=time.perf_counter(); recovered=store.recover(); recover_seconds=time.perf_counter()-start
        assert recovered.state == source
        return dict(variant=variant, size_mib=size_mib, state_bytes=info.metadata['state_bytes'],
                    save_seconds=save_seconds, recover_seconds=recover_seconds,
                    full_cycle_seconds=save_seconds+recover_seconds, baseline_peak_rss_bytes=baseline,
                    save_peak_rss_bytes=after_save, full_cycle_peak_rss_bytes=peak_rss(),
                    state_sha256=info.metadata['state_sha256'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--child', choices=('original','candidate'))
    parser.add_argument('--size-mib', type=int, default=16); parser.add_argument('--correctness-only', action='store_true')
    args=parser.parse_args()
    if args.child:
        if args.size_mib not in (16, 64): raise ValueError('Only the fixed local 16/64 MiB benchmark is approved')
        print(json.dumps(benchmark_child(args.child, args.size_mib))); return
    stream=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT/'validation.log').write_text(stream.getvalue(), encoding='utf8')
    report=dict(schema='isolated-durable-stream-check-v1', local_only=True, tests_run=result.testsRun,
                failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful(),
                real_model_forward=0, GPU_used=False, production_modified=False)
    dump(ROOT/'validation.json', report)
    if not result.wasSuccessful(): print(stream.getvalue()); raise SystemExit(1)
    if args.correctness_only: print(json.dumps(report)); return
    rows=[]
    for size in (16, 64):
        for repetition in range(3):
            order=('original','candidate') if repetition % 2 == 0 else ('candidate','original')
            for variant in order:
                output=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child',variant,'--size-mib',str(size)],
                                      check=False,capture_output=True,text=True)
                if output.returncode:
                    dump(ROOT/'benchmark_child_failure.json', dict(variant=variant, size_mib=size, repetition=repetition,
                         exit_code=output.returncode, stdout=output.stdout, stderr=output.stderr))
                    raise RuntimeError('Bounded local benchmark child failed; preserved output, no automatic retry')
                row=json.loads(output.stdout); row['repetition']=repetition; rows.append(row)
    summaries=[]
    for size in (16, 64):
        selected={name:[r for r in rows if r['size_mib']==size and r['variant']==name] for name in ('original','candidate')}
        assert len({r['state_sha256'] for name in selected for r in selected[name]}) == 1
        summary=dict(size_mib=size)
        for name, values in selected.items():
            summary[name]={key:dict(median=statistics.median([r[key] for r in values]), minimum=min(r[key] for r in values),
                                   maximum=max(r[key] for r in values))
                           for key in ('save_seconds','recover_seconds','full_cycle_seconds','save_peak_rss_bytes','full_cycle_peak_rss_bytes')}
        summary['save_median_change_fraction']=summary['candidate']['save_seconds']['median']/summary['original']['save_seconds']['median']-1
        summary['cycle_median_change_fraction']=summary['candidate']['full_cycle_seconds']['median']/summary['original']['full_cycle_seconds']['median']-1
        summaries.append(summary)
    benchmark=dict(schema='isolated-durable-full-save-benchmark-v1', rows=rows, summaries=summaries,
                   repeats_each_variant_size=3, sizes_mib=[16,64], local_only=True, includes_all_existing_fsync=True,
                   includes_new_serialization_and_complete_previous_SHA=True, includes_separate_full_recover=True,
                   excludes_state_capture_GPU_forward_and_ODR_end_to_end=True, production_deployment=False,
                   limitation='Local Windows cached filesystem and bounded synthetic state; three samples do not establish server/GPU end-to-end benefit')
    dump(ROOT/'benchmark.json', benchmark)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf8'))
    assert sha(Path(manifest['origin_path']).read_bytes()) == manifest['origin_sha256']
    print(json.dumps(dict(validation=report, benchmark=summaries)))


if __name__ == '__main__': main()
