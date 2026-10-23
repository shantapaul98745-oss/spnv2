import copy
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'odr003_exploratory_20261023' / 'implementation'))
from state_adapter import rng_state, restore_rng, equal_tree
from frozen_backend import (HEGeometryView, FrozenScorers, DeterministicImageProvider,
                            stream_records, import_bound_module, sha256)


class Dataset:
    def __init__(self, n=3):
        self.manifest = {'items': [{'sample_id': str(i), 'image_sha256': str(i % 10)*64} for i in range(n)]}
        self.random = False
        self.extra = {}
        self.offset = 0

    def __getitem__(self, i):
        if self.random:
            random.random()
        return dict(sample_id=str(i), image=torch.ones(3, 2, 2) * (i+self.offset), **self.extra)


def provider(ds, capacity=1):
    return DeterministicImageProvider(ds, preprocess_sha256='a'*64, rng_state=rng_state,
                                     restore_rng=restore_rng, equal_tree=equal_tree,
                                     cache_capacity=capacity)


class BackendTests(unittest.TestCase):
    def test_he_view_retains_real_model_and_methods(self):
        class Model:
            head_names = ['heatmap', 'efficientpose', 'segmentation']
            heads = [object(), object(), object()]
            backbone = object()
            def named_buffers(self):
                return [('buffer', self.backbone)]
        m = Model()
        p = HEGeometryView(m)
        self.assertEqual(list(p.head_names), m.head_names[:2])
        self.assertEqual(list(p.heads), m.heads[:2])
        self.assertIs(p.backbone, m.backbone)
        self.assertEqual(p.named_buffers(), m.named_buffers())
        self.assertEqual(m.head_names, ['heatmap', 'efficientpose', 'segmentation'])

    def test_protection_keeps_full_geometry_original_entry(self):
        m = types.SimpleNamespace(head_names=['heatmap', 'efficientpose', 'segmentation'], heads=[1, 2, 3])
        calls = []
        def extract(model, *args, **kwargs):
            calls.append((model, kwargs))
            return {'reprojection': .02, 'heat_peaks': [1]*11}
        def decide(*args):
            calls.append(args)
            return True, 'threshold_pass', .5
        cfg = types.SimpleNamespace(TEST=types.SimpleNamespace(HEATMAP_THRESHOLD=.5))
        s = FrozenScorers(model=m, adapter=None, cfg=cfg, camera=None, points=None,
                         calibration={'threshold': .2}, prediction_module=types.SimpleNamespace(extract=extract),
                         gate_module=types.SimpleNamespace(decide=decide))
        s.arrival(None)
        result = s.protection(None)
        self.assertIs(calls[0][0], m)
        self.assertEqual(calls[1][0].head_names, ('heatmap', 'efficientpose'))
        self.assertEqual(calls[1][1], {'profile': 'full'})
        self.assertEqual(result['reprojection'], .02)
        self.assertEqual(s.gate(result), (True, 'threshold_pass', .5))
        self.assertEqual(calls[-1][0], 'G50')

    def test_cache_is_bounded_and_returned_inputs_are_independent(self):
        p = provider(Dataset())
        x = p('0'); x.add_(20)
        self.assertTrue(torch.equal(p('0'), torch.zeros_like(x)))
        p('1'); p('2')
        self.assertEqual(len(p.cache), 1)
        self.assertEqual(p.materializations, 3)
        self.assertEqual(len(p.tensor_sha256), 3)

    def test_deterministic_reconstruction_and_rng_unchanged(self):
        p = provider(Dataset())
        before = rng_state()
        expected = p('0')
        p('1')
        self.assertTrue(torch.equal(expected, p('0')))
        self.assertTrue(equal_tree(before, rng_state()))

    def test_random_preprocess_is_rejected_and_rng_restored(self):
        ds = Dataset(); ds.random = True
        p = provider(ds); before = rng_state()
        with self.assertRaisesRegex(RuntimeError, 'consumed RNG'):
            p('0')
        self.assertTrue(equal_tree(before, rng_state()))

    def test_no_label_keys_or_foreign_input(self):
        ds = Dataset(); ds.extra = {'translation_gt': [0, 0, 1]}
        p = provider(ds)
        with self.assertRaises(ValueError): p('0')
        with self.assertRaises(ValueError): p('not_observed')

    def test_resume_validates_reconstructed_tensor_and_identity(self):
        ds = Dataset(); p = provider(ds); p('0')
        state = p.state_dict()
        p2 = provider(ds); p2.load_state_dict(state)
        self.assertEqual(len(p2.cache), 0)
        self.assertTrue(torch.equal(p('0'), p2('0')))
        ds.offset = 1
        p2.cache.clear()
        with self.assertRaisesRegex(ValueError, 'tensor changed'): p2('0')
        altered = copy.deepcopy(state); altered['preprocess_sha256'] = 'b'*64
        with self.assertRaises(ValueError): p2.load_state_dict(altered)

    def test_only_exact_frozen_sizes(self):
        ds = Dataset(1024)
        self.assertEqual(len(stream_records(ds, 'a'*64, limit=128)), 128)
        with self.assertRaises(ValueError): stream_records(ds, 'a'*64, limit=129)
        with self.assertRaises(ValueError): stream_records(Dataset(128), 'a'*64, limit=128)

    def test_import_requires_hash_and_absolute_file_identity(self):
        name = 'test_only_bound_backend_stub'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/f'{name}.py'; path.write_text('VALUE = 1\n')
            binding = dict(path=str(path), sha256=sha256(path))
            try:
                m = import_bound_module(name, binding)
                self.assertEqual(m.VALUE, 1)
                path.write_text('VALUE = 2\n')
                with self.assertRaises(ValueError): import_bound_module(name, binding)
            finally:
                sys.modules.pop(name, None)
                sys.path[:] = [p for p in sys.path if p != tmp]


if __name__ == '__main__':
    unittest.main()
