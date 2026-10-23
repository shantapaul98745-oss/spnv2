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
                            stream_records, import_bound_module, sha256, convert_legacy_empty_state)


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


class RandomDataset(Dataset):
    """Exercises every global CPU RNG, including visibly stochastic outputs."""
    def __init__(self, n=6):
        super().__init__(n)
        self.extra_draw = False
        self.fail = False

    def __getitem__(self, i):
        values = [random.random(), float(np.random.random()), float(torch.rand(()))]
        if self.extra_draw:
            random.random()
        if self.fail:
            raise RuntimeError('Synthetic image decode failure')
        return dict(sample_id=str(i), image=torch.tensor(values).view(3, 1, 1).expand(3, 2, 2).clone())


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

    def test_probability_draw_preserves_original_tensor_and_isolates_rng(self):
        ds = Dataset(); ds.random = True
        p = provider(ds); before = rng_state()
        original = ds[0]['image'].unsqueeze(0)
        restore_rng(before)
        self.assertTrue(torch.equal(p('0'), original))
        self.assertTrue(equal_tree(before, rng_state()))
        self.assertFalse(equal_tree(p.initial_data_rng, p.data_rng))
        first_position = copy.deepcopy(p.data_rng)
        p('1')
        second_position = copy.deepcopy(p.data_rng)
        self.assertFalse(equal_tree(first_position, second_position))
        p('0')  # Evicted image: reconstruct without consuming new arrival RNG.
        self.assertTrue(equal_tree(second_position, p.data_rng))
        self.assertTrue(equal_tree(before, rng_state()))

    def test_all_global_rngs_are_isolated_and_per_image_replay_is_exact(self):
        ds = RandomDataset()
        p = provider(ds)
        before = rng_state()
        original = {}
        for sid in ('0', '1', '2', '3'):
            original[sid] = ds[int(sid)]['image'].unsqueeze(0)
        expected_data_rng = rng_state()
        restore_rng(before)
        for sid in original:
            self.assertTrue(torch.equal(p(sid), original[sid]))
            self.assertTrue(equal_tree(before, rng_state()))
        self.assertTrue(equal_tree(expected_data_rng, p.data_rng))
        data_position = copy.deepcopy(p.data_rng)
        for sid in reversed(original):
            self.assertTrue(torch.equal(p(sid), original[sid]))
        self.assertTrue(equal_tree(data_position, p.data_rng))
        self.assertTrue(equal_tree(before, rng_state()))

    def test_extra_protection_reads_do_not_change_arrival_trajectory(self):
        before = rng_state()
        reference, disabled, protected = [provider(RandomDataset()) for _ in range(3)]
        arrivals = [[], [], []]
        for i in range(6):
            for j, p in enumerate((reference, disabled, protected)):
                arrivals[j].append(p(str(i)))
            for old in range(i + 1):
                protected(str(old))
        for j in (1, 2):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(arrivals[0], arrivals[j])))
            self.assertTrue(equal_tree(reference.state_dict(), (disabled, protected)[j-1].state_dict()))
        self.assertTrue(equal_tree(before, rng_state()))

    def test_checkpoint_restores_data_position_and_every_image_replay_state(self):
        p = provider(RandomDataset())
        images = {sid: p(sid) for sid in ('0', '1', '2')}
        state = p.state_dict()
        expected_next = p('3')
        expected_state = p.state_dict()
        random.random(); np.random.random(); torch.rand(())
        p2 = provider(RandomDataset())
        before = rng_state()
        p2.load_state_dict(state)
        self.assertTrue(equal_tree(before, rng_state()))
        for sid in reversed(images):
            self.assertTrue(torch.equal(images[sid], p2(sid)))
        self.assertTrue(torch.equal(expected_next, p2('3')))
        self.assertTrue(equal_tree(expected_state, p2.state_dict()))
        self.assertTrue(equal_tree(before, rng_state()))

    def test_failure_does_not_advance_data_position_or_leak_model_rng(self):
        ds = RandomDataset(); ds.fail = True
        p = provider(ds)
        state, before = p.state_dict(), rng_state()
        with self.assertRaisesRegex(RuntimeError, 'decode failure'):
            p('0')
        self.assertTrue(equal_tree(state, p.state_dict()))
        self.assertTrue(equal_tree(before, rng_state()))
        ds.fail = False
        p('0')
        self.assertEqual(p.materialization_order, ['0'])

    def test_changed_random_consumption_is_rejected_on_reconstruction(self):
        ds = RandomDataset(); p = provider(ds)
        p('0'); p('1')
        state, before = p.state_dict(), rng_state()
        ds.extra_draw = True
        with self.assertRaisesRegex(ValueError, 'RNG consumption changed'):
            p('0')
        self.assertTrue(equal_tree(state, p.state_dict()))
        self.assertTrue(equal_tree(before, rng_state()))

    def test_data_state_copies_do_not_alias_live_rng(self):
        p = provider(RandomDataset()); p('0')
        state = p.state_dict()
        state['data_rng']['cpu'].zero_()
        state['rng_replay']['0']['before']['cpu'].zero_()
        self.assertFalse(equal_tree(state['data_rng'], p.data_rng))
        self.assertFalse(equal_tree(state['rng_replay']['0']['before'], p.rng_replay['0']['before']))

    def test_corrupt_replay_chain_and_v1_state_fail_closed_without_rng_leak(self):
        p = provider(RandomDataset()); p('0'); p('1')
        state, before = p.state_dict(), rng_state()
        corruptions = []
        old = copy.deepcopy(state); old['schema'] = 'odr003-round2-image-provider-v1'; corruptions.append(old)
        old = copy.deepcopy(state); old['materialization_order'] = ['0', '0']; corruptions.append(old)
        old = copy.deepcopy(state); old['rng_replay'].pop('0'); corruptions.append(old)
        old = copy.deepcopy(state); old['data_rng'] = copy.deepcopy(old['initial_data_rng']); corruptions.append(old)
        old = copy.deepcopy(state); old['rng_replay']['1']['before'] = copy.deepcopy(old['initial_data_rng']); corruptions.append(old)
        old = copy.deepcopy(state); old['rng_replay']['1']['after']['cpu'] = None; corruptions.append(old)
        for altered in corruptions:
            with self.subTest(altered=altered.get('schema')):
                with self.assertRaises(ValueError): p.load_state_dict(altered)
                self.assertTrue(equal_tree(state, p.state_dict()))
                self.assertTrue(equal_tree(before, rng_state()))

    def test_callback_cuda_slot_is_preserved_including_replay_and_resume(self):
        # CPU-only CI can still prove the provider retains every callback field.
        # Actual CUDA states use the identical unchanged state_adapter callback.
        cuda_slot = {'value': 19}
        class CudaSlotDataset(Dataset):
            def __getitem__(self, i):
                cuda_slot['value'] += 1
                return super().__getitem__(i)
        def capture():
            state = rng_state(); state['synthetic_cuda_slot'] = cuda_slot['value']; return state
        def restore(state):
            restore_rng(state); cuda_slot['value'] = state['synthetic_cuda_slot']
        def make():
            return DeterministicImageProvider(CudaSlotDataset(), preprocess_sha256='a'*64,
                                             rng_state=capture, restore_rng=restore, equal_tree=equal_tree,
                                             cache_capacity=1)
        p = make(); before = capture()
        p('0'); p('1'); p('0')
        self.assertTrue(equal_tree(before, capture()))
        self.assertEqual(p.data_rng['synthetic_cuda_slot'], 21)
        self.assertEqual(p.rng_replay['0']['before']['synthetic_cuda_slot'], 19)
        p2 = make(); p2.load_state_dict(p.state_dict()); p2('0')
        self.assertTrue(equal_tree(before, capture()))
        self.assertTrue(equal_tree(p.state_dict(), p2.state_dict()))

    def test_explicit_zero_progress_legacy_migration_preserves_identity_and_rng(self):
        p = provider(RandomDataset())
        before = rng_state()
        old = {key: value for key, value in p.state_dict().items()
               if key in {'schema', 'manifest_sha256', 'manifest_semantic_sha256',
                          'preprocess_sha256', 'tensor_sha256', 'data_rng_policy'}}
        old['schema'] = 'odr003-round2-image-provider-v1'
        old['data_rng_policy'] = 'deterministic_preprocessing_no_RNG_consumption'
        original = copy.deepcopy(old)
        state = convert_legacy_empty_state(old, initial_data_rng=before)
        self.assertTrue(equal_tree(old, original))
        self.assertTrue(equal_tree(before, rng_state()))
        self.assertTrue(equal_tree(p.state_dict(), state))
        p.load_state_dict(state)
        self.assertTrue(equal_tree(before, rng_state()))
        state['data_rng']['cpu'].zero_()
        self.assertFalse(equal_tree(state['data_rng'], before))
        self.assertTrue(equal_tree(before, rng_state()))
        for key in ('manifest_sha256', 'manifest_semantic_sha256', 'preprocess_sha256'):
            self.assertEqual(original[key], p.state_dict()[key])

    def test_legacy_migration_refuses_nonempty_unknown_or_malformed_state(self):
        p = provider(Dataset())
        old = {key: value for key, value in p.state_dict().items()
               if key in {'schema', 'manifest_sha256', 'manifest_semantic_sha256',
                          'preprocess_sha256', 'tensor_sha256', 'data_rng_policy'}}
        old.update(schema='odr003-round2-image-provider-v1',
                   data_rng_policy='deterministic_preprocessing_no_RNG_consumption')
        malformed = []
        bad = copy.deepcopy(old); bad['tensor_sha256']['0'] = 'a'*64; malformed.append(bad)
        bad = copy.deepcopy(old); bad['unknown'] = True; malformed.append(bad)
        bad = copy.deepcopy(old); bad['manifest_sha256'] = 'x'*64; malformed.append(bad)
        bad = copy.deepcopy(old); bad['data_rng_policy'] = 'new policy'; malformed.append(bad)
        for bad in malformed:
            with self.assertRaises(ValueError):
                convert_legacy_empty_state(bad, initial_data_rng=rng_state())
        with self.assertRaises(ValueError):
            convert_legacy_empty_state(old, initial_data_rng={})

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
