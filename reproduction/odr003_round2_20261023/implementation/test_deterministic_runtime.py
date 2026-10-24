"""Strict backend contracts with an injected Torch facade; no GPU work."""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import deterministic_runtime as runtime


def fake_torch(initialized=False):
    state = dict(deterministic=False, warn_only=True, precision='high')
    cudnn = SimpleNamespace(benchmark=True, deterministic=False, enabled=True,
                           allow_tf32=True, version=lambda: 8902)
    def algorithms(enabled, *, warn_only):
        state.update(deterministic=enabled, warn_only=warn_only)
    return SimpleNamespace(
        __version__='2.5.1+cu121', version=SimpleNamespace(cuda='12.1'),
        cuda=SimpleNamespace(is_initialized=lambda: initialized),
        backends=SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        are_deterministic_algorithms_enabled=lambda: state['deterministic'],
        is_deterministic_algorithms_warn_only_enabled=lambda: state['warn_only'],
        use_deterministic_algorithms=algorithms,
        get_float32_matmul_precision=lambda: state['precision'],
        set_float32_matmul_precision=lambda value: state.update(precision=value))


class DeterministicRuntimeTests(unittest.TestCase):
    def test_environment_is_prepared_before_cuda_and_not_silently_rewritten(self):
        env = {}
        self.assertEqual(runtime.prepare_environment(environ=env, torch_module=fake_torch()), ':4096:8')
        self.assertEqual(env, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        for initialized in (False, True):
            with self.assertRaises(RuntimeError):
                runtime.prepare_environment(environ={'CUBLAS_WORKSPACE_CONFIG': ':16:8'},
                                            torch_module=fake_torch(initialized))
        with self.assertRaisesRegex(RuntimeError, 'after CUDA initialization'):
            runtime.prepare_environment(environ={}, torch_module=fake_torch(True))

    def test_apply_overrides_frozen_nondeterministic_defaults_without_changing_rng(self):
        torch = fake_torch()
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            receipt = runtime.apply(torch)
            self.assertFalse(receipt['cudnn_benchmark'])
            self.assertTrue(receipt['cudnn_deterministic'])
            self.assertFalse(receipt['cuda_matmul_allow_tf32'])
            self.assertFalse(receipt['cudnn_allow_tf32'])
            self.assertTrue(receipt['torch_deterministic_algorithms'])
            self.assertFalse(receipt['warn_only'])
            self.assertEqual(receipt['float32_matmul_precision'], 'highest')
            self.assertEqual(runtime.validate_live(receipt, torch), receipt)
            self.assertEqual(runtime.validate_saved_receipt(receipt), receipt)

    def test_missing_environment_fails_before_enabling_algorithms(self):
        torch = fake_torch()
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'not prepared'):
            runtime.apply(torch)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_saved_receipt_validation_is_pure_and_rejects_each_policy_change(self):
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            receipt = runtime.apply(fake_torch())
        changes = dict(cudnn_benchmark=True, cudnn_deterministic=False, cudnn_allow_tf32=True,
                       cuda_matmul_allow_tf32=True, torch_deterministic_algorithms=False,
                       warn_only=True, float32_matmul_precision='high',
                       cublas_workspace_config=':16:8', policy_sha256='0'*64,
                       torch_version='', cuda_version=None, cudnn_version=0)
        with patch.object(runtime, '_torch', side_effect=AssertionError('pure checker imported Torch')):
            runtime.validate_saved_receipt(receipt)
            for key, value in changes.items():
                changed = copy.deepcopy(receipt); changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runtime.validate_saved_receipt(changed)

    def test_recovery_rejects_version_or_live_flag_drift(self):
        torch = fake_torch()
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
            receipt = runtime.apply(torch)
            torch.__version__ = 'another-version'
            with self.assertRaisesRegex(RuntimeError, 'receipt differs'):
                runtime.validate_live(receipt, torch)
            torch.__version__ = receipt['torch_version']
            torch.backends.cudnn.benchmark = True
            with self.assertRaisesRegex(RuntimeError, 'flags changed'):
                runtime.validate_live(receipt, torch)

    def test_backend_failure_is_not_retried_or_downgraded(self):
        torch = fake_torch()
        calls = []
        def unsupported(*args, **kwargs):
            calls.append((args, kwargs)); raise RuntimeError('unsupported deterministic operation')
        torch.use_deterministic_algorithms = unsupported
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}), \
             self.assertRaisesRegex(RuntimeError, 'unsupported'):
            runtime.apply(torch)
        self.assertEqual(calls, [((True,), {'warn_only': False})])


if __name__ == '__main__':
    unittest.main()
