"""Strict deterministic FP32 execution; no scientific update-rule changes.

The environment is established before CUDA initialization.  Frozen source
configuration may set cuDNN flags, so ``apply`` is called after its seed helper
and before constructing/warming/forwarding the real model.  The persisted
receipt is compared on recovery rather than accepting a different backend.
"""
from __future__ import annotations

import os
import sys
import hashlib
import json

from round2_protocol import frozen_protocol


POLICY_VERSION = "strict-cuda-fp32-v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def prepare_environment(*, environ=None, torch_module=None):
    environment = os.environ if environ is None else environ
    torch = sys.modules.get("torch") if torch_module is None else torch_module
    previous = environment.get("CUBLAS_WORKSPACE_CONFIG")
    initialized = bool(torch is not None and torch.cuda.is_initialized())
    if initialized and previous != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("Cannot establish deterministic cuBLAS after CUDA initialization")
    if previous not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG; do not silently change backend")
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return CUBLAS_WORKSPACE_CONFIG


def _torch(module):
    if module is not None:
        return module
    import torch
    return torch


def current_receipt(torch_module=None):
    torch = _torch(torch_module)
    return dict(schema="odr003-round2-deterministic-runtime-v1",
                policy_version=POLICY_VERSION,
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                policy_sha256=_policy_sha(),
                torch_deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_enabled=bool(torch.backends.cudnn.enabled),
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version())


def _policy_sha():
    return hashlib.sha256(json.dumps(frozen_protocol()['deterministic_execution'],
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_saved_receipt(receipt):
    """Validate saved policy/flags/version metadata without importing Torch."""
    if not isinstance(receipt, dict):
        raise ValueError('Missing deterministic execution receipt')
    policy = frozen_protocol()['deterministic_execution']
    keys = ('cublas_workspace_config', 'torch_deterministic_algorithms', 'warn_only',
            'cudnn_benchmark', 'cudnn_deterministic', 'cuda_matmul_allow_tf32',
            'cudnn_allow_tf32', 'float32_matmul_precision')
    if (receipt.get('schema') != 'odr003-round2-deterministic-runtime-v1'
            or receipt.get('policy_version') != POLICY_VERSION
            or receipt.get('policy_sha256') != _policy_sha()
            or any(type(receipt.get(key)) is not type(policy[key]) or receipt[key] != policy[key] for key in keys)
            or type(receipt.get('cudnn_enabled')) is not bool
            or not all(isinstance(receipt.get(key), str) and receipt[key] for key in ('torch_version', 'cuda_version'))
            or type(receipt.get('cudnn_version')) is not int or receipt['cudnn_version'] <= 0):
        raise ValueError('Saved deterministic policy/flags/versions differ')
    return receipt


def validate(receipt=None, torch_module=None):
    actual = current_receipt(torch_module)
    expected = dict(cublas_workspace_config=CUBLAS_WORKSPACE_CONFIG,
                    torch_deterministic_algorithms=True, warn_only=False,
                    cudnn_benchmark=False, cudnn_deterministic=True,
                    cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False,
                    float32_matmul_precision='highest')
    if any(actual[key] != value for key, value in expected.items()):
        raise RuntimeError("Strict deterministic execution flags changed")
    if receipt is not None and receipt != actual:
        raise RuntimeError("Deterministic backend/version receipt differs on recovery")
    validate_saved_receipt(actual)
    return actual


validate_live = validate


def apply(torch_module=None):
    torch = _torch(torch_module)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("Deterministic cuBLAS environment was not prepared before CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=False)
    return validate(torch_module=torch)
