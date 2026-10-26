"""Versioned retention adapter for the unchanged, frozen DurableRunStore.

Install only in a newly admitted migration process.  The policy factory must
explicitly approve each exact writable output root.  Unapproved/read-only
roots are rejected *before* the frozen constructor can create files.  Keep a
separate, unadapted module for any read-only source inspection.

No model dependency, serializer replacement, or scientific-state transform is
introduced here.  The frozen serializer is invoked once, by the frozen save.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Callable, Mapping

from reproduction.storage_management_20261027 import storage_policy


ADAPTER_VERSION = "spnv2-sysu-storage-adapter-v1"


class StorageApprovalError(storage_policy.StoragePolicyError):
    """The caller did not explicitly approve this exact writable root."""


@dataclass(frozen=True)
class WritableRootPolicy:
    """An explicit approval for one canonical absolute output/store root.

    ``policy_factory(root, identity)`` returns this object only for authorized
    new-output stores, including their subsequent safe resumptions.  Returning
    ``None`` (for a source/read-only/unrecognized root) refuses construction.
    A broad ancestor approval never authorizes its descendants implicitly.
    """

    root: str | Path
    policy: Mapping


def _canonical_explicit_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.absolute() != path.resolve():
        raise StorageApprovalError("Writable roots must be absolute, canonical, and free of symlink aliases")
    return path


def install_on_module(old_durable_module, policy_factory):
    """Install and return a subclass; never rewrite the frozen module file.

    The two-argument factory receives ``(Path(root), dict(identity))`` and must
    return ``WritableRootPolicy`` for that exact root.  Its policy is validated
    before the original constructor, then initialized under its writer lock.
    Identity arguments/defaults and frozen serialization remain unchanged.

    Read-only fallback is intentionally unsupported: denial is explicit and
    makes no directory, identity, lock, policy, or checkpoint writes.
    """
    if not callable(policy_factory):
        raise TypeError("An explicit writable-root policy factory is required")
    base = old_durable_module.DurableRunStore
    if getattr(base, "_spnv2_storage_adapter_version", None) is not None:
        if (base._spnv2_storage_adapter_version == ADAPTER_VERSION
                and base._spnv2_policy_factory is policy_factory):
            return base
        raise StorageApprovalError("DurableRunStore already has a different storage adapter binding")
    base_save = base.save
    default_serializer = base_save.__kwdefaults__["serializer"]
    default_validator = old_durable_module.validate_run_state

    class PolicyDurableRunStore(base):
        _spnv2_storage_adapter_version = ADAPTER_VERSION
        _spnv2_policy_factory = staticmethod(policy_factory)

        def __init__(self, root: str | Path, identity: Mapping[str, str], *,
                     state_validator: Callable[[Any], None] = default_validator,
                     failure_injector: Callable[[str], None] | None = None):
            canonical_root = _canonical_explicit_root(root)
            approval = policy_factory(canonical_root, dict(identity))
            if not isinstance(approval, WritableRootPolicy):
                raise StorageApprovalError("This root has no explicit writable storage approval")
            if _canonical_explicit_root(approval.root) != canonical_root:
                raise StorageApprovalError("Writable storage approval names a different root")
            approved_policy = copy.deepcopy(dict(approval.policy))
            storage_policy.validate(approved_policy)
            # The exact original root/identity and custom validator are passed
            # through; no migration/source identity is substituted.
            super().__init__(root, identity, state_validator=state_validator,
                             failure_injector=failure_injector)
            with self._lock():
                self._storage_policy = storage_policy.initialize(self.root, approved_policy)

        def save(self, state, scientific_cursors=None, *, serializer=default_serializer):
            def checked_serializer(value):
                data = serializer(value)
                # Let the frozen save produce its original non-bytes error.
                if isinstance(data, bytes):
                    storage_policy.before_write(self.root, len(data))
                return data

            info = base_save(self, state, scientific_cursors, serializer=checked_serializer)
            # The original save has completed all durability barriers before
            # retention can inspect or remove superseded, unpinned generations.
            with self._lock():
                storage_policy.after_publish(self)
            return info

    PolicyDurableRunStore.__name__ = "PolicyDurableRunStore"
    PolicyDurableRunStore.__qualname__ = "PolicyDurableRunStore"
    old_durable_module.DurableRunStore = PolicyDurableRunStore
    return PolicyDurableRunStore
