"""Zero-model stdlib contracts, executed against the actual frozen writer.

SPNV2_FROZEN_DURABLE_STATE must name the approved migration snapshot file.
All writes are confined to an automatically removed temporary test directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import uuid

from reproduction.storage_management_20261027 import storage_policy
from reproduction.sysu_migration_20261027 import storage_adapter_v1 as adapter


FROZEN = Path(os.environ["SPNV2_FROZEN_DURABLE_STATE"])
IDENTITY = dict(run_id="storage-adapter-contract", source_sha256="s" * 64,
                config_sha256="c" * 64, manifest_sha256="m" * 64)


def load_frozen():
    name = "frozen_durable_contract_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, FROZEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy(peak=1024 * 1024):
    return dict(schema="spnv2-storage-policy-v1", retain_latest=2,
                estimated_peak_bytes=peak, protected_generations={})


def validate(state):
    if set(state) != {"step", "payload"} or type(state["step"]) is not int:
        raise ValueError("test state must contain only an integer step and opaque payload")


def cursors(n):
    return dict(observed=n, candidates=n, commits=n, rollbacks=0, next_index=n)


def serializer(state):
    return b"fixed-serializer-v1\x00" + json.dumps(state, sort_keys=True).encode("ascii")


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spnv2-storage-adapter-contract-")
        self.root = Path(self.tmp.name).resolve()
        self.original = load_frozen()
        self.adapted_module = load_frozen()
        self.approved = {}

        def factory(root, identity):
            self.assertEqual(identity, IDENTITY)
            value = self.approved.get(root)
            return None if value is None else adapter.WritableRootPolicy(root, value)

        self.factory = factory
        self.store_type = adapter.install_on_module(self.adapted_module, factory)

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, name="adapted", peak=1024 * 1024, **kwargs):
        root = self.root / name
        self.approved[root] = policy(peak)
        return self.store_type(root, IDENTITY, state_validator=validate, **kwargs)

    def save(self, store, n):
        return store.save(dict(step=n, payload=["opaque", n]), cursors(n), serializer=serializer)

    def test_exact_frozen_state_bytes_identity_and_one_serializer_call(self):
        frozen = self.original.DurableRunStore(self.root / "original", IDENTITY, state_validator=validate)
        adapted = self.store()
        state = dict(step=1, payload=["bytes", 123])
        old_callback, new_callback = mock.Mock(side_effect=serializer), mock.Mock(side_effect=serializer)
        old_info = frozen.save(state, cursors(1), serializer=old_callback)
        new_info = adapted.save(state, cursors(1), serializer=new_callback)
        old_callback.assert_called_once_with(state)
        new_callback.assert_called_once_with(state)
        self.assertEqual((old_info.path / "state.bin").read_bytes(), (new_info.path / "state.bin").read_bytes())
        self.assertEqual(new_info.metadata["state_sha256"], old_info.metadata["state_sha256"])
        self.assertEqual(new_info.metadata["scientific_cursors"], old_info.metadata["scientific_cursors"])
        self.assertEqual((frozen.root / "identity.json").read_bytes(), (adapted.root / "identity.json").read_bytes())
        self.assertEqual(adapted.identity, frozen.identity)
        # The default callback is also the exact frozen callback, not a reimplementation.
        self.assertIs(inspect.signature(type(adapted).save).parameters["serializer"].default,
                      inspect.signature(type(frozen).save).parameters["serializer"].default)
        # Each load has its own default validator function object; compare the
        # adapted signature to its own exact frozen parent, not another load.
        self.assertEqual(inspect.signature(type(adapted).__init__),
                         inspect.signature(type(adapted).__mro__[1].__init__))

    def test_latest_two_and_pinned_generation_survive_under_writer_lock(self):
        store = self.store()
        first = self.save(store, 1)
        first_bytes = (first.path / "state.bin").read_bytes()
        storage_policy.pin(store, 1, "frozen migration reference before publication")
        after_publish = storage_policy.after_publish
        checked_calls = []

        def checked_after_publish(candidate):
            with self.assertRaises(self.adapted_module.ConcurrentWriter):
                with candidate._lock():
                    self.fail("retention did not own the writer lock")
            checked_calls.append(candidate)
            return after_publish(candidate)

        with mock.patch.object(storage_policy, "after_publish", side_effect=checked_after_publish):
            for n in (2, 3, 4):
                self.save(store, n)
        self.assertEqual(len(checked_calls), 3)
        self.assertEqual([n for n, _ in store._paths()], [4, 3, 1])
        self.assertEqual((first.path / "state.bin").read_bytes(), first_bytes)
        self.assertEqual(storage_policy.policy(store.root)["protected_generations"][first.path.name]["state_sha256"],
                         hashlib.sha256(first_bytes).hexdigest())
        journal = [json.loads(line) for line in (store.root / "retention_journal.jsonl").read_text().splitlines()]
        self.assertEqual([v["generation"] for v in journal if v["event"] == "delete_completed"], [2])
        self.assertTrue((store.root / "storage_usage.json").is_file())

    def test_disk_budget_refuses_before_checkpoint_write_without_serializing_twice(self):
        store = self.store(peak=1)
        callback = mock.Mock(side_effect=serializer)
        state = dict(step=1, payload=["opaque"])
        with self.assertRaises(storage_policy.StoragePolicyError):
            store.save(state, cursors(1), serializer=callback)
        callback.assert_called_once_with(state)
        self.assertEqual(list(store.generations.iterdir()), [])
        event = json.loads((store.root / "retention_journal.jsonl").read_text().splitlines()[-1])
        self.assertEqual(event["event"], "disk_estimate_exceeded")
        self.assertEqual(event["new_state_bytes"], len(serializer(state)))

    def test_unapproved_and_mismatched_roots_make_no_files(self):
        source = self.root / "source-readonly"
        with self.assertRaises(adapter.StorageApprovalError):
            self.store_type(source, IDENTITY, state_validator=validate)
        self.assertFalse(source.exists())
        module = load_frozen()
        wrong_root = self.root / "other-output"
        mismatched = adapter.install_on_module(module, lambda root, identity: adapter.WritableRootPolicy(wrong_root, policy()))
        with self.assertRaises(adapter.StorageApprovalError):
            mismatched(source, IDENTITY, state_validator=validate)
        self.assertFalse(source.exists())
        self.assertFalse(wrong_root.exists())

    def test_invalid_policy_and_alias_reject_before_original_constructor(self):
        output = self.root / "invalid-policy"
        self.approved[output] = policy()
        self.approved[output]["retain_latest"] = 1
        with self.assertRaises(storage_policy.StoragePolicyError):
            self.store_type(output, IDENTITY, state_validator=validate)
        self.assertFalse(output.exists())
        target = self.root / "target"
        target.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(adapter.StorageApprovalError):
            self.store_type(alias, IDENTITY, state_validator=validate)
        self.assertEqual(list(target.iterdir()), [])

    def test_failed_frozen_save_keeps_pending_evidence_and_skips_retention(self):
        store = self.store()
        self.save(store, 1)
        self.save(store, 2)

        def fail(point):
            if point == "after_state_fsync":
                raise RuntimeError("injected durable writer interruption")

        store.failure_injector = fail
        callback = mock.Mock(side_effect=serializer)
        with mock.patch.object(storage_policy, "after_publish") as retained:
            with self.assertRaisesRegex(RuntimeError, "injected durable writer"):
                store.save(dict(step=3, payload=[]), cursors(3), serializer=callback)
        callback.assert_called_once()
        retained.assert_not_called()
        self.assertEqual([n for n, _ in store._paths()], [2, 1])
        pending = list(store.generations.glob(".pending-*"))
        self.assertEqual(len(pending), 1)
        self.assertEqual((pending[0] / "state.bin").read_bytes(), serializer(dict(step=3, payload=[])))

    def test_frozen_validation_and_nonbytes_serializer_errors_are_preserved(self):
        store = self.store()
        callback = mock.Mock(return_value="not bytes")
        with self.assertRaisesRegex(ValueError, "test state"):
            store.save({"unexpected": 1}, cursors(1), serializer=callback)
        callback.assert_not_called()
        with mock.patch.object(storage_policy, "before_write") as before:
            with self.assertRaisesRegex(TypeError, "serializer must return bytes"):
                store.save(dict(step=1, payload=[]), cursors(1), serializer=callback)
        callback.assert_called_once()
        before.assert_not_called()
        self.assertEqual(list(store.generations.iterdir()), [])

    def test_installation_is_idempotent_only_for_same_factory(self):
        self.assertIs(adapter.install_on_module(self.adapted_module, self.factory), self.store_type)
        with self.assertRaises(adapter.StorageApprovalError):
            adapter.install_on_module(self.adapted_module, lambda root, identity: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
