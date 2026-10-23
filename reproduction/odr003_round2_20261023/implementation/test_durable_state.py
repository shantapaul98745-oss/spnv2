"""Local stdlib contracts.  No torch, model forward, GPU, or remote execution."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from durable_state import (
    CheckpointCorruption, ConcurrentWriter, DurableRunStore, IdentityMismatch,
    JournalCorruption, SAVE_FAILURE_POINTS, scientific_cursors_from_state,
    validate_run_state,
)


IDENTITY = {"run_id": "local-contract-only", "source_sha256": "a" * 64,
            "config_sha256": "b" * 64, "manifest_sha256": "c" * 64,
            "code_sha256": "d" * 64}


def example_state(observed=0, commits=0, rollbacks=0):
    rng = random.Random(1234).getstate()
    return {
        "schema_version": 1,
        "identity": dict(IDENTITY),
        "learning": {"model": {"weight": [1.0, 2.0]}, "optimizer": {"step": commits},
                     "buffers": {}, "modes": {"training": True}, "gradients": None,
                     "caches": {}, "scaler": None},
        "pools": {name: {"slots": [], "candidate_count": 0, "replacement_count": 0,
                         "rng_state": rng} for name in ("uniform", "vulnerable")},
        "window": {"image_ids": [], "rollback_state": None},
        "pending_observations": [],
        "counters": {"observed": observed, "accepted": 4 * (commits + rollbacks),
                     "candidates": commits + rollbacks, "commits": commits,
                     "rollbacks": rollbacks, "tail_discarded": 0,
                     "consecutive_rollbacks": 0, "max_consecutive_rollbacks": 0},
        "sampling": {"next_index": observed, "manifest_sha256": IDENTITY["manifest_sha256"]},
        "rng": {name: rng for name in ("compute", "data", "uniform_pool", "vulnerable_pool", "random_control")},
        "transaction": {"id": None, "phase": "collecting", "protection_ids": [],
                        "pre_signals": None, "rw_decision": None},
        "ledger_boundary": {"sequence": 0, "byte_offset": 0},
    }


class InjectedCrash(Exception):
    pass


class DurableCheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "one-run"
        self.store = DurableRunStore(self.root, IDENTITY)

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_complete_pools_partial_window_rng(self):
        state = example_state(7, 1)
        state["counters"]["accepted"] = 6
        state["window"] = {"image_ids": ["img5", "img6"], "rollback_state": {"model": [0.4]}}
        state["learning"]["gradients"] = {"parameter": [0.1, 0.2]}
        state["pending_observations"] = [
            {"image_id": "img5", "observation_index": 5, "arrival_record": {"fragile": True}},
            {"image_id": "img6", "observation_index": 6, "arrival_record": {"fragile": False}},
        ]
        state["pools"]["uniform"].update({
            "candidate_count": 4, "slots": [{
                "image_id": "img1", "observation_index": 1,
                "asset_sha256": "e" * 64, "preprocess_identity": "v1", "arrival_record": {"k": 5},
            }],
        })
        state["transaction"] = {"id": "window-2", "phase": "collecting", "protection_ids": ["img1", "img5"],
                                "pre_signals": {"img1": [0.1, 0.2]}, "rw_decision": None}
        info = self.store.save(state)
        recovered = self.store.recover()
        self.assertEqual(recovered.state, state)
        self.assertEqual(recovered.scientific_cursors, scientific_cursors_from_state(state))
        self.assertEqual(info.generation, recovered.info.generation)
        self.assertEqual(recovered.state["window"]["image_ids"], ["img5", "img6"])

    def test_every_publication_interruption_boundary(self):
        for point in SAVE_FAILURE_POINTS:
            with self.subTest(point=point), tempfile.TemporaryDirectory() as temp:
                store = DurableRunStore(Path(temp), IDENTITY)
                old, new = example_state(4, 1), example_state(8, 2)
                store.save(old)
                def crash(name):
                    if name == point:
                        raise InjectedCrash(name)
                store.failure_injector = crash
                with self.assertRaises(InjectedCrash):
                    store.save(new)
                reopened = DurableRunStore(Path(temp), IDENTITY)
                result = reopened.recover()
                before_publication = point in SAVE_FAILURE_POINTS[:3]
                self.assertEqual(result.state, old if before_publication else new)
                # Pointer damage/absence is not a new source of scientific truth.
                (Path(temp) / "CURRENT").write_text("broken hint", encoding="utf-8")
                self.assertEqual(reopened.recover().state, result.state)

    def test_first_unpublished_generation_is_not_recoverable(self):
        self.store.failure_injector = lambda name: (_ for _ in ()).throw(InjectedCrash()) if name == "after_metadata_fsync" else None
        with self.assertRaises(InjectedCrash):
            self.store.save(example_state(4, 1))
        self.assertIsNone(self.store.recover())
        self.assertTrue(list((self.root / "checkpoints").glob(".pending-*")))

    def test_latest_state_corruption_falls_back_and_preserves_evidence(self):
        first = self.store.save(example_state(4, 1))
        second = self.store.save(example_state(8, 2))
        blob = second.path / "state.bin"
        blob.write_bytes(blob.read_bytes()[:-5] + b"wrong")
        recovered = self.store.recover()
        self.assertEqual(recovered.info.generation, first.generation)
        self.assertEqual(recovered.scientific_cursors["commits"], 1)
        self.assertEqual(recovered.rejected_generations[0]["generation"], second.generation)
        self.assertTrue(blob.exists())
        self.store.save(example_state(8, 2))
        self.assertEqual(self.store.recover().info.generation, 3)

    def test_metadata_corruption_falls_back(self):
        self.store.save(example_state(4, 1))
        newest = self.store.save(example_state(8, 2))
        (newest.path / "metadata.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.store.recover().scientific_cursors["observed"], 4)

    def test_all_corrupt_is_fail_closed_not_fresh_run(self):
        info = self.store.save(example_state(4, 1))
        (info.path / "state.bin").write_bytes(b"broken")
        with self.assertRaises(CheckpointCorruption):
            self.store.recover()

    def test_checksum_is_checked_before_deserialization(self):
        info = self.store.save(example_state(4, 1))
        (info.path / "state.bin").write_bytes(b"not-the-saved-state")
        called = []
        def forbidden(data):
            called.append(data)
            raise AssertionError("must never load a checksum-invalid blob")
        with self.assertRaises(CheckpointCorruption):
            self.store.recover(deserializer=forbidden)
        self.assertEqual(called, [])

    def test_identity_mismatch_rejected(self):
        self.store.save(example_state(4, 1))
        for key in IDENTITY:
            changed = dict(IDENTITY, **{key: "different"})
            with self.subTest(key=key), self.assertRaises(IdentityMismatch):
                DurableRunStore(self.root, changed)
        wrong_state = example_state(8, 2)
        wrong_state["identity"]["source_sha256"] = "f" * 64
        with self.assertRaises(IdentityMismatch):
            self.store.save(wrong_state)

    def test_foreign_generation_not_silently_ignored(self):
        self.store.save(example_state(4, 1))
        newest = self.store.save(example_state(8, 2))
        metadata_path = newest.path / "metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["identity"]["run_id"] = "foreign"
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        metadata_path.write_bytes(encoded)
        (newest.path / "metadata.sha256").write_text(hashlib.sha256(encoded).hexdigest(), encoding="ascii")
        with self.assertRaises(IdentityMismatch):
            self.store.recover()

    def test_commit_in_log_is_not_a_durable_commit(self):
        self.store.save(example_state(4, 1))
        self.store.append_event("commit", {"observed": 8, "commits": 2}, event_id="unpublished-commit")
        recovered = self.store.recover()
        self.assertEqual(recovered.scientific_cursors["next_index"], 4)
        self.assertEqual(recovered.scientific_cursors["commits"], 1)
        self.assertEqual(len(self.store.read_journal()[0]), 1)

    def test_durable_cursors_are_not_recounted_and_recover_is_idempotent(self):
        state = example_state(8, 2)
        self.store.save(state)
        first = self.store.recover()
        second = self.store.recover()
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.info, second.info)
        self.assertEqual(self.store.read_journal()[0], [])
        # Starting from the durable next_index yields only the unseen suffix.
        self.assertEqual(list(range(first.scientific_cursors["next_index"], 12)), [8, 9, 10, 11])
        with self.assertRaises(ValueError):
            self.store.save(example_state(4, 1))

    def test_work_recomputation_is_separate_from_science(self):
        self.store.save(example_state(4, 1))
        first = self.store.begin_work("candidate", {"protect": 24, "adapt": 4}, logical_key="window-2")
        self.store.finish_work(first, completed_forwards={"protect": 24, "adapt": 4}, compute_seconds=2.5)
        # No new complete checkpoint: all this completed work must be recomputed.
        recovered = self.store.recover()
        replay = self.store.begin_work("candidate", {"protect": 24, "adapt": 4}, logical_key="window-2", replay_of=first)
        self.store.finish_work(replay, completed_forwards={"protect": 24, "adapt": 4}, compute_seconds=2.6)
        self.store.record_queue_wait(120.0, reason="shared host busy", event_id="wait-1")
        self.store.save(example_state(8, 2))
        summary = self.store.work_summary()
        self.assertEqual(recovered.scientific_cursors["commits"], 1)
        self.assertEqual(self.store.recover().scientific_cursors["commits"], 2)
        self.assertEqual(summary["forwards_confirmed_completed"], {"protect": 48, "adapt": 8})
        self.assertEqual(summary["replay_forwards_confirmed_completed"], {"protect": 24, "adapt": 4})
        self.assertEqual(summary["queue_wait_seconds_recorded"], 120.0)
        self.assertAlmostEqual(summary["compute_seconds_confirmed"], 5.1)

    def test_crash_work_is_unknown_not_falsely_exact(self):
        unknown = self.store.begin_work("protection", {"HE": 24}, logical_key="window-1")
        replay = self.store.begin_work("protection", {"HE": 24}, logical_key="window-1", replay_of=unknown)
        self.store.finish_work(replay, completed_forwards={"HE": 24}, compute_seconds=1.0)
        summary = self.store.work_summary()
        self.assertEqual(summary["forwards_execution_lower_bound"], {"HE": 24})
        self.assertEqual(summary["forwards_execution_upper_bound"], {"HE": 48})
        self.assertEqual(summary["inflight_unknown_attempt_ids"], [unknown])
        self.assertTrue(summary["compute_time_incomplete"])

    def test_idempotent_journal_ack_does_not_double_charge(self):
        attempt = self.store.begin_work("score", {"HE": 1}, logical_key="obs-1", attempt_id="fixed-ack")
        self.store.begin_work("score", {"HE": 1}, logical_key="obs-1", attempt_id=attempt)
        args = dict(completed_forwards={"HE": 1}, compute_seconds=0.1)
        self.store.finish_work(attempt, **args)
        self.store.finish_work(attempt, **args)
        self.assertEqual(self.store.work_summary()["forwards_confirmed_completed"], {"HE": 1})
        with self.assertRaises(ValueError):
            self.store.finish_work(attempt, completed_forwards={"HE": 1}, compute_seconds=0.2)

    def test_torn_journal_tail_preserved_and_new_segment_linked(self):
        self.store.append_event("note", {"value": 1}, event_id="one")
        first_path = next((self.root / "journal").glob("*.jsonl"))
        with first_path.open("ab") as handle:
            handle.write(b'{"event_id":"unconfirmed')
        before = first_path.read_bytes()
        self.store.append_event("note", {"value": 2}, event_id="two")
        records, diagnostics = self.store.read_journal()
        self.assertEqual([r["sequence"] for r in records], [1, 2])
        self.assertEqual(first_path.read_bytes(), before)
        self.assertEqual(len(list((self.root / "journal").glob("*.jsonl"))), 2)
        self.assertEqual(len(diagnostics), 1)

    def test_boundary_tracks_complete_prefix_not_torn_work(self):
        self.assertEqual(self.store.journal_boundary()["sequence"], 0)
        first = self.store.append_event("note", {"x": 1})
        good = self.store.journal_boundary()
        path = next((self.root / "journal").glob("*.jsonl"))
        with path.open("ab") as handle:
            handle.write(b'{"incomplete":')
        self.assertEqual(self.store.journal_boundary(), good)
        state = example_state(4, 1)
        state["ledger_boundary"] = good
        self.store.save(state)
        self.assertEqual(self.store.recover().state["ledger_boundary"]["sha256"], first["sha256"])

    def test_complete_journal_corruption_fails_closed(self):
        self.store.append_event("note", {"value": 1})
        path = next((self.root / "journal").glob("*.jsonl"))
        path.write_bytes(path.read_bytes().replace(b'"value":1', b'"value":9'))
        with self.assertRaises(JournalCorruption):
            self.store.read_journal()

    def test_custom_serialization_and_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            def validate(state):
                if set(state) != {"custom_state", "whole_runtime"}:
                    raise ValueError("custom full contract not met")
            custom = DurableRunStore(temp, IDENTITY, state_validator=validate)
            state = {"custom_state": [1, 2], "whole_runtime": {"model_and_rng": "opaque-test"}}
            cursors = {"observed": 4, "candidates": 1, "commits": 1, "rollbacks": 0, "next_index": 4}
            custom.save(state, cursors, serializer=lambda s: json.dumps(s).encode())
            self.assertEqual(custom.recover(deserializer=json.loads).state, state)

    def test_schema_rejects_incomplete_runtime(self):
        for key in ("pools", "window", "pending_observations", "rng", "transaction", "ledger_boundary"):
            state = example_state()
            del state[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.save(state)
        state = example_state(1)
        state["window"]["image_ids"] = ["x"]
        with self.assertRaises(ValueError):
            self.store.save(state)
        state = example_state(4, 1)
        state["sampling"]["next_index"] = 3
        with self.assertRaises(ValueError):
            self.store.save(state)

    def test_future_or_mismatched_scientific_cursors_rejected(self):
        cursors = scientific_cursors_from_state(example_state(4, 1))
        cursors["commits"] = 2
        with self.assertRaises(ValueError):
            self.store.save(example_state(4, 1), cursors)

    def test_writer_exclusion(self):
        with self.store._lock():
            with self.assertRaises(ConcurrentWriter):
                self.store.append_event("note", {"x": 1})

    def test_negative_or_nonfinite_work_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.begin_work("score", {"HE": -1}, logical_key="obs")
        attempt = self.store.begin_work("score", {"HE": 1}, logical_key="obs")
        with self.assertRaises(ValueError):
            self.store.finish_work(attempt, completed_forwards={"HE": 2}, compute_seconds=0.1)
        with self.assertRaises(ValueError):
            self.store.record_queue_wait(float("nan"), reason="bad")


if __name__ == "__main__":
    unittest.main(verbosity=2)
