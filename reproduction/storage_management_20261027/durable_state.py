"""Complete checkpoint generations and a separate, durable work journal.

This module has no model, torch, GPU, or process-management dependency.  Its
default pickle loader is ONLY for this run's own trusted checkpoint directory.
The checksum is checked before deserialization; it is integrity detection, not
authentication of an untrusted pickle.

A published ``generation-*`` directory is the scientific authority.  CURRENT
is only a hint, and journal events (even events called ``commit``) never advance
scientific cursors.  A crash may leave unpublished ``.pending-*`` evidence.

The work journal is independent of scientific state.  Reserve work BEFORE
dispatch and finish it AFTER completion.  Hardware execution and a disk write
cannot be atomic: unfinished reservations are reported as unknown, with lower
and upper bounds, rather than invented exact forward counts.  Recomputed work
gets a new attempt ID and is charged again.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping


FORMAT_VERSION = 1
GENERATION_RE = re.compile(r"generation-(\d{20})$")
SEGMENT_RE = re.compile(r"segment-(\d{20})\.jsonl$")
REQUIRED_IDENTITY = ("run_id", "source_sha256", "config_sha256", "manifest_sha256")
CURSOR_NAMES = ("observed", "candidates", "commits", "rollbacks", "next_index")
SAVE_FAILURE_POINTS = (
    "after_state_fsync", "after_metadata_fsync", "after_pending_directory_fsync",
    "after_generation_rename", "after_generation_parent_fsync", "after_current_fsync",
    "after_current_replace", "after_current_parent_fsync",
)


class DurableStateError(RuntimeError):
    pass


class IdentityMismatch(DurableStateError):
    pass


class CheckpointCorruption(DurableStateError):
    pass


class JournalCorruption(DurableStateError):
    pass


class ConcurrentWriter(DurableStateError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _mapping(value: Any, name: str, required: tuple[str, ...]) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(f"{name} missing required state: {missing}")
    return value


def validate_run_state(state: Any) -> None:
    """Default complete-state contract; a caller may inject its own validator.

    Opaque learning objects and RNG states are deliberately not inspected.
    Their tensor/state equality belongs to the model adapter's independent
    continuous-vs-resumed verification.  Presence alone is not that proof.
    """
    state = _mapping(state, "state", (
        "schema_version", "identity", "learning", "pools", "window",
        "pending_observations", "counters", "sampling", "rng", "transaction",
        "ledger_boundary",
    ))
    if state["schema_version"] != 1:
        raise ValueError("unsupported complete-state schema")
    _mapping(state["identity"], "identity", REQUIRED_IDENTITY)
    _mapping(state["learning"], "learning", (
        "model", "optimizer", "buffers", "modes", "gradients", "caches", "scaler",
    ))
    pools = _mapping(state["pools"], "pools", ("uniform", "vulnerable"))
    for name in ("uniform", "vulnerable"):
        pool = _mapping(pools[name], f"pools.{name}", (
            "slots", "candidate_count", "replacement_count", "rng_state",
        ))
        if not isinstance(pool["slots"], list) or len(pool["slots"]) > 4:
            raise ValueError(f"pools.{name}.slots must be a list of at most four records")
        for slot in pool["slots"]:
            _mapping(slot, f"pools.{name}.slot", (
                "image_id", "observation_index", "asset_sha256", "preprocess_identity",
                "arrival_record",
            ))
        count = _nonnegative_int(pool["candidate_count"], f"pools.{name}.candidate_count")
        replacement = _nonnegative_int(pool["replacement_count"], f"pools.{name}.replacement_count")
        if count < len(pool["slots"]) or replacement > count:
            raise ValueError(f"pools.{name} counts cannot explain its contents")
    window = _mapping(state["window"], "window", ("image_ids", "rollback_state"))
    if not isinstance(window["image_ids"], list) or len(window["image_ids"]) > 4:
        raise ValueError("window.image_ids must contain at most four IDs")
    if window["image_ids"] and window["rollback_state"] is None:
        raise ValueError("a partial/full window requires its complete rollback baseline")
    if not isinstance(state["pending_observations"], list):
        raise ValueError("pending_observations must retain ordered records")
    for record in state["pending_observations"]:
        _mapping(record, "pending_observation", ("image_id", "observation_index", "arrival_record"))
    counters = _mapping(state["counters"], "counters", (
        "observed", "accepted", "candidates", "commits", "rollbacks", "tail_discarded",
        "consecutive_rollbacks", "max_consecutive_rollbacks",
    ))
    for name, value in counters.items():
        _nonnegative_int(value, f"counters.{name}")
    if counters["accepted"] > counters["observed"]:
        raise ValueError("accepted observations exceed observed observations")
    resolved = counters["commits"] + counters["rollbacks"]
    if not resolved <= counters["candidates"] <= resolved + 1:
        raise ValueError("candidate count must contain resolved windows and at most one pending transaction")
    if counters["consecutive_rollbacks"] > counters["max_consecutive_rollbacks"]:
        raise ValueError("consecutive rollback maximum is inconsistent")
    sampling = _mapping(state["sampling"], "sampling", ("next_index", "manifest_sha256"))
    if _nonnegative_int(sampling["next_index"], "sampling.next_index") != counters["observed"]:
        raise ValueError("observed must equal next_index at the saved processing boundary")
    if sampling["manifest_sha256"] != state["identity"]["manifest_sha256"]:
        raise ValueError("sampling manifest does not match complete-state identity")
    _mapping(state["rng"], "rng", ("compute", "data", "uniform_pool", "vulnerable_pool", "random_control"))
    transaction = _mapping(state["transaction"], "transaction", (
        "id", "phase", "protection_ids", "pre_signals", "rw_decision",
    ))
    if not isinstance(transaction["phase"], str) or not transaction["phase"]:
        raise ValueError("transaction phase must be explicit")
    if not isinstance(transaction["protection_ids"], list):
        raise ValueError("transaction protection IDs must retain exact order")
    boundary = _mapping(state["ledger_boundary"], "ledger_boundary", ("sequence", "byte_offset"))
    _nonnegative_int(boundary["sequence"], "ledger_boundary.sequence")
    _nonnegative_int(boundary["byte_offset"], "ledger_boundary.byte_offset")


def scientific_cursors_from_state(state: Mapping) -> dict[str, int]:
    return {
        "observed": state["counters"]["observed"],
        "candidates": state["counters"]["candidates"],
        "commits": state["counters"]["commits"],
        "rollbacks": state["counters"]["rollbacks"],
        "next_index": state["sampling"]["next_index"],
    }


def _validate_cursors(cursors: Mapping) -> dict[str, int]:
    _mapping(cursors, "scientific_cursors", CURSOR_NAMES)
    result = {name: _nonnegative_int(value, f"scientific_cursors.{name}")
              for name, value in cursors.items()}
    if result["observed"] != result["next_index"]:
        raise ValueError("scientific observed and next_index must be equal")
    if result["commits"] + result["rollbacks"] > result["candidates"]:
        raise ValueError("scientific resolved windows exceed candidate windows")
    return result


def _fsync_directory(path: Path) -> None:
    """POSIX directory durability barrier; Windows lacks a stdlib equivalent.

    File fsync and os.replace still apply on Windows, but power-loss directory
    persistence cannot be promised by this stdlib implementation there.  The
    intended Linux execution host supports the additional directory barriers.
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class CheckpointInfo:
    generation: int
    path: Path
    metadata: dict

    @property
    def scientific_cursors(self) -> dict[str, int]:
        return dict(self.metadata["scientific_cursors"])


@dataclass(frozen=True)
class RecoveredCheckpoint:
    state: Any
    info: CheckpointInfo
    rejected_generations: tuple[dict, ...]

    @property
    def scientific_cursors(self) -> dict[str, int]:
        return self.info.scientific_cursors


class DurableRunStore:
    """One trusted run's complete-state store with advisory writer exclusion.

    Identity requires run_id/source_sha256/config_sha256/manifest_sha256; add
    code/calibration/rule identities as appropriate.  All keys are compared.
    The checkpoint serializer may be replaced with a torch-aware bytes callback.
    A custom state validator must check the caller's *complete* runtime schema.
    """

    def __init__(self, root: str | Path, identity: Mapping[str, str], *,
                 state_validator: Callable[[Any], None] = validate_run_state,
                 failure_injector: Callable[[str], None] | None = None,
                 retention_policy: Mapping[str, Any] | None = None):
        _mapping(identity, "identity", REQUIRED_IDENTITY)
        if any(not isinstance(k, str) or not isinstance(v, str) or not v
               for k, v in identity.items()):
            raise ValueError("identity keys and values must be nonempty strings")
        self.root = Path(root)
        self.identity = dict(identity)
        self.identity_fingerprint = _sha(_json_bytes(self.identity))
        self.state_validator = state_validator
        self.failure_injector = failure_injector
        self._journal_cache = None
        self._journal_event_index = {}
        self.generations = self.root / "checkpoints"
        self.journal = self.root / "journal"
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations.mkdir(exist_ok=True)
        self.journal.mkdir(exist_ok=True)
        with self._lock():
            import storage_policy
            storage_policy.initialize(self.root, retention_policy)
            identity_path = self.root / "identity.json"
            if identity_path.exists():
                try:
                    actual = json.loads(identity_path.read_bytes())
                except (OSError, ValueError) as exc:
                    raise IdentityMismatch("run identity record is unreadable") from exc
                if actual != self.identity:
                    raise IdentityMismatch("run/source/config/manifest or other identity differs")
            else:
                # Do not bless an orphaned pre-existing store with a new identity.
                if list(self.generations.iterdir()) or list(self.journal.iterdir()):
                    raise IdentityMismatch("existing run evidence has no authoritative identity")
                temp = self.root / f".identity-{uuid.uuid4().hex}"
                _write_fsynced(temp, _json_bytes(self.identity))
                os.replace(temp, identity_path)
                _fsync_directory(self.root)
                _fsync_directory(self.root.parent)

    @contextlib.contextmanager
    def _lock(self):
        path = self.root / ".writer.lock"
        handle = path.open("a+b")
        locked = False
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                raise ConcurrentWriter("another writer owns this run store") from exc
            yield
        finally:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _point(self, name: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(name)

    def _paths(self) -> list[tuple[int, Path]]:
        return sorted(((int(m.group(1)), path) for path in self.generations.iterdir()
                       if (m := GENERATION_RE.fullmatch(path.name))), reverse=True)

    def save(self, state: Any, scientific_cursors: Mapping[str, int] | None = None, *,
             serializer: Callable[[Any], bytes] = pickle.dumps) -> CheckpointInfo:
        """Atomically publish a complete generation; never infer commits from WAL.

        The caller owns state snapshot consistency and must not mutate tensors
        during serialization.  This API does not make a shallow tensor copy safe.
        """
        self.state_validator(state)
        if isinstance(state, Mapping) and "identity" in state and state["identity"] != self.identity:
            raise IdentityMismatch("state identity differs from store identity")
        cursors = _validate_cursors(scientific_cursors if scientific_cursors is not None
                                    else scientific_cursors_from_state(state))
        if self.state_validator is validate_run_state and cursors != scientific_cursors_from_state(state):
            raise ValueError("metadata scientific cursors differ from the complete state")
        data = serializer(state)
        if not isinstance(data, bytes):
            raise TypeError("serializer must return bytes")
        with self._lock():
            import storage_policy
            storage_policy.before_write(self.root, len(data))
            paths = self._paths()
            # Corrupt latest generations may be bypassed, but a valid durable
            # cursor must never be moved backwards by an accidental replay.
            previous = self._recover_unlocked(pickle.loads, deserialize=False)
            if previous is not None:
                old = previous.info.scientific_cursors
                for key, old_value in old.items():
                    if key not in cursors or cursors[key] < old_value:
                        raise ValueError(f"scientific cursor {key} would move backwards")
            generation = (paths[0][0] if paths else 0) + 1
            pending = self.generations / f".pending-{generation:020d}-{uuid.uuid4().hex}"
            published = self.generations / f"generation-{generation:020d}"
            pending.mkdir()
            _write_fsynced(pending / "state.bin", data)
            self._point("after_state_fsync")
            records, diagnostics = self._read_journal_unlocked()
            metadata = {
                "format_version": FORMAT_VERSION,
                "generation": generation,
                "created_time_ns": time.time_ns(),
                "identity": self.identity,
                "identity_fingerprint": self.identity_fingerprint,
                "state_sha256": _sha(data),
                "state_bytes": len(data),
                "scientific_cursors": cursors,
                "journal_sequence_at_save": records[-1]["sequence"] if records else 0,
                "journal_tail_sha256_at_save": records[-1]["sha256"] if records else None,
                "journal_torn_tail_count_at_save": len(diagnostics),
                "directory_fsync_supported": os.name != "nt",
            }
            metadata_bytes = _json_bytes(metadata)
            _write_fsynced(pending / "metadata.json", metadata_bytes)
            _write_fsynced(pending / "metadata.sha256", (_sha(metadata_bytes) + "\n").encode("ascii"))
            self._point("after_metadata_fsync")
            _fsync_directory(pending)
            self._point("after_pending_directory_fsync")
            os.rename(pending, published)
            self._point("after_generation_rename")
            _fsync_directory(self.generations)
            self._point("after_generation_parent_fsync")
            current_temp = self.root / f".CURRENT-{uuid.uuid4().hex}"
            _write_fsynced(current_temp, _json_bytes({
                "generation": generation, "metadata_sha256": _sha(metadata_bytes),
            }))
            self._point("after_current_fsync")
            os.replace(current_temp, self.root / "CURRENT")
            self._point("after_current_replace")
            _fsync_directory(self.root)
            self._point("after_current_parent_fsync")
            storage_policy.after_publish(self)
            return CheckpointInfo(generation, published, metadata)

    def _read_generation(self, number: int, path: Path, deserializer: Callable[[bytes], Any],
                         *, deserialize: bool) -> tuple[Any, CheckpointInfo]:
        try:
            metadata_bytes = (path / "metadata.json").read_bytes()
            expected = (path / "metadata.sha256").read_text("ascii").strip()
            if _sha(metadata_bytes) != expected:
                raise CheckpointCorruption("metadata checksum mismatch")
            metadata = json.loads(metadata_bytes)
            if metadata["identity"] != self.identity or metadata["identity_fingerprint"] != self.identity_fingerprint:
                raise IdentityMismatch(f"generation {number} belongs to another run/protocol")
            if metadata["format_version"] != FORMAT_VERSION or metadata["generation"] != number:
                raise CheckpointCorruption("generation/version mismatch")
            _validate_cursors(metadata["scientific_cursors"])
            blob = (path / "state.bin").read_bytes()
            if len(blob) != metadata["state_bytes"] or _sha(blob) != metadata["state_sha256"]:
                raise CheckpointCorruption("state checksum or length mismatch")
            state = deserializer(blob) if deserialize else None
            if deserialize:
                self.state_validator(state)
                if isinstance(state, Mapping) and "identity" in state and state["identity"] != self.identity:
                    raise IdentityMismatch("deserialized state has a different identity")
                if self.state_validator is validate_run_state:
                    if scientific_cursors_from_state(state) != metadata["scientific_cursors"]:
                        raise CheckpointCorruption("scientific cursors differ from complete state")
            return state, CheckpointInfo(number, path, metadata)
        except IdentityMismatch:
            raise
        except Exception as exc:
            if isinstance(exc, CheckpointCorruption):
                raise
            raise CheckpointCorruption(f"invalid complete generation: {type(exc).__name__}: {exc}") from exc

    def _recover_unlocked(self, deserializer: Callable[[bytes], Any], *, deserialize: bool = True):
        rejected = []
        for number, path in self._paths():
            try:
                state, info = self._read_generation(number, path, deserializer, deserialize=deserialize)
                return RecoveredCheckpoint(state, info, tuple(rejected))
            except CheckpointCorruption as exc:
                rejected.append({"generation": number, "path": str(path), "reason": str(exc)})
        if rejected:
            raise CheckpointCorruption(f"no complete recoverable checkpoint remains: {rejected}")
        return None

    def recover(self, *, deserializer: Callable[[bytes], Any] = pickle.loads) -> RecoveredCheckpoint | None:
        """Read the newest complete generation; leave corrupt evidence untouched."""
        with self._lock():
            return self._recover_unlocked(deserializer)

    def _segments(self) -> list[tuple[int, Path]]:
        return sorted((int(m.group(1)), path) for path in self.journal.iterdir()
                      if (m := SEGMENT_RE.fullmatch(path.name)))

    def _read_journal_unlocked(self) -> tuple[list[dict], list[dict]]:
        segments = self._segments()
        signature = self._journal_signature(segments)
        if self._journal_cache is not None and self._journal_cache[0] == signature:
            return self._journal_cache[1], self._journal_cache[2]
        records, diagnostics = [], []
        previous_hash = None
        seen_ids = set()
        for segment_number, path in segments:
            raw = path.read_bytes()
            lines = raw.splitlines(keepends=True)
            offset = 0
            for index, line in enumerate(lines):
                if not line.endswith(b"\n"):
                    if index != len(lines) - 1:
                        raise JournalCorruption("nonterminal incomplete journal record")
                    diagnostics.append({"segment": segment_number, "byte_offset": offset,
                                        "unconfirmed_tail_bytes": len(line), "tail_sha256": _sha(line)})
                    break
                try:
                    record = json.loads(line)
                    checksum = record["sha256"]
                    unsigned = {key: value for key, value in record.items() if key != "sha256"}
                    if _sha(_json_bytes(unsigned)) != checksum:
                        raise ValueError("checksum mismatch")
                    if record["identity_fingerprint"] != self.identity_fingerprint:
                        raise IdentityMismatch("journal identity differs")
                    if record["sequence"] != len(records) + 1 or record["previous_sha256"] != previous_hash:
                        raise ValueError("journal chain/sequence mismatch")
                    if record["event_id"] in seen_ids:
                        raise ValueError("duplicate event ID")
                except IdentityMismatch:
                    raise
                except Exception as exc:
                    raise JournalCorruption(f"invalid durable journal record at {path}:{offset}: {exc}") from exc
                records.append(record)
                seen_ids.add(record["event_id"])
                previous_hash = checksum
                offset += len(line)
        self._journal_cache = (signature, records, diagnostics)
        self._journal_event_index = {r["event_id"]: r for r in records}
        return records, diagnostics

    @staticmethod
    def _journal_signature(segments):
        return tuple((number, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
                     for number, path in segments for stat in [path.stat()])

    def read_journal(self) -> tuple[list[dict], list[dict]]:
        with self._lock():
            # Callers may inspect/mutate the returned data without changing the
            # checked in-memory cache used by later append operations.
            return copy.deepcopy(self._read_journal_unlocked())

    def journal_boundary(self) -> dict:
        """Return an exact audit boundary suitable for complete-state capture.

        ``byte_offset`` is the complete prefix in the named segment; a torn
        suffix is deliberately excluded.  Sequence/hash identify the chain
        across segment rotation.  This is audit context, never a commit cursor.
        """
        with self._lock():
            records, diagnostics = self._read_journal_unlocked()
            segments = self._segments()
            if not segments:
                return {"sequence": 0, "byte_offset": 0, "segment": None, "sha256": None}
            number, path = segments[-1]
            offset = path.stat().st_size
            if diagnostics and diagnostics[-1]["segment"] == number:
                offset = diagnostics[-1]["byte_offset"]
            return {"sequence": records[-1]["sequence"] if records else 0,
                    "byte_offset": offset, "segment": number,
                    "sha256": records[-1]["sha256"] if records else None}

    def _append_unlocked(self, kind: str, payload: Mapping, event_id: str | None = None) -> dict:
        if not isinstance(kind, str) or not kind or not isinstance(payload, Mapping):
            raise ValueError("journal kind and mapping payload are required")
        event_id = event_id or uuid.uuid4().hex
        records, diagnostics = self._read_journal_unlocked()
        old = self._journal_event_index.get(event_id)
        if old is not None:
            if old["kind"] != kind or old["payload"] != dict(payload):
                raise ValueError("event ID reused with different content")
            return copy.deepcopy(old)  # Retry a disk acknowledgment, not computation.
        segments = self._segments()
        segment_number = segments[-1][0] if segments else 1
        # A torn suffix is evidence, never silently truncated.  Continue in a
        # fresh segment linked to the last complete record.
        if diagnostics and diagnostics[-1]["segment"] == segment_number:
            segment_number += 1
        path = self.journal / f"segment-{segment_number:020d}.jsonl"
        is_new = not path.exists()
        record = {
            "sequence": len(records) + 1, "time_ns": time.time_ns(),
            "event_id": event_id, "kind": kind, "payload": dict(payload),
            "identity_fingerprint": self.identity_fingerprint,
            "previous_sha256": records[-1]["sha256"] if records else None,
        }
        record["sha256"] = _sha(_json_bytes(record))
        encoded = _json_bytes(record) + b"\n"
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if is_new:
            _fsync_directory(self.journal)
        # Preserve the already checked prefix instead of rereading the complete
        # journal for every model-call reservation.  File identity/size/mtime
        # changes invalidate it, including writes from another store instance.
        record = json.loads(encoded)
        records.append(record)
        self._journal_event_index[event_id] = record
        self._journal_cache = (self._journal_signature(self._segments()), records, diagnostics)
        return copy.deepcopy(record)

    def append_event(self, kind: str, payload: Mapping, *, event_id: str | None = None) -> dict:
        """Append audit evidence.  This NEVER changes scientific durable state."""
        with self._lock():
            return self._append_unlocked(kind, payload, event_id)

    def begin_work(self, kind: str, forwards: Mapping[str, int], *,
                   logical_key: str, replay_of: str | None = None,
                   attempt_id: str | None = None) -> str:
        """Durably reserve an upper bound before dispatch; new replay = new ID.

        Each category count is the maximum number of forwards this attempt may
        execute.  Splitting into small batches narrows crash-time uncertainty.
        Passing an existing attempt_id only retries the *reservation write*;
        it is never authorization to dispatch the same computation again.
        """
        if not kind or not logical_key:
            raise ValueError("work kind and logical key are required")
        counts = {str(k): _nonnegative_int(v, f"forwards.{k}") for k, v in forwards.items()}
        attempt_id = attempt_id or uuid.uuid4().hex
        with self._lock():
            self._read_journal_unlocked()
            if replay_of is not None and f"work-start:{replay_of}" not in self._journal_event_index:
                raise ValueError("replay_of must refer to an existing work attempt")
            self._append_unlocked("work_started", {
                "attempt_id": attempt_id, "work_kind": kind, "forwards_reserved": counts,
                "logical_key": logical_key, "replay_of": replay_of,
            }, event_id=f"work-start:{attempt_id}")
        return attempt_id

    def finish_work(self, attempt_id: str, *, completed_forwards: Mapping[str, int],
                    compute_seconds: float, status: str = "completed") -> dict:
        """Confirm executed work after required device synchronization, if any."""
        if status not in ("completed", "failed", "aborted"):
            raise ValueError("invalid work completion status")
        if not math.isfinite(compute_seconds) or compute_seconds < 0:
            raise ValueError("compute_seconds must be finite and nonnegative")
        counts = {str(k): _nonnegative_int(v, f"completed_forwards.{k}")
                  for k, v in completed_forwards.items()}
        with self._lock():
            self._read_journal_unlocked()
            start_record = self._journal_event_index.get(f"work-start:{attempt_id}")
            if start_record is None:
                raise ValueError("cannot finish an unreserved attempt")
            started = start_record["payload"]
            reserved = started["forwards_reserved"]
            if any(k not in reserved or value > reserved[k] for k, value in counts.items()):
                raise ValueError("completed forwards exceed the pre-dispatch reservation")
            return self._append_unlocked("work_finished", {
                "attempt_id": attempt_id, "completed_forwards": counts,
                "compute_seconds": float(compute_seconds), "status": status,
            }, event_id=f"work-finish:{attempt_id}")

    def record_queue_wait(self, seconds: float, *, reason: str, event_id: str | None = None) -> dict:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("queue wait must be finite and nonnegative")
        return self.append_event("queue_wait", {"seconds": float(seconds), "reason": reason}, event_id=event_id)

    def work_summary(self) -> dict:
        """Report compute separately from waiting and unresolved crash work."""
        records, diagnostics = self.read_journal()
        started = {r["payload"]["attempt_id"]: r["payload"]
                   for r in records if r["kind"] == "work_started"}
        finished = {r["payload"]["attempt_id"]: r["payload"]
                    for r in records if r["kind"] == "work_finished"}
        if set(finished) - set(started):
            raise JournalCorruption("finished work without reservation")
        confirmed, reserved_total, unknown, replay_confirmed = {}, {}, {}, {}
        for key, reservation in started.items():
            for category, value in reservation["forwards_reserved"].items():
                reserved_total[category] = reserved_total.get(category, 0) + value
            if key not in finished:
                for category, value in reservation["forwards_reserved"].items():
                    unknown[category] = unknown.get(category, 0) + value
            else:
                for category, value in finished[key]["completed_forwards"].items():
                    confirmed[category] = confirmed.get(category, 0) + value
                    if reservation["replay_of"] is not None:
                        replay_confirmed[category] = replay_confirmed.get(category, 0) + value
        categories = set(confirmed) | set(unknown)
        return {
            "attempts_started": len(started), "attempts_finished": len(finished),
            "inflight_unknown_attempt_ids": sorted(set(started) - set(finished)),
            "forwards_confirmed_completed": confirmed,
            "forwards_reserved_attempt_upper_bound": reserved_total,
            "forwards_inflight_unknown_upper_bound": unknown,
            "forwards_execution_lower_bound": {k: confirmed.get(k, 0) for k in sorted(categories)},
            "forwards_execution_upper_bound": {k: confirmed.get(k, 0) + unknown.get(k, 0)
                                                for k in sorted(categories)},
            "replay_forwards_confirmed_completed": replay_confirmed,
            "compute_seconds_confirmed": sum(p["compute_seconds"] for p in finished.values()),
            "compute_time_incomplete": bool(set(started) - set(finished)),
            "queue_wait_seconds_recorded": sum(r["payload"]["seconds"] for r in records if r["kind"] == "queue_wait"),
            "torn_journal_tails": diagnostics,
            "scientific_counts_source": "latest complete recoverable checkpoint only; never this journal",
        }
