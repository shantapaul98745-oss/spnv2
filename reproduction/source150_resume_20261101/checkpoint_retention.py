"""Bounded native checkpoint publication. This module has no torch dependency.

The scientific writer supplies the unchanged native payload and serializer.
Only checkpoint.pth.tar is intercepted. Immutable hard links retain its previous
version; all writes create new files, so a retained inode is never overwritten.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

GIB = 1024 ** 3
SCHEMA = "spnv2-source150-native-retention-v1"
FIXED_LIMITS = {
    "max_checkpoint_bytes": GIB,
    "max_run_bytes": 4 * GIB,
    "max_total_bytes": 40 * GIB,
    "min_free_bytes": 20 * GIB,
}


class RetentionError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fsync_directory(path):
    if os.name == "nt":
        return
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def usage(root):
    """Count unique immutable inodes once, including pending failure evidence."""
    root = Path(root)
    seen = set()
    total = 0
    for path in root.rglob("*") if root.exists() else ():
        if path.is_symlink():
            raise RetentionError("Unexpected symlink in controlled output tree: " + str(path))
        if path.is_file():
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                total += stat.st_size
                seen.add(key)
    return total


def validate_policy(policy):
    if policy.get("schema") != SCHEMA:
        raise RetentionError("A versioned native retention policy is required")
    if policy.get("retain_latest") != 2:
        raise RetentionError("Exactly two complete ordinary checkpoints are required")
    for key, fixed in FIXED_LIMITS.items():
        if policy.get(key) != fixed:
            raise RetentionError("Fixed storage limit differs: " + key)
    run = Path(policy["run_output"])
    total = Path(policy["total_output_root"])
    if not run.is_absolute() or not total.is_absolute() or run == total or total not in run.parents:
        raise RetentionError("Run output must lie strictly inside the owned total output root")
    if run.resolve() != run or total.resolve() != total:
        raise RetentionError("Output path contains a symlink or noncanonical component")
    return policy


class NativeCheckpointRetention:
    def __init__(self, policy, *, disk_usage=shutil.disk_usage):
        self.policy = validate_policy(dict(policy))
        self.root = Path(policy["run_output"])
        self.total = Path(policy["total_output_root"])
        self.current = self.root / "checkpoint.pth.tar"
        self.previous = self.root / "checkpoint.previous.pth.tar"
        self.metadata = self.root / "checkpoint_retention.json"
        self.journal = self.root / "checkpoint_retention.jsonl"
        self.disk_usage = disk_usage

    def event(self, kind, **fields):
        value = dict(event=kind, time_ns=str(time.time_ns()), **fields)
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _limits(self):
        run_bytes = usage(self.root)
        total_bytes = usage(self.total)
        probe = self.root
        while not probe.exists():
            probe = probe.parent
        free = self.disk_usage(probe).free
        reserve = self.policy["max_checkpoint_bytes"]
        evidence = dict(run_bytes=run_bytes, total_bytes=total_bytes, free_bytes=free,
                        incoming_reserved_bytes=reserve)
        if (run_bytes + reserve > self.policy["max_run_bytes"] or
                total_bytes + reserve > self.policy["max_total_bytes"] or
                free < self.policy["min_free_bytes"] + reserve):
            if self.root.exists():
                self.event("storage_limit_refused_before_write", **evidence)
            raise RetentionError("Storage estimate or disk safety reserve exceeded: " + json.dumps(evidence))
        return evidence

    def preflight(self):
        """No output directory is created: the original trainer owns creation."""
        return self._limits()

    @contextlib.contextmanager
    def _lock(self):
        path = self.root / ".checkpoint-retention.lock"
        with path.open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def _validate_existing(self):
        for path in (self.current, self.previous, self.metadata):
            if path.is_symlink():
                raise RetentionError("Checkpoint symlink is forbidden: " + str(path))
        if not self.metadata.exists():
            if self.current.exists() or self.previous.exists():
                raise RetentionError("Existing checkpoint has no bound retention metadata")
            return None
        old = json.loads(self.metadata.read_text())
        if old.get("policy") != self.policy:
            raise RetentionError("Checkpoint retention policy identity changed")
        for name in ("current", "previous"):
            row = old.get(name)
            path = self.current if name == "current" else self.previous
            if row is None:
                if path.exists():
                    raise RetentionError("Unrecorded checkpoint exists: " + str(path))
            elif (not path.is_file() or path.stat().st_size != row["bytes"] or
                  sha256(path) != row["sha256"]):
                raise RetentionError("Retained checkpoint checksum mismatch: " + str(path))
        return old

    def save(self, value, path, writer):
        path = Path(path)
        if path != self.current:
            return writer(value, path)
        if not self.root.is_dir():
            raise RetentionError("Original trainer must own the output directory before saving")
        with self._lock():
            old = self._validate_existing()
            evidence = self._limits()
            token = uuid.uuid4().hex
            pending = self.root / (".checkpoint.pending-" + token + ".pth.tar")
            previous_link = self.root / (".checkpoint.previous-" + token + ".tmp")
            self.event("checkpoint_write_intent", pending=str(pending), **evidence)
            try:
                # Original serializer, payload and checkpoint format stay unchanged.
                writer(value, pending)
                if pending.is_symlink() or not pending.is_file():
                    raise RetentionError("Native writer did not publish a regular checkpoint")
                size = pending.stat().st_size
                if size > self.policy["max_checkpoint_bytes"]:
                    raise RetentionError("Actual checkpoint exceeds the reserved file bound")
                # Explicit fsync plus readback SHA precede changing either recovery path.
                with pending.open("r+b") as stream:
                    os.fsync(stream.fileno())
                digest = sha256(pending)
                if pending.stat().st_size != size:
                    raise RetentionError("Checkpoint changed during checksum validation")
                previous = None
                if old is not None:
                    previous = dict(old["current"])
                    self.event("previous_rotation_intent", previous_replaced=old.get("previous"),
                               retained_current=previous, new_sha256=digest)
                    os.link(self.current, previous_link)
                    os.replace(previous_link, self.previous)
                    fsync_directory(self.root)
                os.replace(pending, self.current)
                fsync_directory(self.root)
                record = dict(schema=SCHEMA, policy=self.policy,
                              current=dict(path=str(self.current), bytes=size, sha256=digest),
                              previous=({**previous, "path": str(self.previous)} if previous else None))
                atomic_json(self.metadata, record)
                self.event("checkpoint_published", current=record["current"], previous=record["previous"],
                           run_bytes=usage(self.root), total_bytes=usage(self.total))
                return None
            except BaseException as exc:
                # Keep incomplete/pending evidence; never infer success or retry silently.
                self.event("checkpoint_write_failed", pending=str(pending),
                           pending_exists=pending.exists(), error=type(exc).__name__ + ": " + str(exc))
                raise
