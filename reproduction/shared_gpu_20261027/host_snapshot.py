"""Linux read-only collector for host_gate; no scheduling or process actions.

Example registry (owned by the caller UID and not group/world writable):
  {"schema":"spnv2.round2.registry.v1", "project_id":"odr003_round2_20261023",
   "records":[{"record_path":"/project/round2/launches/run.json",
                "record_sha256":"<host_gate.launch_record_sha256(full_record)>"}]}
An optional ``expected_record`` must hash to the same canonical digest. Every
record path must resolve under explicit project_root. Missing registry is not
treated as an empty registry. To start the first worker, create a trusted empty
registry explicitly before collecting evidence.

Root entrypoint integration:
  before = collect_snapshot(registry, project_root=root, project_id=project)
  with caller_owned_project_locks:
      after = collect_snapshot(registry, project_root=root, project_id=project)
      decision = assess_collected_launch_boundary(before, after,
                      project_lock_held=True, task_kind="gpu")
      # Caller may start inside decision.valid_until; this module never starts.

Other GPU PIDs receive only public nvidia-smi PID/name/memory classification.
Only exact registered PIDs are probed for worker identity. Their environment is
read through a pinned /proc directory handle only after boot/start/full command
and caller UID match. Only SPNV2_RUN_NONCE is retained from that environment.
The single known desktop executable may receive public stat/cmdline/status/exe
checks; its environment, cwd and underlying executable file are never read.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
import stat
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Mapping

import host_gate

REGISTRY_SCHEMA = "spnv2.round2.registry.v1"
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
MEMINFO_PATH = "/proc/meminfo"
NONCE_KEY = "SPNV2_RUN_NONCE"
MAX_READ_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_fd(fd: int) -> bytes:
    chunks = []
    total = 0
    while True:
        data = os.read(fd, min(65536, MAX_READ_BYTES + 1 - total))
        if not data:
            return b"".join(chunks)
        chunks.append(data); total += len(data)
        _require(total <= MAX_READ_BYTES, "read exceeds bounded metadata size")


class LinuxProcHandle:
    """A pinned proc directory avoids following a reused numeric PID to its env."""
    def __init__(self, fd: int):
        self.fd = fd

    def read_bytes(self, name: str) -> bytes:
        _require(name in {"stat", "cmdline", "status", "environ"}, "proc read outside allowlist")
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC, dir_fd=self.fd)
        try:
            return _read_fd(fd)
        finally:
            os.close(fd)

    def readlink(self, name: str) -> str:
        _require(name in {"exe", "cwd"}, "proc link outside allowlist")
        return os.readlink(name, dir_fd=self.fd)


class LinuxReadOnlyFS:
    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def read_bytes(self, path: str) -> bytes:
        _require(path in {BOOT_ID_PATH, MEMINFO_PATH}, "only public boot-id and meminfo paths are allowed here")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            return _read_fd(fd)
        finally:
            os.close(fd)

    def read_owned_bytes(self, path: str, expected_uid: int) -> bytes:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            _require(stat.S_ISREG(before.st_mode) and before.st_uid == expected_uid,
                     "registry/launch record is not a caller-owned regular file")
            _require(before.st_mode & 0o022 == 0, "registry/launch record is writable by another UID/group")
            data = _read_fd(fd)
            after = os.fstat(fd)
            _require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                     == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                     "registry/launch record changed while being read")
            return data
        finally:
            os.close(fd)

    @contextmanager
    def open_proc(self, pid: int):
        # O_PATH pins the original proc inode. Relative reads do not resolve a
        # subsequently recycled /proc/<pid> directory belonging to another task.
        fd = os.open("/proc/" + str(pid), os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            yield LinuxProcHandle(fd)
        finally:
            os.close(fd)


def system_command_runner(argv: list[str]) -> CommandResult:
    result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=15, check=False, shell=False)
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _json(data: bytes) -> dict[str, Any]:
    _require(len(data) <= MAX_READ_BYTES, "oversized JSON metadata")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("nonfinite JSON metadata")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    _require(isinstance(value, dict), "metadata must be a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def _inside(fs, path: str, root: str) -> str:
    resolved = PurePosixPath(fs.realpath(str(path)))
    boundary = PurePosixPath(fs.realpath(str(root)))
    _require(resolved.is_absolute() and boundary.is_absolute() and resolved.is_relative_to(boundary),
             "registry/launch path outside this project root")
    return str(resolved)


def _boot_id(fs) -> str:
    value = fs.read_bytes(BOOT_ID_PATH).decode("ascii").strip()
    _require(bool(value) and len(value) <= 128, "missing or invalid boot ID")
    return value



def _host_memory(fs) -> tuple[int, int]:
    """Public physical-host memory only; no process/private-file or model read."""
    values = {}
    for line in fs.read_bytes(MEMINFO_PATH).decode("ascii").splitlines():
        name, sep, rest = line.partition(":")
        if name not in {"MemAvailable", "MemTotal"}: continue
        _require(sep and name not in values, "Duplicate host memory field")
        parts = rest.split()
        _require(len(parts) == 2 and parts[0].isdigit() and parts[1] == "kB", "Invalid public host memory field")
        values[name] = int(parts[0])
    _require(set(values) == {"MemAvailable", "MemTotal"}
             and 0 <= values["MemAvailable"] <= values["MemTotal"] and values["MemTotal"] > 0,
             "MemAvailable/MemTotal missing or inconsistent")
    return values["MemAvailable"] // 1024, values["MemTotal"] // 1024


def _registry(fs, path: str, root: str, project_id: str, uid: int):
    path = _inside(fs, path, root)
    raw = fs.read_owned_bytes(path, uid)
    data = _json(raw)
    _require(data.get("schema") == REGISTRY_SCHEMA and data.get("project_id") == project_id,
             "registry schema/project mismatch")
    entries = data.get("records")
    _require(isinstance(entries, list), "registry records list missing")
    result = []; nonces = set()
    for entry in entries:
        _require(isinstance(entry, dict) and {"record_path", "record_sha256"} <= entry.keys(),
                 "incomplete registry entry")
        record_path = _inside(fs, entry["record_path"], root)
        observed = _json(fs.read_owned_bytes(record_path, uid))
        host_gate._validate_record(observed)
        _require(observed["project_id"] == project_id and observed["uid"] == uid,
                 "registered worker is not owned by this project/caller UID")
        _require(host_gate.launch_record_sha256(observed) == entry["record_sha256"], "launch record digest changed")
        expected = copy.deepcopy(entry.get("expected_record", observed))
        host_gate._validate_record(expected)
        _require(host_gate.launch_record_sha256(expected) == entry["record_sha256"], "expected complete launch record differs")
        _require(observed["run_nonce"] not in nonces, "duplicate registered run nonce")
        nonces.add(observed["run_nonce"])
        result.append({"record_path": record_path, "record_sha256": entry["record_sha256"],
                       "expected_record": expected, "observed_record": observed})
    return path, raw, result


def _stat(data: bytes, pid: int) -> dict[str, Any]:
    text = data.decode("utf-8", errors="strict")
    left, right = text.find("("), text.rfind(")")
    _require(left > 0 and right > left and int(text[:left].strip()) == pid, "proc stat PID mismatch")
    fields = text[right + 1:].split()
    _require(len(fields) >= 20, "proc stat truncated")
    starttime = int(fields[19])
    _require(starttime > 0, "invalid proc starttime")
    return {"pid": pid, "state": fields[0], "starttime": starttime}


def _command(data: bytes):
    _require(bool(data) and data.endswith(b"\x00"), "live command line empty or truncated")
    argv = [part.decode("utf-8", errors="strict") for part in data.split(b"\x00")[:-1]]
    digest = hashlib.sha256(data).hexdigest()
    _require(host_gate.command_sha256(argv) == digest, "full command-line reconstruction differs")
    return argv, digest


def _uids(data: bytes) -> tuple[int, ...]:
    lines = [line for line in data.decode("utf-8", errors="strict").splitlines() if line.startswith("Uid:")]
    _require(len(lines) == 1, "proc UID evidence missing")
    values = tuple(int(value) for value in lines[0].split()[1:])
    _require(len(values) == 4 and all(value >= 0 for value in values), "proc UID evidence malformed")
    return values


def _nonce(data: bytes) -> str | None:
    prefix = NONCE_KEY.encode("ascii") + b"="
    values = [entry[len(prefix):] for entry in data.split(b"\x00") if entry.startswith(prefix)]
    _require(len(values) <= 1, "duplicate run nonce environment key")
    return values[0].decode("ascii", errors="strict") if values else None


def _probe_worker(fs, entry: Mapping[str, Any], boot_id: str, uid: int) -> dict[str, Any]:
    expected = entry["expected_record"]
    check = {"expected_record": copy.deepcopy(expected), "observed_record": copy.deepcopy(entry["observed_record"]),
             "probe_succeeded": False, "process": None, "absence_confirmed": False}
    def absent(reason, evidence):
        check.update(probe_succeeded=True, absence_confirmed=True, absence_reason=reason, absence_evidence=evidence)
        return check
    if expected["boot_id"] != boot_id:
        return absent("record_from_previous_boot", {"current_boot_id": boot_id, "record_boot_id": expected["boot_id"]})
    pid = expected["pid"]
    try:
        with fs.open_proc(pid) as proc:
            try:
                first = _stat(proc.read_bytes("stat"), pid)
            except FileNotFoundError:
                return absent("proc_stat_absent", {"pid": pid, "boot_id": boot_id})
            if first["starttime"] != expected["starttime"]:
                return absent("PID_recycled_original_worker_gone", {**first, "expected_starttime": expected["starttime"], "boot_id": boot_id})
            if first["state"] in {"Z", "X"}:
                return absent("recorded_worker_terminal", {**first, "boot_id": boot_id})
            try:
                argv, digest = _command(proc.read_bytes("cmdline"))
                actual = {**first, "boot_id": boot_id, "argv": argv, "cmdline_sha256": digest,
                          "identity_complete": False, "run_nonce": None, "role": "unverified_registered_process"}
                check["process"] = actual
                # A mismatching public identity may now be somebody else's work.
                # Do not inspect its environment or cwd.
                if digest != expected["cmdline_sha256"]:
                    check["error"] = "public_command_mismatch_private_reads_skipped"
                    return check
                uids = _uids(proc.read_bytes("status")); actual["uid"] = uids[0]
                if any(value != uid for value in uids) or uids[0] != expected["uid"]:
                    check["error"] = "UID_mismatch_private_reads_skipped"
                    return check
                actual["run_nonce"] = _nonce(proc.read_bytes("environ"))
                # Never construct actual nonce from the expected launch record.
                if actual["run_nonce"] != expected["run_nonce"]:
                    actual["identity_complete"] = True
                    check.update(probe_succeeded=True, error="live_run_nonce_mismatch_cwd_skipped")
                    return check
                cwd = proc.readlink("cwd")
                actual["cwd"] = cwd
                if fs.realpath(cwd) != fs.realpath(expected["cwd"]):
                    check["error"] = "live_cwd_differs_from_launch"
                    return check
                second = _stat(proc.read_bytes("stat"), pid)
                argv_after, hash_after = _command(proc.read_bytes("cmdline"))
                _require(second["starttime"] == first["starttime"] and second["state"] not in {"Z", "X"}
                         and hash_after == digest and argv_after == argv and _uids(proc.read_bytes("status")) == uids,
                         "live process changed during identity collection")
                actual.update(state=second["state"], identity_complete=True, role="experiment_worker")
                check["probe_succeeded"] = True
                return check
            except FileNotFoundError:
                # A missing environ/cwd alone is not proof that a process ended.
                try:
                    _stat(proc.read_bytes("stat"), pid)
                except FileNotFoundError:
                    check["process"] = None
                    return absent("proc_gone_during_pinned_probe", {"pid": pid, "boot_id": boot_id})
                check["error"] = "required_proc_field_missing_but_process_still_present"
                return check
    except FileNotFoundError:
        return absent("proc_directory_absent", {"pid": pid, "boot_id": boot_id})
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        check["error"] = "worker_probe_unavailable:" + type(error).__name__
        return check


def _desktop_process(fs, metadata: dict[str, Any], boot_id: str) -> dict[str, Any]:
    pid = metadata["pid"]
    with fs.open_proc(pid) as proc:
        first = _stat(proc.read_bytes("stat"), pid)
        argv, digest = _command(proc.read_bytes("cmdline"))
        uids = _uids(proc.read_bytes("status"))
        exe = proc.readlink("exe")
        last = _stat(proc.read_bytes("stat"), pid)
        _require(first["starttime"] == last["starttime"] and last["state"] not in {"Z", "X"}, "desktop process changed")
        _require(exe == host_gate.DESKTOP_EXE and argv[0] == host_gate.DESKTOP_EXE, "desktop executable mismatch")
        return {**metadata, **last, "argv": argv, "cmdline_sha256": digest, "boot_id": boot_id,
                "uid": uids[0], "exe": exe, "role": "desktop", "run_nonce": None, "identity_complete": True}


def _csv_rows(result: CommandResult, columns: int):
    _require(result.returncode == 0, "nvidia-smi command failed")
    parsed = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or all(not part.strip() for part in row):
            continue
        _require(len(row) == columns, "unexpected nvidia-smi CSV columns")
        parsed.append([part.strip() for part in row])
    return parsed


def collect_snapshot(registry_path: str, *, project_root: str, project_id: str, gpu_index: int = 0,
                     fs=None, command_runner: Callable[[list[str]], CommandResult] | None = None,
                     clock: Callable[[], float] | None = None,
                     snapshot_id_factory: Callable[[], str] | None = None,
                     current_uid: int | None = None) -> dict[str, Any]:
    """Collect once; injecting FS/commands/clock allows tests with no host access."""
    _require(type(gpu_index) is int and gpu_index >= 0, "invalid GPU index")
    if (fs is None or command_runner is None) and sys.platform != "linux":
        raise RuntimeError("Real collection is Linux-only; tests must inject filesystem and command runner")
    fs = fs if fs is not None else LinuxReadOnlyFS()
    command_runner = command_runner or system_command_runner
    clock = clock or time.time
    snapshot_id_factory = snapshot_id_factory or (lambda: str(uuid.uuid4()))
    uid = os.getuid() if current_uid is None else current_uid
    _require(type(uid) is int and uid >= 0, "invalid caller UID")
    snapshot = dict(snapshot_id=snapshot_id_factory(), captured_at=float(clock()), boot_id="", gpu_uuid="", gpu_index=gpu_index,
        free_mib=0, used_mib=0, query_ok=False, inventory_complete=False, gpu_processes=[],
        worker_inventory_complete=False, registered_run_nonces=[], worker_checks=[],
        registry_path=str(registry_path), registry_sha256=None, project_id=project_id,
        collection_errors=[], verified_own_gpu_pids=[], foreign_gpu_pids=[], desktop_gpu_pids=[],
        read_only=True, foreign_private_files_read=False,
        host_mem_available_mib=None, host_mem_total_mib=None, host_memory_query_ok=False,
        host_memory_sampled_at=None, resource_policy_sha256=host_gate.resource_policy_binding()["sha256"])
    try:
        snapshot["boot_id"] = _boot_id(fs)
    except (OSError, ValueError, UnicodeError) as error:
        snapshot["collection_errors"].append("boot_identity_unavailable:" + type(error).__name__)
    try:
        available, total = _host_memory(fs)
        snapshot.update(host_mem_available_mib=available, host_mem_total_mib=total,
                        host_memory_query_ok=True, host_memory_sampled_at=float(clock()))
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        snapshot["collection_errors"].append("host_memory_unavailable:" + type(error).__name__)
    entries = []; raw_registry = None
    try:
        registry_path, raw_registry, entries = _registry(fs, registry_path, project_root, project_id, uid)
        snapshot["registry_path"] = registry_path
        snapshot["registry_sha256"] = hashlib.sha256(raw_registry).hexdigest()
        snapshot["registered_run_nonces"] = [entry["expected_record"]["run_nonce"] for entry in entries]
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        snapshot["collection_errors"].append("registry_untrusted_or_unavailable:" + type(error).__name__)
    try:
        gpu_rows = _csv_rows(command_runner(["nvidia-smi", "--id=" + str(gpu_index),
            "--query-gpu=uuid,memory.free,memory.used", "--format=csv,noheader,nounits"]), 3)
        _require(len(gpu_rows) == 1, "requested GPU not uniquely identified")
        gpu_uuid, free, used = gpu_rows[0]
        _require(bool(gpu_uuid) and int(free) >= 0 and int(used) >= 0, "invalid GPU memory evidence")
        snapshot.update(gpu_uuid=gpu_uuid, free_mib=int(free), used_mib=int(used))
        process_rows = _csv_rows(command_runner(["nvidia-smi", "--id=" + str(gpu_index),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]), 4)
        # Observe registered process births after the GPU PID inventory, so an
        # earlier ownership probe cannot bless a subsequently recycled GPU PID.
        if raw_registry is not None and snapshot["boot_id"]:
            snapshot["worker_checks"] = [_probe_worker(fs, entry, snapshot["boot_id"], uid) for entry in entries]
            snapshot["worker_inventory_complete"] = all(check["probe_succeeded"] for check in snapshot["worker_checks"])
        for row_uuid, pid_text, name, memory_text in process_rows:
            if row_uuid != gpu_uuid:
                continue
            pid = int(pid_text); _require(pid > 0, "invalid GPU PID")
            memory = int(memory_text) if memory_text.isdigit() else None
            metadata = dict(pid=pid, process_name=name, memory_mib=memory, role="foreign",
                            boot_id=snapshot["boot_id"], identity_complete=False)
            matching = [check for check in snapshot["worker_checks"] if check.get("process") is not None
                        and check["process"].get("pid") == pid
                        and host_gate.verify_ownership(check["process"], check["expected_record"], check["observed_record"]).is_own]
            if matching:
                metadata.update(matching[0]["process"])
                snapshot["verified_own_gpu_pids"].append(pid)
            elif name == host_gate.DESKTOP_EXE and memory is not None and 0 <= memory <= host_gate.MAX_DESKTOP_MIB:
                try:
                    metadata = _desktop_process(fs, metadata, snapshot["boot_id"])
                    snapshot["desktop_gpu_pids"].append(pid)
                except (OSError, ValueError, TypeError, UnicodeError):
                    snapshot["collection_errors"].append("desktop_public_identity_unavailable:" + str(pid))
                    snapshot["foreign_gpu_pids"].append(pid)
            else:
                snapshot["foreign_gpu_pids"].append(pid)
            snapshot["gpu_processes"].append(metadata)
        snapshot["query_ok"] = True; snapshot["inventory_complete"] = True
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        snapshot["collection_errors"].append("GPU_inventory_unavailable:" + type(error).__name__)
    # Detect registration or source-of-truth changes during collection. A failed
    # recheck is uncertainty, not confirmed process absence or an empty registry.
    if raw_registry is not None:
        try:
            _require(fs.read_owned_bytes(registry_path, uid) == raw_registry, "registry changed during collection")
            for entry in entries:
                latest = _json(fs.read_owned_bytes(entry["record_path"], uid))
                _require(host_gate.launch_record_sha256(latest) == entry["record_sha256"], "launch record changed during collection")
        except (OSError, ValueError, TypeError, UnicodeError):
            snapshot["worker_inventory_complete"] = False
            snapshot["collection_errors"].append("registry_or_launch_record_changed_during_collection")
    try:
        _require(_boot_id(fs) == snapshot["boot_id"], "boot changed during collection")
    except (OSError, ValueError, UnicodeError):
        snapshot["query_ok"] = False; snapshot["worker_inventory_complete"] = False
        snapshot["collection_errors"].append("boot_identity_not_stable")
    snapshot["collection_finished_at"] = float(clock())
    return snapshot


def assess_collected_launch_boundary(before: Mapping[str, Any], after: Mapping[str, Any], *,
                                     project_lock_held: bool, now: float | None = None,
                                     task_kind: str = "gpu") -> host_gate.AdmissionDecision:
    """Consume caller-collected snapshots; acquire/release no locks whatsoever."""
    decision = host_gate.assess_launch_boundary(before, after, now=time.time() if now is None else now,
                                               project_lock_held=project_lock_held, task_kind=task_kind)
    if any(before.get(key) != after.get(key) for key in ("registry_path", "registry_sha256", "project_id")):
        return replace(decision, allow_start=False, boundary_verified=False, valid_until=None,
                       reasons=decision.reasons + ("launch_registry_changed_between_snapshots",))
    return decision
