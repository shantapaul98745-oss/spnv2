"""Pure-data admission and accounting for shared-host SPNv2 work.

This module never queries a host, sleeps, polls, launches, signals or kills a
process. Callers supply independently collected observations. In particular a
process nonce must be observed from the live process/launch handshake, never
copied from the expected record merely to make verification pass.

Snapshot contract:
  snapshot_id, captured_at (host clock seconds), boot_id, gpu_uuid, gpu_index,
  free_mib, used_mib, query_ok, inventory_complete, gpu_processes,
  worker_inventory_complete, registered_run_nonces, worker_checks.
Each worker_check has expected_record, observed_record, probe_succeeded,
process (live /proc identity or None), and absence_confirmed when process=None.
The worker inventory covers experiment workers, not the waiting queue manager.

Both GPU work and heavy CPU model verification use the same resource gate.
Lightweight read-only monitoring does not use a model-task admission token.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import copy
from typing import Any, Mapping, Sequence

LAUNCH_SCHEMA = "spnv2.round2.launch.v1"
DESKTOP_EXE = "/usr/libexec/gnome-remote-desktop-daemon"
MIN_FREE_MIB = 14500
MAX_USED_MIB = 1200
MAX_DESKTOP_MIB = 256
MIN_WAIT_SECONDS = 5 * 60
MAX_SNAPSHOT_AGE_SECONDS = 60
LAUNCH_SNAPSHOT_MAX_AGE_SECONDS = 5
IDENTITY_FIELDS = ("boot_id", "pid", "starttime", "cmdline_sha256", "run_nonce", "uid")
RECORD_FIELDS = {"schema", "project_id", "run_id", "run_nonce", "boot_id", "pid", "starttime",
                 "argv", "cmdline_sha256", "uid", "cwd", "session", "log_path", "launched_at", "role"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)



RESOURCE_POLICY_SCHEMA = "spnv2-shared-gpu-resource-policy-v1"
RESOURCE_POLICY = None
RESOURCE_POLICY_BINDING = None


def configure_policy(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one explicit immutable capacity policy; never infer free resources."""
    global RESOURCE_POLICY, RESOURCE_POLICY_BINDING
    _require(isinstance(binding, Mapping) and set(binding) == {"path", "sha256"}, "Exact resource-policy binding required")
    data = Path(binding["path"]).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == binding["sha256"], "Resource-policy bytes changed")
    policy = json.loads(data.decode("utf8"))
    _require(policy.get("schema") == RESOURCE_POLICY_SCHEMA
             and policy.get("authorization") == "user_2026-10-27_shared_gpu_capacity_admission"
             and policy.get("project_id") == "g50p_risk_diagnostic_20261025"
             and policy.get("resource_admission_only") is True
             and policy.get("allow_foreign_gpu_processes") is True
             and policy.get("ignore_gpu_utilization") is True,
             "Shared GPU authorization/scope incomplete")
    for field in ("gpu_peak_estimate_mib", "gpu_safety_margin_mib", "min_free_gpu_mib",
                  "host_memory_peak_estimate_mib", "host_memory_safety_margin_mib", "min_host_mem_available_mib"):
        _integer(policy.get(field), field, 1)
    _require(policy["min_free_gpu_mib"] == policy["gpu_peak_estimate_mib"] + policy["gpu_safety_margin_mib"],
             "GPU threshold must include the explicit peak estimate and safety margin")
    _require(policy["min_host_mem_available_mib"] == policy["host_memory_peak_estimate_mib"] + policy["host_memory_safety_margin_mib"],
             "Host threshold must include the explicit peak estimate and safety margin")
    _require(policy.get("snapshot_max_age_seconds") == MAX_SNAPSHOT_AGE_SECONDS
             and policy.get("launch_snapshot_max_age_seconds") == LAUNCH_SNAPSHOT_MAX_AGE_SECONDS
             and policy.get("resource_retry_seconds") == MIN_WAIT_SECONDS,
             "Original snapshot freshness and300-second resource cadence must remain")
    if RESOURCE_POLICY_BINDING is not None:
        _require(dict(binding) == RESOURCE_POLICY_BINDING and policy == RESOURCE_POLICY, "Resource policy cannot hot-change")
    RESOURCE_POLICY, RESOURCE_POLICY_BINDING = copy.deepcopy(policy), dict(binding)
    return copy.deepcopy(policy)


def shared_gpu_enabled() -> bool:
    return RESOURCE_POLICY is not None and RESOURCE_POLICY_BINDING is not None


def resource_policy_binding() -> dict[str, Any]:
    _require(shared_gpu_enabled(), "Explicit shared GPU resource policy not configured")
    return dict(RESOURCE_POLICY_BINDING)


def _check_capacity(snapshot: Mapping[str, Any], *, now: float, reasons: list[str]) -> None:
    _require(shared_gpu_enabled(), "Explicit shared GPU resource policy not configured")
    _require(snapshot.get("resource_policy_sha256") == RESOURCE_POLICY_BINDING["sha256"], "Snapshot resource-policy binding differs")
    _require(snapshot.get("host_memory_query_ok") is True, "Host memory observation incomplete")
    available = _integer(snapshot.get("host_mem_available_mib"), "host_mem_available_mib")
    total = _integer(snapshot.get("host_mem_total_mib"), "host_mem_total_mib", 1)
    _require(available <= total, "Host available memory exceeds physical total")
    sampled = _time(snapshot.get("host_memory_sampled_at"), "host_memory_sampled_at")
    finished = _time(snapshot.get("collection_finished_at"), "collection_finished_at")
    _require(_time(snapshot["captured_at"], "captured_at") <= sampled <= finished <= now,
             "Host memory sample is outside this snapshot interval")
    _require(0 <= now-sampled <= MAX_SNAPSHOT_AGE_SECONDS, "Host memory sample stale or from the future")
    if _integer(snapshot.get("free_mib"), "free_mib") < RESOURCE_POLICY["min_free_gpu_mib"]:
        reasons.append("insufficient_free_GPU_memory")
    _integer(snapshot.get("used_mib"), "used_mib")
    if available < RESOURCE_POLICY["min_host_mem_available_mib"]:
        reasons.append("insufficient_available_host_memory")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def _time(value: Any, name: str) -> float:
    _require(type(value) in (float, int) and math.isfinite(value) and value >= 0, name + " must be finite nonnegative seconds")
    return float(value)


def _text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), name + " must be a nonempty string")
    return value


def command_sha256(argv: Sequence[str]) -> str:
    """SHA256 of exact UTF-8 /proc-style argv bytes, including the final NUL."""
    _require(isinstance(argv, (list, tuple)) and bool(argv), "full argv required")
    _require(all(isinstance(x, str) and "\x00" not in x for x in argv) and bool(argv[0]), "invalid argv")
    return hashlib.sha256(("\x00".join(argv) + "\x00").encode("utf-8")).hexdigest()


def launch_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash every JSON field, including extra fields; never an identity subset."""
    _require(isinstance(record, Mapping), "launch record must be an object")
    return hashlib.sha256(json.dumps(dict(record), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _validate_record(record: Mapping[str, Any]) -> None:
    _require(isinstance(record, Mapping) and RECORD_FIELDS <= record.keys(), "incomplete launch record")
    _require(record["schema"] == LAUNCH_SCHEMA and record["role"] == "worker", "wrong launch schema/role")
    for key in ("project_id", "run_id", "run_nonce", "boot_id", "cwd", "session", "log_path"):
        _text(record[key], key)
    _require(len(record["run_nonce"]) >= 16, "run nonce too short")
    _integer(record["pid"], "pid", 1)
    _integer(record["starttime"], "starttime", 1)
    _integer(record["uid"], "uid")
    _time(record["launched_at"], "launched_at")
    _require(record["cmdline_sha256"] == command_sha256(record["argv"]), "launch full-command hash mismatch")
    launch_record_sha256(record)


def _validate_process(process: Mapping[str, Any]) -> None:
    _require(isinstance(process, Mapping) and process.get("identity_complete") is True, "live process identity incomplete")
    _text(process.get("boot_id"), "process boot_id")
    _integer(process.get("pid"), "process pid", 1)
    _integer(process.get("starttime"), "process starttime", 1)
    _integer(process.get("uid"), "process uid")
    _require(process.get("state") in {"R", "S", "D", "T", "t", "I", "Z", "X", "P"}, "unknown process state")
    _require(process.get("cmdline_sha256") == command_sha256(process.get("argv")), "observed full-command hash mismatch")


@dataclass(frozen=True)
class OwnershipResult:
    is_own: bool
    reasons: tuple[str, ...]
    launch_record_sha256: str | None = None


def verify_ownership(process: Mapping[str, Any], expected_record: Mapping[str, Any],
                     observed_record: Mapping[str, Any]) -> OwnershipResult:
    """No UID, executable name, command prefix or PID-only ownership inference."""
    try:
        _validate_record(expected_record)
        _validate_record(observed_record)
        _validate_process(process)
        expected_hash = launch_record_sha256(expected_record)
        if launch_record_sha256(observed_record) != expected_hash:
            return OwnershipResult(False, ("complete_launch_record_mismatch",))
        mismatch = tuple("identity_mismatch:" + key for key in IDENTITY_FIELDS
                         if process.get(key) != expected_record[key])
        if mismatch:
            return OwnershipResult(False, mismatch)
        return OwnershipResult(True, (), expected_hash)
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        return OwnershipResult(False, ("invalid_ownership_evidence:" + str(error),))


@dataclass(frozen=True)
class AdmissionDecision:
    allow_start: bool
    reasons: tuple[str, ...]
    task_kind: str
    snapshot_ids: tuple[str, ...]
    owned_worker_pids: tuple[int, ...] = ()
    desktop_pids: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()
    evaluated_at: float | None = None
    valid_until: float | None = None
    boundary_verified: bool = False


def _desktop_allowed(process: Mapping[str, Any], boot_id: str) -> bool:
    try:
        _validate_process(process)
        return (process["boot_id"] == boot_id
                and process.get("exe") == DESKTOP_EXE
                and process["argv"][0] == DESKTOP_EXE
                and process.get("role") == "desktop"
                and process["state"] not in {"Z", "X"}
                and type(process.get("memory_mib")) is int
                and 0 <= process["memory_mib"] <= MAX_DESKTOP_MIB)
    except (ValueError, TypeError, KeyError):
        return False


def assess_snapshot(snapshot: Mapping[str, Any], *, now: float,
                    task_kind: str = "gpu") -> AdmissionDecision:
    """One observation only. GPU utilization is deliberately not an admission rule."""
    reasons: list[str] = []
    own: set[int] = set()
    desktop: set[int] = set()
    snapshot_id = str(snapshot.get("snapshot_id", "missing")) if isinstance(snapshot, Mapping) else "missing"
    try:
        _require(task_kind in {"gpu", "heavy_cpu"}, "task_kind must be gpu or heavy_cpu")
        _require(isinstance(snapshot, Mapping), "snapshot must be an object")
        _text(snapshot.get("snapshot_id"), "snapshot_id")
        boot_id = _text(snapshot.get("boot_id"), "boot_id")
        _text(snapshot.get("gpu_uuid"), "gpu_uuid")
        _integer(snapshot.get("gpu_index"), "gpu_index")
        age = _time(now, "now") - _time(snapshot.get("captured_at"), "captured_at")
        _require(0 <= age <= MAX_SNAPSHOT_AGE_SECONDS, "snapshot stale or from the future")
        _require(snapshot.get("query_ok") is True and snapshot.get("inventory_complete") is True,
                 "GPU query/inventory incomplete")
        _require(snapshot.get("worker_inventory_complete") is True, "registered-worker probes incomplete")
        processes = snapshot.get("gpu_processes")
        checks = snapshot.get("worker_checks")
        nonces = snapshot.get("registered_run_nonces")
        _require(isinstance(processes, (list, tuple)) and isinstance(checks, (list, tuple)), "process/check inventory required")
        _require(isinstance(nonces, (list, tuple)) and all(isinstance(n, str) and len(n) >= 16 for n in nonces), "run nonce registry required")
        _require(len(nonces) == len(set(nonces)), "duplicate registry nonce")
        checked_nonces = []
        valid_checks = []
        for check in checks:
            _require(isinstance(check, Mapping) and check.get("probe_succeeded") is True, "worker process probe failed")
            expected, observed = check["expected_record"], check["observed_record"]
            _validate_record(expected); _validate_record(observed)
            _require(launch_record_sha256(expected) == launch_record_sha256(observed), "registered launch record changed")
            checked_nonces.append(expected["run_nonce"])
            actual = check.get("process")
            if actual is None:
                _require(check.get("absence_confirmed") is True, "missing process observation is not confirmed absence")
                valid_checks.append((expected, observed))
                continue
            _validate_process(actual)
            _require(actual["pid"] == expected["pid"] and actual["boot_id"] == boot_id, "probe returned a different PID or boot")
            valid_checks.append((expected, observed))
            # A different boot/start proves the original recorded worker is gone;
            # it never makes the replacement process ours. GPU entries still gate.
            if actual["boot_id"] != expected["boot_id"] or actual["starttime"] != expected["starttime"]:
                continue
            identity = verify_ownership(actual, expected, observed)
            if not identity.is_own:
                reasons.append("registered_worker_identity_ambiguous:" + str(actual["pid"]))
            elif actual["state"] not in {"Z", "X"}:
                own.add(actual["pid"])
                reasons.append("existing_owned_worker_blocks_duplicate:" + str(actual["pid"]))
        _require(len(checked_nonces) == len(set(checked_nonces)) and set(checked_nonces) == set(nonces),
                 "worker probes do not cover the complete nonce registry")
        seen_pids = set()
        for process in processes:
            _require(isinstance(process, Mapping), "invalid GPU process entry")
            pid = _integer(process.get("pid"), "GPU pid", 1)
            _require(pid not in seen_pids, "duplicate GPU PID observation")
            seen_pids.add(pid)
            if _desktop_allowed(process, boot_id):
                desktop.add(pid)
                continue
            if process.get("boot_id") != boot_id:
                reasons.append("GPU_process_boot_identity_unknown:" + str(pid))
                continue
            identities = [verify_ownership(process, expected, observed) for expected, observed in valid_checks]
            if any(result.is_own for result in identities):
                own.add(pid)
                reasons.append("owned_GPU_process_still_present:" + str(pid))
            else:
                # Public foreign presence is retained for observation only. The
                # authorized shared policy judges capacity, never another job.
                pass
        _check_capacity(snapshot, now=float(now), reasons=reasons)
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        reasons.append("invalid_snapshot:" + str(error))
    evaluated = float(now) if type(now) in (int, float) and math.isfinite(now) else None
    return AdmissionDecision(not reasons, tuple(dict.fromkeys(reasons)), task_kind, (snapshot_id,),
                             tuple(sorted(own)), tuple(sorted(desktop)),
                             ("single_snapshot_only_not_launch_boundary", "desktop_role_is_not_experiment_ownership",
                              "foreign_GPU_presence_and_utilization_are_not_capacity_rejections"),
                             evaluated_at=evaluated)


def assess_launch_boundary(before: Mapping[str, Any], after: Mapping[str, Any], *, now: float,
                           project_lock_held: bool, task_kind: str = "gpu") -> AdmissionDecision:
    """Two fresh checks around the project lock, immediately before launch.

    The caller obtains its real project lock between observations. That lock
    prevents project duplicates only. Other users do not participate in it, so
    two passing snapshots cannot promise a globally atomic GPU reservation.
    """
    first = assess_snapshot(before, now=now, task_kind=task_kind)
    second = assess_snapshot(after, now=now, task_kind=task_kind)
    reasons = list(first.reasons) + list(second.reasons)
    if project_lock_held is not True:
        reasons.append("project_lock_not_held")
    try:
        _require(before["snapshot_id"] != after["snapshot_id"], "two distinct observations required")
        _require(_time(before["captured_at"], "first captured_at") < _time(after["captured_at"], "second captured_at"),
                 "launch observations must be ordered and distinct")
        _require(0 <= _time(now, "now") - _time(after["captured_at"], "second captured_at") <= LAUNCH_SNAPSHOT_MAX_AGE_SECONDS,
                 "final launch snapshot is not immediate")
        _require(all(before[key] == after[key] for key in ("boot_id", "gpu_uuid", "gpu_index")), "host boot or GPU changed between observations")
        _require(set(before["registered_run_nonces"]) == set(after["registered_run_nonces"]), "worker registry changed during admission")
    except (ValueError, TypeError, KeyError) as error:
        reasons.append("invalid_launch_boundary:" + str(error))
    evaluated = float(now) if type(now) in (int, float) and math.isfinite(now) else None
    return AdmissionDecision(not reasons, tuple(dict.fromkeys(reasons)), task_kind,
                             first.snapshot_ids + second.snapshot_ids,
                             tuple(sorted(set(first.owned_worker_pids + second.owned_worker_pids))),
                             tuple(sorted(set(first.desktop_pids + second.desktop_pids))),
                             ("project_lock_is_not_a_global_GPU_reservation",
                              "another_user_can_start_after_the_last_snapshot",
                              "no_signals_or_process_actions_are_authorized_by_this_decision"),
                             evaluated_at=evaluated,
                             valid_until=float(after["captured_at"])+LAUNCH_SNAPSHOT_MAX_AGE_SECONDS if not reasons else None,
                             boundary_verified=not reasons)


def wait_advice(decision: AdmissionDecision, requested_seconds: int = MIN_WAIT_SECONDS) -> dict[str, Any]:
    """Return one low-frequency scheduling suggestion; never wait or poll here."""
    _integer(requested_seconds, "requested_seconds")
    if decision.allow_start:
        return {"action": "use_fresh_launch_boundary_before_start", "retry_after_seconds": None,
                "poll_loop": False, "sleep_performed": False}
    return {"action": "one_later_resource_check", "retry_after_seconds": max(MIN_WAIT_SECONDS, requested_seconds),
            "reasons": list(decision.reasons), "poll_loop": False, "sleep_performed": False,
            "heavy_CPU_model_validation_must_also_wait": True, "notify_only_on_meaningful_change": True}


class RunAccounting:
    """In-memory wait/compute ledger using caller-supplied host-clock timestamps.

    Compute includes active GPU/CPU model work and its validation, not queue wait.
    begin_wait marks a gap after a compute phase has naturally ended; it neither
    suspends nor stops any process. No timers or side effects are implemented.
    """
    def __init__(self, run_id: str, queued_at: float):
        self.run_id = _text(run_id, "run_id")
        self._since = _time(queued_at, "queued_at")
        self._phase = "waiting"
        self._wait = self._compute = 0.0
        self._ever_computed = False
        self._events = [{"event": "queue_wait_started", "run_id": run_id, "at": self._since}]

    def _elapsed(self, at: float) -> float:
        timestamp = _time(at, "event time")
        _require(timestamp >= self._since, "accounting clock moved backwards")
        return timestamp - self._since

    def begin_compute(self, at: float, admission: AdmissionDecision) -> None:
        _require(self._phase == "waiting", "compute may begin only from waiting")
        _require(admission.allow_start and admission.boundary_verified and len(admission.snapshot_ids) == 2,
                 "compute requires a passing two-snapshot launch boundary")
        timestamp = _time(at, "compute start")
        _require(admission.evaluated_at is not None and admission.valid_until is not None
                 and admission.evaluated_at <= timestamp <= admission.valid_until,
                 "admission expired or compute starts before its resource check")
        elapsed = self._elapsed(at); self._wait += elapsed
        self._events.extend([{"event": "queue_wait_ended", "run_id": self.run_id, "at": float(at), "segment_seconds": elapsed},
                             {"event": "compute_started", "run_id": self.run_id, "at": float(at), "snapshot_ids": list(admission.snapshot_ids)}])
        self._phase = "computing"; self._since = float(at); self._ever_computed = True

    def begin_wait(self, at: float, reason: str) -> None:
        _require(self._phase == "computing", "wait transition requires a naturally completed compute phase")
        reason = _text(reason, "reason")
        elapsed = self._elapsed(at); self._compute += elapsed
        self._events.extend([{"event": "compute_ended", "run_id": self.run_id, "at": float(at), "segment_seconds": elapsed},
                             {"event": "queue_wait_started", "run_id": self.run_id, "at": float(at), "reason": reason}])
        self._phase = "waiting"; self._since = float(at)

    def finish(self, at: float, outcome: str) -> None:
        _require(self._phase != "finished", "run already finished")
        _require(outcome in {"completed", "failed", "cancelled"}, "unknown terminal outcome")
        _require(outcome != "completed" or self._ever_computed, "no compute evidence for completed outcome")
        elapsed = self._elapsed(at)
        if self._phase == "waiting":
            self._wait += elapsed; event = "queue_wait_ended"
        else:
            self._compute += elapsed; event = "compute_ended"
        self._events.extend([{"event": event, "run_id": self.run_id, "at": float(at), "segment_seconds": elapsed},
                             {"event": "run_finished", "run_id": self.run_id, "at": float(at), "outcome": outcome}])
        self._phase = "finished"; self._since = float(at)

    def snapshot(self, at: float | None = None) -> dict[str, Any]:
        waiting, computing = self._wait, self._compute
        if at is not None:
            elapsed = self._elapsed(at)
            if self._phase == "waiting": waiting += elapsed
            elif self._phase == "computing": computing += elapsed
        return {"schema": "spnv2.resource-accounting.v1", "run_id": self.run_id, "phase": self._phase,
                "queue_wait_seconds": waiting, "compute_seconds": computing,
                "accounted_seconds": waiting + computing, "events": json.loads(json.dumps(self._events))}
