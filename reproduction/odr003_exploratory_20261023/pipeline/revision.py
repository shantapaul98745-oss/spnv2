"""Explicit append-only manifest revisions; no file writes or process operations.

The caller must validate both manifests, hold the manager lock, archive the returned
audit durably, and only then publish initialize_state(amended, new_manifest).
"""
import copy
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta

_RUNTIME_FIELDS = {
    "runtime", "history", "error", "accepted_at", "started_at", "event_error",
    "progress_warning",
}
_STATUSES = {
    "queued", "pending", "limited", "accepted", "failed", "running",
    "launching", "awaiting_verification",
}


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Revision evidence must be finite JSON data") from error


def canonical_sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _manifest_tasks(manifest, label):
    if not isinstance(manifest, dict):
        raise ValueError(label + " manifest must be an object")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(label + " tasks must be a list")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(label + " task must be an object")
        ident = task.get("id")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident)
                or ident in seen):
            raise ValueError(label + " requires unique safe task identities")
        seen.add(ident)
        if task.get("status", "queued") not in {"queued", "pending", "limited"}:
            raise ValueError(label + " manifest cannot pre-accept execution evidence")
        if _RUNTIME_FIELDS.intersection(task):
            raise ValueError(label + " manifest embeds execution evidence")
    return tasks


def _validate_previous(previous, old_manifest, old_tasks):
    if not isinstance(previous, dict):
        raise ValueError("Previous state must be an object")
    if previous.get("manifest_sha256") != canonical_sha256(old_manifest):
        raise ValueError("Old manifest hash does not match previous state binding")
    if previous.get("plan_id") != old_manifest.get("plan_id"):
        raise ValueError("Previous state plan_id does not match old manifest")
    tasks = previous.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(old_tasks):
        raise ValueError("Previous state must contain every original task exactly once")
    for task, spec in zip(tasks, old_tasks):
        if not isinstance(task, dict) or task.get("id") != spec["id"]:
            raise ValueError("Previous state task identities/order differ from old manifest")
        if task.get("status") not in _STATUSES:
            raise ValueError("Previous state has invalid task status")
        for key, value in spec.items():
            if key != "status" and (key not in task or _canonical(task[key]) != _canonical(value)):
                raise ValueError("Previous state task definition differs: " + spec["id"] + "." + key)
        if "runtime" in task and not isinstance(task["runtime"], dict):
            raise ValueError("Previous task runtime must be an object")
        if "history" in task and (not isinstance(task["history"], list)
                                  or any(not isinstance(x, dict) for x in task["history"])):
            raise ValueError("Previous task history must be a list of objects")
    completed = previous.get("completed")
    if (not isinstance(completed, list) or any(not isinstance(x, str) for x in completed)
            or len(set(completed)) != len(completed)
            or set(completed) != {x["id"] for x in tasks if x["status"] == "accepted"}):
        raise ValueError("Previous completed evidence disagrees with accepted tasks")
    if not isinstance(previous.get("current"), dict):
        raise ValueError("Previous current evidence must be an object")
    active = previous.get("active")
    if active is not None and active not in {x["id"] for x in old_tasks}:
        raise ValueError("Previous active identity is absent from old manifest")
    current_id = previous["current"].get("task_id")
    if current_id is not None and current_id not in {x["id"] for x in old_tasks}:
        raise ValueError("Previous current identity is absent from old manifest")
    # Also reject non-JSON and nonfinite nested runtime/audit evidence.
    _canonical(previous)


def extend_state(previous, old_manifest, new_manifest, *, caller_context=None, timestamp=None):
    """Return (amended_previous, audit) without modifying inputs or disk.

    Only manifest_sha256 changes in amended_previous. initialize_state subsequently
    adds new queued tasks and performs its normal restart identity revalidation.
    Exact original definitions include all command/config/env/session/log fields.
    Both manifests still require the manager's ordinary validate_manifest checks.
    """
    old_tasks = _manifest_tasks(old_manifest, "Old")
    new_tasks = _manifest_tasks(new_manifest, "New")
    _validate_previous(previous, old_manifest, old_tasks)
    old_digest, new_digest = canonical_sha256(old_manifest), canonical_sha256(new_manifest)
    if not isinstance(old_manifest.get("plan_id"), str) or not old_manifest["plan_id"]:
        raise ValueError("A nonempty plan_id is required")
    if len(new_tasks) <= len(old_tasks):
        raise ValueError("Revision must strictly append at least one task")
    if _canonical(new_tasks[:len(old_tasks)]) != _canonical(old_tasks):
        raise ValueError("Original tasks must remain an exact canonical prefix")
    old_fixed = {k: v for k, v in old_manifest.items() if k not in {"tasks", "backlog", "scope"}}
    new_fixed = {k: v for k, v in new_manifest.items() if k not in {"tasks", "backlog", "scope"}}
    if _canonical(old_fixed) != _canonical(new_fixed):
        raise ValueError("Only tasks, backlog, and scope may change")
    context = {} if caller_context is None else copy.deepcopy(caller_context)
    if not isinstance(context, dict):
        raise ValueError("caller_context must be an object")
    _canonical(context)
    at = timestamp or datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    if not isinstance(at, str) or not at:
        raise ValueError("timestamp must be a nonempty string")
    amended = copy.deepcopy(previous)
    amended["manifest_sha256"] = new_digest
    audit = {
        "version": 1, "operation": "append_only_manifest_extension",
        "timestamp": at, "caller_context": context, "plan_id": old_manifest["plan_id"],
        "old_manifest_sha256": old_digest, "new_manifest_sha256": new_digest,
        "previous_state_sha256": canonical_sha256(previous),
        "old_task_count": len(old_tasks), "new_task_count": len(new_tasks),
        "original_tasks": copy.deepcopy(old_tasks),
        "appended_tasks": copy.deepcopy(new_tasks[len(old_tasks):]),
        "old_manifest_metadata": copy.deepcopy({k: v for k, v in old_manifest.items() if k != "tasks"}),
        "new_manifest_metadata": copy.deepcopy({k: v for k, v in new_manifest.items() if k != "tasks"}),
        "preserved_task_ids": [t["id"] for t in old_tasks],
        "accepted_task_ids": list(previous["completed"]),
        "active_task_id": previous.get("active"),
    }
    _canonical(audit)
    return amended, audit
