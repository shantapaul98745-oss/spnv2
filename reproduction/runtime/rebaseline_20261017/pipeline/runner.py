"""Durable serial SPNv2 queue. Launch this manager itself in the project screen wrapper."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from progress import Reporter, age_seconds, atomic_json, now, read_json
from task_entry import process_alive, process_identity

BASE = Path("/home/DATA/Downloads/UsrSourceCode/LuXinghai")
PLAN = BASE / "spnv2_runs/plans/rebaseline_20261017"
PYTHON = "/home/ps/anaconda3/envs/SPNv2/bin/python"
WRAPPER = BASE / "spnv2_runs/scripts/run_in_screen_guarded.sh"
REPORT = BASE / "paper/SPNv2_复现实时进度.md"
EVENT_FIELDS = {
    "event", "phase", "epoch", "epochs", "batch", "batches", "loss", "lr",
    "update_attempts", "optimizer_updates", "amp_skips", "samples_seen",
    "checkpoint", "source", "resume", "message", "error", "path", "queue_blocked", "timestamp",
    "score", "metrics", "summary", "n_samples", "validation_samples", "heatmap_rejections", "heatmap_valid", "domain", "config", "output",
}
TERMINAL = {"accepted", "failed", "limited"}


def inside(path, root=BASE):
    resolved = Path(path).resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise ValueError("Path outside this project: " + str(path))
    return resolved


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_command(command):
    return (isinstance(command, list) and bool(command)
            and all(isinstance(x, str) and x and "\x00" not in x for x in command))


def validate_manifest(manifest):
    if manifest.get("version", 1) != 1 or manifest.get("plan_id") != PLAN.name:
        raise ValueError("Unsupported manifest version or plan_id")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    seen, sessions, logs = set(), set(), set()
    inside(manifest.get("report_path", str(REPORT)), BASE / "paper")
    for task in tasks:
        if task.get("status", "queued") not in {"queued", "pending", "limited"}:
            raise ValueError("Manifest cannot pre-accept execution evidence")
        ident = task.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in seen:
            raise ValueError("Unique safe task id required")
        requires = task.get("requires", [])
        if not isinstance(requires, list) or any(x not in seen for x in requires):
            raise ValueError("Dependencies must precede task " + ident)
        seen.add(ident)
        if task.get("status") == "limited":
            if not task.get("reason"):
                raise ValueError("Limited task requires reason: " + ident)
            continue
        if not valid_command(task.get("command")):
            raise ValueError("command must be a nonempty argv list: " + ident)
        if task.get("verify_command") is not None and not valid_command(task["verify_command"]):
            raise ValueError("Invalid verify_command: " + ident)
        if not task.get("verify_command") and not task.get("expected_outputs"):
            raise ValueError("Each executable task needs verification evidence: " + ident)
        inside(task["cwd"])
        session = task["session"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
            raise ValueError("Unsafe screen session: " + ident)
        for phase_session in (session, session + "_verify"):
            if phase_session in sessions:
                raise ValueError("Reused screen session: " + phase_session)
            sessions.add(phase_session)
        log = str(inside(task["log"], BASE / "spnv2_runs"))
        for phase_log in (log, log + ".verify"):
            if phase_log in logs:
                raise ValueError("Reused log: " + phase_log)
            logs.add(phase_log)
        for expected in task.get("expected_outputs", []):
            path = expected["path"] if isinstance(expected, dict) else expected
            inside(path)
        gpu = task.get("gpu", False)
        if gpu is not False and not isinstance(gpu, dict):
            raise ValueError("gpu must be false or resource object: " + ident)
        if isinstance(gpu, dict):
            for key in ("index", "min_free_mb", "max_used_mb"):
                if key in gpu and (not isinstance(gpu[key], int) or gpu[key] < 0):
                    raise ValueError("Invalid GPU resource field " + key)


def asset_hash_error(task):
    """Recheck pinned executable/config assets immediately before each phase."""
    assets = (task.get("metadata") or {}).get("asset_sha256", {})
    if not isinstance(assets, dict):
        return "Invalid pinned asset map"
    for name, expected in assets.items():
        try:
            path = inside(name)
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                return "Invalid pinned asset SHA256: " + str(name)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                return "Pinned asset changed; no launch: " + str(path)
        except (OSError, ValueError, TypeError) as error:
            return "Pinned asset unavailable; no launch: " + str(name) + " " + repr(error)
    return None


def initialize_state(previous, manifest):
    digest = manifest_hash(manifest)
    bound = previous.get("manifest_sha256")
    if bound and bound != digest:
        raise ValueError("Manifest changed: refuse to reuse or relaunch any recorded work")
    previous_tasks = {t["id"]: t for t in previous.get("tasks", [])}
    if not bound and any(t.get("status") in {"accepted", "running", "launching", "awaiting_verification"}
                         or t.get("runtime") for t in previous_tasks.values()):
        raise ValueError("Unbound prior execution state requires manual evidence reconciliation")
    state = copy.deepcopy(previous)
    state.update({"manifest_sha256": digest, "plan_id": manifest["plan_id"]})
    state.setdefault("created_at", now())
    state.setdefault("notes", [])
    state.setdefault("completed", [])
    state.setdefault("current", {})
    state.setdefault("error", None)
    state["backlog"] = copy.deepcopy(manifest.get("backlog", []))
    state["scope"] = manifest.get("scope", "full")
    state["tasks"] = []
    for spec in manifest["tasks"]:
        task = copy.deepcopy(spec)
        old = previous_tasks.get(task["id"], {}) if bound else {}
        for key in ("status", "runtime", "history", "error", "accepted_at", "started_at", "event_error", "progress_warning"):
            if key in old:
                task[key] = copy.deepcopy(old[key])
        task.setdefault("status", "queued")
        if task["status"] == "pending":
            task["status"] = "queued"
        state["tasks"].append(task)
    # Accepted results can only survive a byte-equivalent canonical manifest.
    state["completed"] = [t["id"] for t in state["tasks"] if t["status"] == "accepted"]
    for task in state["tasks"]:
        if task.get("runtime") and task["status"] not in TERMINAL:
            state["current"]["process_state"] = "unverified after manager restart"
            if task["status"] == "running":
                task["status"] = "launching"
    return state


def bind_execution_state(previous, manifest, old_manifest=None, caller_context=None):
    """Explicit prefix-only extension under the caller's exclusive manager lock.

    Existing worker identity is retained; initialize_state then re-observes that
    same process rather than starting an already recorded phase again.
    """
    validate_manifest(manifest)
    if old_manifest is None:
        return initialize_state(previous, manifest)
    validate_manifest(old_manifest)
    from revision import extend_state
    amended, audit = extend_state(previous, old_manifest, manifest,
                                 caller_context=caller_context)
    state = initialize_state(amended, manifest)
    directory = PLAN / "revisions"
    directory.mkdir(parents=True, exist_ok=True)
    name = now().replace(":", "-") + "_" + str(os.getpid()) + "_" + audit["new_manifest_sha256"][:16] + ".json"
    destination = directory / name
    if destination.exists():
        raise ValueError("Refuse to overwrite a manifest revision audit")
    atomic_json(destination, audit)
    state.setdefault("manifest_revisions", []).append(str(destination))
    state.setdefault("notes", []).append(
        "队列已作显式追加修订：原有" + str(audit["old_task_count"]) + "项定义及运行证据保留，现共"
        + str(audit["new_task_count"]) + "项；修订证据：" + str(destination))
    return state


def expected_outputs_error(task):
    for expected in task.get("expected_outputs", []):
        spec = expected if isinstance(expected, dict) else {"path": expected}
        path = inside(spec["path"])
        if not path.is_file():
            return "Expected output is not a file: " + str(path)
        if path.stat().st_size < spec.get("min_bytes", 1):
            return "Expected output is too small: " + str(path)
        if spec.get("sha256"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != spec["sha256"]:
                return "Expected output hash mismatch: " + str(path)
    return None


def gpu_available(spec):
    """Read resource information only; never manage any GPU process."""
    if spec is False or spec is None:
        return True, None
    index = str(spec.get("index", 0))
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
        used, free = [int(x.strip()) for x in result.stdout.strip().split(",")]
        processes = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
        desktop = {"Xorg", "Xwayland", "gnome-shell", "kwin_wayland", "kwin_x11"}
        occupied = []
        for line in processes.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            remote_desktop = (len(parts) >= 3 and parts[1] == "/usr/libexec/gnome-remote-desktop-daemon"
                              and parts[2].isdigit() and int(parts[2]) <= 256)
            if len(parts) >= 2 and Path(parts[1]).name not in desktop and not remote_desktop:
                occupied.append(parts[0] + " " + parts[1])
        if occupied:
            return False, "Other GPU compute process: " + "; ".join(occupied)
        if free < spec.get("min_free_mb", 0) or used > spec.get("max_used_mb", 2 ** 31):
            return False, "GPU memory used=" + str(used) + " MiB free=" + str(free) + " MiB"
        return True, None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return False, "GPU resource query unavailable: " + repr(error)


class Queue:
    def __init__(self, manifest_path, manifest, state, reporter):
        self.manifest_path = Path(manifest_path)
        self.manifest = manifest
        self.state = state
        self.reporter = reporter
        self.stop_requested = False
        self.last_resource_check = 0
        self.resource_ok = False
        self.events_path = PLAN / "events/queue.events.jsonl"

    def event(self, name, task=None, **fields):
        value = {"at": now(), "event": name, "task": task.get("id") if task else None}
        value.update(fields)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def fail(self, task, message):
        task["status"], task["error"] = "failed", message
        self.state.update({"status": "failed", "active": task["id"], "error": message})
        self.event("task_failed", task, message=message)
        self.reporter.publish(force=True)

    def set_current(self, task):
        runtime = task.get("runtime") or {}
        current = self.state.setdefault("current", {})
        current.update({
            "task_id": task["id"], "execution_phase": runtime.get("phase", "queued"),
            "phase": current.get("phase", "main") if runtime.get("phase") == "main" else runtime.get("phase", "queued"),
            "log": runtime.get("log", task.get("log")),
            "command": runtime.get("command", task.get("command")),
            "outputs": task.get("expected_outputs"),
            "source": (task.get("metadata") or {}).get("source", current.get("source")),
            "resume": (task.get("metadata") or {}).get("resume", current.get("resume")),
            "started_at": task.get("started_at"),
        })

    def prepare_launch(self, task, phase):
        if self.stop_requested:
            return False
        gpu = task.get("gpu", False) if phase == "main" else task.get("verify_gpu", False)
        if gpu:
            ready, reason = gpu_available(gpu)
            if self.stop_requested:
                return False
            if not ready:
                self.state.update(status="waiting_resources", resource_wait=reason, active=task["id"])
                self.reporter.publish(force=True)
                return False
        if self.reporter.write_failed or self.state.get("downstream_blocked"):
            return False
        if manifest_hash(read_json(self.manifest_path)) != self.state["manifest_sha256"]:
            self.fail(task, "Manifest changed while manager was running; no launch")
            return False
        asset_error = asset_hash_error(task)
        if asset_error:
            self.fail(task, asset_error)
            return False
        runtime = task.get("runtime")
        if runtime:
            if runtime["phase"] == phase:
                # A persisted launch intent is never launched twice.
                return False
            task.setdefault("history", []).append(copy.deepcopy(runtime))
        log = task["log"] + (".verify" if phase == "verify" else "")
        session = task["session"] + ("_verify" if phase == "verify" else "")
        prefix = PLAN / "events" / (task["id"] + "." + phase)
        runtime = {
            "phase": phase, "session": session, "log": log,
            "config": str(prefix) + ".config.json", "record": str(prefix) + ".process.json",
            "result": str(prefix) + ".result.json",
            "command": task["verify_command"] if phase == "verify" else task["command"],
            "launched_at": now(), "tail_offset": 0, "tail_pending": "", "launch_attempted": True,
        }
        for path in (runtime["record"], runtime["result"], runtime["config"], log, log + ".exitcode"):
            if Path(path).exists():
                self.fail(task, "Refuse launch over existing evidence: " + path)
                return False
        if self.stop_requested:
            return False
        task["runtime"] = runtime
        task["status"] = "launching" if phase == "main" else "awaiting_verification"
        task.setdefault("started_at", now())
        self.state.update({"active": task["id"], "status": "launching", "resource_wait": None})
        self.set_current(task)
        self.state["current"].update({"process_state": "launch intent; no live PID evidence", "pid": None,
                                      "starttime": None, "screen_id": None})
        config = dict(runtime)
        environment = dict(task.get("env", {}))
        if phase == "verify":
            environment.update(task.get("verify_env", {}))
            if not task.get("verify_gpu", False):
                environment["CUDA_VISIBLE_DEVICES"] = ""
        config.update({"cwd": task["cwd"], "env": environment})
        atomic_json(runtime["config"], config)
        if not self.reporter.publish(force=True):
            # Intent is durable in memory; never launch if publication failed.
            # On restart an existing config without a launch is treated as ambiguous and fails closed.
            return False
        self.event("launch_intent", task, phase=phase, command=runtime["command"], session=session, log=log)
        if self.stop_requested:
            runtime["launch_cancelled_before_spawn"] = True
            self.reporter.publish(force=True)
            return False
        result = subprocess.run(
            ["bash", str(WRAPPER), session, log, PYTHON, "-B", "-s",
             str(Path(__file__).with_name("task_entry.py")), "--config", runtime["config"]],
            cwd=task["cwd"], capture_output=True, text=True, timeout=30)
        runtime["wrapper_launch_rc"] = result.returncode
        runtime["wrapper_launch_output"] = result.stdout + result.stderr
        self.event("wrapper_launch", task, phase=phase, exit_code=result.returncode,
                   output=runtime["wrapper_launch_output"])
        if result.returncode != 0:
            self.fail(task, "Screen launch rejected: " + runtime["wrapper_launch_output"])
            return False
        self.reporter.publish(force=True)
        return True

    def tail_events(self, task):
        runtime = task["runtime"]
        path = Path(runtime["log"])
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(runtime.get("tail_offset", 0))
            # Drain incrementally; bounded chunks prevent logs monopolizing heartbeat.
            data = stream.read(2 * 1024 * 1024)
            runtime["tail_offset"] = stream.tell()
        if not data:
            return
        raw = runtime.get("tail_pending", "") + data.decode("utf-8", errors="replace")
        lines = raw.split("\n")
        runtime["tail_pending"] = lines.pop()[-1024 * 1024:]
        for line in lines:
            if not line.startswith("SPNV2_EVENT "):
                continue
            try:
                value = json.loads(line[len("SPNV2_EVENT "):])
                if not isinstance(value, dict):
                    raise ValueError("event is not an object")
                event = {k: v for k, v in value.items() if k in EVENT_FIELDS}
                if any(isinstance(v, float) and not math.isfinite(v) for v in event.values()):
                    raise ValueError("nonfinite event metric")
                current = self.state.setdefault("current", {})
                phase_changed = bool(event.get("phase") and event["phase"] != current.get("phase"))
                current.update(event)
                if event.get("event") == "checkpoint":
                    current["checkpoint"] = event.get("path")
                elif event.get("event") == "resume":
                    current["resume"] = event.get("path")
                if event.get("event") in {"validation_end", "evaluation_complete"}:
                    self.state["last_evaluation"] = copy.deepcopy(event)
                current["last_event_at"] = now()
                self.event("task_event", task, payload=event)
                if event.get("event") == "progress_write_warning":
                    task["progress_warning"] = event
                    self.state["downstream_blocked"] = True
                    self.state["reporting_warning"] = {"task_id": task["id"], "event": event,
                        "reason": "Progress sink write failed; current process may finish, all downstream launches blocked pending manual reconciliation"}
                elif event.get("event") in {"error", "failed", "fatal"} or event.get("error"):
                    task["event_error"] = event
                    self.state["error"] = "Task reported an error; waiting for its exact process exit: " + json.dumps(event, ensure_ascii=False)
                frequent = {"update", "progress", "observation", "evaluation_progress", "validation_progress"}
                # Keep every raw/audit event, but rewrite a potentially large full
                # state only every 60 seconds unless its phase or milestone changes.
                self.reporter.publish(force=phase_changed or event.get("event") not in frequent)
            except (ValueError, TypeError) as error:
                task["event_error"] = {"error": "Malformed task event: " + repr(error)}
                self.state["error"] = task["event_error"]["error"]
                self.event("invalid_task_event", task, message=repr(error))
                self.reporter.publish(force=True)

    def observe(self, task):
        previous_observation = (self.state.get("status"), task.get("status"),
                                (self.state.get("current") or {}).get("process_state"))
        runtime = task["runtime"]
        self.set_current(task)
        record = read_json(runtime["record"])
        result = read_json(runtime["result"])
        self.tail_events(task)
        live = process_alive(record)
        current = self.state["current"]
        if record:
            current.update({k: record.get(k) for k in ("pid", "starttime", "screen_id")})
            current["execution_phase"] = runtime["phase"]
        if live:
            current["process_state"] = "live PID and /proc starttime verified"
            task["status"] = "running" if runtime["phase"] == "main" else "awaiting_verification"
            self.state["status"] = "running" if runtime["phase"] == "main" else "awaiting_verification"
            first_observation = not runtime.get("observed_live")
            if first_observation:
                runtime["observed_live"] = True
                self.event("process_live", task, phase=runtime["phase"], record=record)
            current_observation = (self.state["status"], task["status"], current["process_state"])
            if first_observation or current_observation != previous_observation:
                self.reporter.publish(force=True)
            return
        current["process_state"] = "recorded process is not live" if record else "no recorded live PID"
        log_path = Path(runtime["log"])
        if log_path.exists() and runtime.get("tail_offset", 0) < log_path.stat().st_size:
            return  # Drain every final event before judging successful completion.
        exit_file = Path(runtime["log"] + ".exitcode")
        wrapper_rc = None
        if exit_file.exists():
            try:
                wrapper_rc = int(exit_file.read_text().strip())
            except ValueError:
                self.fail(task, "Malformed wrapper exit evidence")
                return
        if result is None or wrapper_rc is None:
            entry_live = process_alive((record or {}).get("entry"))
            if entry_live:
                current["process_state"] += "; entry is finishing"
                return
            if runtime.get("missing_since") is None:
                runtime["missing_since"] = now()
            if (age_seconds(runtime["missing_since"]) or 0) >= 30:
                self.fail(task, "No live exact process and missing durable result/exit evidence; no automatic relaunch")
            return
        if record and (result.get("pid") != record.get("pid")
                       or str(result.get("starttime")) != str(record.get("starttime"))):
            self.fail(task, "Result/process identity mismatch")
            return
        current["exit_code"] = result.get("exit_code")
        runtime["exit_code"] = result.get("exit_code")
        runtime["wrapper_exit_code"] = wrapper_rc
        if result.get("exit_code") != 0 or wrapper_rc != 0 or result.get("error"):
            self.fail(task, "Task exit failed: command=" + str(result.get("exit_code"))
                      + " wrapper=" + str(wrapper_rc) + " " + str(result.get("error", "")))
            return
        if not record:
            self.fail(task, "Successful exit lacks exact process identity")
            return
        if task.get("event_error"):
            self.fail(task, "Task emitted error event: " + json.dumps(task["event_error"], ensure_ascii=False))
            return
        if task.get("progress_warning"):
            task["status"] = "awaiting_verification"
            changed = self.state.get("status") != "reporting_blocked"
            self.state["status"] = "reporting_blocked"
            if changed:
                self.event("downstream_blocked", task, warning=task["progress_warning"])
            self.reporter.publish(force=changed)
            return
        if not self.reporter.publish(force=True):
            return
        if runtime["phase"] == "main":
            task["status"] = "awaiting_verification"
            self.state["status"] = "awaiting_verification"
            self.event("main_complete", task, exit_code=0)
            if not self.reporter.publish(force=True):
                return
            if task.get("verify_command"):
                self.prepare_launch(task, "verify")
                return
        error = expected_outputs_error(task)
        if error:
            self.fail(task, error)
            return
        task.update({"status": "accepted", "accepted_at": now()})
        if task["id"] not in self.state["completed"]:
            self.state["completed"].append(task["id"])
        self.state.update({"status": "accepted", "active": task["id"]})
        self.event("task_accepted", task)
        self.reporter.publish(force=True)

    def step(self):
        if self.stop_requested:
            return "stopped"
        # A publication failure is retried; a healthy current task is still monitored.
        if self.reporter.write_failed:
            self.reporter.publish(force=True)
        tasks = self.state["tasks"]
        failed = next((t for t in tasks if t["status"] == "failed"), None)
        if failed:
            self.state.update({"status": "failed", "active": failed["id"], "error": failed.get("error")})
            self.reporter.publish()
            return "failed"
        active = next((t for t in tasks if t.get("runtime") and t["status"] not in TERMINAL), None)
        if active:
            self.observe(active)
            self.reporter.publish()
            return "active"
        task = next((t for t in tasks if t["status"] not in TERMINAL), None)
        if task is None:
            terminal = "phase_complete" if self.state.get("backlog") else ("complete_with_limits" if any(t["status"] == "limited" for t in tasks) else "complete")
            changed = self.state.get("status") != terminal
            self.state.update({"status": terminal, "active": None, "resource_wait": None})
            self.reporter.publish(force=changed)
            return "complete"
        accepted = set(self.state["completed"])
        unmet = [key for key in task.get("requires", []) if key not in accepted]
        if unmet:
            self.state.update({"status": "blocked", "active": task["id"],
                               "resource_wait": "Prerequisites not accepted: " + ", ".join(unmet)})
            self.reporter.publish()
            return "blocked"
        if self.reporter.write_failed:
            return "report_blocked"
        if time.monotonic() - self.last_resource_check >= 5:
            self.resource_ok, reason = gpu_available(task.get("gpu", False))
            self.last_resource_check = time.monotonic()
            self.state["resource_wait"] = reason
        if not self.resource_ok:
            self.state.update({"status": "waiting_resources", "active": task["id"]})
            task["status"] = "queued"
            self.reporter.publish()
            return "waiting_resources"
        # Recheck immediately before each GPU launch, never reuse a previous task's gate.
        ready, reason = gpu_available(task.get("gpu", False))
        if not ready:
            self.state.update({"status": "waiting_resources", "resource_wait": reason, "active": task["id"]})
            self.reporter.publish(force=True)
            return "waiting_resources"
        if self.stop_requested:
            return "stopped"
        self.state["current"] = {}
        self.prepare_launch(task, "main")
        self.resource_ok = False
        self.last_resource_check = 0
        return "launched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--extend-from", help="Explicit previous bound manifest; append-only revision at this manager restart")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--poll-interval", type=float, default=1)
    args = parser.parse_args()
    manifest_path = inside(args.manifest, PLAN)
    if args.interval <= 0 or args.poll_interval <= 0:
        parser.error("intervals must be positive")
    if not os.environ.get("STY"):
        parser.error("Launch the queue/reporter in the project screen wrapper")
    PLAN.mkdir(parents=True, exist_ok=True)
    with (PLAN / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A progress writer/queue manager already holds this plan lock")
        previous = read_json(PLAN / "status.json", {})
        manifest = read_json(manifest_path, {}) if args.report_only else read_json(manifest_path)
        if args.extend_from and args.report_only:
            parser.error("--extend-from is only valid for an execution queue")
        if not args.report_only:
            old_manifest_path = inside(args.extend_from, PLAN) if args.extend_from else None
            old_manifest = read_json(old_manifest_path) if old_manifest_path else None
            state = bind_execution_state(previous, manifest, old_manifest,
                caller_context={"operation": "explicit_manager_restart_append", "pid": os.getpid(),
                                "old_manifest": str(old_manifest_path) if old_manifest_path else None,
                                "new_manifest": str(manifest_path)})
        else:
            state = copy.deepcopy(previous)
            state.setdefault("tasks", [])
            if any(t.get("runtime") and t.get("status") not in TERMINAL for t in state["tasks"]):
                raise SystemExit("Active execution evidence exists: resume the queue instead of report-only")
        report_path = inside(manifest.get("report_path", str(REPORT)), BASE / "paper")
        reporter = Reporter(PLAN / "status.json", report_path, state, interval=args.interval,
                            warning_path=PLAN / "logs/reporter.warnings.log")
        state["manager"] = {"pid": os.getpid(), "starttime": process_identity(os.getpid())["starttime"], "started_at": now(), "mode": "report-only" if args.report_only else "queue"}
        queue = Queue(manifest_path, manifest, state, reporter)
        def stop(signum, frame):
            queue.stop_requested = True
            state["manager"]["signal"] = signum
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGHUP, stop)
        reporter.publish(force=True)
        stage_signature = None
        try:
            while not queue.stop_requested:
                if args.report_only:
                    stage = read_json(PLAN / "stage.json", {})
                    signature = json.dumps(stage, ensure_ascii=False, sort_keys=True)
                    if signature != stage_signature:
                        for key in ("status", "active", "notes", "current", "error"):
                            if key in stage:
                                state[key] = copy.deepcopy(stage[key])
                        stage_signature = signature
                        reporter.publish(force=True)
                    reporter.publish()
                else:
                    outcome = queue.step()
                    if outcome in {"failed", "complete"} and not reporter.write_failed:
                        # Remain alive for independent 60-second terminal status heartbeat.
                        state["manager"]["terminal"] = outcome
                time.sleep(min(args.poll_interval, 5))
        except BaseException as error:
            state.update({"status": "manager_error", "error": repr(error)})
            queue.event("manager_error", message=repr(error))
            reporter.publish(force=True)
            raise
        finally:
            state["manager"]["stopped_at"] = now()
            if queue.stop_requested:
                state["status"] = "monitor_stopped"
                state.setdefault("notes", []).append(
                    "管理器收到停止信号；未终止任何任务。恢复时先核验记录PID及启动标识，禁止重复启动。")
            reporter.publish(force=True)


if __name__ == "__main__":
    main()
