"""Run exactly one queue command; durable identity and result evidence."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from progress import atomic_json, now, read_json


def process_identity(pid):
    try:
        text = Path("/proc/" + str(int(pid)) + "/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return {"pid": int(pid), "starttime": fields[19], "state": fields[0]}
    except (OSError, ValueError, IndexError, TypeError):
        return None


def process_alive(record):
    if not record or not record.get("pid") or not record.get("starttime"):
        return False
    actual = process_identity(record["pid"])
    return bool(actual and actual["starttime"] == str(record["starttime"]) and actual["state"] != "Z")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    record_path, result_path = Path(config["record"]), Path(config["result"])
    if record_path.exists() or result_path.exists():
        raise SystemExit("Refusing duplicate command entry: existing identity/result")
    command = config["command"]
    print("SPNV2_ENTRY " + json.dumps({
        "command": command, "cwd": config["cwd"], "phase": config["phase"],
        "stdout": config["log"], "stderr": config["log"], "config": args.config,
    }, ensure_ascii=False), flush=True)
    child = None
    interrupted = []

    def forward(signum, frame):
        interrupted.append(signum)
        # Only this entry's exact child can receive a forwarded signal.
        if child is not None and child.poll() is None and process_alive(identity):
            child.send_signal(signum)

    identity = None
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGHUP, forward)
    started = now()
    environment = os.environ.copy()
    environment.update({
        "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MPLBACKEND": "Agg",
    })
    cache_root = Path(config["config"]).parent / "cache" / Path(config["config"]).name.removesuffix(".config.json")
    cache_root.mkdir(parents=True, exist_ok=True)
    environment.update({"MPLCONFIGDIR": str(cache_root / "matplotlib"),
                        "XDG_CACHE_HOME": str(cache_root / "xdg")})
    environment.update(config.get("env") or {})
    try:
        child = subprocess.Popen(command, cwd=config["cwd"], env=environment)
        identity = process_identity(child.pid)
        if identity is None:
            raise RuntimeError("Cannot record task process identity")
        record = dict(identity)
        record.update({
            "entry": process_identity(os.getpid()), "started_at": started,
            "screen_id": os.environ.get("STY"), "command": command, "cwd": config["cwd"],
            "stdout": config["log"], "stderr": config["log"],
            "config": args.config, "phase": config["phase"],
        })
        atomic_json(record_path, record)
        rc = child.wait()
        atomic_json(result_path, {
            "exit_code": rc, "started_at": started, "ended_at": now(),
            "pid": identity["pid"], "starttime": identity["starttime"],
            "signals": interrupted, "command": command, "phase": config["phase"],
        })
        print("SPNV2_ENTRY_RESULT " + json.dumps({"exit_code": rc, "phase": config["phase"]}), flush=True)
        return rc if rc >= 0 else 128 - rc
    except BaseException as error:
        # Never leave an unrecorded healthy command behind if recording itself failed.
        if child is not None and child.poll() is None and not record_path.exists():
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        atomic_json(result_path, {
            "exit_code": 125, "started_at": started, "ended_at": now(),
            "pid": child.pid if child else None,
            "starttime": (identity or {}).get("starttime"), "error": repr(error),
            "command": command, "phase": config["phase"],
        })
        raise


if __name__ == "__main__":
    sys.exit(main())
