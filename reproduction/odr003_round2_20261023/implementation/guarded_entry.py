"""One owned screen attempt, with shared-host admission before heavy imports.

No signals, process termination, sleeps, or probing model runs. A busy host exits
75 and the caller may inspect it again after at least thirty minutes. All project
locks remain held for this attempt's lifetime. They serialize our experiments;
they are not a reservation enforceable on other lab users.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time

from frozen_backend import sha256
from host_gate import command_sha256, launch_record_sha256, assess_launch_boundary, verify_ownership

BASE = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
PLAN = BASE/'spnv2_runs/plans/odr003_round2_20261023'
LOCKS = (BASE/'spnv2_runs/plans/rebaseline_20261017/pipeline.lock',
         BASE/'spnv2_runs/plans/odr003_exploratory_20261023/pipeline.lock', PLAN/'pipeline.lock')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


@contextmanager
def project_lock(path):
    import fcntl
    # Opening an existing lock never truncates or replaces its inode.
    handle = Path(path).open('a+b')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def own_launch_record(task, log_path):
    nonce = os.environ.get('SPNV2_RUN_NONCE', '')
    if len(nonce) < 16 or not os.environ.get('STY'):
        raise RuntimeError('An independent guarded screen and fresh launch nonce are required')
    proc = Path('/proc')/str(os.getpid())
    stat = (proc/'stat').read_text()
    raw_argv = (proc/'cmdline').read_bytes()
    if not raw_argv.endswith(b'\0'):
        raise ValueError('Incomplete own process command line')
    argv = [part.decode() for part in raw_argv.split(b'\0')[:-1]]
    return dict(schema='spnv2.round2.launch.v1', project_id='odr003_round2_20261023',
                run_id=task['run_id'], run_nonce=nonce,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=os.getpid(), starttime=int(stat[stat.rfind(')')+2:].split()[19]),
                uid=os.getuid(), argv=argv, cmdline_sha256=command_sha256(argv),
                cwd=str(Path.cwd()), session=os.environ['STY'], log_path=str(log_path),
                launched_at=time.time(), role='worker')


def assert_task_path(task_path):
    task_path = Path(task_path).resolve(strict=True)
    if not task_path.is_relative_to(PLAN/'tasks') or task_path.suffix != '.json':
        raise ValueError('Task must be an immutable JSON under the current plan/tasks')
    task = json.loads(task_path.read_text())
    if task.get('schema') != 'odr003-round2-task-v1':
        raise ValueError('Wrong round-two task schema')
    for name, digest in task['implementation_sha256'].items():
        if Path(name).name != name or sha256(Path(__file__).parent/name) != digest:
            raise ValueError('Execution implementation binding mismatch: '+name)
    if not {'guarded_entry.py', 'host_gate.py', 'host_snapshot.py', 'frozen_backend.py'} <= task['implementation_sha256'].keys():
        raise ValueError('All shared-host admission dependencies must be bound')
    return task


def resource_conflict(snapshot, own_record):
    """During an admitted run, the exact own PID is allowed; nobody else is."""
    if (not snapshot.get('query_ok') or not snapshot.get('inventory_complete')
            or not snapshot.get('worker_inventory_complete') or snapshot.get('collection_errors')):
        return 'resource_observation_incomplete'
    matched = [check for check in snapshot.get('worker_checks', [])
               if check.get('expected_record', {}).get('run_nonce') == own_record['run_nonce']]
    if (len(matched) != 1 or own_record['run_nonce'] not in snapshot.get('registered_run_nonces', [])
            or not matched[0].get('probe_succeeded') or matched[0].get('process') is None
            or launch_record_sha256(matched[0]['expected_record']) != launch_record_sha256(own_record)
            or not verify_ownership(matched[0]['process'], own_record, matched[0]['observed_record']).is_own):
        return 'own_worker_registration_unverified'
    if snapshot.get('foreign_gpu_pids'):
        return 'foreign_compute_arrived'
    owned = set(snapshot.get('verified_own_gpu_pids', []))
    if owned - {own_record['pid']}:
        return 'another_owned_worker_arrived'
    for check in snapshot.get('worker_checks', []):
        process = check.get('process')
        if process is not None and process['pid'] != own_record['pid']:
            return 'another_registered_live_worker'
    return None


def main(task_path, log_path):
    attempt_started = time.monotonic()
    if os.name != 'posix':
        raise RuntimeError('Actual shared-host entry requires Linux')
    task = assert_task_path(task_path)
    from host_snapshot import collect_snapshot
    registry_path = PLAN/'launches/registry.json'
    nonce = os.environ.get('SPNV2_RUN_NONCE', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{16,128}', nonce):
        raise ValueError('A safe fresh launch nonce is required')
    attempt_dir = PLAN/'launches'/nonce
    attempt_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(attempt_dir/'attempt_started.json', dict(started_at=time.time(),
                task_path=str(task_path), task_sha256=sha256(task_path), run_nonce=nonce))
    def outcome(value):
        value = {**value, 'attempt_elapsed_seconds': time.monotonic()-attempt_started,
                 'finished_at': time.time()}
        atomic_json(attempt_dir/'outcome.json', value)
        return value
    try:
        with ExitStack() as stack:
            for lock in LOCKS:
                stack.enter_context(project_lock(lock))
            if not registry_path.exists():
                atomic_json(registry_path, dict(schema='spnv2.round2.registry.v1',
                            project_id='odr003_round2_20261023', records=[]))
            before = collect_snapshot(registry_path, project_root=PLAN, project_id='odr003_round2_20261023')
            after = collect_snapshot(registry_path, project_root=PLAN, project_id='odr003_round2_20261023')
            admission = assess_launch_boundary(before, after, now=time.time(),
                                               project_lock_held=True, task_kind='gpu')
            atomic_json(attempt_dir/'admission.json', dict(before=before, after=after,
                        decision=asdict(admission), task_sha256=sha256(task_path)))
            if not admission.allow_start:
                outcome(dict(status='waiting_resources', exit_code=75,
                            reasons=admission.reasons, retry_not_before=time.time()+1800,
                            shared_model_imported=False, experiments_started=0))
                return 75
            if time.time() > admission.valid_until:
                raise RuntimeError('Launch admission expired before ownership registration')
            record = own_launch_record(task, log_path)
            record_path = attempt_dir/'launch.json'
            atomic_json(record_path, record)
            registry = json.loads(registry_path.read_text())
            registry['records'].append(dict(record_path=str(record_path),
                          record_sha256=launch_record_sha256(record), expected_record=record))
            atomic_json(registry_path, registry)
            def conflict_check():
                return resource_conflict(collect_snapshot(registry_path, project_root=PLAN,
                                         project_id='odr003_round2_20261023'), record)
            # Read back our actual birth/nonce and registry before importing any
            # model module, even though this worker has not allocated CUDA yet.
            conflict = conflict_check()
            if conflict or time.time() > admission.valid_until:
                reason = conflict or 'admission_expired_after_registration'
                outcome(dict(status='waiting_resources', exit_code=75,
                            reason=reason, retry_not_before=time.time()+1800, experiments_started=0))
                return 75
            if task.get('action', 'run') == 'verify_engineering':
                from verify_engineering_phase import execute
            elif task.get('action', 'run') == 'run':
                from run_round2 import execute
            else:
                raise ValueError('Unknown experiment action')
            result = execute(Path(task_path), admission=asdict(admission), launch_record=record,
                              conflict_check=conflict_check)
            outcome(result)
            return int(result['exit_code'])
    except BlockingIOError:
        outcome(dict(status='waiting_project_lock', exit_code=75,
                    retry_not_before=time.time()+1800, experiments_started=0))
        return 75
    except Exception as error:
        outcome(dict(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task-json', required=True)
    p.add_argument('--log-path', required=True)
    args = p.parse_args()
    raise SystemExit(main(args.task_json, args.log_path))
