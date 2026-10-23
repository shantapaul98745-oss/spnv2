"""Owned server queue for eight engineering trajectories and independent review.

Waiting is server-side, low frequency and read-only. Never signals any process.
An ambiguous launch or unexpected failure stops advancement. This queue cannot
activate the historical 150 source-training epochs or complete ODR templates.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from guarded_entry import PLAN, BASE, atomic_json, project_lock, LOCKS
from host_gate import assess_snapshot, launch_record_sha256, command_sha256
from host_snapshot import collect_snapshot
from frozen_backend import sha256, canonical_sha
from round2_protocol import build_engineering_matrix

PYTHON = '/home/ps/anaconda3/envs/SPNv2/bin/python'
WRAPPER = BASE/'spnv2_runs/scripts/run_in_screen_guarded.sh'
PROJECT = 'odr003_round2_20261023'
RESOURCE_INTERVAL = 1800


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def process_matches(record):
    """Only inspect the exact process recorded for this owned attempt."""
    try:
        proc = Path('/proc')/str(record['pid'])
        stat = (proc/'stat').read_text()
        parts = stat[stat.rfind(')')+2:].split()
        cmd = (proc/'cmdline').read_bytes()
        return (int(parts[19]) == record['starttime']
                and hashlib.sha256(cmd).hexdigest() == record['cmdline_sha256']
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id'])
    except FileNotFoundError:
        return False


def write_progress(state):
    state['heartbeat_at'] = time.time()
    state['queue_wait_seconds_reported'] = (state['queue_wait_seconds'] +
            (time.time()-state['waiting_since'] if state['waiting_since'] is not None else 0.))
    atomic_json(PLAN/'status.json', state)
    active = state.get('active')
    lines = ['# SPNv2 实时进度', '',
             '当前方向：已确认 G50-P 第二轮；最多18次完整 ODR，先完成8条工程轨迹。',
             f"状态：{state['status']}",
             f"已完成工程轨迹：{len(state['completed'])}/8；完整 ODR：0/18（尚未启用）。",
             f"当前任务：{active['run_id'] if active else '无运行 worker'}",
             f"排队等待：{state['queue_wait_seconds_reported']:.1f} 秒，与计算时间分开。",
             '共享主机有其他计算进程时等待；不暂停、终止或重启其他人的任务。',
             '旧150轮 source 训练保持暂缓，目标模式保持暂停。',
             f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}", '']
    if state.get('reason'): lines.extend(['说明：'+str(state['reason']), ''])
    path = BASE/'paper/SPNv2_复现实时进度.md'
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf8') as handle:
        handle.write('\n'.join(lines)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _new_state(index_path):
    return dict(schema='odr003-round2-queue-status-v1', status='preparing',
                index_path=str(index_path), index_sha256=sha256(index_path), active=None,
                completed=[], attempts=[], queue_wait_seconds=0., waiting_since=time.time(),
                next_resource_check=0., verified=False, manager_pid=os.getpid())


def current_manager_identity():
    own_proc = Path('/proc')/str(os.getpid())
    stat = (own_proc/'stat').read_text()
    raw_command = (own_proc/'cmdline').read_bytes()
    return dict(pid=os.getpid(), starttime=int(stat[stat.rfind(')')+2:].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                argv=[x.decode() for x in raw_command.split(b'\0')[:-1]],
                cmdline_sha256=hashlib.sha256(raw_command).hexdigest(),
                session=os.environ['STY'], started_at=time.time(), manager_lock=str(PLAN/'manager.lock'))


def _launch(state, task_path, run_id):
    nonce = uuid.uuid4().hex
    session = 'spnv2_g50p2_'+nonce[:16]
    log = PLAN/'logs'/(run_id+'_'+nonce+'.log')
    log.parent.mkdir(exist_ok=True)
    attempt = dict(run_id=run_id, task_path=str(task_path), task_sha256=sha256(task_path),
                   nonce=nonce, session=session, log_path=str(log), queued_at=time.time(),
                   launch_state='intent_persisted',
                   launch_directory=str(PLAN/'launches'/nonce))
    state['active'] = attempt
    state['attempts'].append(attempt)
    state['status'] = 'launching'
    write_progress(state)  # Failure here prevents launching the worker.
    env = dict(os.environ, SPNV2_RUN_NONCE=nonce, PYTHONNOUSERSITE='1', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONHASHSEED='2021', TORCH_HOME=str(BASE/'spnv2_runs/env/cache/torch'))
    command = ['bash', str(WRAPPER), session, str(log), PYTHON, '-B', '-s', '-u',
               str(PLAN/'implementation/guarded_entry.py'), '--task-json', str(task_path), '--log-path', str(log)]
    result = subprocess.run(command, cwd=PLAN, env=env, capture_output=True, text=True, timeout=20)
    attempt['wrapper_returncode'] = result.returncode
    attempt['wrapper_output'] = result.stdout+result.stderr
    attempt['launch_state'] = 'wrapper_returned'
    if result.returncode:
        state.update(status='blocked', reason='Wrapper launch failed; preserve ambiguous attempt, no retry')
    else:
        state['status'] = 'running'
        if state['waiting_since'] is not None:
            state['queue_wait_seconds'] += time.time()-state['waiting_since']
            state['waiting_since'] = None
    write_progress(state)


def _finish_attempt(state):
    a = state['active']
    exit_path = Path(a['log_path']+'.exitcode')
    launch_dir = Path(a['launch_directory'])
    if not exit_path.exists():
        record_path = launch_dir/'launch.json'
        if record_path.exists():
            record = read(record_path)
            if not process_matches(record) and time.time()-a['queued_at'] > 120:
                state.update(status='blocked', reason='Owned worker disappeared without wrapper exit evidence')
        elif time.time()-a['queued_at'] > 120:
            state.update(status='blocked', reason='Launch identity unavailable; do not issue a duplicate launch')
        return
    rc = int(exit_path.read_text().strip())
    a['exit_code'] = rc
    outcome_path = launch_dir/'outcome.json'
    if not outcome_path.exists():
        state.update(status='blocked', reason='Worker exit lacks owned attempt outcome')
        return
    outcome = read(outcome_path)
    a['outcome'] = outcome
    matching = [i for i, item in enumerate(state['attempts']) if item['nonce'] == a['nonce']]
    if len(matching) != 1:
        state.update(status='blocked', reason='Attempt ledger identity missing or duplicated')
        return
    state['attempts'][matching[0]] = a
    if rc != outcome['exit_code']:
        state.update(status='blocked', reason='Wrapper and owned outcome exit codes disagree')
        return
    if rc == 75:
        state.update(status='waiting_resources', active=None, waiting_since=time.time(),
                     next_resource_check=time.time()+RESOURCE_INTERVAL)
    elif rc == 76 and outcome['status'] == 'controlled_interruption':
        task = read(a['task_path'])
        point = outcome.get('point')
        previous = [item['outcome'].get('point') for item in state['attempts']
                    if item['run_id'] == a['run_id'] and item['nonce'] != a['nonce']
                    and item.get('outcome', {}).get('status') == 'controlled_interruption']
        if (sha256(a['task_path']) != a['task_sha256'] or task.get('stage') != 'engineering'
                or task.get('definition', {}).get('trajectory') != 'P_interrupted_resumed'
                or point not in ('candidate_2_tentative', 'observation_31')
                or point in previous or len(previous) >= 2):
            state.update(status='blocked', reason='Unapproved/repeated controlled interruption; no further resume')
            return
        state.update(status='resume_required', active=None, waiting_since=time.time(), next_resource_check=0.)
    elif rc == 0 and outcome['status'] == 'main_complete':
        report = Path(outcome['report'])
        record_path = launch_dir/'launch.json'
        state['completed'].append(dict(run_id=a['run_id'], log_path=a['log_path'],
                  exit_code_path=str(exit_path), report_path=str(report), report_sha256=sha256(report),
                  launch_record_path=str(record_path), launch_record_sha256=canonical_sha(read(record_path)),
                  outcome_path=str(outcome_path),
                  attempts=[copy for copy in state['attempts'] if copy['run_id'] == a['run_id']]))
        state.update(status='between_tasks', active=None, waiting_since=time.time(), next_resource_check=0.)
    elif rc == 0 and outcome['status'] in ('engineering_passed', 'engineering_held'):
        state.update(status=outcome['status'], active=None, verified=True,
                     reason=outcome.get('reason'), waiting_since=None)
    else:
        state.update(status='blocked', reason='Unexpected experiment/verification failure; no automatic retry')


def verification_task(index, state):
    receipts = dict(schema='odr003-round2-engineering-receipts-v1', runs=state['completed'],
                    all_attempts=state['attempts'], queue_wait_seconds=state['queue_wait_seconds'])
    receipts_path = PLAN/'engineering_receipts.json'
    if receipts_path.exists():
        if read(receipts_path)['runs'] != receipts['runs']:
            raise ValueError('Completed engineering receipts changed')
    else:
        atomic_json(receipts_path, receipts)
    task = dict(schema='odr003-round2-task-v1', action='verify_engineering', stage='verification',
                run_id='g50p2_engineering_verification', protocol=read(index['tasks'][0]['task_path'])['protocol'],
                implementation_sha256=index['implementation_sha256'],
                engineering_index=dict(path=state['index_path'], sha256=state['index_sha256']),
                synthetic_contracts=index['synthetic_contracts'],
                execution_receipts=dict(path=str(receipts_path), sha256=sha256(receipts_path)))
    path = PLAN/'tasks/g50p2_engineering_verification.json'
    if path.exists() and read(path) != task: raise ValueError('Verification task identity changed')
    if not path.exists(): atomic_json(path, task)
    return path, task['run_id']


def main(index_path):
    if os.name != 'posix' or not os.environ.get('STY'):
        raise RuntimeError('Server manager must use its own guarded screen')
    index_path = Path(index_path).resolve(strict=True)
    if index_path != PLAN/'engineering_index.json': raise ValueError('Unexpected plan index')
    index = read(index_path)
    validate_engineering_index(index)
    for name, expected in index['implementation_sha256'].items():
        if sha256(PLAN/'implementation'/name) != expected:
            raise ValueError('Frozen implementation drift')
    with project_lock(PLAN/'manager.lock'):
        status_path = PLAN/'status.json'
        state = read(status_path) if status_path.exists() else _new_state(index_path)
        known_ids = {item['run_id'] for item in index['tasks']}
        completed_ids = [item['run_id'] for item in state['completed']]
        if len(completed_ids) != len(set(completed_ids)) or not set(completed_ids) <= known_ids:
            raise ValueError('Completed records duplicate or escape this engineering matrix')
        if not state.get('active'):
            with ExitStack() as stack:
                for path in LOCKS: stack.enter_context(project_lock(path))
                # Do not overwrite another healthy project's progress/pipeline.
        if state['index_sha256'] != sha256(index_path): raise ValueError('Queue identity changed')
        state['manager_pid'] = os.getpid()
        state['manager'] = current_manager_identity()
        registry = PLAN/'launches/registry.json'
        if not registry.exists(): atomic_json(registry, dict(schema='spnv2.round2.registry.v1',project_id=PROJECT,records=[]))
        last_write = 0.
        while True:
            if state['status'] in ('blocked', 'engineering_passed', 'engineering_held'):
                write_progress(state)
                return 0 if state['status'] != 'blocked' else 1
            if state['active']:
                _finish_attempt(state)
                if state['active'] is None or state['status'] == 'blocked':
                    write_progress(state)
                    last_write = time.monotonic()
            elif time.time() >= state['next_resource_check']:
                snapshot = collect_snapshot(registry, project_root=PLAN, project_id=PROJECT)
                decision = assess_snapshot(snapshot, now=time.time(), task_kind='gpu')
                state['resource_snapshot'] = snapshot
                if decision.allow_start:
                    done = {r['run_id'] for r in state['completed']}
                    pending = [r for r in index['tasks'] if r['run_id'] not in done]
                    if pending:
                        task_path, run_id = Path(pending[0]['task_path']), pending[0]['run_id']
                        if sha256(task_path) != pending[0]['task_sha256']: raise ValueError('Task identity changed')
                    else:
                        task_path, run_id = verification_task(index, state)
                    _launch(state, task_path, run_id)
                else:
                    state.update(status='waiting_resources', reason=list(decision.reasons),
                                 next_resource_check=time.time()+RESOURCE_INTERVAL)
            if time.monotonic()-last_write >= 60:
                write_progress(state); last_write = time.monotonic()
            # This is the dedicated server's event loop, not chat polling. GPU
            # reads occur at launch boundaries or at the 30-minute wait interval.
            time.sleep(5)


def validate_engineering_index(index):
    rows = index.get('tasks', [])
    if (index.get('schema') != 'odr003-round2-engineering-index-v1' or len(rows) != 8
            or [item['definition'] for item in rows] != build_engineering_matrix()
            or len({item['run_id'] for item in rows}) != 8):
        raise ValueError('Only the exact eight unique engineering trajectories may enter this queue')
    for item in rows:
        path = Path(item['task_path']).resolve(strict=True)
        if not path.is_relative_to(PLAN/'tasks') or sha256(path) != item['task_sha256']:
            raise ValueError('Engineering task binding/path changed')
        task = read(path)
        if (task['run_id'] != item['run_id'] or task['definition'] != item['definition']
                or task.get('stage') != 'engineering' or task.get('action', 'run') != 'run'
                or task['implementation_sha256'] != index['implementation_sha256']):
            raise ValueError('Non-engineering or inconsistent task in engineering-only queue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    raise SystemExit(main(parser.parse_args().index))
