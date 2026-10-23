"""Owned autonomous queue for engineering and the approved finite exploration.

Waiting is server-side, low frequency and read-only. Never signals any process.
An ambiguous launch or unexpected failure stops advancement. Passed independent
engineering unlocks P6, accepted P counts unlock controls, and all accepted runs
unlock the audited summary. Historical source training remains outside scope.
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
import complete_phase_tools as complete_tools
from audit_revision import validate_revision
from queue_control import pending_refreshes, acknowledge

PYTHON = '/home/ps/anaconda3/envs/SPNv2/bin/python'
WRAPPER = BASE/'spnv2_runs/scripts/run_in_screen_guarded.sh'
PROJECT = 'odr003_round2_20261023'
RESOURCE_INTERVAL = 300


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
             f"已完成工程轨迹：{len(state['completed'])}/8；完整 ODR已独立验收：{len(state.get('complete_accepted', {}))}/18。",
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
                next_resource_check=0., verified=False, manager_pid=os.getpid(),
                complete_main={}, complete_accepted={}, complete_index=None,
                engineering_verified=False, complete_phase='not_started', resource_refresh_history=[])


def current_manager_identity():
    own_proc = Path('/proc')/str(os.getpid())
    stat = (own_proc/'stat').read_text()
    raw_command = (own_proc/'cmdline').read_bytes()
    return dict(pid=os.getpid(), starttime=int(stat[stat.rfind(')')+2:].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                argv=[x.decode() for x in raw_command.split(b'\0')[:-1]],
                cmdline_sha256=hashlib.sha256(raw_command).hexdigest(),
                session=os.environ['STY'], started_at=time.time(), manager_lock=str(PLAN/'manager.lock'),
                manager_nonce=os.environ['SPNV2_MANAGER_NONCE'])


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
        receipt = dict(run_id=a['run_id'], log_path=a['log_path'],
                  exit_code_path=str(exit_path), report_path=str(report), report_sha256=sha256(report),
                  launch_record_path=str(record_path), launch_record_sha256=canonical_sha(read(record_path)),
                  outcome_path=str(outcome_path),
                  attempts=[item for item in state['attempts'] if item['run_id'] == a['run_id']])
        task = read(a['task_path'])
        if task['stage'] == 'engineering':
            if a['run_id'] in {row['run_id'] for row in state['completed']}:
                raise ValueError('Already completed engineering result cannot be replaced')
            state['completed'].append(receipt)
        elif task['stage'] == 'complete':
            if a['run_id'] in state['complete_main']:
                raise ValueError('Already completed ODR result cannot be replaced')
            state['complete_main'][a['run_id']] = receipt
        else:
            raise ValueError('Model report from unauthorized stage')
        state.update(status='between_tasks', active=None, waiting_since=time.time(), next_resource_check=0.)
    elif rc == 0 and outcome['status'] in ('engineering_passed', 'engineering_held'):
        state.update(status=outcome['status'], active=None, verified=True,
                     engineering_verified=True, reason=outcome.get('reason'),
                     waiting_since=time.time() if outcome['status'] == 'engineering_passed' else None)
    elif rc == 0 and outcome['status'] == 'complete_run_verified':
        verify_spec = read(a['task_path'])
        science_task = complete_tools.checked(verify_spec['run_task'])
        science_id = science_task['run_id']
        if science_id not in state['complete_main'] or science_id in state['complete_accepted']:
            raise ValueError('Verification has no unique pending complete result')
        proof_path = Path(verify_spec['proof_path'])
        proof = read(proof_path)
        report_path = Path(state['complete_main'][science_id]['report_path'])
        if (proof.get('status') != 'passed' or proof.get('report_sha256') != sha256(report_path)
                or outcome.get('run_id') != science_id or Path(outcome['proof']) != proof_path):
            raise ValueError('Independent complete proof does not bind the actual report')
        state['complete_accepted'][science_id] = dict(report=complete_tools.binding(report_path),
                proof=complete_tools.binding(proof_path), verification_attempt=a,
                run_task=verify_spec['run_task'])
        state.update(status='between_tasks', active=None, waiting_since=time.time(), next_resource_check=0.)
    elif rc == 0 and outcome['status'] == 'round2_delivered':
        task = read(a['task_path'])
        proof = read(task['proof_path'])
        if (proof.get('status') != 'passed' or proof.get('summary_sha256') != sha256(task['summary_path'])
                or task.get('action') != 'summarize_round2'):
            raise ValueError('Final summary lacks independent delivery proof')
        state.update(status='complete', active=None, verified=True, waiting_since=None,
                     delivery=dict(summary=complete_tools.binding(task['summary_path']),
                                   proof=complete_tools.binding(task['proof_path']), verification_attempt=a),
                     reason='Approved finite round-two exploration delivered; optional old stages remain deferred')
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
    if index.get('pre_forward_repair') is not None:
        task['pre_forward_repair'] = index['pre_forward_repair']
    # Preserve the failed verifier's task bytes and process evidence. Only its
    # CPU audit is rerun; completed scientific trajectories remain immutable.
    filename = ('g50p2_engineering_verification_audit_repair.json' if index.get('auditor_repair')
                else 'g50p2_engineering_verification.json')
    path = PLAN/'tasks'/filename
    if path.exists() and read(path) != task: raise ValueError('Verification task identity changed')
    if not path.exists(): atomic_json(path, task)
    return path, task['run_id']


def next_task(index, state):
    """Only immutable accepted evidence unlocks the next authorized task."""
    done = {row['run_id'] for row in state['completed']}
    pending = [row for row in index['tasks'] if row['run_id'] not in done]
    if pending:
        if index.get('auditor_repair'):
            raise ValueError('Auditor-only revision cannot launch/repeat an engineering trajectory')
        row = pending[0]
        if sha256(row['task_path']) != row['task_sha256']:
            raise ValueError('Engineering task identity changed')
        return Path(row['task_path']), row['run_id']
    if not state.get('engineering_verified'):
        return verification_task(index, state)
    if state.get('status') == 'engineering_held':
        return None
    if state.get('complete_index') is None:
        path, compiled = complete_tools.prepare_P(PLAN, index)
        state.update(complete_index=complete_tools.binding(path), complete_phase='P', status='between_tasks')
    else:
        compiled = complete_tools.checked(state['complete_index'])
    accepted = state['complete_accepted']
    for row in compiled['tasks']:
        if row['run_id'] in accepted:
            complete_tools.checked(accepted[row['run_id']]['report'])
            complete_tools.checked(accepted[row['run_id']]['proof'])
            continue
        receipt = state['complete_main'].get(row['run_id'])
        if receipt is not None:
            return complete_tools.verify_task(PLAN, compiled, row, receipt,
                         queue_wait_seconds=state['queue_wait_seconds'])
        if sha256(row['task_path']) != row['task_sha256']:
            raise ValueError('Complete task identity changed')
        return Path(row['task_path']), row['run_id']
    if compiled['phase'] == 'P':
        path, compiled = complete_tools.prepare_controls(PLAN, index, compiled, accepted)
        state.update(complete_index=complete_tools.binding(path), complete_phase='controls')
        return next_task(index, state)
    return complete_tools.summary_task(PLAN, state['complete_index']['path'], compiled, state)


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
        for key, value in dict(complete_main={}, complete_accepted={}, complete_index=None,
                               engineering_verified=state.get('verified', False),
                               complete_phase='not_started', resource_refresh_history=[]).items():
            state.setdefault(key, value)
        known_ids = {item['run_id'] for item in index['tasks']}
        completed_ids = [item['run_id'] for item in state['completed']]
        if len(completed_ids) != len(set(completed_ids)) or not set(completed_ids) <= known_ids:
            raise ValueError('Completed records duplicate or escape this engineering matrix')
        if index.get('auditor_repair') and set(completed_ids) != known_ids:
            raise ValueError('Auditor-only revision requires all eight completed producers')
        if len(state['complete_main']) > 18 or not set(state['complete_accepted']) <= set(state['complete_main']):
            raise ValueError('Complete ledger exceeds authorized scope or lacks actual results')
        if not state.get('active'):
            with ExitStack() as stack:
                for path in LOCKS: stack.enter_context(project_lock(path))
        if state['index_sha256'] != sha256(index_path): raise ValueError('Queue identity changed')
        state['manager_pid'] = os.getpid()
        state['manager'] = current_manager_identity()
        state['resource_interval_seconds'] = RESOURCE_INTERVAL
        state['next_resource_check'] = min(state['next_resource_check'], time.time()+RESOURCE_INTERVAL)
        registry = PLAN/'launches/registry.json'
        if not registry.exists(): atomic_json(registry, dict(schema='spnv2.round2.registry.v1',project_id=PROJECT,records=[]))
        last_write = 0.
        while True:
            if state['status'] in ('blocked', 'complete', 'engineering_held'):
                write_progress(state)
                return 1 if state['status'] == 'blocked' else 0
            requests = pending_refreshes(PLAN, state['manager'], now=time.time())
            if state['active']:
                _finish_attempt(state)
                if requests:
                    snapshot = collect_snapshot(registry, project_root=PLAN, project_id=PROJECT)
                    state['resource_snapshot'] = snapshot
                    acknowledge(PLAN, requests, snapshot=snapshot, status='sampled_existing_worker_preserved', now=time.time())
                    state['resource_refresh_history'].extend(r['request_id'] for r in requests)
                if state['active'] is None or state['status'] == 'blocked':
                    write_progress(state); last_write = time.monotonic()
            else:
                task = next_task(index, state)
                if task is None:
                    write_progress(state)
                    return 0
                if requests or time.time() >= state['next_resource_check']:
                    snapshot = collect_snapshot(registry, project_root=PLAN, project_id=PROJECT)
                    decision = assess_snapshot(snapshot, now=time.time(), task_kind='gpu')
                    state['resource_snapshot'] = snapshot
                    state['resource_sampled_at'] = snapshot.get('captured_at')
                    if requests:
                        acknowledge(PLAN, requests, snapshot=snapshot,
                                    status='idle_admitted' if decision.allow_start else 'sampled_busy', now=time.time())
                        state['resource_refresh_history'].extend(r['request_id'] for r in requests)
                    if decision.allow_start:
                        state['reason'] = None
                        _launch(state, *task)
                    else:
                        state.update(status='waiting_resources', reason=list(decision.reasons),
                                     next_resource_check=time.time()+RESOURCE_INTERVAL)
            if time.monotonic()-last_write >= 60:
                write_progress(state); last_write = time.monotonic()
            # Five-second private mailbox reads do not query the GPU. Ordinary
            # resource snapshots use the explicitly authorized five-minute wait.
            time.sleep(5)


def validate_engineering_index(index):
    producer_code = (validate_revision(index, PLAN/'implementation', plan=PLAN)
                     if index.get('auditor_repair') else index['implementation_sha256'])
    rows = index.get('tasks', [])
    if (index.get('schema') != 'odr003-round2-engineering-index-v1' or len(rows) != 8
            or [item['definition'] for item in rows] != build_engineering_matrix()
            or len({item['run_id'] for item in rows}) != 8):
        raise ValueError('Only the exact eight unique engineering trajectories may enter this queue')
    if index.get('pre_forward_repair_required'):
        binding = index.get('pre_forward_repair')
        if not binding or sha256(binding['path']) != binding['sha256']:
            raise ValueError('Zero-progress repair evidence must pass and be bound before dispatch')
    for item in rows:
        path = Path(item['task_path']).resolve(strict=True)
        if not path.is_relative_to(PLAN/'tasks') or sha256(path) != item['task_sha256']:
            raise ValueError('Engineering task binding/path changed')
        task = read(path)
        if (task['run_id'] != item['run_id'] or task['definition'] != item['definition']
                or task.get('stage') != 'engineering' or task.get('action', 'run') != 'run'
                or task['implementation_sha256'] != producer_code):
            raise ValueError('Non-engineering or inconsistent task in engineering-only queue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    raise SystemExit(main(parser.parse_args().index))
