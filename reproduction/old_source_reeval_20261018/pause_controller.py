"""One-shot, bounded old-source evaluation while the exact phi6 process is stopped.

Does not edit the production manifest/state or restart any production process.
The independent watchdog is armed before SIGSTOP. Both recovery paths first
remove only this run's evaluation processes, then resume the recorded trainer.
"""
import argparse
import contextlib
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
P = BASE / 'spnv2_runs/plans/rebaseline_20261017'
Q = BASE / 'spnv2_runs/plans/old_source_reeval_20261018'
PYTHON = '/home/ps/anaconda3/envs/SPNv2/bin/python'
WRAPPER = BASE / 'spnv2_runs/scripts/run_in_screen_guarded.sh'
ENTRY = P / 'pipeline/task_entry.py'
REPORT = BASE / 'paper/SPNv2_旧模型补评进度.md'
TRAIN = {'pid': 2584192, 'starttime': '527383895'}
MANAGER = {'pid': 1851843, 'starttime': '526573814'}
MANAGER_COMMAND = [PYTHON, '-B', '-s', '-u', str(P / 'pipeline/runner.py'), '--manifest',
                   str(P / 'queue_observed_stage_20261018.json'), '--revise-pending-from',
                   str(P / 'queue_v2.json')]
for _task in ('eval_phi3_synthetic', 'eval_phi3_lightbox', 'eval_phi3_sunlamp',
              'eval_phi6_synthetic', 'eval_phi6_lightbox', 'eval_phi6_sunlamp'):
    MANAGER_COMMAND += ['--pending-task', _task]
PRODUCTION_SHA = 'a6f82b1efc62ebb6223e1cfec4919e44de35424b53709230f39b0655ea3e27a5'
MAX_PAUSE = 30 * 60
MAX_TASK = 20 * 60
HEARTBEAT_TIMEOUT = 90


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inside(path, root=Q):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path == root or root in path.parents, 'Path outside run: ' + str(path))
    return path


def identity(pid, proc=Path('/proc')):
    try:
        directory = proc / str(int(pid))
        stat = (directory / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        command = (directory / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        return dict(pid=int(pid), starttime=fields[19], state=fields[0], ppid=int(fields[1]),
                    command=[s.decode(errors='replace') for s in command])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def matches(actual, expected):
    return bool(actual and expected and actual['pid'] == int(expected['pid'])
                and str(actual['starttime']) == str(expected['starttime'])
                and actual.get('state') != 'Z'
                and (not expected.get('command') or actual.get('command') == expected['command']))


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid, 0)
    # Linux x86_64/aarch64 generic syscall numbers; refuse other platforms.
    require(os.uname().machine in {'x86_64', 'aarch64'}, 'pidfd fallback unavailable on this architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(434, int(pid), 0)
    if result < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    return result


def _pidfd_signal(expected, signum):
    fd = _pidfd_open(int(expected['pid']))
    try:
        require(matches(identity(expected['pid']), expected), 'Identity changed after pidfd open')
        if hasattr(signal, 'pidfd_send_signal'):
            signal.pidfd_send_signal(fd, signum, None, 0)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.syscall(424, fd, int(signum), ctypes.c_void_p(), 0) < 0:
                raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')
    finally:
        os.close(fd)


def signal_exact(expected, signum, reader=None, sender=None):
    reader, sender = reader or identity, sender or _pidfd_signal
    if not matches(reader(expected['pid']), expected):
        return False
    sender(expected, signum)
    return True


@contextlib.contextmanager
def lock(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def lock_owned(pid, path, text=None):
    stat = Path(path).stat()
    for line in (Path('/proc/locks').read_text() if text is None else text).splitlines():
        words = line.split()
        if len(words) < 6 or words[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        try:
            major, minor, inode = words[5].split(':')
            if int(words[4]) == pid and int(inode) == stat.st_ino and os.makedev(int(major, 16), int(minor, 16)) == stat.st_dev:
                return True
        except ValueError:
            continue
    return False


def production_gate(state, worker, manager, manifest, clock=None):
    """Pure production-state gate, also used by CPU fixtures."""
    require(canonical(manifest) == PRODUCTION_SHA and state.get('manifest_sha256') == PRODUCTION_SHA,
            'Production manifest binding changed')
    require(state.get('status') == 'running' and state.get('active') == 'train_phi6_EHS',
            'Production is not the expected running phi6 task')
    for key in ('error', 'downstream_blocked', 'reporting_warning'):
        require(not state.get(key), 'Production warning/error: ' + key)
    require(matches(worker, TRAIN) and matches(manager, MANAGER), 'Production process identity changed')
    require(matches(manager, state.get('manager')), 'Recorded manager differs')
    task = next((t for t in state.get('tasks', []) if t.get('id') == 'train_phi6_EHS'), None)
    require(task and task.get('status') == 'running' and task.get('runtime', {}).get('phase') == 'main',
            'Production training phase changed')
    require(not task.get('event_error') and not task.get('progress_warning'), 'Production task has an unresolved warning')
    require(worker.get('command') == task['command'], 'Training command changed')
    require(manager.get('command') == MANAGER_COMMAND, 'Manager command changed')
    current = state.get('current', {})
    require(int(current.get('pid', -1)) == TRAIN['pid'] and str(current.get('starttime')) == TRAIN['starttime'],
            'Current training record differs')
    stamp = dt.datetime.fromisoformat(state['heartbeat_at']).timestamp()
    age = (time.time() if clock is None else clock) - stamp
    require(-5 <= age <= 180, 'Production heartbeat stale')
    return task


def check_production():
    state = read(P / 'status.json')
    task = production_gate(state, identity(TRAIN['pid']), identity(MANAGER['pid']),
                           read(P / 'queue_observed_stage_20261018.json'))
    require(lock_owned(MANAGER['pid'], P / 'pipeline.lock'), 'Manager no longer holds production lock')
    record = read(task['runtime']['record'])
    require(matches(identity(TRAIN['pid']), record), 'Durable training identity differs')
    log = Path(task['log'])
    return dict(worker=identity(TRAIN['pid']), manager=identity(MANAGER['pid']),
                manifest_sha256=state['manifest_sha256'], log=str(log), log_offset=log.stat().st_size,
                last_event=state.get('current', {}), checked_at=now())


def gpu():
    def query(args):
        return subprocess.run(['nvidia-smi', '-i', '0'] + args, check=True, text=True,
                              capture_output=True, timeout=15).stdout
    values = query(['--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
    used, free, utilization = [int(x.strip()) for x in values.strip().split(',')]
    rows = query(['--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
    processes = []
    for line in rows.splitlines():
        fields = [x.strip() for x in line.split(',')]
        require(len(fields) == 3 and fields[0].isdigit(), 'Unrecognized GPU process information')
        processes.append(dict(pid=int(fields[0]), name=fields[1], memory=fields[2]))
    return dict(used_mb=used, free_mb=free, utilization=utilization, processes=processes)


def gpu_gate(value, allowed):
    desktop = {'Xorg', 'Xwayland', 'gnome-shell', 'kwin_wayland', 'kwin_x11'}
    for row in value['processes']:
        exempt = Path(row['name']).name in desktop or (
            row['name'] == '/usr/libexec/gnome-remote-desktop-daemon'
            and str(row['memory']).isdigit() and int(row['memory']) <= 256)
        require(row['pid'] in allowed or exempt, 'Unrelated GPU process: ' + str(row))
    require(value['free_mb'] >= 4500, 'Insufficient free GPU memory for the authorized evaluation')


def publish(state, force=False):
    if not force and time.monotonic() - state.get('_published_monotonic', 0) < 5:
        return
    state['heartbeat_at'] = now()
    state['heartbeat_unix'] = time.time()
    state['_published_monotonic'] = time.monotonic()
    atomic(Q / 'status.json', state)
    if force or time.monotonic() - state.get('_md_monotonic', 0) >= 60:
        state['_md_monotonic'] = time.monotonic()
        text = ['# SPNv2 旧模型补评进度', '', '- 更新时间：' + now(),
                '- 状态：' + state['status'], '- 当前任务：' + str(state.get('active')),
                '- 原训练PID：2584192；启动标识：527383895',
                '- 仅暂停原进程；不重训、不重载、不切换生产管理器。',
                '- 已验收：' + ', '.join(state.get('accepted', [])),
                '- 异常：' + str(state.get('error')), '- 本轮记录：' + str(Q),
                '- 独立恢复记录：' + str(Q / 'recovery.json'), '']
        atomic(REPORT, '\n'.join(text))


def validate_manifest(manifest):
    require(manifest.get('root') == str(Q) and manifest.get('production_plan') == str(P), 'Wrong comparison root')
    require(manifest.get('expected_task_count') == 3 and len(manifest['tasks']) == 3, 'Expected exactly three evaluations')
    require({t['domain_id'] for t in manifest['tasks']} == {'synthetic', 'lightbox', 'sunlamp'}, 'Unexpected domains')
    require(manifest['protocol'].get('no_training') is True and manifest['protocol'].get('no_odr') is True,
            'Only source inference is authorized')
    sessions, paths = set(), set()
    for task in manifest['tasks']:
        for phase, command in [('main', task['command']), ('verify', task['verify_command'])]:
            require(isinstance(command, list) and command and all(isinstance(x, str) and x for x in command), 'Invalid argv')
            require(command[0] == PYTHON and not any(Path(x).name in {'train.py', 'odr.py'} for x in command),
                    'Unexpected executable/training command')
            suffix = '' if phase == 'main' else '_verify'
            session = task['session'] + suffix
            log = str(inside(task['log'] + ('' if phase == 'main' else '.verify')))
            require(session.replace('_', '').replace('-', '').isalnum() and session not in sessions and log not in paths,
                    'Duplicate/unsafe session or log')
            sessions.add(session); paths.add(log)
        inside(task['verified_report'])
    require(read(manifest['preflight'])['status'] == 'passed', 'CPU preparation not accepted')
    for path, expected in manifest['shared_assets'].items():
        require(sha(path) == expected, 'Pinned comparison asset changed: ' + path)


def phase_config(task, phase):
    prefix = Q / 'events' / (task['id'] + '.' + phase)
    suffix = '' if phase == 'main' else '.verify'
    env = dict(task.get('env', {}))
    if phase == 'verify':
        env.update(task.get('verify_env', {})); env['CUDA_VISIBLE_DEVICES'] = ''
    return dict(phase=phase, session=task['session'] + ('' if phase == 'main' else '_verify'),
                config=str(prefix) + '.config.json', record=str(prefix) + '.process.json',
                result=str(prefix) + '.result.json', log=task['log'] + suffix, cwd=task['cwd'], env=env,
                command=task['command'] if phase == 'main' else task['verify_command'])


def all_configs(manifest):
    return [phase_config(t, phase) for t in manifest['tasks'] for phase in ('main', 'verify')]


def process_inventory():
    return {int(p.name): item for p in Path('/proc').iterdir() if p.name.isdigit()
            for item in [identity(int(p.name))] if item and item['state'] != 'Z'}


def owned_processes(configs, inventory, records):
    """Exact recorded children or exact unique entry/launcher argv; then PPID descendants."""
    selected = {}
    for config in configs:
        for path in ('config', 'record', 'result', 'log'):
            inside(config[path])
        record = records.get(config['record'])
        if record and matches(inventory.get(record.get('pid')), record):
            require(record.get('command') == config['command'], 'Owned record command mismatch')
            selected[record['pid']] = inventory[record['pid']]
        # Covers a controller failure during screen/entry launch before child registration.
        for pid, item in inventory.items():
            argv = item['command']
            entry = str(ENTRY) in argv and '--config' in argv and config['config'] in argv
            launcher = str(WRAPPER) in argv and config['log'] in argv and config['config'] in argv
            if entry or launcher:
                selected[pid] = item
    changed = True
    while changed:
        changed = False
        for pid, item in inventory.items():
            if item['ppid'] in selected and pid not in selected:
                selected[pid] = item; changed = True
    require(not ({TRAIN['pid'], MANAGER['pid'], os.getpid()} & set(selected)), 'Recovery scope contains protected process')
    return list(selected.values())


def cleanup_owned(manifest):
    configs = [c for c in all_configs(manifest) if Path(c['config']).exists()]
    records = {c['record']: read(c['record']) for c in configs}
    owned = owned_processes(configs, process_inventory(), records)
    atomic(Q / 'cleanup_scope.json', dict(at=now(), processes=owned))
    for item in reversed(owned):
        signal_exact(item, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(matches(identity(p['pid']), p) for p in owned):
        time.sleep(.25)
    for item in reversed(owned):
        if matches(identity(item['pid']), item):
            signal_exact(item, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(matches(identity(p['pid']), p) for p in owned):
        time.sleep(.25)
    require(not any(matches(identity(p['pid']), p) for p in owned), 'Owned evaluation process did not exit')
    # Any surviving registered/spawning actor must block SIGCONT, not overlap GPU work.
    require(not owned_processes(configs, process_inventory(), records), 'Owned evaluation actor remains')
    return [p['pid'] for p in owned]


def recover(manifest, reason):
    with lock(Q / 'recovery.lock'):
        previous = read(Q / 'recovery.json')
        if previous and previous.get('status') == 'resumed':
            return previous
        atomic(Q / 'recovery_started.json', dict(at=now(), reason=reason))
        cleaned = cleanup_owned(manifest)
        armed = read(Q / 'pause_intent.json')
        result = dict(at=now(), reason=reason, cleaned_pids=cleaned, trainer=TRAIN)
        if not armed:
            result['status'] = 'no_pause_was_armed'
        else:
            require(armed['trainer']['pid'] == TRAIN['pid'] and str(armed['trainer']['starttime']) == TRAIN['starttime'],
                    'Pause ownership identity mismatch')
            require(matches(identity(TRAIN['pid']), TRAIN), 'Original trainer no longer exists; refuse another PID')
            require(signal_exact(armed['trainer'], signal.SIGCONT), 'Unable to resume exact trainer')
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                actual = identity(TRAIN['pid'])
                if matches(actual, armed['trainer']) and actual['state'] not in {'T', 't'}:
                    result.update(status='resumed', after_identity=actual)
                    break
                time.sleep(.25)
            require(result.get('status') == 'resumed', 'Trainer did not leave stopped state')
        atomic(Q / 'recovery.json', result)
        return result


def watchdog(manifest):
    require(os.environ.get('STY'), 'Watchdog must run in its own screen')
    with lock(Q / 'watchdog.lock', nonblocking=True):
        owner = read(Q / 'owner.json')
        atomic(Q / 'watchdog_ready.json', dict(status='ready', at=now(), identity=identity(os.getpid()), owner=owner))
        while True:
            state = read(Q / 'status.json', {})
            if read(Q / 'recovery.json', {}).get('status') in {'resumed', 'no_pause_was_armed'}:
                return
            age = time.time() - state.get('heartbeat_unix', 0)
            intent = read(Q / 'pause_intent.json')
            reason = None
            if not matches(identity(owner['pid']), owner):
                reason = 'controller exited'
            elif age > HEARTBEAT_TIMEOUT:
                reason = 'controller heartbeat stale'
            elif intent and time.monotonic() - intent['paused_monotonic'] >= MAX_PAUSE:
                reason = 'maximum pause deadline reached'
            atomic(Q / 'watchdog.json', dict(at=now(), status='guarding', controller_age_seconds=age, reason=reason))
            if reason:
                try:
                    result = recover(manifest, 'watchdog: ' + reason)
                    state.update(status='watchdog_recovered', error=reason, recovery=result)
                    publish(state, force=True)
                    return
                except BaseException as error:
                    atomic(Q / 'watchdog_recovery_error.json', dict(at=now(), error=repr(error)))
                    # Preserve evidence and retry recovery only; never launch evaluation.
            time.sleep(2)


def run_phase(task, phase, state, manifest_sha):
    config = phase_config(task, phase)
    with lock(Q / 'recovery.lock'):
        require(not (Q / 'recovery_started.json').exists(), 'Recovery has begun; no new task')
        require(sha(Q / 'manifest.json') == manifest_sha, 'Comparison manifest changed')
        require(matches(identity(TRAIN['pid']), TRAIN) and identity(TRAIN['pid'])['state'] in {'T', 't'},
                'Trainer is not paused; do not launch comparison')
        check_production()
        if phase == 'main':
            resources = gpu(); gpu_gate(resources, {TRAIN['pid']})
            require(resources['utilization'] <= 5, 'GPU still active before evaluation')
        for path in (config['config'], config['record'], config['result'], config['log'], config['log'] + '.exitcode'):
            require(not Path(path).exists(), 'Refuse repeated phase or overwritten evidence: ' + path)
        atomic(config['config'], config)
        state.update(status='evaluating' if phase == 'main' else 'verifying', active=task['id'], phase=phase,
                     runtime=config)
        publish(state, force=True)
        launched = subprocess.run(['bash', str(WRAPPER), config['session'], config['log'], PYTHON, '-B', '-s',
                                   str(ENTRY), '--config', config['config']], cwd=task['cwd'],
                                  capture_output=True, text=True, timeout=30)
        atomic(Q / 'events' / (task['id'] + '.' + phase + '.launch.json'),
               dict(at=now(), exit_code=launched.returncode, stdout=launched.stdout, stderr=launched.stderr))
        require(launched.returncode == 0, 'Screen task launcher failed')
    started = time.monotonic()
    while True:
        require(not (Q / 'recovery_started.json').exists(), 'Watchdog recovery has begun')
        require(time.monotonic() - started < MAX_TASK, 'Task exceeded twenty-minute limit')
        require(time.monotonic() - state['paused_monotonic'] < MAX_PAUSE, 'Pause exceeded thirty-minute limit')
        record, result = read(config['record']), read(config['result'])
        exit_path = Path(config['log'] + '.exitcode')
        if result is not None and exit_path.exists():
            require(record and record['command'] == config['command'], 'Missing/mismatched process evidence')
            require(result.get('pid') == record['pid'] and str(result.get('starttime')) == str(record['starttime']), 'Exit identity mismatch')
            require(result.get('exit_code') == 0 and not result.get('error') and int(exit_path.read_text().strip()) == 0,
                    'Evaluation/verification failed; preserve output')
            require(not matches(identity(record['pid']), record), 'Exit recorded while child still live')
            state.setdefault('phase_exits', []).append(dict(task=task['id'], phase=phase, record=record, result=result))
            return
        if record:
            require(record.get('command') == config['command'], 'Process command mismatch')
            state['current_process'] = record
        publish(state)
        time.sleep(1)


def controller(manifest):
    require(os.environ.get('STY'), 'Controller must run in its own screen')
    with lock(Q / 'controller.lock', nonblocking=True):
        require(not (Q / 'owner.json').exists(), 'One-shot controller already has ownership evidence; no automatic rerun')
        validate_manifest(manifest)
        before = check_production()
        require(before['worker']['state'] not in {'T', 't'}, 'Trainer already stopped by another owner')
        fd = _pidfd_open(TRAIN['pid']); os.close(fd)  # Fail before STOP if safe signaling is unsupported.
        owner = identity(os.getpid()); atomic(Q / 'owner.json', owner)
        state = dict(status='preparing_watchdog', active=None, accepted=[], before=before, owner=owner,
                     manifest_sha256=sha(Q / 'manifest.json'), error=None)
        publish(state, force=True)
        try:
            wd_session = 'spnv2_old_reeval_watchdog_20261018'
            launched = subprocess.run(['bash', str(WRAPPER), wd_session, str(Q / 'logs/watchdog.log'),
                                       PYTHON, '-B', '-s', str(Path(__file__).resolve()), '--mode', 'watchdog',
                                       '--manifest', str(Q / 'manifest.json')], cwd=Q,
                                      capture_output=True, text=True, timeout=30)
            require(launched.returncode == 0, 'Independent watchdog launch failed')
            deadline = time.monotonic() + 15
            while not (Q / 'watchdog_ready.json').exists() and time.monotonic() < deadline:
                publish(state); time.sleep(.25)
            ready = read(Q / 'watchdog_ready.json')
            require(ready and ready['owner'] == owner and matches(identity(ready['identity']['pid']), ready['identity']),
                    'Watchdog handshake failed; training has not been paused')
            with lock(Q / 'recovery.lock'):
                require(not (Q / 'recovery_started.json').exists(), 'Watchdog recovery already begun')
                before = check_production()
                intent = dict(at=now(), paused_monotonic=time.monotonic(), trainer=before['worker'], owner=owner,
                              log_offset=before['log_offset'], last_event=before['last_event'])
                atomic(Q / 'pause_intent.json', intent)
                state.update(status='pausing', paused_monotonic=intent['paused_monotonic'])
                publish(state, force=True)
                require(signal_exact(intent['trainer'], signal.SIGSTOP), 'Exact trainer could not be stopped')
            deadline = time.monotonic() + 15
            stable = 0
            while time.monotonic() < deadline:
                actual = identity(TRAIN['pid']); require(matches(actual, intent['trainer']), 'Trainer identity disappeared')
                if actual['state'] in {'T', 't'}:
                    values = gpu(); gpu_gate(values, {TRAIN['pid']})
                    stable = stable + 1 if values['utilization'] <= 5 else 0
                    if stable >= 2:
                        atomic(Q / 'pause_confirmed.json', dict(at=now(), trainer=actual, gpu=values)); break
                publish(state); time.sleep(1)
            require(stable >= 2, 'Stopped trainer/GPU idle confirmation timed out')
            for task in manifest['tasks']:
                run_phase(task, 'main', state, state['manifest_sha256'])
                run_phase(task, 'verify', state, state['manifest_sha256'])
                proof = read(task['verified_report'])
                require(proof and proof.get('status') == 'passed', 'Independent result verification missing')
                state['accepted'].append(task['id']); publish(state, force=True)
            state.update(status='evaluations_accepted', active=None)
        except BaseException as error:
            state.update(status='failed', error=repr(error)); publish(state, force=True)
        finally:
            try:
                result = recover(manifest, 'controller finally')
                state['recovery'] = result
                state['after'] = check_production()
                state['training_resumed'] = result['status'] == 'resumed'
            except BaseException as error:
                state.update(status='recovery_needs_attention', recovery_error=repr(error))
            publish(state, force=True)
        require(state.get('training_resumed') or not (Q / 'pause_intent.json').exists(), 'Training recovery requires attention')
        require(not state.get('error'), 'Evaluation stopped; original training was resumed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['controller', 'watchdog'], default='controller')
    parser.add_argument('--manifest', default=str(Q / 'manifest.json'))
    args = parser.parse_args()
    require(Path(args.manifest).resolve() == Q / 'manifest.json', 'Only this fixed comparison manifest is allowed')
    manifest = read(args.manifest)
    if args.mode == 'watchdog':
        watchdog(manifest)
    else:
        def interrupted(signum, frame):
            raise RuntimeError('Controller received signal ' + str(signum))
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        controller(manifest)


if __name__ == '__main__':
    main()
