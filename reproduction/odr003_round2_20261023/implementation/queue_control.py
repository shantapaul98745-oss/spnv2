"""Private resource-refresh mailbox; never signal or restart a manager.

The five-second mailbox poll performs filesystem reads only. GPU collection is
requested once per fresh request ID, independently of the 300-second schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid

from guarded_entry import PLAN, atomic_json
from frozen_backend import canonical_sha


def private_read(path):
    path = Path(path)
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o022 or metadata.st_size > 1024*1024):
        raise ValueError('Untrusted manager/control metadata')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError('Manager metadata inode changed')
        with os.fdopen(fd, 'r', encoding='utf8', closefd=False) as handle:
            return json.load(handle)
    finally:
        os.close(fd)


def target_identity(manager):
    keys = ('pid', 'starttime', 'boot_id', 'cmdline_sha256', 'manager_nonce', 'manager_lock')
    result = {key: manager[key] for key in keys}
    if len(result['manager_nonce']) < 16:
        raise ValueError('Manager launch nonce is missing')
    return result


def verify_manager_live(manager, *, plan=PLAN):
    """Read only the exact registered manager and its lock; no PID search."""
    target = target_identity(manager)
    proc = Path('/proc')/str(target['pid'])
    metadata = proc.stat()
    if metadata.st_uid != os.getuid():
        raise ValueError('Manager UID differs')
    raw_stat = (proc/'stat').read_text()
    raw_command = (proc/'cmdline').read_bytes()
    environment = (proc/'environ').read_bytes().split(b'\0')
    nonce = b'SPNV2_MANAGER_NONCE='+target['manager_nonce'].encode()
    if (int(raw_stat[raw_stat.rfind(')')+2:].split()[19]) != target['starttime']
            or hashlib.sha256(raw_command).hexdigest() != target['cmdline_sha256']
            or Path('/proc/sys/kernel/random/boot_id').read_text().strip() != target['boot_id']
            or nonce not in environment):
        raise ValueError('Manager live birth/command/nonce identity differs')
    argv = [part.decode() for part in raw_command.split(b'\0') if part]
    expected = [str(Path(plan)/'implementation/queue_round2.py'), '--index', str(Path(plan)/'engineering_index.json')]
    if argv[-3:] != expected or Path(target['manager_lock']) != Path(plan)/'manager.lock':
        raise ValueError('Manager command or project lock differs')
    lock = Path(target['manager_lock']).stat()
    device = f'{os.major(lock.st_dev):02x}:{os.minor(lock.st_dev):02x}:{lock.st_ino}'
    owned = [line for line in Path('/proc/locks').read_text().splitlines()
             if f' {target["pid"]} ' in line and device in line and ' FLOCK ' in line and ' WRITE ' in line]
    if len(owned) != 1:
        raise ValueError('Exact manager does not exclusively hold its recorded lock')
    return target


def request_refresh(plan=PLAN, *, reason='explicit_user_start_or_idle'):
    plan = Path(plan)
    state = private_read(plan/'status.json')
    if state.get('status') in ('blocked', 'complete', 'engineering_held'):
        raise ValueError('Terminal/blocked manager cannot be woken to bypass a stop condition')
    target = verify_manager_live(state['manager'], plan=plan)
    request_id = uuid.uuid4().hex
    request = dict(schema='odr003-round2-resource-refresh-v1', request_id=request_id,
                   created_at=time.time(), reason=str(reason), target=target,
                   target_sha256=canonical_sha(target), action='refresh_resources')
    path = plan/'control/requests'/(request_id+'.json')
    atomic_json(path, request)
    return dict(request_id=request_id, request_path=str(path), target=target,
                no_signal_sent=True, no_manager_restarted=True)


def pending_refreshes(plan, manager, *, now=None):
    """Return fresh addressed requests; malformed metadata fails closed."""
    plan = Path(plan)
    now = time.time() if now is None else now
    root = plan/'control/requests'
    if not root.exists():
        return []
    target = target_identity(manager)
    requests = []
    for path in sorted(root.glob('*.json')):
        if (plan/'control/acks'/path.name).exists():
            continue
        row = private_read(path)
        if (row.get('schema') != 'odr003-round2-resource-refresh-v1'
                or row.get('action') != 'refresh_resources'
                or row.get('request_id')+'.json' != path.name
                or row.get('target_sha256') != canonical_sha(row.get('target'))):
            raise ValueError('Malformed resource refresh request')
        if row['target'] != target or not -5 <= now-row['created_at'] <= 3600:
            atomic_json(plan/'control/acks'/path.name,
                        dict(request_id=row['request_id'], status='ignored_stale_or_other_manager',
                             acknowledged_at=now, sampled_resources=False))
            continue
        requests.append(row)
    return requests


def acknowledge(plan, requests, *, snapshot, status, now=None):
    now = time.time() if now is None else now
    for request in requests:
        atomic_json(Path(plan)/'control/acks'/(request['request_id']+'.json'),
                    dict(request_id=request['request_id'], status=status, acknowledged_at=now,
                         snapshot_id=snapshot.get('snapshot_id'), sampled_at=snapshot.get('captured_at'),
                         sampled_resources=True, target=request['target']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=str(PLAN))
    parser.add_argument('--reason', default='explicit_user_start_or_idle')
    args = parser.parse_args()
    print(json.dumps(request_refresh(args.plan, reason=args.reason), ensure_ascii=False))
