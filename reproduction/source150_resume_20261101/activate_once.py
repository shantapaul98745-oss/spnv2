"""One-time explicit source-branch cutover, not a retry command."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    revision = Path(__file__).resolve().parent
    deployment = revision / 'deployment'
    binding = json.loads((deployment / 'binding.json').read_text())
    plan = Path(binding['plan'])
    sys.path.insert(0, str(plan / 'pipeline'))
    from progress import atomic_json, now
    started = time.time()
    with (plan / 'pipeline.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (revision / 'cutover.json').exists():
            raise RuntimeError('One-time cutover was already performed; use ordinary manager entry')
        if sha(plan / 'status.json') != binding['old_status_sha256']:
            raise RuntimeError('Original state changed after preparation')
        for path, digest in binding['files_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('Continuation binding changed: ' + path)
        live = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                if not (proc / 'exe').resolve().name.startswith(('python', 'pypy')):
                    continue
                cmd = (proc / 'cmdline').read_bytes()
                if str(plan).encode() in cmd:
                    live.append({'pid': int(proc.name), 'command': cmd.decode(errors='replace')})
            except OSError:
                continue
        if live:
            raise RuntimeError('Related live Python process prevents cutover: ' + repr(live))
        state = json.loads((deployment / 'state.json').read_text())
        old = json.loads((plan / 'status.json').read_text())
        assert state['tasks'][:52] == old['tasks'][:52]
        assert len(state['completed']) == 52
        assert all(t['status'] == 'queued' for t in state['tasks'][52:])
        receipt = dict(at=now(), operation='explicit_source150_branch_resume',
                       old_status_sha256=sha(plan / 'status.json'),
                       old_state_archive=str(revision / 'candidate/old55_status.json'),
                       manifest=str(deployment / 'manifest.json'),
                       manifest_canonical_sha256=state['manifest_sha256'],
                       original52_records_preserved=True,
                       retired_research_branch_not_accepted=True,
                       science_source_unchanged=True, scientific_workers_found=live,
                       metadata_seconds=time.time() - started)
        atomic_json(revision / 'cutover.json', receipt)
        atomic_json(plan / 'status.json', state)
    # The old manager is gone. The original runner reacquires the same exclusive
    # lock and validates the new canonical binding before launching anything.
    command = [binding['python'], '-B', '-s', '-u', str(revision / 'manager_entry.py'),
               '--manifest', str(deployment / 'manifest.json')]
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
