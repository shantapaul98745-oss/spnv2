"""Narrow queued-only observer revisions, with immutable started task evidence."""
import copy
import re
from pathlib import PurePosixPath
from revision import _canonical, canonical_sha256, extend_state


def _wrapped_command(old, new, wrapper, mode):
    if not isinstance(old, list) or not isinstance(new, list) or len(old) < 3:
        raise ValueError('Observer commands must retain the original Python command')
    entry = 1
    while entry < len(old) and old[entry] in {'-B', '-s', '-u'}:
        entry += 1
    if (entry + 1 >= len(old) or PurePosixPath(old[entry]).name != 'source_artifacts.py'
            or old[entry + 1] != mode):
        raise ValueError('Only source evaluate/verify_eval entry points may be wrapped')
    expected = old[:entry] + [wrapper, '--base-entry', old[entry]] + old[entry + 1:]
    if new != expected:
        raise ValueError('Observer must preserve interpreter, arguments and base entry exactly')


def revise_queued_observers(previous, old_manifest, new_manifest, *, task_ids,
                            caller_context=None, timestamp=None):
    """Append tasks and wrap explicitly named, never-started source evaluations.

    No running/accepted task, original asset hash, env, dependency, session, output,
    GPU gate or original command argument can change. This is not a general queue
    editing API. Pure function; caller must hold lock and durably save its audit.
    """
    if (not isinstance(task_ids, list) or not task_ids
            or any(not isinstance(x, str) for x in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ValueError('Require unique explicitly named queued evaluation identities')
    old_tasks = old_manifest.get('tasks', [])
    new_tasks = new_manifest.get('tasks', [])
    if (len(new_tasks) < len(old_tasks)
            or [x.get('id') for x in new_tasks[:len(old_tasks)]] != [x.get('id') for x in old_tasks]):
        raise ValueError('Existing identities and order must be preserved')
    # Reuse the strict extension validation, including bound old-state integrity,
    # appended task schemas, nonfinite rejection and fixed top-level metadata.
    baseline = copy.deepcopy(new_manifest)
    baseline['tasks'][:len(old_tasks)] = copy.deepcopy(old_tasks)
    amended, audit = extend_state(previous, old_manifest, baseline,
                                  caller_context=caller_context, timestamp=timestamp)
    changed = []
    for index, (old, new) in enumerate(zip(old_tasks, new_tasks)):
        if _canonical(old) == _canonical(new):
            continue
        ident = old['id']; evidence = previous['tasks'][index]
        if (ident not in task_ids or evidence['status'] not in {'queued', 'pending'}
                or ident in {previous.get('active'), previous.get('current', {}).get('task_id')}
                or {'runtime', 'history', 'started_at', 'accepted_at', 'error', 'event_error',
                    'progress_warning'}.intersection(evidence)):
            raise ValueError('Cannot revise started, selected or unlisted task: ' + ident)
        if old.get('kind') != 'full_source_evaluation':
            raise ValueError('Only source evaluation tasks may receive an observer')
        fields = {k for k in set(old) | set(new) if _canonical(old.get(k)) != _canonical(new.get(k))}
        if not {'command', 'verify_command'}.issubset(fields) or fields - {'command', 'verify_command', 'metadata'}:
            raise ValueError('Only paired evaluation commands and observer metadata may change')
        before = old.get('metadata', {}); after = new.get('metadata', {})
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError('Observer metadata must be objects')
        for key, value in before.items():
            if key != 'asset_sha256' and (key not in after or _canonical(after[key]) != _canonical(value)):
                raise ValueError('Original evaluation metadata must be retained')
        old_assets = before.get('asset_sha256', {}); assets = after.get('asset_sha256', {})
        if not isinstance(old_assets, dict) or not isinstance(assets, dict):
            raise ValueError('Observer must bind all original and new assets')
        if any(assets.get(k) != v for k, v in old_assets.items()):
            raise ValueError('Original asset bindings cannot change')
        if any(not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch('[a-f0-9]{64}', v)
               for k, v in assets.items()):
            raise ValueError('Asset hashes must be complete SHA256 values')
        wrapper = after.get('pending_observer_entry')
        if (not isinstance(wrapper, str) or not PurePosixPath(wrapper).is_absolute()
                or PurePosixPath(wrapper).name != 'capture_source.py' or wrapper not in assets):
            raise ValueError('Require a hash-bound source observer wrapper')
        _wrapped_command(old['command'], new['command'], wrapper, 'evaluate')
        _wrapped_command(old['verify_command'], new['verify_command'], wrapper, 'verify_eval')
        replacement = copy.deepcopy(new); replacement['status'] = evidence['status']
        amended['tasks'][index] = replacement
        changed.append(dict(id=ident, old=copy.deepcopy(old), new=copy.deepcopy(new),
                            previous_status=evidence['status']))
    if set(task_ids) != {x['id'] for x in changed}:
        raise ValueError('Declared pending identities must equal actual observer revisions')
    amended['manifest_sha256'] = canonical_sha256(new_manifest)
    audit.update(operation='queued_source_observer_revision_and_append', version=2,
                 new_manifest_sha256=amended['manifest_sha256'],
                 pending_observer_revisions=changed,
                 declared_pending_task_ids=list(task_ids),
                 preserved_task_ids=[x['id'] for x in old_tasks if x['id'] not in task_ids])
    _canonical(audit)
    return amended, audit
