"""Two focused terminal-handoff scenarios and the actual next verification."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import time
from unittest import mock


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    from budget import atomic_json
    started=time.monotonic()
    task=json.loads(Path(task_path).read_text());R=Path(task['revision_dir']);D=R.parent.parent
    if not admission['allow_start'] or not admission['boundary_verified'] or conflict_check():
        raise ValueError('Fresh shared-host admission required')
    spec=importlib.util.spec_from_file_location('_identity_handoff_manager',R/'candidate/manager.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    import host_entry
    before=json.loads((R/'before/status.json').read_text())
    source=before['active'];original_outcome=json.loads((Path(source['launch_directory'])/'outcome.json').read_text())
    original_launch=json.loads((Path(source['launch_directory'])/'launch.json').read_text())
    root=R/'metadata_test';root.mkdir(exist_ok=False)
    def fixture(name):
        s=copy.deepcopy(before);item=copy.deepcopy(source);folder=root/name;folder.mkdir()
        launch=folder/'launch';launch.mkdir();atomic_json(launch/'launch.json',original_launch)
        item.update(log_path=str(folder/'worker.log'),launch_directory=str(launch),queued_at=time.time()-121)
        s['attempts']=[item if x['nonce']==item['nonce'] else x for x in s['attempts']]
        s['active']=item
        return s,item,launch
    # Reproduce the real PermissionError and valid terminal files appearing in
    # the same call. The old producer report is only read, never recomputed.
    s,item,launch=fixture('terminal_arrives')
    def exit_race(record):
        Path(item['log_path']+'.exitcode').write_text('0\n')
        atomic_json(launch/'outcome.json',original_outcome)
        raise PermissionError('simulated proc/environ terminal race')
    accepted=copy.deepcopy(s['accepted'])
    with mock.patch.object(host_entry,'live_identity',side_effect=exit_race),\
         mock.patch.object(m,'launch',side_effect=AssertionError('No downstream dispatch in finish')):
        m.finish(s)
    if s['active'] is not None or source['run_id'] not in s['main'] or s['accepted']!=accepted:
        raise ValueError('Terminal race was not reconciled without scientific reexecution/acceptance')
    # An unreadable identity without terminal evidence is not alive/dead/owned.
    # Keep that exact attempt and block after the bounded handoff grace.
    s,item,launch=fixture('terminal_missing');clock=[1000.]
    with mock.patch.object(host_entry,'live_identity',side_effect=PermissionError('still unreadable')) as probe,\
         mock.patch.object(m.time,'monotonic',side_effect=lambda:clock[0]),\
         mock.patch.object(m,'launch',side_effect=AssertionError('No downstream dispatch during handoff')):
        m.finish(s)
        if s['status']!='awaiting_terminal_evidence' or s['active'] is not item:
            raise ValueError('Unverified identity was not kept pending')
        if item['terminal_evidence_handoff']['identity_status']!='unverified':
            raise ValueError('Permission failure was treated as ownership')
        clock[0]+=m.TERMINAL_HANDOFF_GRACE_SECONDS+1
        m.finish(s)
        if s['status']!='blocked' or s['active'] is not item or probe.call_count!=1:
            raise ValueError('Missing terminal evidence did not fail closed')
    idx=json.loads((R/'index_candidate.json').read_text());real_sha=m.sha
    m.sha=lambda p:real_sha(R/'candidate/manager.py') if Path(p)==D/'implementation/manager.py' else real_sha(p)
    m.validate_index(idx)
    reconciled=json.loads((R/'reconciled_state.json').read_text())
    reconciled['index']=dict(path=str(D/'index.json'),sha256=real_sha(R/'index_candidate.json'))
    path,rid=m.next_task(idx,reconciled);v=json.loads(path.read_text())
    if v['action']!='verify_pair' or v['run_task']!=reconciled['main'][source['run_id']]['task']:
        raise ValueError('Recovery would rerun the saved main instead of verifying it')
    if v.get('budget_policy')!=idx['budget_policy'] or len(reconciled['accepted'])!=26:
        raise ValueError('Current budget policy or accepted set changed')
    old_read,old_sha=host_entry.read,host_entry.sha
    try:
        host_entry.read=lambda p:idx if Path(p)==D/'index.json' else old_read(p)
        host_entry.sha=m.sha
        host_entry.validate_task(path,D/'index.json')
    finally:
        host_entry.read,host_entry.sha=old_read,old_sha
    proof=dict(schema='g50p-terminal-handoff-targeted-validation-v1',status='passed',
        model_forwards=0,models_constructed=False,focused_scenarios=2,measured_seconds=time.monotonic()-started,
        permission_race_consumes_actual_terminal_evidence=True,
        unreadable_identity_never_assumed_alive_dead_or_owned=True,
        missing_evidence_keeps_active_then_blocks=True,no_downstream_dispatch_during_handoff=True,
        accepted_count_unchanged=26,completed_producer_not_reexecuted=True,
        actual_next_task_action='verify_pair',real_independent_verification_still_pending=True,
        candidate_index=dict(path=str(R/'index_candidate.json'),sha256=real_sha(R/'index_candidate.json')),
        candidate_manager=dict(path=str(R/'candidate/manager.py'),sha256=real_sha(R/'candidate/manager.py')),
        generated_verifier=dict(path=str(path),sha256=real_sha(path)),
        time_hard_stop_enabled=budget.snapshot()['time_hard_stop_enabled'])
    output=R/'targeted_validation.json'
    if output.exists():raise ValueError('Do not repeat successful validation')
    atomic_json(output,proof)
    return dict(status='terminal_handoff_validated',exit_code=0,model_forwards=0,
                proof=str(output),proof_sha256=real_sha(output))
