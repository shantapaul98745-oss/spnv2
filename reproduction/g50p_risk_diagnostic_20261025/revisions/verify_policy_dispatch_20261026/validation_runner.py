"""Validate the actual generated verifier and preservation of its saved main."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import time


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    from budget import atomic_json
    started=time.monotonic()
    task=json.loads(Path(task_path).read_text());R=Path(task['revision_dir']);D=R.parent.parent
    if not admission['allow_start'] or not admission['boundary_verified'] or conflict_check():
        raise ValueError('Actual shared-host admission required')
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);return module
    manager=load('_verify_policy_manager',R/'candidate/manager.py')
    idx=json.loads((R/'index_candidate.json').read_text());state=json.loads((R/'before/status.json').read_text())
    real_sha=manager.sha
    manager.sha=lambda p:real_sha(R/'candidate/manager.py') if Path(p)==D/'implementation/manager.py' else real_sha(p)
    manager.validate_index(idx)
    rid='g50p_diag_v1_lightbox_s2023_w0087_notimelimit1'
    if len(state['accepted'])!=21 or rid in state['accepted'] or rid not in state['main']:
        raise ValueError('Pending completed-main boundary changed')
    before_main=copy.deepcopy(state['main'][rid]);before_accepted=copy.deepcopy(state['accepted'])
    state['index']=dict(path=str(D/'index.json'),sha256=real_sha(R/'index_candidate.json'))
    path,run_id=manager.next_task(idx,state)
    produced=json.loads(path.read_text())
    if produced['action']!='verify_pair' or produced.get('budget_policy')!=idx['budget_policy']:
        raise ValueError('Actual generated verifier lacks the bound policy')
    if produced['run_task']!=before_main['task'] or run_id!=rid+'_verify_'+state['index']['sha256'][:12]:
        raise ValueError('Saved main was rebound or verifier identity not versioned')
    old=json.loads((R/'before/status.json').read_text())['active']
    if path==Path(old['task_path']) or real_sha(old['task_path'])!=old['task_sha256']:
        raise ValueError('Old failed verifier task was overwritten')
    host=load('_verify_policy_host',D/'implementation/host_entry.py')
    original_read=host.read;host.read=lambda p:idx if Path(p)==D/'index.json' else original_read(p)
    host.sha=manager.sha
    host.validate_task(path,D/'index.json')
    bad=copy.deepcopy(produced);bad.pop('budget_policy')
    host.read=lambda p:idx if Path(p)==D/'index.json' else bad if Path(p)==path else original_read(p)
    try:host.validate_task(path,D/'index.json')
    except ValueError:missing_rejected=True
    else:raise ValueError('Original missing-policy failure path was weakened')
    changed=copy.deepcopy(idx);changed['preserved_main'][rid]['main_state']['report']['sha256']='0'*64
    try:manager.validate_index(changed)
    except ValueError:tamper_rejected=True
    else:raise ValueError('Tampered pending main was accepted')
    if state['main'][rid]!=before_main or state['accepted']!=before_accepted:
        raise ValueError('Metadata validation changed scientific completion state')
    proof=dict(schema='g50p-verify-policy-dispatch-validation-v1',status='passed',model_forwards=0,
        models_constructed=False,measured_seconds=time.monotonic()-started,
        candidate_index=dict(path=str(R/'index_candidate.json'),sha256=real_sha(R/'index_candidate.json')),
        candidate_manager=dict(path=str(R/'candidate/manager.py'),sha256=real_sha(R/'candidate/manager.py')),
        actual_generated_verifier=dict(path=str(path),sha256=real_sha(path)),
        generated_policy_matches_current_index=True,original_missing_policy_rejected=missing_rejected,
        old_failed_task_preserved=True,pending_main_original_task_preserved=True,
        pending_main_tamper_rejected=tamper_rejected,accepted_count_unchanged=21,
        saved_main_not_reexecuted=True,real_independent_verification_still_pending=True,
        time_hard_stop_enabled=budget.snapshot()['time_hard_stop_enabled'])
    out=R/'targeted_validation.json'
    if out.exists():raise ValueError('Do not repeat completed validation')
    atomic_json(out,proof)
    return dict(status='verify_policy_dispatch_validated',exit_code=0,model_forwards=0,
        proof=str(out),proof_sha256=real_sha(out))
