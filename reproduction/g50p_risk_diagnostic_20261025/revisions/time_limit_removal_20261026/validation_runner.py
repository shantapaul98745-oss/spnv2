"""Real bound metadata validation; no scientific state/model is loaded."""
from pathlib import Path
import copy
import importlib.util
import json
import shutil
import sys
import time


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    from budget import atomic_json
    started=time.monotonic()
    task=json.loads(Path(task_path).read_text())
    R=Path(task['revision_dir'])
    D=R.parent.parent
    if not admission['allow_start'] or not admission['boundary_verified'] or conflict_check():
        raise ValueError('Fresh actual shared-host admission required')
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    manager=load('_time_revision_manager',R/'candidate/manager.py')
    idx=json.loads((R/'index_candidate.json').read_text())
    state=json.loads((R/'before/status.json').read_text())
    real_sha=manager.sha
    def candidate_sha(path):
        p=Path(path)
        replacement=R/'candidate'/p.name
        return real_sha(replacement) if p.parent==D/'implementation' and replacement.exists() else real_sha(p)
    manager.sha=candidate_sha
    manager.validate_index(idx)
    if len(idx['preserved_pairs'])!=21 or not state['pilot_admitted']:
        raise ValueError('Accepted pair/pilot reuse changed')
    for rid,spec in idx['preserved_pairs'].items():
        if state['accepted'][rid]!=spec['accepted_state'] or state['main'][rid]!=spec['main_state']:
            raise ValueError('Saved accepted pair state changed')
    selected,run_id=manager.next_task(idx,state)
    expected='g50p_diag_v1_lightbox_s2023_w0087_notimelimit1'
    if run_id!=expected:
        raise ValueError('Next task repeats an accepted window or changes order')
    next_task=json.loads(selected.read_text())
    host=load('_time_revision_host',R/'candidate/host_entry.py')
    real_read,real_host_sha=host.read,host.sha
    host.read=lambda p: idx if Path(p)==D/'index.json' else real_read(p)
    host.sha=candidate_sha
    host.validate_task(selected,D/'index.json')
    bad=copy.deepcopy(next_task);bad['budget_policy']={}
    host.read=lambda p: idx if Path(p)==D/'index.json' else bad if Path(p)==selected else real_read(p)
    try:host.validate_task(selected,D/'index.json')
    except ValueError:policy_tamper_rejected=True
    else:raise ValueError('Unbound task policy was accepted')
    host.read=real_read;host.sha=real_host_sha
    original=json.loads((R/'before/protocol.json').read_text())
    revised=json.loads((R/'protocol_time_unlimited.json').read_text())
    for key in ['max_active_seconds','time_hard_stop_enabled','budget_policy','budget_amendment','original_protocol']:
        original.pop(key,None);revised.pop(key,None)
    if original!=revised:
        raise ValueError('Non-time scientific protocol fields changed')
    import budget as new_budget
    clone=R/'metadata_validation_budget'
    shutil.copytree(R/'before/budget',clone)
    prefix=(clone/'journal.jsonl').read_bytes()
    initial_identity=(clone/'identity.json').read_bytes()
    class Clock:
        def __init__(self):self.now=time.monotonic()
        def monotonic(self):return self.now
        def time(self):return time.time()
    clock=Clock()
    b=new_budget.GlobalBudget(clone,dict(project_id=idx['diagnostic_id'],
        frozen_inputs_sha256=idx['frozen_inputs']['sha256']),initial_cost_receipt=idx['initial_cost_receipt'],
        budget_policy=idx['budget_policy'],activate_policy=True,clock=clock)
    before=b.snapshot()
    if before['active_seconds_charged_upper']!=14400 or before['physical_forwards_reserved_upper']!=17150:
        raise ValueError('Historical charges changed during migration')
    a=b.begin_attempt(launch_record,allow_model_forwards=False)
    clock.now+=100000.
    a.check_time('beyond old cumulative deadline')
    a.close('metadata_contract_no_model');a.check_final_receipt()
    if (clone/'identity.json').read_bytes()!=initial_identity or not (clone/'journal.jsonl').read_bytes().startswith(prefix):
        raise ValueError('Original accounting identity or prefix was rewritten')
    if b.snapshot()['physical_forwards_reserved_upper']!=17150:
        raise ValueError('Model-free test changed physical-forward charges')
    seed=json.loads((R/'resume_origin.json').read_text())
    if seed['old_failed_physical_forwards']!=191 or seed['already_persisted_branch_predictions']!=0:
        raise ValueError('Interrupted-window work was misclassified')
    current=json.loads(Path(seed['seed_current']['path']).read_text())
    generation=Path(seed['seed_directory'])/'checkpoints'/('generation-'+str(current['generation']).zfill(20))
    meta=json.loads((generation/'metadata.json').read_text())
    if real_sha(generation/'metadata.json')!=current['metadata_sha256'] or meta['scientific_cursors']!=seed['scientific_cursors']:
        raise ValueError('Seed complete-checkpoint metadata changed')
    if meta['scientific_cursors']['next_index']!=512 or meta['scientific_cursors']['candidates']!=77:
        raise ValueError('Seed is not the last complete recoverable boundary')
    if any((Path(next_task['output_dir'])/name).exists() for name in ['failure.json','candidate_binding.json','progress_accept.json','progress_rollback.json']):
        raise ValueError('Resume would overwrite or ignore persisted branch evidence')
    if budget.deadline_monotonic is not None or budget.snapshot()['time_hard_stop_enabled'] is not False:
        raise ValueError('Actual admitted validation retained the old deadline')
    proof=dict(schema='g50p-time-limit-removal-targeted-validation-v1',status='passed',
        physical_model_forwards=0,models_constructed=False,measured_seconds=time.monotonic()-started,
        candidate_index=dict(path=str(R/'index_candidate.json'),sha256=real_sha(R/'index_candidate.json')),
        candidate_code={p.name:dict(path=str(p),sha256=real_sha(p)) for p in (R/'candidate').glob('*.py')},
        accepted_pairs_preserved=21,A_reused_without_reexecution=True,pilot_proof_reused=True,
        actual_next_run_id=expected,policy_tamper_rejected=policy_tamper_rejected,
        original_protocol_science_fields_equal=True,real_historical_journal_prefix_preserved=True,
        real_historical_identity_bytes_preserved=True,historical_time_charge_retained=14400.,
        historical_forward_upper_retained=17150,old_deadline_exceeded_in_fixture_without_stop=True,
        seed_complete_next_index=512,seed_complete_candidates=77,old_failure_191_forwards_retained=True,
        actual_admitted_budget_deadline=None,actual_time_hard_stop_enabled=False,
        new_actual_window_scientific_verification_still_required=True)
    path=R/'targeted_validation.json'
    if path.exists():raise ValueError('Validation proof exists; do not repeat')
    atomic_json(path,proof)
    return dict(status='time_limit_removal_metadata_validated',exit_code=0,model_forwards=0,
                proof=str(path),proof_sha256=real_sha(path))
