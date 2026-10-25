"""Own the finite metadata repair: admitted validation, then safe manager cutover.

No model execution here. Resource waits are separate. Failed validation never
retries; only an explicit resource exit75 can receive a fresh admitted attempt.
"""
from contextlib import ExitStack
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

D = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
R = D/'revisions/admission_template_fix_20261025'
P = D.parent.parent.parent/'spnv2'
sys.path.insert(0, str(D/'implementation'))
from budget import GlobalBudget, atomic_json
from host_entry import project_lock, LOCKS


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(p):
    return dict(path=str(p), sha256=sha(p))


def own_identity():
    p = Path('/proc')/str(os.getpid())
    stat = (p/'stat').read_text()
    return dict(pid=os.getpid(), starttime=int(stat[stat.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cmdline_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),
        manager_nonce=os.environ['SPNV2_MANAGER_NONCE'], session=os.environ['STY'])


def ensure_no_worker():
    for item in read(D/'launches/registry.json')['records']:
        record = item['expected_record']
        proc = Path('/proc')/str(record['pid'])
        if proc.exists():
            stat = (proc/'stat').read_text()
            if int(stat[stat.rfind(')')+2:].split()[19]) == record['starttime']:
                raise ValueError('A registered diagnostic worker is still alive')


def publish_candidate(config, control):
    with ExitStack() as locks:
        for lock in LOCKS:
            locks.enter_context(project_lock(lock))
        ensure_no_worker()
        if sha(D/'index.json') != config['old_index_sha256'] or sha(D/'status.json') != config['old_status_sha256']:
            raise ValueError('Production index/status changed during repair wait')
        proof = read(R/'targeted_validation.json')
        if proof['status'] != 'passed' or proof['model_forwards'] != 0:
            raise ValueError('Targeted metadata validation has not passed')
        idx = read(R/'index_candidate.json')
        if sha(R/'index_candidate.json') != config['new_index_sha256']:
            raise ValueError('Candidate index changed')
        if proof['candidate_index'] != binding(R/'index_candidate.json'):
            raise ValueError('Validation did not bind this candidate index')
        manager = R/'candidate/manager.py'
        if proof['candidate_manager'] != binding(manager) or sha(manager) != idx['implementation_sha256']['manager.py']:
            raise ValueError('Candidate manager changed')
        if sha(P/'reproduction/g50p_risk_diagnostic_20261025/manager.py') != sha(manager):
            raise ValueError('Published/runtime candidate bytes differ')
        current = read(D/'index.json')
        for name, digest in current['implementation_sha256'].items():
            if sha(D/'implementation'/name) != digest:
                raise ValueError('Production implementation changed while waiting: '+name)
        budget = GlobalBudget(D/'budget', dict(project_id=idx['diagnostic_id'],
            frozen_inputs_sha256=idx['frozen_inputs']['sha256']), initial_cost_receipt=idx['initial_cost_receipt']).snapshot()
        if budget['unresolved_attempt_ids'] or budget['active_time_bound_violation_attempt_ids']:
            raise ValueError('Unresolved budget prevents manager cutover')
        if budget['remaining_active_seconds_lower'] <= 60 or budget['remaining_forward_reservations'] <= 0:
            raise ValueError('Budget exhausted before repaired admission')
        state = read(D/'status.json')
        for run_id, spec in idx['preserved_pairs'].items():
            if state['accepted'][run_id] != spec['accepted_state'] or state['main'][run_id] != spec['main_state']:
                raise ValueError('Accepted pair changed; never reconstruct/rebind it')
        shutil.copyfile(D/'budget/journal.jsonl', R/'before/budget_journal_at_cutover.jsonl')
        nonce = uuid.uuid4().hex
        session = 'spnv2_riskdiag_'+nonce[:12]
        log = D/'logs'/('manager_auditfix_'+nonce+'.log')
        atomic_json(R/'cutover_intent.json', dict(created_at=time.time(), manager_nonce=nonce,
            old_index=binding(R/'before/index.json'), new_index=binding(R/'index_candidate.json'),
            published_commit=config['published_commit'], budget_before=budget,
            accepted_pairs_preserved=list(state['accepted']), validation=binding(R/'targeted_validation.json')))
        for source, destination in [(manager, D/'implementation/manager.py'), (R/'index_candidate.json', D/'index.json')]:
            temporary = destination.with_name(destination.name+'.auditfix.tmp')
            shutil.copyfile(source, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            if sha(source) != sha(destination):
                raise ValueError('Atomic byte-preserving publication failed')
        state.setdefault('resolved_pre_admission_failures', []).append(dict(
            original_attempt=state['active'], resolution='stale audit-template code binding; pre-heavy-import exit1',
            failure_evidence=binding(R/'failure_receipt.json'), control_cost=binding(R/'control_cost_settlement.json')))
        state.update(index=binding(D/'index.json'), active=None, status='running', next_resource_check=0.0,
            reason='Versioned audit templates repaired; both accepted pilots reused; no science rerun')
        state['queue_wait_seconds'] += control['resource_wait_seconds']
        state.setdefault('technical_revisions', []).append(dict(name='admission_template_fix_20261025',
            previous_index=binding(R/'before/index.json'), targeted_validation=binding(R/'targeted_validation.json'),
            published_commit=config['published_commit']))
        atomic_json(D/'status.json', state)
        deployment = dict(schema='g50p-diagnostic-deployment-v1', accepted_at=time.time(),
            status='audit_template_repair_validated_pilot_admission_pending',
            published_commit=config['published_commit'], index_sha256=sha(D/'index.json'),
            producer_commit=idx['preserved_pairs'][idx['tasks'][0]['run_id']]['accepted_state'].get('producer_commit',
                '3ef5cbd32f16bcfcb650f4b5d7a04a69e718aa47'),
            frozen_inputs_sha256=idx['frozen_inputs']['sha256'], implementation_sha256=idx['implementation_sha256'],
            prior_deployment=binding(R/'before/deployment_accepted.json'),
            targeted_validation=binding(R/'targeted_validation.json'), old_producer_files_unchanged=50,
            original_A_preserved=True, accepted_pairs_preserved=list(state['accepted']), tasks=48,
            technical_revision='admission_template_fix_20261025', settled_budget_before=budget,
            initial_cost_receipt=idx['initial_cost_receipt'])
        atomic_json(D/'deployment_accepted.json', deployment)
        atomic_json(R/'deployment_accepted.json', deployment)
        intent = dict(created_at=time.time(), nonce=nonce, session=session, log_path=str(log),
            index_sha256=sha(D/'index.json'), published_commit=config['published_commit'],
            previous_intent=binding(R/'before/manager_launch_intent.json'))
        atomic_json(D/'manager_launch_intent.json', intent)
        atomic_json(R/'manager_launch_intent.json', intent)
        return intent


def main():
    config = read(R/'controller_config.json')
    if sha(__file__) != config['controller_sha256']:
        raise ValueError('Controller source changed')
    control = dict(schema='g50p-audit-template-repair-controller-v1', identity=own_identity(),
        status='starting', resource_wait_seconds=0.0, attempts=[], published_commit=config['published_commit'])
    def save():
        control['updated_at'] = time.time()
        atomic_json(R/'controller_status.json', control)
    try:
        with project_lock(R/'controller.lock'):
            with project_lock(D/'manager.lock'):
                ensure_no_worker()
                while True:
                    if sha(D/'index.json') != config['old_index_sha256']:
                        raise ValueError('Original bound index changed before validation')
                    nonce = uuid.uuid4().hex
                    log = D/'logs'/('auditfix_validate_'+nonce+'.log')
                    task = D/'tasks/admission_template_fix_validation_20261025.json'
                    env = dict(os.environ, SPNV2_RUN_NONCE=nonce, PYTHONNOUSERSITE='1', PYTHONUTF8='1',
                        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
                    cmd = ['bash', str(D.parent.parent/'scripts/run_in_screen_guarded.sh'),
                        'g50p_auditfix_'+nonce[:12], str(log), '/home/ps/anaconda3/envs/SPNv2/bin/python',
                        '-B','-s','-u',str(R/'probe_entry.py'),'--task-json',str(task),
                        '--index',str(D/'index.json'),'--log-path',str(log)]
                    attempt = dict(nonce=nonce, log_path=str(log), task_sha256=sha(task), started_at=time.time())
                    control['attempts'].append(attempt)
                    control.update(status='validating', active=attempt)
                    save()
                    subprocess.run(cmd,cwd=D,env=env,check=True)
                    exit_path = Path(str(log)+'.exitcode')
                    while not exit_path.exists():
                        time.sleep(1)
                    code = int(exit_path.read_text().strip())
                    outcome = read(D/'launches'/nonce/'outcome.json')
                    if outcome['exit_code'] != code:
                        raise ValueError('Validation outcome/wrapper mismatch')
                    attempt.update(exit_code=code, outcome=outcome)
                    control['active'] = None
                    if code == 75:
                        control['status'] = 'waiting_resources'
                        save()
                        wait_started = time.monotonic()
                        time.sleep(300)
                        control['resource_wait_seconds'] += time.monotonic()-wait_started
                        continue
                    if code != 0 or outcome['status'] != 'audit_template_fix_validated':
                        raise ValueError('Targeted validation failed; preserve and stop without retry')
                    intent = publish_candidate(config, control)
                    break
            # Release the old manager lock before launching the new manager.
            env = dict(os.environ, SPNV2_MANAGER_NONCE=intent['nonce'])
            cmd = ['bash',str(D.parent.parent/'scripts/run_in_screen_guarded.sh'),intent['session'],intent['log_path'],
                '/usr/bin/python3','-B','-u',str(D/'implementation/manager.py'),'--index',str(D/'index.json')]
            subprocess.run(cmd,cwd=D,env=env,check=True)
            control.update(status='repaired_manager_dispatched', manager_launch=intent,
                           new_index=binding(D/'index.json'), scientific_admission_not_claimed=True)
            save()
        return 0
    except Exception as error:
        control.update(status='blocked', error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
