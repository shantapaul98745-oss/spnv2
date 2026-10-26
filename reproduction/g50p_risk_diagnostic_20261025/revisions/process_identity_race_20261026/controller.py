"""Admit the focused repair check, then resume only saved-main verification."""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

D=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
R=D/'revisions/process_identity_race_20261026'
P=D.parent.parent.parent/'spnv2'
sys.path.insert(0,str(D/'implementation'))
from budget import GlobalBudget,atomic_json
from host_entry import project_lock,LOCKS

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(p),sha256=sha(p))

def identity():
    p=Path('/proc')/str(os.getpid());s=(p/'stat').read_text()
    return dict(pid=os.getpid(),starttime=int(s[s.rfind(')')+2:].split()[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cmdline_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),
        manager_nonce=os.environ['SPNV2_MANAGER_NONCE'],session=os.environ['STY'])

def no_owned_worker():
    for item in read(D/'launches/registry.json')['records']:
        r=item['expected_record'];p=Path('/proc')/str(r['pid'])
        if p.exists():
            s=(p/'stat').read_text()
            if int(s[s.rfind(')')+2:].split()[19])==r['starttime']:
                raise ValueError('Owned worker remains live; do not cut over')

def deploy(cfg,control):
    with ExitStack() as locks:
        for path in LOCKS:locks.enter_context(project_lock(path))
        no_owned_worker()
        if sha(D/'index.json')!=cfg['old_index_sha256'] or sha(D/'status.json')!=cfg['old_status_sha256']:
            raise ValueError('Original runtime changed during resource wait')
        proof=read(R/'targeted_validation.json');idx=read(R/'index_candidate.json')
        if proof['status']!='passed' or proof['model_forwards']!=0:
            raise ValueError('Bound validation has not passed')
        if proof['candidate_index']!=binding(R/'index_candidate.json') or sha(R/'index_candidate.json')!=cfg['new_index_sha256']:
            raise ValueError('Candidate index changed')
        if proof['candidate_manager']!=binding(R/'candidate/manager.py'):
            raise ValueError('Candidate manager changed')
        if sha(P/'reproduction/g50p_risk_diagnostic_20261025/manager.py')!=idx['implementation_sha256']['manager.py']:
            raise ValueError('Archive/runtime code mismatch')
        old=read(D/'index.json')
        for name,digest in old['implementation_sha256'].items():
            if sha(D/'implementation'/name)!=digest:raise ValueError('Runtime code changed while waiting')
        state=read(R/'reconciled_state.json')
        if state['active'] is not None or len(state['accepted'])!=26:
            raise ValueError('Reconciliation boundary changed')
        for rid,spec in idx['preserved_pairs'].items():
            if state['accepted'][rid]!=spec['accepted_state'] or state['main'][rid]!=spec['main_state']:
                raise ValueError('Accepted evidence changed')
        for rid,spec in idx['preserved_main'].items():
            if state['main'][rid]!=spec['main_state']:raise ValueError('Saved completed main changed')
        b=GlobalBudget(D/'budget',dict(project_id=idx['diagnostic_id'],frozen_inputs_sha256=idx['frozen_inputs']['sha256']),
                       initial_cost_receipt=idx['initial_cost_receipt'],budget_policy=idx['budget_policy'])
        before=b.snapshot()
        if before['unresolved_attempt_ids'] or before['physical_forwards_reserved_upper']>40000:
            raise ValueError('Unresolved execution or physical-budget violation')
        if before['time_hard_stop_enabled'] is not False:raise ValueError('Time hard stop unexpectedly restored')
        record=identity();record.update(run_nonce=record['manager_nonce'],run_id='g50p_process_handoff_cutover_cost')
        cost=b.begin_attempt(record,allow_model_forwards=False)
        nonce=uuid.uuid4().hex;session='spnv2_riskdiag_'+nonce[:12]
        log=D/'logs'/('manager_process_handoff_'+nonce+'.log')
        atomic_json(R/'cutover_intent.json',dict(created_at=time.time(),manager_nonce=nonce,
            previous_index=binding(R/'before/index.json'),new_index=binding(R/'index_candidate.json'),
            validation=binding(R/'targeted_validation.json'),published_commit=cfg['published_commit']))
        for src,dst in [(R/'candidate/manager.py',D/'implementation/manager.py'),(R/'index_candidate.json',D/'index.json')]:
            tmp=dst.with_name(dst.name+'.handoff.tmp');shutil.copyfile(src,tmp);os.chmod(tmp,0o600);os.replace(tmp,dst)
            if sha(src)!=sha(dst):raise ValueError('Byte-preserving publication failed')
        prior=state.pop('partial_delivery',None)
        if prior:state.setdefault('prior_partial_deliveries',[]).append(prior)
        state.setdefault('reconciled_manager_exits',[]).append(dict(receipt=binding(R/'reconciliation_receipt.json'),
            before_status=binding(R/'before/status.json'),worker_main_reused=True))
        state.update(index=binding(D/'index.json'),active=None,status='running',next_resource_check=0.,
            reason='Terminal identity-read race corrected; saved w16 main reconciled; independent verification only')
        state['queue_wait_seconds']+=control['resource_wait_seconds']
        state.setdefault('technical_revisions',[]).append(dict(name='process_identity_race_20261026',
            before_index=binding(R/'before/index.json'),validation=binding(R/'targeted_validation.json'),published_commit=cfg['published_commit']))
        atomic_json(D/'status.json',state)
        measured=time.monotonic()-cost.record['started_monotonic']
        external=cfg['local_validation_process_seconds']
        close=dict(attempt_id=cost.attempt_id,status='manager_terminal_handoff_control_cost_settled',
            measured_seconds_before_finalization=measured+external,current_cutover_seconds=measured,
            measured_external_local_validation_process_seconds=external,
            manager_failure_conservative_overhead_seconds=12.,finalization_reserved_upper_seconds=2.,
            charged_seconds_upper=measured+external+14.,unresolved_forward_tokens=[],exact_total_time_claim=False,
            scope='metadata cutover and two local software contracts; conservative manager-error overhead; zero model work')
        b._append('attempt_closed',close);cost.closed=True;b._attempt=None
        after=b.snapshot();atomic_json(R/'accounting_receipt.json',dict(before=before,after=after,cost=close,
            reconciliation=binding(R/'reconciliation_receipt.json'),model_forwards_added=0))
        dep=dict(schema='g50p-diagnostic-deployment-v1',accepted_at=time.time(),
            status='terminal_handoff_fix_validated_saved_main_independent_verification_pending',
            published_commit=cfg['published_commit'],index_sha256=sha(D/'index.json'),
            producer_commit='3ef5cbd32f16bcfcb650f4b5d7a04a69e718aa47',
            frozen_inputs_sha256=idx['frozen_inputs']['sha256'],implementation_sha256=idx['implementation_sha256'],
            budget_policy=idx['budget_policy'],effective_protocol=idx['protocol'],time_hard_stop_enabled=False,
            max_physical_forwards=40000,old_producer_files_unchanged=50,A_preserved=True,
            accepted_pairs_preserved=26,completed_main_pending_verification='g50p_diag_v1_sunlamp_s2021_w0016',
            targeted_validation=binding(R/'targeted_validation.json'),prior_deployment=binding(R/'before/deployment_accepted.json'),
            accounting_receipt=binding(R/'accounting_receipt.json'))
        atomic_json(D/'deployment_accepted.json',dep);atomic_json(R/'deployment_accepted.json',dep)
        launch=dict(created_at=time.time(),nonce=nonce,session=session,log_path=str(log),
            index_sha256=sha(D/'index.json'),published_commit=cfg['published_commit'],
            previous_intent=binding(R/'before/manager_launch_intent.json'))
        atomic_json(D/'manager_launch_intent.json',launch);atomic_json(R/'manager_launch_intent.json',launch)
        return launch

def main():
    cfg=read(R/'controller_config.json')
    if sha(__file__)!=cfg['controller_sha256']:raise ValueError('Controller changed')
    c=dict(schema='g50p-process-identity-repair-controller-v1',identity=identity(),status='starting',
        attempts=[],active=None,resource_wait_seconds=0.,published_commit=cfg['published_commit'])
    def save():c['updated_at']=time.time();atomic_json(R/'controller_status.json',c)
    try:
        with project_lock(R/'controller.lock'):
            with project_lock(D/'manager.lock'):
                no_owned_worker()
                while True:
                    if sha(D/'index.json')!=cfg['old_index_sha256']:raise ValueError('Runtime index changed')
                    nonce=uuid.uuid4().hex;log=D/'logs'/('process_handoff_validate_'+nonce+'.log')
                    task=D/'tasks/process_identity_race_validation_20261026.json'
                    cmd=['bash',str(D.parent.parent/'scripts/run_in_screen_guarded.sh'),'g50p_handoff_'+nonce[:12],
                        str(log),'/home/ps/anaconda3/envs/SPNv2/bin/python','-B','-s','-u',str(R/'validation_entry.py'),
                        '--task-json',str(task),'--index',str(D/'index.json'),'--log-path',str(log)]
                    a=dict(nonce=nonce,log_path=str(log),task_sha256=sha(task),started_at=time.time())
                    c['attempts'].append(a);c.update(status='validating',active=a);save()
                    env=dict(os.environ,SPNV2_RUN_NONCE=nonce,PYTHONNOUSERSITE='1',PYTHONUTF8='1',
                        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
                    subprocess.run(cmd,cwd=D,env=env,check=True)
                    ep=Path(str(log)+'.exitcode')
                    while not ep.exists():time.sleep(1)
                    code=int(ep.read_text().strip());out=read(D/'launches'/nonce/'outcome.json')
                    if out['exit_code']!=code:raise ValueError('Actual exit/outcome mismatch')
                    a.update(exit_code=code,outcome=out);c['active']=None
                    if code==75:
                        c['status']='waiting_resources';save();started=time.monotonic();time.sleep(300)
                        c['resource_wait_seconds']+=time.monotonic()-started;continue
                    if code!=0 or out['status']!='terminal_handoff_validated':
                        raise ValueError('Focused validation failed; preserve and stop')
                    launch=deploy(cfg,c);break
            env=dict(os.environ,SPNV2_MANAGER_NONCE=launch['nonce'])
            subprocess.run(['bash',str(D.parent.parent/'scripts/run_in_screen_guarded.sh'),launch['session'],
                launch['log_path'],'/usr/bin/python3','-B','-u',str(D/'implementation/manager.py'),
                '--index',str(D/'index.json')],cwd=D,env=env,check=True)
            c.update(status='repaired_manager_dispatched',manager_launch=launch,new_index=binding(D/'index.json'),
                     independent_scientific_acceptance_not_claimed=True);save()
        return 0
    except Exception as error:
        c.update(status='blocked',error_type=type(error).__name__,error=str(error));save();raise

if __name__=='__main__':raise SystemExit(main())
