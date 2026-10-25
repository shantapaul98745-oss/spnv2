"""Fixed diagnostic queue: saved-state A, two planned pilots, then up to 48.

No historical manager/task is edited or restarted. Every heavy action uses a
fresh independent owned screen and the shared-host entry. A failed/ambiguous
attempt blocks advancement; resource-yield 75 alone is resumable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from budget import GlobalBudget, atomic_json
from host_entry import PLAN, BASE, PROJECT, host_modules, project_lock, read, sha

PYTHON='/home/ps/anaconda3/envs/SPNv2/bin/python'
WRAPPER=BASE/'spnv2_runs/scripts/run_in_screen_guarded.sh'
RESOURCE_INTERVAL=300


def binding(path): return dict(path=str(path),sha256=sha(path))


def checked(item):
    if sha(item['path'])!=item['sha256']: raise ValueError('Diagnostic immutable binding changed')
    return read(item['path'])


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value: raise ValueError('Frozen diagnostic artifact changed')
    else: atomic_json(path,value)
    return path


def validate_preserved_A(index,row,task,spec):
    """One exact already-completed A record, never a generic code-drift waiver."""
    action=task.get('action')
    if action not in ('extract_a','analyze_a'):
        raise ValueError('Only already accepted zero-forward A preprocessing can preserve an old implementation map')
    if spec['task']!=dict(path=row['task_path'],sha256=row['task_sha256']):
        raise ValueError('Preserved A task binding differs from original row')
    original=checked(spec['original_index'])
    if (row not in original.get('preprocessing',[]) or task['implementation_sha256']!=original['implementation_sha256']
            or original['frozen_inputs']!=index['frozen_inputs'] or original['host_modules']!=index['host_modules']):
        raise ValueError('Preserved A original index, code, inputs or host admission differs')
    module='extract_A.py' if action=='extract_a' else 'analyze_existing.py'
    for name in ('host_entry.py','budget.py',module):
        old=task['implementation_sha256'].get(name)
        if old is None or old!=index['implementation_sha256'].get(name) or sha(PLAN/'implementation'/name)!=old:
            raise ValueError('An actual A execution dependency changed: '+name)
    proof,outcome,launch=checked(spec['proof']),checked(spec['outcome']),checked(spec['launch_record'])
    expected_status='a_extracted' if action=='extract_a' else 'a_analyzed'
    if (checked(spec['exit_code'])!=0 or proof.get('status')!='passed' or proof.get('model_forwards')!=0
            or proof.get('frozen_inputs')!=index['frozen_inputs']
            or outcome.get('exit_code')!=0 or outcome.get('status')!=expected_status
            or outcome.get('proof')!=spec['proof']['path'] or outcome.get('proof_sha256')!=spec['proof']['sha256']):
        raise ValueError('Preserved A lacks the exact actual exit, zero-forward proof and outcome')
    state_record=spec['state_record'];attempt=state_record['attempt']
    argv=launch['argv']
    if (state_record.get('proof')!=spec['proof'] or attempt.get('run_id')!=row['run_id']
            or attempt.get('task_path')!=row['task_path'] or attempt.get('task_sha256')!=row['task_sha256']
            or attempt.get('exit_code')!=0 or attempt.get('outcome')!=outcome
            or launch.get('project_id')!=PROJECT or launch.get('run_id')!=row['run_id']
            or launch.get('run_nonce')!=attempt.get('nonce') or len(launch.get('run_nonce',''))<16
            or type(launch.get('pid')) is not int or launch['pid']<=0
            or type(launch.get('starttime')) is not int or launch['starttime']<=0 or not launch.get('boot_id')
            or '--task-json' not in argv or argv[argv.index('--task-json')+1]!=row['task_path']
            or launch.get('log_path')!=attempt.get('log_path')
            or spec['exit_code']['path']!=attempt['log_path']+'.exitcode'
            or Path(spec['outcome']['path'])!=Path(attempt['launch_directory'])/'outcome.json'
            or Path(spec['launch_record']['path'])!=Path(attempt['launch_directory'])/'launch.json'):
        raise ValueError('Preserved A state/launch/task/nonce lineage differs')
    command_hash=hashlib.sha256(('\0'.join(argv)+'\0').encode()).hexdigest()
    if launch.get('cmdline_sha256')!=command_hash:
        raise ValueError('Preserved A launch command checksum differs')
    return True


def validate_preserved_pair(index, row, task, spec):
    """Reuse one already accepted immutable producer/verifier, never rerun it."""
    task_binding = dict(path=row['task_path'], sha256=row['task_sha256'])
    original = checked(spec['original_index'])
    if (spec['task'] != task_binding or row not in original['tasks']
            or task['implementation_sha256'] != original['implementation_sha256']
            or original['frozen_inputs'] != index['frozen_inputs']
            or original['host_modules'] != index['host_modules']):
        raise ValueError('Preserved pair original task/index/input lineage differs')
    for name, digest in original['implementation_sha256'].items():
        source = spec['original_code'][name]
        if source['sha256'] != digest or sha(source['path']) != digest:
            raise ValueError('Preserved original execution code changed: '+name)
    accepted, main = spec['accepted_state'], spec['main_state']
    proof, report = checked(accepted['proof']), checked(main['report'])
    if (proof['status'] != 'passed' or proof['task_binding'] != task_binding
            or proof['report_sha256'] != main['report']['sha256']
            or report['task_sha256'] != row['task_sha256']
            or not report['full_scientific_checkpoint_equal']
            or not report['original_trace_reproduced_stepwise']
            or main['task'] != task_binding or main['run_id'] != row['run_id']
            or accepted['report'] != main['report']):
        raise ValueError('Preserved pair lacks exact accepted scientific proof')
    attempts = main['attempts'] + [accepted['verification_attempt']]
    if main['attempts'][-1]['exit_code'] != 0:
        raise ValueError('Preserved pair main did not finish successfully')
    for attempt in attempts:
        directory = Path(attempt['launch_directory'])
        outcome, launch = read(directory/'outcome.json'), read(directory/'launch.json')
        argv = launch['argv']
        if (attempt['exit_code'] not in (0,75) or read(attempt['log_path']+'.exitcode') != attempt['exit_code']
                or outcome != attempt['outcome'] or outcome['exit_code'] != attempt['exit_code']
                or sha(attempt['task_path']) != attempt['task_sha256']
                or launch['task_sha256'] != attempt['task_sha256']
                or launch['run_id'] != attempt['run_id'] or launch['run_nonce'] != attempt['nonce']
                or launch['implementation_sha256'] != original['implementation_sha256']
                or launch['index_sha256'] != spec['original_index']['sha256']
                or argv[argv.index('--task-json')+1] != attempt['task_path']
                or hashlib.sha256(('\0'.join(argv)+'\0').encode()).hexdigest() != launch['cmdline_sha256']):
            raise ValueError('Preserved pair actual launch/exit/nonce binding changed')
    verification = accepted['verification_attempt']
    if (verification['exit_code'] != 0 or proof['verifier_task_sha256'] != verification['task_sha256']
            or verification['outcome']['proof_sha256'] != accepted['proof']['sha256']):
        raise ValueError('Preserved pair verifier receipt differs')
    return True


def validate_index(index):
    rows=index.get('tasks',[])
    if (index.get('schema')!='g50p-risk-diagnostic-index-v1' or len(rows)!=48
            or len({row['run_id'] for row in rows})!=48
            or any(type(row.get('pilot')) is not bool for row in rows)
            or [row['pilot'] for row in rows]!=[True,True]+[False]*46):
        raise ValueError('Only two fixed included pilots followed by the remaining 46 windows are authorized')
    if index.get('complete_ODR_runs',0)!=0: raise ValueError('This diagnostic authorizes zero complete ODR')
    for kind,action in [('pilot_admission','verify_pilots'),('summary','summarize_diagnostic')]:
        template=index[kind+'_template']
        if template.get('action')!=action or template.get('implementation_sha256')!=index['implementation_sha256']:
            raise ValueError('Audit template implementation/action differs: '+kind)
        path=index.get('audit_task_paths',{}).get(kind)
        if path is not None and (Path(path).parent!=PLAN/'tasks' or Path(path).suffix!='.json'):
            raise ValueError('Versioned audit task path escapes diagnostic tasks')
    preserved=index.get('preserved_preprocessing',{})
    if not set(preserved)<={row['run_id'] for row in index.get('preprocessing',[])}:
        raise ValueError('Preserved preprocessing allowance escapes original A rows')
    preserved_pairs=index.get('preserved_pairs',{})
    if not set(preserved_pairs)<={row['run_id'] for row in rows}:
        raise ValueError('Preserved pairs escape the fixed48 rows')
    for row in [*index.get('preprocessing',[]),*rows]:
        task=checked(dict(path=row['task_path'],sha256=row['task_sha256']))
        if task['run_id']!=row['run_id']:
            raise ValueError('Diagnostic index/task identity mismatch')
        if row['run_id'] in preserved:
            validate_preserved_A(index,row,task,preserved[row['run_id']])
        elif row['run_id'] in preserved_pairs:
            validate_preserved_pair(index,row,task,preserved_pairs[row['run_id']])
        elif task['implementation_sha256']!=index['implementation_sha256']:
            raise ValueError('Diagnostic index/task implementation mismatch')
        if row in rows and task['action']!='run_pair': raise ValueError('Pair window must use the fixed run_pair entry')
    # Root's independently prepared manifest validates exact pair/window IDs,
    # immutable source checkpoints and data boundaries. Absence fails closed.
    from freeze_protocol import validate_index as validate_frozen
    validate_frozen(index)


def manager_identity():
    proc=Path('/proc')/str(os.getpid()); raw=(proc/'stat').read_text(); cmd=(proc/'cmdline').read_bytes()
    return dict(pid=os.getpid(),starttime=int(raw[raw.rfind(')')+2:].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                cmdline_sha256=hashlib.sha256(cmd).hexdigest(),
                manager_nonce=os.environ['SPNV2_MANAGER_NONCE'],session=os.environ['STY'])


def write_state(state):
    state['heartbeat_at']=time.time(); atomic_json(PLAN/'status.json',state)


def new_state(index_path):
    return dict(schema='g50p-risk-diagnostic-manager-v1',status='preparing',index=binding(index_path),
                active=None,attempts=[],main={},accepted={},preprocessing={},pilot_admitted=False,
                queue_wait_seconds=0.,waiting_since=time.time(),next_resource_check=0.,
                complete_ODR_runs=0,planned_unique_windows=48)


def launch(state,task_path,run_id,index_path):
    nonce=uuid.uuid4().hex; session='g50pdiag_'+nonce[:16]
    log=PLAN/'logs'/(run_id+'_'+nonce+'.log'); log.parent.mkdir(exist_ok=True)
    attempt=dict(run_id=run_id,task_path=str(task_path),task_sha256=sha(task_path),nonce=nonce,
                 session=session,log_path=str(log),launch_directory=str(PLAN/'launches'/nonce),
                 queued_at=time.time(),launch_state='intent_persisted')
    state['active']=attempt; state['attempts'].append(attempt);state['status']='launching'
    if state['waiting_since'] is not None:
        waited=time.time()-state['waiting_since'];state['queue_wait_seconds']+=waited
        state['waiting_since']=None
    write_state(state)
    command=['bash',str(WRAPPER),session,str(log),PYTHON,'-B','-s','-u',
             str(PLAN/'implementation/host_entry.py'),'--task-json',str(task_path),
             '--index',str(index_path),'--log-path',str(log)]
    env={**os.environ,'SPNV2_RUN_NONCE':nonce,'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUTF8':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0',
         'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'2021','MPLBACKEND':'Agg'}
    result=subprocess.run(command,cwd=PLAN,env=env,capture_output=True,text=True,timeout=20)
    attempt.update(wrapper_returncode=result.returncode,wrapper_output=result.stdout+result.stderr,
                   launch_state='wrapper_returned')
    state['status']='running' if result.returncode==0 else 'blocked'
    write_state(state)


def finish(state):
    item=state['active'];exit_path=Path(item['log_path']+'.exitcode')
    if not exit_path.exists():
        directory=Path(item['launch_directory']);record=directory/'launch.json'
        if time.time()-item['queued_at']>120:
            if not record.exists():
                state.update(status='blocked',reason='Ambiguous launch identity missing; no duplicate launch')
            else:
                from host_entry import live_identity
                try: alive=live_identity(read(record))
                except FileNotFoundError: alive=False
                if not alive: state.update(status='blocked',reason='Owned worker exited without complete exit evidence')
        return
    code=int(exit_path.read_text().strip());outcome_path=Path(item['launch_directory'])/'outcome.json'
    if not outcome_path.exists():
        state.update(status='blocked',reason='Actual wrapper exit lacks owned outcome');return
    outcome=read(outcome_path);item.update(exit_code=code,outcome=outcome)
    if outcome.get('exit_code')!=code:
        state.update(status='blocked',reason='Wrapper and owned outcome exit codes differ');return
    if code==75:
        state.update(status='waiting_resources',active=None,waiting_since=time.time(),
                     next_resource_check=time.time()+RESOURCE_INTERVAL);return
    if code!=0:
        state.update(status='budget_stopped' if outcome.get('status')=='budget_stopped' else 'blocked',
                     reason='Preserve failed diagnostic attempt; no automatic retry');return
    task=read(item['task_path']);action=task['action']
    if action=='run_pair' and outcome['status']=='pair_main_complete':
        report=Path(outcome['report'])
        if item['run_id'] in state['main']: raise ValueError('A completed pair cannot be rerun')
        state['main'][item['run_id']]=dict(run_id=item['run_id'],task=binding(item['task_path']),
            report=binding(report),log_path=item['log_path'],exit_code_path=str(exit_path),
            outcome_path=str(outcome_path),launch_record_path=str(Path(item['launch_directory'])/'launch.json'),
            attempts=[copy.deepcopy(row) for row in state['attempts'] if row['run_id']==item['run_id']])
    elif action=='verify_pair' and outcome['status']=='pair_verified':
        source=checked(task['run_task']);run_id=source['run_id'];proof_path=Path(outcome['proof'])
        proof=read(proof_path)
        if (run_id not in state['main'] or run_id in state['accepted'] or proof.get('status')!='passed'
                or proof.get('report_sha256')!=state['main'][run_id]['report']['sha256']):
            raise ValueError('Independent pair proof does not match actual main evidence')
        state['accepted'][run_id]=dict(report=state['main'][run_id]['report'],proof=binding(proof_path),
                                      verification_attempt=copy.deepcopy(item))
    elif action in ('extract_a','analyze_a') and outcome['status'] in ('a_extracted','a_analyzed'):
        proof_path=Path(outcome['proof']);proof=read(proof_path)
        if proof.get('status')!='passed' or proof.get('model_forwards',0)!=0:
            raise ValueError('Saved-state-only A evidence did not pass without forwarding')
        state['preprocessing'][item['run_id']]=dict(proof=binding(proof_path),attempt=copy.deepcopy(item))
    elif action=='verify_pilots' and outcome['status']=='pilot_admitted':
        proof_path=Path(outcome['proof']);proof=read(proof_path)
        if proof.get('status')!='passed' or proof.get('allow_remaining_46') is not True:
            raise ValueError('Two-window correctness/cost pilot admission did not pass')
        state['pilot_admitted']=True;state['pilot_proof']=binding(proof_path)
    elif action=='verify_pilots' and outcome['status']=='diagnostic_held':
        state.update(status='held',active=None,reason=outcome.get('reason'));return
    elif action=='summarize_diagnostic' and outcome['status']=='diagnostic_delivered':
        proof_path=Path(outcome['proof']);proof=read(proof_path)
        summary_path=Path(outcome['summary'])
        if proof.get('status')!='passed' or proof.get('summary_sha256')!=sha(summary_path):
            raise ValueError('Diagnostic delivery lacks the actual summary/proof binding')
        final_budget_path=Path(item['launch_directory'])/'budget_closed.json'
        final_budget=read(final_budget_path)
        if (final_budget.get('schema')!='g50p-diagnostic-budget-snapshot-v1'
                or final_budget.get('unresolved_attempt_ids') or final_budget.get('active_time_bound_violation_attempt_ids')):
            raise ValueError('Final delivery requires the actual settled budget receipt after process exit')
        complete=proof.get('complete_scope') is True and proof.get('completed_windows')==48 and len(state['accepted'])==48
        final_manifest=dict(schema='g50p-diagnostic-delivery-manifest-v1',status='complete' if complete else 'partial',
            summary=binding(summary_path),proof=binding(proof_path),final_settled_budget=binding(final_budget_path),
            active_seconds_charged_upper=final_budget['active_seconds_charged_upper'],
            initial_cost_receipts=final_budget.get('initial_cost_receipts',[]),
            physical_forwards_reserved_upper=final_budget['physical_forwards_reserved_upper'],
            queue_wait_seconds=state['queue_wait_seconds'],complete_ODR_runs=0,
            independently_accepted_windows=len(state['accepted']),scientific_scope_complete=complete,
            note='The summary running-time snapshot reserves its live deadline; this post-exit receipt supplies settled cost.')
        manifest_path=immutable(PLAN/'delivery/final_manifest.json',final_manifest)
        state.update(status='complete' if complete else 'partial_delivery_required',active=None,
                     delivery={**outcome,'final_manifest':binding(manifest_path),
                               'final_settled_budget':binding(final_budget_path)});return
    else:
        state.update(status='blocked',reason='Unexpected diagnostic action/outcome; preserve and block');return
    state.update(status='between_tasks',active=None,waiting_since=time.time(),next_resource_check=0.)


def verification_task(index,row,state):
    science=checked(dict(path=row['task_path'],sha256=row['task_sha256']))
    receipt=state['main'][row['run_id']]
    task=dict(schema='g50p-risk-diagnostic-task-v1',action='verify_pair',run_id=row['run_id']+'_verify',
              implementation_sha256=index['implementation_sha256'],run_task=receipt['task'],
              execution_receipt=receipt,all_attempts=receipt['attempts'],
              proof_path=str(Path(science['output_dir'])/'proof.json'),index_path=str(PLAN/'index.json'))
    if 'index_binding' in science: task['index_binding']=science['index_binding']
    path=immutable(PLAN/'tasks'/(task['run_id']+'.json'),task)
    return path,task['run_id']


def audit_task(index,state,kind):
    filename='pilot' if kind=='pilot_admission' else 'final'
    default=PLAN/'tasks'/('verify_pilots.json' if filename=='pilot' else 'summarize_diagnostic.json')
    path=Path(index.get('audit_task_paths',{}).get(kind,str(default)))
    if path.exists(): return path,read(path)['run_id']
    selected=index['tasks'][:2] if filename=='pilot' else index['tasks']
    receipts=dict(schema='g50p-diagnostic-execution-receipts-v1',
        runs=[state['main'][row['run_id']] for row in selected if row['run_id'] in state['main']],
        accepted={row['run_id']:state['accepted'][row['run_id']] for row in selected if row['run_id'] in state['accepted']},
        planned_run_ids=[row['run_id'] for row in selected],
        missing_main_run_ids=[row['run_id'] for row in selected if row['run_id'] not in state['main']],
        missing_independent_acceptance_run_ids=[row['run_id'] for row in selected if row['run_id'] not in state['accepted']],
        all_attempts=state['attempts'],queue_wait_seconds=state['queue_wait_seconds'],
        preprocessing=state['preprocessing'])
    receipt_suffix='_'+state['index']['sha256'][:12] if index.get('audit_task_paths') else ''
    receipts_path=immutable(PLAN/(filename+'_receipts'+receipt_suffix+'.json'),receipts)
    task=copy.deepcopy(index[kind+'_template']);task['execution_receipts']=binding(receipts_path)
    task['index_path']=str(PLAN/'index.json')
    path=immutable(path,task)
    return path,task['run_id']


def partial_delivery(index,state):
    """Tiny metadata handoff when no further budgeted computation is allowed.

    This is deliberately not a scientific 'passed' proof or a claim that all
    forty-eight windows completed. A guard can report/analyse this preserved
    subset without restarting a worker or consuming another model forward.
    """
    suffix='_'+state['index']['sha256'][:12] if index.get('technical_revision') else ''
    manifest_path=PLAN/'delivery'/('partial_manifest'+suffix+'.json')
    if manifest_path.exists():
        state['partial_delivery']=binding(manifest_path)
        return manifest_path
    reason=state.get('reason') or state['status']
    planned=[row['run_id'] for row in index['tasks']]
    dispatched={row['run_id'] for row in state['attempts'] if row['run_id'] in planned}
    accepted={key:value for key,value in state['accepted'].items() if key in planned}
    preserved=[]
    for row in index['tasks']:
        if row['run_id'] not in dispatched:continue
        task=checked(dict(path=row['task_path'],sha256=row['task_sha256']))
        output=Path(task['output_dir'])
        names=('report.json','failure.json','yield.json','candidate_binding.json',
               'progress_accept.json','progress_rollback.json')
        preserved.append(dict(run_id=row['run_id'],output_dir=str(output),
            saved_metadata_paths=[str(output/name) for name in names if (output/name).is_file()],
            isolated_checkpoint_directory=str(output/'isolated_durable'),
            independent_acceptance=row['run_id'] in accepted))
    budget_state=None;budget_error=None
    try:
        budget_state=GlobalBudget(PLAN/'budget',dict(project_id=PROJECT,
                                 frozen_inputs_sha256=index['frozen_inputs']['sha256']),
                                 initial_cost_receipt=index.get('initial_cost_receipt')).snapshot()
    except Exception as error: budget_error=dict(type=type(error).__name__,message=str(error))
    receipts=dict(schema='g50p-diagnostic-partial-receipts-v1',status='partial_preserved',
        stop_status=state['status'],stop_reason=reason,created_at=time.time(),
        complete_ODR_runs=0,planned_unique_windows=48,
        independently_accepted=accepted,main_results=state['main'],
        main_without_independent_acceptance=sorted(set(state['main'])-set(accepted)),
        not_executed=[key for key in planned if key not in dispatched],
        attempted_without_complete_main=sorted(dispatched-set(state['main'])),
        no_complete_main=[key for key in planned if key not in state['main']],
        preserved_own_output_locations=preserved,
        all_attempts=state['attempts'],preprocessing=state['preprocessing'],
        queue_wait_seconds=state['queue_wait_seconds'],budget=budget_state,budget_read_error=budget_error,
        new_model_forwards_in_delivery=0,no_automatic_retry=True,scientific_scope_complete=False)
    receipts_path=immutable(PLAN/'delivery'/('partial_receipts'+suffix+'.json'),receipts)
    manifest=dict(schema='g50p-diagnostic-partial-delivery-v1',status='partial_delivery_required',
                  stop_reason=reason,index=state['index'],receipts=binding(receipts_path),
                  accepted_window_count=len(accepted),main_window_count=len(state['main']),
                  attempted_window_count=len(dispatched),
                  planned_window_count=48,complete_ODR_runs=0,
                  independent_final_summary_status='not_run_no_budget_or_admission',
                  scientific_scope_complete=False,guard_must_report_preserved_subset=True,
                  no_further_model_dispatch=True)
    immutable(manifest_path,manifest);state['partial_delivery']=binding(manifest_path)
    return manifest_path


def next_task(index,state):
    for row in index.get('preprocessing',[]):
        if row['run_id'] not in state['preprocessing']:
            checked(dict(path=row['task_path'],sha256=row['task_sha256']))
            return Path(row['task_path']),row['run_id']
    for i,row in enumerate(index['tasks']):
        if i==2 and not state['pilot_admitted']: return audit_task(index,state,'pilot_admission')
        if row['run_id'] in state['accepted']:
            checked(state['accepted'][row['run_id']]['proof']);continue
        if row['run_id'] in state['main']: return verification_task(index,row,state)
        checked(dict(path=row['task_path'],sha256=row['task_sha256']))
        return Path(row['task_path']),row['run_id']
    return audit_task(index,state,'summary')


def main(index_path):
    if os.name!='posix' or not os.environ.get('STY') or len(os.environ.get('SPNV2_MANAGER_NONCE',''))<16:
        raise ValueError('Diagnostic manager requires its own screen and nonce')
    index_path=Path(index_path).resolve(strict=True)
    if index_path!=PLAN/'index.json': raise ValueError('Wrong isolated diagnostic index')
    index=read(index_path);validate_index(index)
    for name,digest in index['implementation_sha256'].items():
        if sha(PLAN/'implementation'/name)!=digest: raise ValueError('Diagnostic frozen code changed')
    gate,collector=host_modules(index)
    with project_lock(PLAN/'manager.lock'):
        state=read(PLAN/'status.json') if (PLAN/'status.json').exists() else new_state(index_path)
        if state['index']!=binding(index_path): raise ValueError('Existing diagnostic manager belongs to another index')
        for run_id,spec in index.get('preserved_preprocessing',{}).items():
            if state['preprocessing'].get(run_id)!=spec['state_record']:
                raise ValueError('Preserved accepted A state changed or is missing; never rerun under a new implementation')
        for run_id,spec in index.get('preserved_pairs',{}).items():
            if state['accepted'].get(run_id)!=spec['accepted_state'] or state['main'].get(run_id)!=spec['main_state']:
                raise ValueError('Preserved accepted pair state changed; never rerun')
        state['manager']=manager_identity();registry=PLAN/'launches/registry.json'
        if not registry.exists(): atomic_json(registry,dict(schema='spnv2.round2.registry.v1',project_id=PROJECT,records=[]))
        last_write=0.
        while True:
            if state['status'] in ('blocked','budget_stopped','held','complete','partial_delivery_required'):
                if state['status'] in ('blocked','budget_stopped','held'):
                    prior=state['status'];partial_delivery(index,state)
                    if prior in ('budget_stopped','held'):state['status']='partial_delivery_required'
                write_state(state);return 1 if state['status']=='blocked' else 0
            if state['active']:
                finish(state)
            else:
                task_path,run_id=next_task(index,state)
                if time.time()>=state['next_resource_check']:
                    snapshot=collector.collect_snapshot(registry,project_root=PLAN,project_id=PROJECT)
                    decision=gate.assess_snapshot(snapshot,now=time.time(),task_kind='gpu')
                    state['resource_snapshot']=snapshot;state['resource_sampled_at']=snapshot.get('captured_at')
                    if decision.allow_start: launch(state,task_path,run_id,index_path)
                    else: state.update(status='waiting_resources',reason=list(decision.reasons),
                                       next_resource_check=time.time()+RESOURCE_INTERVAL)
            if time.monotonic()-last_write>=60:
                write_state(state);last_write=time.monotonic()
            time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',required=True)
    raise SystemExit(main(parser.parse_args().index))
