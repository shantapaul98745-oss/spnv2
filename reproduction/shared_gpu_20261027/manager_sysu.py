"""Fixed diagnostic queue: saved-state A, two planned pilots, then up to 48.

No historical manager/task is edited or restarted. Every heavy action uses a
fresh independent owned screen and the shared-host entry. A failed/ambiguous
attempt blocks advancement; resource-yield 75 alone is resumable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import sys
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
TERMINAL_HANDOFF_GRACE_SECONDS=30
SYSU_DEPLOYMENT_BINDING=None


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



_RESOURCE_API_CACHE={}
def resource_api(index):
    item=index.get('resource_revision')
    if item is None:raise ValueError('Explicit resource amendment required')
    key=item['sha256']
    if key not in _RESOURCE_API_CACHE:
        rev=checked(item);code=rev['validator']
        expected=str(BASE/'spnv2/reproduction/shared_gpu_20261027/resource_amendment.py')
        if code['path']!=expected or sha(code['path'])!=code['sha256']:raise ValueError('Resource bridge code binding differs')
        spec=importlib.util.spec_from_file_location('spnv2_bound_resource_revision',code['path'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.load(index)
        _RESOURCE_API_CACHE[key]=module
    return _RESOURCE_API_CACHE[key]

def resource_host_modules_compatible(index,original):
    return resource_api(index).host_modules_compatible(index,original)

def validate_time_limit_policy(index):
    """The only authorized budget revision removes time stopping, not forwards."""
    spec=index.get('budget_policy')
    if spec is None:
        if index.get('time_limit_revision') is not None:
            raise ValueError('Time-limit revision lacks its bound budget policy')
        return None
    policy=checked(spec)
    if (policy.get('schema')!='g50p-diagnostic-budget-policy-v2'
            or policy.get('project_id')!=PROJECT
            or policy.get('frozen_inputs_sha256')!=index['frozen_inputs']['sha256']
            or policy.get('max_physical_forwards')!=40000
            or 'max_active_seconds' not in policy or policy['max_active_seconds'] is not None
            or policy.get('time_hard_stop_enabled') is not False):
        raise ValueError('Budget revision changed more than the authorized time hard stop')
    return policy


def validate_preserved_A_time_dependency(index,task,spec,name,old):
    """Bind exact old A execution bytes while the time-only infrastructure changes.

    This exception is limited to host_entry/budget. The actual A extraction or
    analysis module must still have its original live bytes. Old task/proof,
    outcome, launch, nonce and successful wrapper exit checks remain below.
    """
    if name not in ('host_entry.py','budget.py') or validate_time_limit_policy(index) is None:
        raise ValueError('An actual A scientific dependency changed: '+name)
    revision=checked(index['time_limit_revision'])
    if (revision.get('schema')!='g50p-time-limit-removal-revision-v1'
            or revision.get('authorization')!='user_2026-10-26_remove_cumulative_time_limit'
            or revision.get('budget_policy')!=index['budget_policy']):
        raise ValueError('Preserved A budget dependency lacks the exact authorized revision')
    before=checked(revision['before_index'])
    current=index['implementation_sha256'].get(name)
    expected_live=current
    if name=='host_entry.py':
        resource=resource_api(index).load(index)
        current=resource['before_implementation_sha256']['host_entry.py']
    original_code=spec.get('original_execution_code',{}).get(name)
    if (before['frozen_inputs']!=index['frozen_inputs']
            or not resource_host_modules_compatible(index,before)
            or before['implementation_sha256'].get(name)!=old
            or not isinstance(original_code,dict) or original_code.get('sha256')!=old
            or sha(original_code['path'])!=old
            or revision.get('modified_implementation_sha256',{}).get(name)!=current
            or current is None or sha(PLAN/'implementation'/name)!=expected_live):
        raise ValueError('Preserved A old/new time-dependency bytes differ: '+name)
    return True


def validate_preserved_A(index,row,task,spec):
    """One exact already-completed A record, never a generic code-drift waiver."""
    action=task.get('action')
    if action not in ('extract_a','analyze_a'):
        raise ValueError('Only already accepted zero-forward A preprocessing can preserve an old implementation map')
    if spec['task']!=dict(path=row['task_path'],sha256=row['task_sha256']):
        raise ValueError('Preserved A task binding differs from original row')
    original=checked(spec['original_index'])
    if (row not in original.get('preprocessing',[]) or task['implementation_sha256']!=original['implementation_sha256']
            or original['frozen_inputs']!=index['frozen_inputs'] or not resource_host_modules_compatible(index,original)):
        raise ValueError('Preserved A original index, code, inputs or host admission differs')
    module='extract_A.py' if action=='extract_a' else 'analyze_existing.py'
    for name in ('host_entry.py','budget.py',module):
        old=task['implementation_sha256'].get(name)
        if old is None:
            raise ValueError('An actual A execution dependency is missing: '+name)
        if old!=index['implementation_sha256'].get(name) or sha(PLAN/'implementation'/name)!=old:
            validate_preserved_A_time_dependency(index,task,spec,name,old)
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
            or not resource_host_modules_compatible(index,original)):
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


def validate_preserved_main(index,row,task,spec):
    """Preserve a completed producer while its independent verifier is pending.

    This proves only the existing main's immutable completion and lineage. It
    never adds the run to accepted; the normal verifier/finish path must do so.
    A later accepted proof does not invalidate preservation of the same main.
    """
    task_binding=dict(path=row['task_path'],sha256=row['task_sha256'])
    original=checked(spec['original_index'])
    if (spec['task']!=task_binding or row not in original['tasks']
            or task.get('action')!='run_pair'
            or task['implementation_sha256']!=original['implementation_sha256']
            or original['frozen_inputs']!=index['frozen_inputs']
            or not resource_host_modules_compatible(index,original)
            or original.get('budget_policy')!=index.get('budget_policy')):
        raise ValueError('Preserved pending main original task/index/input/policy lineage differs')
    for name,digest in original['implementation_sha256'].items():
        source=spec['original_code'][name]
        if source['sha256']!=digest or sha(source['path'])!=digest:
            raise ValueError('Preserved pending-main execution code changed: '+name)
    main=spec['main_state'];report=checked(main['report'])
    if (main['task']!=task_binding or main['run_id']!=row['run_id']
            or report.get('schema')!='g50p-risk-diagnostic-pair-report-v1'
            or report.get('status')!='pair_main_complete_pending_independent_verification'
            or report['run_id']!=row['run_id'] or report['task_sha256']!=row['task_sha256']
            or report['selection']!=task['selection'] or report['producer']!=task['producer']
            or report['panel']!=task['panel']
            or report['excluded_match_fields']!=['ledger_boundary']
            or not report['full_scientific_checkpoint_equal']
            or not report['original_trace_reproduced_stepwise']
            or report['matched_scientific_sha256']!=report['expected_scientific_sha256']
            or not all(report['branch_results'][branch]['learning_unchanged']
                       and report['branch_results'][branch]['producer_provider_unchanged']
                       for branch in ['accept','rollback'])
            or report['gt_boundary']['forbidden_attempts']!=[]
            or report['gt_boundary']['restored_phase']!='adaptation'):
        raise ValueError('Preserved pending main lacks the original complete scientific producer report')
    attempts=main['attempts']
    if (not attempts or attempts[-1]['exit_code']!=0
            or sum(a['exit_code']==0 for a in attempts)!=1
            or len({a['nonce'] for a in attempts})!=len(attempts)):
        raise ValueError('Preserved pending main does not have one unique successful producer completion')
    for attempt in attempts:
        directory=Path(attempt['launch_directory'])
        outcome,launch=read(directory/'outcome.json'),read(directory/'launch.json')
        argv=launch['argv']
        if (attempt['exit_code'] not in (0,75)
                or read(attempt['log_path']+'.exitcode')!=attempt['exit_code']
                or outcome!=attempt['outcome'] or outcome['exit_code']!=attempt['exit_code']
                or attempt['run_id']!=row['run_id'] or attempt['task_path']!=row['task_path']
                or attempt['task_sha256']!=row['task_sha256'] or sha(attempt['task_path'])!=row['task_sha256']
                or launch.get('project_id')!=PROJECT or launch['run_id']!=row['run_id']
                or launch['task_sha256']!=row['task_sha256']
                or launch['index_sha256']!=spec['original_index']['sha256']
                or launch['implementation_sha256']!=original['implementation_sha256']
                or launch['run_nonce']!=attempt['nonce'] or len(launch['run_nonce'])<16
                or type(launch.get('pid')) is not int or launch['pid']<=0
                or type(launch.get('starttime')) is not int or launch['starttime']<=0
                or not launch.get('boot_id') or launch.get('log_path')!=attempt['log_path']
                or '--task-json' not in argv or argv[argv.index('--task-json')+1]!=row['task_path']
                or '--index' not in argv or argv[argv.index('--index')+1]!=str(PLAN/'index.json')
                or hashlib.sha256(('\0'.join(argv)+'\0').encode()).hexdigest()!=launch['cmdline_sha256']):
            raise ValueError('Preserved pending-main actual launch/exit/nonce binding changed')
    last=attempts[-1];directory=Path(last['launch_directory'])
    outcome=read(directory/'outcome.json');launch=read(directory/'launch.json')
    launch_digest=hashlib.sha256(json.dumps(launch,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (main['log_path']!=last['log_path'] or main['exit_code_path']!=last['log_path']+'.exitcode'
            or main['outcome_path']!=str(directory/'outcome.json')
            or main['launch_record_path']!=str(directory/'launch.json')
            or outcome.get('status')!='pair_main_complete' or outcome.get('run_id')!=row['run_id']
            or outcome.get('report')!=main['report']['path']
            or outcome.get('report_sha256')!=main['report']['sha256']
            or report['launch_record_sha256']!=launch_digest):
        raise ValueError('Preserved pending-main report/outcome/provenance binding differs')
    return True


def validate_index(index):
    resource_api(index).load(index)
    if index.get('sysu_task_revision')!='shared_gpu_20261027':
        raise ValueError('Wrong explicit SYSU task revision')
    if set(index.get('migration_extensions',{}))!={'namespace_runner','bootstrap','dispatch_config','storage_adapter','storage_policy'}:
        raise ValueError('Incomplete exact migration extension map')
    stream = index.get('streaming_migration')
    if stream is None: raise ValueError('Explicit streaming dependency plan is required')
    code=stream['code']
    if code['path']!=str(BASE/'spnv2/reproduction/sysu_migration_20261027/streaming_assets_v1.py') or sha(code['path'])!=code['sha256']:
        raise ValueError('Streaming dependency helper binding differs')
    spec=importlib.util.spec_from_file_location('spnv2_streaming_bound',code['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.load_plan(index)
    validate_time_limit_policy(index)
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
        if path is not None and (Path(path).parent!=PLAN/'tasks'/index['sysu_task_revision'] or Path(path).suffix!='.json'):
            raise ValueError('Versioned audit task path escapes diagnostic tasks')
    preserved=index.get('preserved_preprocessing',{})
    if not set(preserved)<={row['run_id'] for row in index.get('preprocessing',[])}:
        raise ValueError('Preserved preprocessing allowance escapes original A rows')
    preserved_pairs=index.get('preserved_pairs',{})
    if not set(preserved_pairs)<={row['run_id'] for row in rows}:
        raise ValueError('Preserved pairs escape the fixed48 rows')
    preserved_main=index.get('preserved_main',{})
    if (not set(preserved_main)<={row['run_id'] for row in rows}
            or set(preserved_main)&set(preserved_pairs)):
        raise ValueError('Preserved pending main escapes or overlaps the fixed48 preservation rows')
    for row in [*index.get('preprocessing',[]),*rows]:
        task=checked(dict(path=row['task_path'],sha256=row['task_sha256']))
        if task['run_id']!=row['run_id']:
            raise ValueError('Diagnostic index/task identity mismatch')
        if row['run_id'] in preserved:
            validate_preserved_A(index,row,task,preserved[row['run_id']])
        elif row['run_id'] in preserved_pairs:
            validate_preserved_pair(index,row,task,preserved_pairs[row['run_id']])
        elif row['run_id'] in preserved_main:
            validate_preserved_main(index,row,task,preserved_main[row['run_id']])
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


def sysu_worker_command(index_path, task_path, log_path):
    if SYSU_DEPLOYMENT_BINDING is None:
        raise ValueError('Explicit SYSU deployment receipt is required')
    index=read(index_path)
    extension=index.get('migration_extensions',{}).get('dispatch_config')
    if extension is None:
        raise ValueError('Index lacks the exact migration dispatcher binding')
    path=Path(extension['path'])
    if sha(path)!=extension['sha256']:
        raise ValueError('Migration dispatcher code changed')
    code_dir=str(path.parent)
    if code_dir not in sys.path: sys.path.insert(0,code_dir)
    spec=importlib.util.spec_from_file_location('sysu_dispatch_config',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    task_binding=binding(task_path)
    config=module.compose_single_task(SYSU_DEPLOYMENT_BINDING,task_binding)
    config_binding=module.write_config_version(config)
    bootstrap=index['migration_extensions']['bootstrap']
    if sha(bootstrap['path'])!=bootstrap['sha256']:
        raise ValueError('Migration bootstrap code changed')
    return [PYTHON,'-B','-s','-u',bootstrap['path'],'--config',config_binding['path'],
            '--config-sha256',config_binding['sha256'],'--execute-host-entry',
            str(PLAN/'implementation/host_entry.py'),'--task-json',str(task_path),
            '--index',str(index_path),'--log-path',str(log_path)],config_binding


def launch(state,task_path,run_id,index_path):
    nonce=uuid.uuid4().hex; session='g50pdiag_'+nonce[:16]
    log=PLAN/'logs'/(run_id+'_'+nonce+'.log'); log.parent.mkdir(exist_ok=True)
    worker_argv,runtime_config=sysu_worker_command(index_path,task_path,log)
    attempt=dict(run_id=run_id,task_path=str(task_path),task_sha256=sha(task_path),nonce=nonce,
                 session=session,log_path=str(log),launch_directory=str(PLAN/'launches'/nonce),
                 queued_at=time.time(),launch_state='intent_persisted')
    state['active']=attempt; state['attempts'].append(attempt);state['status']='launching'
    if state['waiting_since'] is not None:
        waited=time.time()-state['waiting_since'];state['queue_wait_seconds']+=waited
        state['waiting_since']=None
    write_state(state)
    command=['bash',str(WRAPPER),session,str(log),*worker_argv]
    env={**os.environ,'SPNV2_RUN_NONCE':nonce,'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUTF8':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0',
         'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'2021','MPLBACKEND':'Agg'}
    result=subprocess.run(command,cwd=PLAN,env=env,capture_output=True,text=True,timeout=20)
    attempt.update(wrapper_returncode=result.returncode,wrapper_output=result.stdout+result.stderr,
                   launch_state='wrapper_returned')
    state['status']='running' if result.returncode==0 else 'blocked'
    write_state(state)


def wait_for_terminal_evidence(state,item,reason,error_type=None):
    """Hold the same active attempt through a short unverified exit handoff.

    No process exception proves ownership, life or death. This grace only waits
    for the existing wrapper/outcome files; it cannot launch or accept a task.
    A healthy, positively verified running worker never starts this deadline.
    """
    now=time.monotonic()
    handoff=item.get('terminal_evidence_handoff')
    boot_id=state.get('manager',{}).get('boot_id')
    if handoff is None:
        handoff=dict(schema='g50p-terminal-evidence-handoff-v1',first_observed_at=time.time(),
            started_monotonic=now,deadline_monotonic=now+TERMINAL_HANDOFF_GRACE_SECONDS,
            grace_seconds=TERMINAL_HANDOFF_GRACE_SECONDS,boot_id=boot_id,
            identity_status='unverified',reason=reason,identity_probe_error_type=error_type)
        item['terminal_evidence_handoff']=handoff
    if handoff['boot_id']!=boot_id:
        state.update(status='blocked',reason='Boot changed during unresolved terminal-evidence handoff; preserve active attempt, no downstream dispatch')
    elif now>=handoff['deadline_monotonic']:
        handoff['timed_out_at']=time.time()
        state.update(status='blocked',reason='Terminal-evidence handoff timed out with identity unverified; preserve active attempt, no downstream dispatch')
    else:
        state.update(status='awaiting_terminal_evidence',reason='Waiting for existing wrapper/outcome evidence; process identity unverified, no downstream dispatch')


def finish(state):
    item=state['active'];exit_path=Path(item['log_path']+'.exitcode')
    directory=Path(item['launch_directory']);outcome_path=directory/'outcome.json'
    def terminal_ready():
        return exit_path.is_file() and outcome_path.is_file()
    if not terminal_ready():
        reason=None;error_type=None
        handoff=item.get('terminal_evidence_handoff')
        if handoff is not None:
            # Once identity became unavailable, only the existing terminal
            # files can close this handoff. Do not reinterpret the PID as alive.
            reason=handoff['reason'];error_type=handoff.get('identity_probe_error_type')
        elif exit_path.exists() or outcome_path.exists():
            reason='incomplete_terminal_evidence'
        elif time.time()-item['queued_at']>120:
            record=directory/'launch.json'
            if not record.is_file():
                reason='launch_identity_missing'
            else:
                from host_entry import live_identity
                try:
                    alive=live_identity(read(record))
                except (PermissionError,FileNotFoundError,ProcessLookupError) as error:
                    # /proc can disappear or deny environ while the worker
                    # exits. This is unverified, never successful ownership or
                    # confirmed death. Other exception classes remain visible.
                    reason='process_identity_unavailable_during_terminal_handoff'
                    error_type=type(error).__name__
                else:
                    if not alive:
                        reason='process_identity_not_verified'
        # The actual failure wrote its valid terminal files in the same second
        # as the /proc exception. Recheck before waiting or blocking.
        if not terminal_ready():
            if reason is not None:
                wait_for_terminal_evidence(state,item,reason,error_type)
            return
    handoff=item.get('terminal_evidence_handoff')
    if handoff is not None:
        handoff.update(resolved_at=time.time(),resolution='existing_terminal_evidence_received')
        if state.get('status')=='awaiting_terminal_evidence':
            state.update(status='running',reason='Terminal evidence arrived; continue normal strict outcome validation')
    # Completed terminal evidence is authoritative; a normally exited worker
    # need not retain its old /proc entry. All existing exit/outcome/report and
    # independent acceptance conditions remain unchanged below.
    code=int(exit_path.read_text().strip())
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
                or final_budget.get('unresolved_attempt_ids')
                or (final_budget.get('time_hard_stop_enabled',True)
                    and final_budget.get('active_time_bound_violation_attempt_ids'))):
            raise ValueError('Final delivery requires the actual settled budget receipt after process exit')
        complete=proof.get('complete_scope') is True and proof.get('completed_windows')==48 and len(state['accepted'])==48
        final_manifest=dict(schema='g50p-diagnostic-delivery-manifest-v1',status='complete' if complete else 'partial',
            summary=binding(summary_path),proof=binding(proof_path),final_settled_budget=binding(final_budget_path),
            active_seconds_charged_upper=final_budget['active_seconds_charged_upper'],
            time_hard_stop_enabled=final_budget.get('time_hard_stop_enabled',True),
            budget_policy=final_budget.get('budget_policy'),
            initial_cost_receipts=final_budget.get('initial_cost_receipts',[]),
            physical_forwards_reserved_upper=final_budget['physical_forwards_reserved_upper'],
            queue_wait_seconds=state['queue_wait_seconds'],complete_ODR_runs=0,
            independently_accepted_windows=len(state['accepted']),scientific_scope_complete=complete,
            note=('Time is accounted without a hard stop under the bound revision; this post-exit receipt supplies settled cost.'
                  if final_budget.get('time_hard_stop_enabled') is False else
                  'The summary running-time snapshot reserves its live deadline; this post-exit receipt supplies settled cost.'))
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
    # A technical manager revision gets a fresh verifier identity/file. The
    # original failed immutable verify task remains available as evidence.
    suffix=state['index']['sha256'][:12]
    task=dict(schema='g50p-risk-diagnostic-task-v1',action='verify_pair',run_id=row['run_id']+'_verify_'+suffix,
              implementation_sha256=index['implementation_sha256'],run_task=receipt['task'],
              execution_receipt=receipt,all_attempts=receipt['attempts'],
              proof_path=str(Path(science['output_dir'])/'proof.json'),index_path=str(PLAN/'index.json'))
    if 'index_binding' in science: task['index_binding']=science['index_binding']
    if index.get('budget_policy') is not None: task['budget_policy']=copy.deepcopy(index['budget_policy'])
    path=immutable(PLAN/'tasks'/index['sysu_task_revision']/(task['run_id']+'.json'),task)
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
                                 initial_cost_receipt=index.get('initial_cost_receipt'),
                                 budget_policy=index.get('budget_policy')).snapshot()
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


def main(index_path,deployment_path,deployment_sha256):
    global SYSU_DEPLOYMENT_BINDING
    SYSU_DEPLOYMENT_BINDING=dict(path=str(deployment_path),sha256=deployment_sha256)
    deployment=checked(SYSU_DEPLOYMENT_BINDING)
    if deployment.get("index")!=binding(index_path):
        raise ValueError("Deployment receipt and diagnostic index differ")
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
        for run_id,spec in index.get('preserved_main',{}).items():
            if state['main'].get(run_id)!=spec['main_state']:
                raise ValueError('Preserved completed main changed; never rerun its model work')
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
                    stream_code=index['streaming_migration']['code']
                    if sha(stream_code['path'])!=stream_code['sha256']:raise ValueError('Streaming helper changed')
                    spec=importlib.util.spec_from_file_location('spnv2_streaming_ready',stream_code['path'])
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                    ready=module.readiness(index,read(task_path))
                    state['dependency_readiness']=ready
                    if not ready['ready']:
                        state.update(status='waiting_verified_dependencies',reason=ready['missing'],next_resource_check=time.time()+RESOURCE_INTERVAL)
                        write_state(state);time.sleep(5);continue
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
    parser.add_argument('--deployment-manifest',required=True);parser.add_argument('--deployment-sha256',required=True)
    args=parser.parse_args();raise SystemExit(main(args.index,args.deployment_manifest,args.deployment_sha256))
