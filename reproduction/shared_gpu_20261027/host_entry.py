"""One owned diagnostic screen, admitted before any model/CPU-heavy import.

The watchdog may signal only the exact registered diagnostic worker via pidfd.
It never searches for or signals another lab user's GPU process.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from budget import GlobalBudget, AReadOnlyBudget, BudgetExhausted, atomic_json

BASE = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
PLAN = BASE/'spnv2_runs/plans/g50p_risk_diagnostic_20261025'
OLD = BASE/'spnv2_runs/plans/odr003_round2_20261023'
PROJECT = 'g50p_risk_diagnostic_20261025'
LOCKS = (BASE/'spnv2_runs/plans/rebaseline_20261017/pipeline.lock',
         BASE/'spnv2_runs/plans/odr003_exploratory_20261023/pipeline.lock',
         OLD/'pipeline.lock', PLAN/'pipeline.lock')


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


@contextmanager
def project_lock(path):
    import fcntl
    handle=Path(path).open('a+b')
    try:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally: handle.close()


def host_modules(index):
    modules={}; resource=index.get('resource_policy')
    if resource is not None:
        expected=PLAN/'revisions/shared_gpu_20261027/resource_policy.json'
        if (set(resource)!={'path','sha256'} or Path(resource['path']).resolve(strict=True)!=expected
                or sha(expected)!=resource['sha256']):
            raise ValueError('Exact authorized shared-resource policy binding is required')
    for name in ('host_gate','host_snapshot'):
        item=index['host_modules'][name]
        path=Path(item['path']).resolve(strict=True)
        expected=(BASE/'spnv2/reproduction/shared_gpu_20261027' if resource is not None else OLD/'implementation')/(name+'.py')
        if path!=expected or sha(path)!=item['sha256']:
            raise ValueError('Only exact version-bound read-only host resource modules are allowed')
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        spec.loader.exec_module(module); modules[name]=module
    if resource is not None: modules['host_gate'].configure_policy(resource)
    return modules['host_gate'],modules['host_snapshot']


def validate_task(task_path,index_path):
    task_path=Path(task_path).resolve(strict=True); index_path=Path(index_path).resolve(strict=True)
    if index_path!=PLAN/'index.json' or not task_path.is_relative_to(PLAN/'tasks'):
        raise ValueError('Diagnostic task/index path escapes the isolated plan')
    index,task=read(index_path),read(task_path)
    if index.get('schema')!='g50p-risk-diagnostic-index-v1' or task.get('schema')!='g50p-risk-diagnostic-task-v1':
        raise ValueError('Wrong diagnostic schema')
    if task.get('action') not in ('extract_a','analyze_a','run_pair','verify_pair','verify_pilots','summarize_diagnostic'):
        raise ValueError('Unapproved diagnostic action')
    if task['implementation_sha256']!=index['implementation_sha256']:
        raise ValueError('Task implementation differs from frozen diagnostic index')
    for name,digest in task['implementation_sha256'].items():
        if Path(name).name!=name or sha(PLAN/'implementation'/name)!=digest:
            raise ValueError('Diagnostic implementation binding changed')
    if task['action']=='run_pair':
        rows=[row for row in index['tasks'] if row['run_id']==task['run_id']]
        if len(rows)!=1 or rows[0]['task_path']!=str(task_path) or rows[0]['task_sha256']!=sha(task_path):
            raise ValueError('Run is not one of the frozen forty-eight diagnostic windows')
    if index.get('budget_policy') is not None and task.get('budget_policy') != index['budget_policy']:
        raise ValueError('Task budget-policy binding differs from the amended index')
    return task,index


def live_identity(record):
    argv=record.get('argv',[])
    if (record.get('project_id')!=PROJECT or str(PLAN/'implementation/host_entry.py') not in argv
            or '--index' not in argv or argv[argv.index('--index')+1]!=str(PLAN/'index.json')
            or '--task-json' not in argv or not Path(argv[argv.index('--task-json')+1]).is_relative_to(PLAN/'tasks')):
        return False
    proc=Path('/proc')/str(record['pid'])
    raw_stat=(proc/'stat').read_text(); command=(proc/'cmdline').read_bytes()
    environment=(proc/'environ').read_bytes().split(b'\0')
    return (proc.stat().st_uid==os.getuid() and
            int(raw_stat[raw_stat.rfind(')')+2:].split()[19])==record['starttime'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id'] and
            hashlib.sha256(command).hexdigest()==record['cmdline_sha256'] and
            b'SPNV2_RUN_NONCE='+record['run_nonce'].encode() in environment)


def launch_record(task,log_path,gate):
    nonce=os.environ.get('SPNV2_RUN_NONCE','')
    if len(nonce)<16 or not os.environ.get('STY'):
        raise ValueError('A unique diagnostic nonce and independent screen are required')
    proc=Path('/proc')/str(os.getpid()); stat=(proc/'stat').read_text()
    argv=[part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
    return dict(schema='spnv2.round2.launch.v1', project_id=PROJECT,run_id=task['run_id'],
                run_nonce=nonce,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=os.getpid(),starttime=int(stat[stat.rfind(')')+2:].split()[19]),uid=os.getuid(),
                argv=argv,cmdline_sha256=gate.command_sha256(argv),cwd=str(Path.cwd()),
                session=os.environ['STY'],log_path=str(log_path),launched_at=time.time(),role='worker')


def conflict(snapshot,record,gate):
    if (not snapshot.get('query_ok') or not snapshot.get('inventory_complete')
            or not snapshot.get('worker_inventory_complete') or snapshot.get('collection_errors')):
        return 'resource_observation_incomplete'
    if snapshot.get('foreign_gpu_pids') and not getattr(gate,'shared_gpu_enabled',lambda:False)(): return 'foreign_compute_arrived'
    checks=[row for row in snapshot.get('worker_checks',[]) if row.get('expected_record',{}).get('run_nonce')==record['run_nonce']]
    if (len(checks)!=1 or not checks[0].get('process')
            or not gate.verify_ownership(checks[0]['process'],record,checks[0]['observed_record']).is_own):
        return 'own_worker_registration_unverified'
    for row in snapshot.get('worker_checks',[]):
        if row.get('process') and row['process']['pid']!=record['pid']: return 'another_owned_worker_present'
    return None


def watchdog(path):
    """System-Python supervisor: exact pidfd only, never a name/UID based kill."""
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PLAN/'launches') or path.name!='watchdog.json':
        raise ValueError('Watchdog record escapes this diagnostic')
    spec=read(path); record=spec['launch_record']
    if (record['project_id']!=PROJECT or not live_identity(record)
            or not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal')):
        raise ValueError('Exact diagnostic identity or real pidfd API unavailable')
    fd=os.pidfd_open(record['pid'],0)
    try:
        if not live_identity(record): raise ValueError('Diagnostic birth changed before pidfd pin')
        atomic_json(path.parent/'watchdog_ready.json',dict(status='ready',pidfd_target=record['pid'],
                    starttime=record['starttime'],deadline_monotonic=spec['kill_at_monotonic']))
        while True:
            if spec['kill_at_monotonic'] is None:
                # Identity/exit monitor only: explicit removal of every derived time kill.
                if select.select([fd],[],[],.25)[0]: return 0
                continue
            remaining=spec['kill_at_monotonic']-time.monotonic()
            if remaining<=0:
                if select.select([fd],[],[],0)[0]: return 0
                late=max(0.,time.monotonic()-spec['kill_at_monotonic'])
                atomic_json(path.parent/'watchdog_timeout.json',dict(status='deadline_stop',
                            launch_record=record,unknown_work_preserved=True,lateness_seconds=late,
                            exact_cost_claim=False,reason='diagnostic_active_time_upper_bound'))
                signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
                return 0
            if select.select([fd],[],[],min(.25,remaining))[0]: return 0
    finally: os.close(fd)


def start_watchdog(attempt_dir,record,budget):
    path=attempt_dir/'watchdog.json'
    # Stop new work sixty seconds before this; reserve ten more seconds for
    # supervisor scheduling/termination instead of saving a heavy state at cap.
    atomic_json(path,dict(launch_record=record,kill_at_monotonic=budget.deadline_monotonic-10. if budget.deadline_monotonic is not None else None,
                        active_time_deadline_monotonic=budget.deadline_monotonic))
    log=(attempt_dir/'watchdog.log').open('ab',buffering=0)
    child=subprocess.Popen(['/usr/bin/python3','-B',str(PLAN/'implementation/host_entry.py'),
                            '--watchdog-json',str(path)],stdout=log,stderr=subprocess.STDOUT,
                           cwd=PLAN,env={**os.environ,'PYTHONNOUSERSITE':'1'})
    started=time.monotonic()
    while not (attempt_dir/'watchdog_ready.json').exists():
        budget.check_time('watchdog_initialization')
        if child.poll() is not None or time.monotonic()-started>5:
            raise RuntimeError('Deadline supervisor did not establish exact pidfd before model import')
        time.sleep(.02)
    return child,log


def execute(task_path,index_path,log_path):
    task,index=validate_task(task_path,index_path)
    gate,collector=host_modules(index)
    nonce=os.environ.get('SPNV2_RUN_NONCE','')
    if not nonce.isalnum() or len(nonce)<16: raise ValueError('Invalid diagnostic launch nonce')
    attempt_dir=PLAN/'launches'/nonce; attempt_dir.mkdir(parents=True,exist_ok=False)
    atomic_json(attempt_dir/'started.json',dict(task_path=str(task_path),task_sha256=sha(task_path),started_at=time.time()))
    def outcome(value):
        atomic_json(attempt_dir/'outcome.json',{**value,'finished_at':time.time()})
        return value
    budget=None
    try:
        with ExitStack() as stack:
            for lock in LOCKS: stack.enter_context(project_lock(lock))
            registry=PLAN/'launches/registry.json'
            if not registry.exists(): atomic_json(registry,dict(schema='spnv2.round2.registry.v1',project_id=PROJECT,records=[]))
            before=collector.collect_snapshot(registry,project_root=PLAN,project_id=PROJECT)
            after=collector.collect_snapshot(registry,project_root=PLAN,project_id=PROJECT)
            admitted=gate.assess_launch_boundary(before,after,now=time.time(),project_lock_held=True,task_kind='gpu')
            atomic_json(attempt_dir/'admission.json',dict(before=before,after=after,decision=asdict(admitted)))
            if not admitted.allow_start:
                return outcome(dict(status='waiting_resources',exit_code=75,models_imported=False))
            record=launch_record(task,log_path,gate)
            record.update(task_sha256=sha(task_path),index_sha256=sha(index_path),
                          implementation_sha256=task['implementation_sha256'])
            atomic_json(attempt_dir/'launch.json',record)
            document=read(registry); document['records'].append(dict(record_path=str(attempt_dir/'launch.json'),
                    record_sha256=gate.launch_record_sha256(record),expected_record=record)); atomic_json(registry,document)
            def conflict_check():
                return conflict(collector.collect_snapshot(registry,project_root=PLAN,project_id=PROJECT),record,gate)
            reason=conflict_check()
            if reason or time.time()>admitted.valid_until:
                return outcome(dict(status='waiting_resources',exit_code=75,reason=reason or 'admission_expired',models_imported=False))
            if task['action'] in ('extract_a','analyze_a'):
                budget=AReadOnlyBudget(PLAN/'cost_A',record,max_seconds=index['a_execution_max_seconds'])
            else:
                global_budget=GlobalBudget(PLAN/'budget',dict(project_id=PROJECT,
                                           frozen_inputs_sha256=index['frozen_inputs']['sha256']),
                                           initial_cost_receipt=index.get('initial_cost_receipt'),budget_policy=index.get('budget_policy'))
                budget=global_budget.begin_attempt(record,allow_model_forwards=task['action']=='run_pair')
            start_watchdog(attempt_dir,record,budget)
            # All heavy dependencies occur after shared-host and durable budget
            # admission, including independent CPU result verification.
            route={'extract_a':('extract_A','execute'),'analyze_a':('analyze_existing','execute'),
                   'run_pair':('pair_runner','execute'),'verify_pair':('verify_pair','execute'),
                   'verify_pilots':('verify_pilots','execute'),'summarize_diagnostic':('summarize_diagnostic','execute')}[task['action']]
            module=__import__(route[0]); runner=getattr(module,route[1])
            result=runner(Path(task_path),admission=asdict(admitted),launch_record=record,
                          conflict_check=conflict_check,budget=budget)
            budget.check_time('after_runner_and_final_io')
            outcome(result)
            closed=budget.close(result['status'])
            # This small final receipt is covered by close's conservative tail.
            atomic_json(attempt_dir/'budget_closed.json',closed)
            if hasattr(budget,'check_final_receipt'): budget.check_final_receipt()
            return result
    except BlockingIOError:
        return outcome(dict(status='waiting_project_lock',exit_code=75,models_imported=False))
    except Exception as error:
        result=dict(status='budget_stopped' if isinstance(error,BudgetExhausted) else 'failed',
                    exit_code=1,error_type=type(error).__name__,error=str(error),preserve_and_block=True)
        outcome(result)
        if budget is not None and not budget.closed:
            try: budget.close(result['status'])
            except Exception: pass  # unresolved allocation remains conservatively fully charged
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--task-json');parser.add_argument('--index')
    parser.add_argument('--log-path');parser.add_argument('--watchdog-json')
    args=parser.parse_args()
    if args.watchdog_json: raise SystemExit(watchdog(args.watchdog_json))
    if not all((args.task_json,args.index,args.log_path)): parser.error('Task, index and log are required')
    raise SystemExit(execute(args.task_json,args.index,args.log_path)['exit_code'])
