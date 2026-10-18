"""One explicitly authorized manager-only cutover; never restart a worker."""
import copy, fcntl, hashlib, json, os, pathlib, select, signal, subprocess, sys, time
P=pathlib.Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D=P/'queue_proposals/observed_stage_20261018'
sys.path.insert(0,str(P/'pipeline'))
import runner
from pending_observers import revise_queued_observers
from progress import atomic_json
from task_entry import process_alive

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
proof=read(D/'preflight.json')
assert (D/'preflight_b.log.exitcode').read_text().strip()=='0'
assert proof['status']=='passed_inactive_proposal' and not proof['activated']
assert sha(D/'proposal.json')==proof['proposal_sha256']=='9d2ae4f7e74e575a779e611c7d872653aa6495989b38c8b8260abf479b899153'
old=read(P/'queue_v2.json');new=read(D/'proposal.json')
assert runner.manifest_hash(new)==proof['proposal_canonical_sha256']
runner.validate_manifest(new)
seen={}
for task in new['tasks']:
    for path,digest in task['metadata'].get('asset_sha256',{}).items():
        if path not in seen:seen[path]=sha(path)
        assert seen[path]==digest, (task['id'],path)
state=read(P/'status.json')
assert state['manifest_sha256']==runner.manifest_hash(old)
assert state['status']=='running' and state['active']=='train_phi3_EHS'
assert not state.get('error') and not state.get('downstream_blocked')
worker=copy.deepcopy(state['current']);manager=copy.deepcopy(state['manager'])
assert worker['pid']==3111276 and str(worker['starttime'])=='517862167'
assert manager['pid']==3298458 and str(manager['starttime'])=='518141786'
assert process_alive(worker) and process_alive(manager)
assert b'source/tools/train.py' in pathlib.Path('/proc',str(worker['pid']),'cmdline').read_bytes()
assert b'pipeline/runner.py' in pathlib.Path('/proc',str(manager['pid']),'cmdline').read_bytes()
revise_queued_observers(state,old,new,task_ids=proof['pending_task_ids'],caller_context={'operation':'authorized_cutover_preflight_only'})
destination=P/'queue_observed_stage_20261018.json'
if destination.exists():
    assert destination.read_bytes()==(D/'proposal.json').read_bytes()
else:
    with destination.open('xb') as f:f.write((D/'proposal.json').read_bytes());f.flush();os.fsync(f.fileno())
receipt_dir=P/'revisions/manager_observed_cutover_20261018_b'
receipt_dir.mkdir(exist_ok=False)
atomic_json(receipt_dir/'before.json',state)
atomic_json(receipt_dir/'intent.json',{
    'operation':'one_explicitly_authorized_graceful_manager_only_restart',
    'user_authorization':'2026-10-18 user answered: allow this one manager-only switch; training continues',
    'old_manager':manager,'worker':{k:worker[k] for k in ['pid','starttime']},
    'manifest':str(destination),'canonical_sha256':proof['proposal_canonical_sha256'],
    'pending_ids':proof['pending_task_ids'],'training_must_not_restart':True,
})
# Reuse the established exact PID/starttime shutdown contract on this Python build.
assert process_alive(manager) and process_alive(worker)
os.kill(manager['pid'],signal.SIGTERM)
deadline=time.monotonic()+15
while process_alive(manager) and time.monotonic()<deadline:time.sleep(.2)
assert not process_alive(manager), 'Old manager has not exited; do not start another'
assert process_alive(worker),'Worker identity changed; preserve evidence, no replacement worker'
with (P/'pipeline.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=read(P/'status.json')
    assert stopped['manager']['pid']==manager['pid'] and stopped['status']=='monitor_stopped'
    revise_queued_observers(stopped,old,new,task_ids=proof['pending_task_ids'],caller_context={'operation':'after_manager_exit_preflight_only'})
    atomic_json(receipt_dir/'stopped.json',stopped)

env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',TORCH_HOME=str(P.parents[1]/'env/cache/torch'))
env.pop('PYTHONPATH',None)
env.pop('CUDA_VISIBLE_DEVICES',None)
cmd=['bash',str(runner.WRAPPER),'rebaseline_queue_observed_20261018',str(P/'logs/queue_manager_observed_20261018.log'),runner.PYTHON,'-B','-s','-u',str(P/'pipeline/runner.py'),'--manifest',str(destination),'--revise-pending-from',str(P/'queue_v2.json')]
for ident in proof['pending_task_ids']:cmd+=['--pending-task',ident]
result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=20)
atomic_json(receipt_dir/'launch.json',dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stdout+result.stderr
# A bounded startup acknowledgement, not a periodic experiment monitoring loop.
deadline=time.monotonic()+15
while True:
    after=read(P/'status.json')
    if after.get('manifest_sha256')==proof['proposal_canonical_sha256'] and after.get('status')=='running' and process_alive(after.get('manager')):
        break
    assert time.monotonic()<deadline, 'No valid startup acknowledgement; preserve state, do not relaunch'
    time.sleep(.2)
assert after['active']=='train_phi3_EHS' and len(after['tasks'])==49
assert after['current']['pid']==worker['pid'] and str(after['current']['starttime'])==str(worker['starttime']) and process_alive(worker)
assert after['manager']['pid']!=manager['pid'] and process_alive(after['manager'])
assert after['tasks'][0]['runtime']['record']==state['tasks'][0]['runtime']['record']
assert not after.get('error') and not after.get('downstream_blocked')
for task in after['tasks'][1:]:assert task['status']=='queued' and 'runtime' not in task
atomic_json(receipt_dir/'after.json',after)
receipt={
    'status':'passed','operation':'authorized_manager_only_observer_revision',
    'worker_pid_unchanged':worker['pid'],'worker_starttime_unchanged':worker['starttime'],
    'training_restarted':False,'old_manager':manager,'new_manager':after['manager'],
    'task_count':49,'original_training_tasks_unchanged':True,
    'six_source_evaluations_wrapped_before_start':True,'fixed_odr_tasks_appended':40,
    'manifest':str(destination),'manifest_sha256':after['manifest_sha256'],
    'revision_audit':after['manifest_revisions'][-1],
    'before_epoch':worker.get('epoch'),'before_batch':worker.get('batch'),
    'after_epoch':after['current'].get('epoch'),'after_batch':after['current'].get('batch'),
    'full_plan_completed':False,
    'normal_recovery':'Use --manifest '+str(destination)+' without any historical revision/extend flags',
}
atomic_json(receipt_dir/'accepted.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
