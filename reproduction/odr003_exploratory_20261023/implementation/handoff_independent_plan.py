"""One controlled handoff from failed preflight to the independent003 plan.
The old attempted run and original52 execution records remain immutable history.
"""
import fcntl,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023');P=N.parent/'rebaseline_20261017'
sys.path.insert(0,str(N/'pipeline'))
import runner
from task_entry import process_alive
from revision import canonical_sha256
runner.PLAN=N
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def main():
 assert os.environ.get('STY')
 d=N/'revisions/independent_plan_handoff';d.mkdir(exist_ok=False)
 q=N/'queue_offline_r1/proposal.json';manifest=read(q);runner.validate_manifest(manifest)
 for t in manifest['tasks']:
  assert runner.asset_hash_error(t) is None,t['id']
  assert not Path(t['log']).exists()
  assert all(not Path(x).exists() for x in t['expected_outputs'])
 runner.initialize_state(read(N/'status.json'),manifest)
 smoke=read(N/'preflight/test_cuda_entry_init_20261023_a.json')
 assert smoke['status']=='passed'
 before=read(P/'status.json');m=before['manager']
 assert before['status']=='failed' and before['active']=='odr003_O_lightbox'
 assert m['pid']==3812181 and str(m['starttime'])=='565051994' and process_alive(m)
 assert len(before['completed'])==52 and all(t['status']=='accepted' for t in before['tasks'][:52])
 assert before['tasks'][52]['status']=='failed' and not process_alive(read(before['tasks'][52]['runtime']['record']))
 assert all(t['status']=='queued' and not t.get('runtime') for t in before['tasks'][53:])
 assert (N/'logs/odr003_O_lightbox.log.exitcode').read_text().strip()=='1'
 failure=read(N/'runs/odr003_O_lightbox/failure.json');assert failure['phase']=='preflight' and failure['message']=='Invalid device argument '
 progress=read(N/'runs/odr003_O_lightbox/progress.json');assert progress['observed']==progress['evaluated']==progress['updates']==0
 cmdline=Path('/proc',str(m['pid']),'cmdline').read_bytes();assert str(P/'pipeline/runner.py').encode() in cmdline
 write(d/'old_before.json',before);write(d/'new_before.json',read(N/'status.json'))
 write(d/'preflight.json',dict(status='passed',repair='Explicit CUDA device initialization before memory statistics',failed_attempt_retained=True,no_old_attempt_forward=True,old52_accepted=True,new_manifest_sha256=canonical_sha256(manifest)))
 os.kill(m['pid'],signal.SIGTERM);deadline=time.monotonic()+15
 while process_alive(m) and time.monotonic()<deadline:time.sleep(.2)
 assert not process_alive(m)
 with (P/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stopped=read(P/'status.json');assert stopped['tasks']==before['tasks']
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 for k in ('STY','WINDOW','PYTHONPATH','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
 cmd=['bash',str(runner.WRAPPER),'spnv2_odr003_manager_r1',str(N/'queue_offline_r1/manager.log'),runner.PYTHON,'-B','-s','-u',str(N/'implementation/launch_plan.py'),'--manifest',str(q)]
 p=subprocess.run(cmd,cwd=N,env=env,capture_output=True,text=True,timeout=20)
 write(d/'launch.json',dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0
 deadline=time.monotonic()+25
 while True:
  s=read(N/'status.json')
  if s.get('manifest_sha256')==canonical_sha256(manifest) and process_alive(s.get('manager')):break
  assert time.monotonic()<deadline,'No new manager acknowledgement'
  time.sleep(.2)
 write(d/'accepted.json',dict(status='passed',manager=s['manager'],new_manifest=str(q),new_manifest_sha256=canonical_sha256(manifest),old_plan_tasks_unchanged=True,old_manager_stopped=True,old52_never_rerun=True,source_and_scientific_parameters_unchanged=True,normal_recovery='launch_plan.py --manifest '+str(q)))
 print(json.dumps(read(d/'accepted.json')))
if __name__=='__main__':main()
