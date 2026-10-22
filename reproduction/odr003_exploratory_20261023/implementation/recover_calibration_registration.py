"""One bounded recovery after rejected manifest location, before state binding."""
import fcntl,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from prepare_calibration_queue import P,N,Q,OLD,PY,read,sha,write
sys.path.insert(0,str(P/'pipeline'))
import runner
from revision import extend_state
from task_entry import process_alive
def main():
 assert os.environ.get('STY')
 d=N/'revisions/manager_calibration_registration_b';d.mkdir(exist_ok=False)
 old_before=read(N/'revisions/manager_calibration_20261023/before.json')
 assert (Q/'manager.log.exitcode').read_text().strip()=='1'
 assert 'Path outside this project' in (Q/'manager.log').read_text()
 with (P/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  s=read(P/'status.json');assert s['status']=='monitor_stopped' and s['active'] is None
  assert not process_alive(s['manager']) and s['tasks']==old_before['tasks'] and s['manifest_sha256']==old_before['manifest_sha256']
  old,new=read(OLD),read(Q/'proposal.json');assert new['tasks'][:52]==old['tasks']
  assert sha(Q/'proposal.json')==read(Q/'verified.json')['proposal_sha256']
  for t in new['tasks'][52:]:
   assert runner.asset_hash_error(t) is None,t['id']
   assert not (P/'events'/(t['id']+'.main.process.json')).exists(),t['id']
   assert not Path(t['log']).exists(),t['id']
  extend_state(s,old,new,caller_context={'operation':'correct_rejected_location_before_first_binding'})
  registered=P/'queue_proposals/odr003_calibration_20261023/proposal.json'
  registered.parent.mkdir(parents=True,exist_ok=False)
  registered.write_bytes((Q/'proposal.json').read_bytes());assert sha(registered)==sha(Q/'proposal.json')
  runner.inside(registered,P)
  write(d/'preflight.json',dict(status='passed',old52_evidence_unchanged=True,no_new_task_ever_started=True,old_manager_exited=True,lock_exclusive=True,manifest=str(registered),sha256=sha(registered)))
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 for k in ('STY','WINDOW','PYTHONPATH','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
 cmd=['bash',str(runner.WRAPPER),'rebaseline_queue_odr003_calibration_b',str(Q/'manager_registration_b.log'),PY,'-B','-s','-u',str(P/'pipeline/runner.py'),'--manifest',str(registered),'--extend-from',str(OLD)]
 launch=subprocess.run(cmd,cwd=P,env=env,capture_output=True,text=True,timeout=20)
 write(d/'launch.json',dict(command=cmd,exit_code=launch.returncode,stdout=launch.stdout,stderr=launch.stderr));assert launch.returncode==0
 end=time.monotonic()+25
 while True:
  after=read(P/'status.json')
  if after.get('manifest_sha256')==runner.manifest_hash(new) and process_alive(after.get('manager')):break
  assert not after.get('error'),after.get('error')
  assert time.monotonic()<end,'No acknowledgement; preserve evidence'
  time.sleep(.2)
 for a,b in zip(old_before['tasks'],after['tasks'][:52]):
  for k in ('status','runtime','history','accepted_at','started_at','error'):assert a.get(k)==b.get(k)
 write(d/'accepted.json',dict(status='passed',old52_records_preserved=True,manager=after['manager'],manifest=str(registered),canonical_sha256=runner.manifest_hash(new),normal_recovery='Use only --manifest '+str(registered),revision=after['manifest_revisions'][-1]))
 print(json.dumps(read(d/'accepted.json')))
if __name__=='__main__':main()
