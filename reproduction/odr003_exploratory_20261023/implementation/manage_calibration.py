"""Verified append-only manager transition for the confirmed 003 calibration."""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from prepare_calibration_queue import B,P,N,Q,OLD,PY,read,sha,write
sys.path.insert(0,str(P/'pipeline'))
import runner
from revision import extend_state
from task_entry import process_alive
from progress import atomic_json,now

def argv(pid):return Path('/proc',str(pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
def owners():
 inode=(P/'pipeline.lock').stat().st_ino
 return [int(x.split()[4]) for x in Path('/proc/locks').read_text().splitlines() if len(x.split())>=6 and x.split()[1]=='FLOCK' and x.split()[5].split(':')[-1]==str(inode)]
def validate():
 old,new=read(OLD),read(Q/'proposal.json');proof=read(Q/'build_proof.json')
 assert len(old['tasks'])==52 and len(new['tasks'])==55 and new['tasks'][:52]==old['tasks']
 assert sha(Q/'proposal.json')==proof['proposal_file_sha256'] and runner.manifest_hash(new)==proof['new_manifest_sha256']
 assert [x['id'] for x in new['tasks'][52:]]==['odr003_O_lightbox','odr003_O_sunlamp','odr003_shared_calibration']
 runner.validate_manifest(new)
 for t in new['tasks'][52:]:
  assert runner.asset_hash_error(t) is None,t['id']
  assert all(not Path(p).exists() for p in [t['log'],t['log']+'.exitcode',*t['expected_outputs']])
 s=read(P/'status.json')
 assert s['status']=='phase_complete' and s['active'] is None and len(s['completed'])==52
 assert not s.get('error') and all(t['status']=='accepted' for t in s['tasks'])
 extend_state(s,old,new,caller_context={'operation':'003_candidate_independent_readback'})
 return old,new,s
def main(mode):
 assert os.environ.get('STY')
 old,new,s=validate()
 if mode=='verify':
  write(Q/'verified.json',dict(status='passed',proposal_sha256=sha(Q/'proposal.json'),canonical_sha256=runner.manifest_hash(new),old52_preserved=True,new_tasks=3,manager=s['manager'],at=now()))
  print('Candidate independently verified; no activation');return
 proof=read(Q/'verified.json');assert proof['status']=='passed' and proof['proposal_sha256']==sha(Q/'proposal.json')
 assert (N/'logs/prepare_calibration_b.log.exitcode').read_text().strip()=='0'
 assert (N/'logs/verify_calibration_candidate_a.log.exitcode').read_text().strip()=='0'
 d=N/'revisions/manager_calibration_20261023';d.mkdir(parents=True,exist_ok=True)
 assert not (d/'intent.json').exists(),'One-time operation already attempted'
 m=copy.deepcopy(s['manager']);ma=argv(m['pid'])
 assert process_alive(m) and owners()==[m['pid']]
 assert str(P/'pipeline/runner.py') in ma and ma[ma.index('--manifest')+1]==str(OLD)
 write(d/'before.json',s);write(d/'intent.json',dict(at=now(),authorization='Confirmed003 scope and standing manager-only switch permission',manager=m,manifest=str(Q/'proposal.json'),manifest_sha256=runner.manifest_hash(new)))
 assert process_alive(m) and argv(m['pid'])==ma and owners()==[m['pid']]
 os.kill(m['pid'],signal.SIGTERM)
 end=time.monotonic()+15
 while process_alive(m) and time.monotonic()<end:time.sleep(.2)
 assert not process_alive(m),'Old manager still alive; never duplicate'
 with (P/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stopped=read(P/'status.json');assert stopped['status']=='monitor_stopped' and stopped['tasks']==s['tasks'] and stopped['active'] is None
  extend_state(stopped,old,new,caller_context={'operation':'003_authorized_after_stop'})
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 for k in ('STY','WINDOW','PYTHONPATH','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
 cmd=['bash',str(runner.WRAPPER),'rebaseline_queue_odr003_calibration_20261023',str(Q/'manager.log'),PY,'-B','-s','-u',str(P/'pipeline/runner.py'),'--manifest',str(Q/'proposal.json'),'--extend-from',str(OLD)]
 launch=subprocess.run(cmd,cwd=P,env=env,capture_output=True,text=True,timeout=20)
 write(d/'launch.json',dict(command=cmd,exit_code=launch.returncode,stdout=launch.stdout,stderr=launch.stderr));assert launch.returncode==0
 end=time.monotonic()+25
 while True:
  after=read(P/'status.json')
  if after.get('manifest_sha256')==runner.manifest_hash(new) and process_alive(after.get('manager')):break
  assert not after.get('error'),after.get('error')
  assert time.monotonic()<end,'No startup acknowledgement; do not rerun controller'
  time.sleep(.2)
 assert owners()==[after['manager']['pid']] and not process_alive(m)
 for before,current in zip(s['tasks'],after['tasks'][:52]):
  for k in ('status','runtime','history','accepted_at','started_at','error'):
   assert before.get(k)==current.get(k),(before['id'],k)
 write(d/'accepted.json',dict(status='passed',at=now(),old_manager=m,new_manager=after['manager'],old52_records_preserved=True,old_tasks_restarted=False,
  manifest=str(Q/'proposal.json'),manifest_sha256=runner.manifest_hash(new),task_count=55,normal_recovery='Use --manifest only, never reuse --extend-from or this controller',revision=after['manifest_revisions'][-1]))
 print('Manager-only calibration stage activated; original52 preserved')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','activate']);a=p.parse_args()
 try:main(a.mode)
 except BaseException as e:
  if a.mode=='activate':
   d=N/'revisions/manager_calibration_20261023';d.mkdir(parents=True,exist_ok=True)
   atomic_json(d/'failure.json',dict(at=now(),error=repr(e),next_action='Preserve evidence; never repeat controller blindly'))
  raise
