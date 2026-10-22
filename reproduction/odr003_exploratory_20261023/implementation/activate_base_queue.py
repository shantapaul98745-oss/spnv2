"""Validate and activate the approved36-run stage without repeating calibration."""
import argparse,copy,fcntl,json,os,signal,subprocess,sys,time,hashlib
from pathlib import Path
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023');P=N.parent/'rebaseline_20261017'
sys.path.insert(0,str(N/'pipeline'))
import runner
from revision import extend_state
from task_entry import process_alive
from progress import atomic_json,now
runner.PLAN=N
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(mode):
 assert os.environ.get('STY')
 q=N/'queue_base';proof=read(q/'build_proof.json');oldpath=Path(proof['old_manifest_path']);newpath=q/'proposal.json'
 old,new=read(oldpath),read(newpath);s=read(N/'status.json')
 assert proof['status']=='passed' and sha(newpath)==proof['proposal_file_sha256']
 assert len(old['tasks'])==3 and len(new['tasks'])==40 and new['tasks'][:3]==old['tasks']
 assert s['status']=='phase_complete' and s['active'] is None and s['completed']==[t['id'] for t in old['tasks']]
 assert all(t['status']=='accepted' for t in s['tasks']) and not s.get('error')
 runner.validate_manifest(new);extend_state(s,old,new,caller_context={'operation':'003_base_independent_preflight'})
 for t in new['tasks'][3:]:
  assert runner.asset_hash_error(t) is None,t['id']
  assert all(not Path(p).exists() for p in [t['log'],t['log']+'.exitcode',*t['expected_outputs']])
 if mode=='verify':
  assert not (q/'verified.json').exists()
  atomic_json(q/'verified.json',dict(status='passed',manifest_sha256=runner.manifest_hash(new),proposal_sha256=sha(newpath),old3_preserved=True,added_adaptations=36,added_summary=1))
  print('3->40 candidate independently verified; no activation');return
 v=read(q/'verified.json');assert v['status']=='passed' and v['proposal_sha256']==sha(newpath)
 assert (N/'logs/prepare_base_a.log.exitcode').read_text().strip()=='0' and (N/'logs/verify_base_a.log.exitcode').read_text().strip()=='0'
 d=N/'revisions/base_stage';d.mkdir(exist_ok=False)
 m=s['manager'];assert process_alive(m)
 cmd=Path('/proc',str(m['pid']),'cmdline').read_bytes()
 assert str(N/'implementation/launch_plan.py').encode() in cmd and str(oldpath).encode() in cmd
 atomic_json(d/'before.json',s);atomic_json(d/'intent.json',dict(at=now(),manager=m,manifest=str(newpath),sha256=runner.manifest_hash(new),authorization='Confirmed v2 36 base runs and standing manager-only switch'))
 os.kill(m['pid'],signal.SIGTERM);end=time.monotonic()+15
 while process_alive(m) and time.monotonic()<end:time.sleep(.2)
 assert not process_alive(m)
 with (P/'pipeline.lock').open('a') as oldlock,(N/'pipeline.lock').open('a') as lock:
  fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stopped=read(N/'status.json');assert stopped['tasks']==s['tasks'] and stopped['active'] is None
  extend_state(stopped,old,new,caller_context={'operation':'003_base_after_manager_stop'})
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 for k in ('STY','WINDOW','PYTHONPATH','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
 command=['bash',str(runner.WRAPPER),'spnv2_odr003_manager_base',str(q/'manager.log'),runner.PYTHON,'-B','-s','-u',str(N/'implementation/launch_plan.py'),'--manifest',str(newpath),'--extend-from',str(oldpath)]
 launch=subprocess.run(command,cwd=N,env=env,capture_output=True,text=True,timeout=20)
 atomic_json(d/'launch.json',dict(command=command,returncode=launch.returncode,stdout=launch.stdout,stderr=launch.stderr));assert launch.returncode==0
 end=time.monotonic()+25
 while True:
  after=read(N/'status.json')
  if after.get('manifest_sha256')==runner.manifest_hash(new) and process_alive(after.get('manager')):break
  assert time.monotonic()<end
  time.sleep(.2)
 for a,b in zip(s['tasks'],after['tasks'][:3]):
  for k in ('status','runtime','history','accepted_at','started_at','error'):assert a.get(k)==b.get(k)
 atomic_json(d/'accepted.json',dict(status='passed',at=now(),manager=after['manager'],old3_records_preserved=True,manifest=str(newpath),manifest_sha256=runner.manifest_hash(new),task_count=40,adaptation_runs=36,normal_recovery='launch_plan.py --manifest '+str(newpath)))
 print(json.dumps(read(d/'accepted.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','activate']);main(p.parse_args().mode)
