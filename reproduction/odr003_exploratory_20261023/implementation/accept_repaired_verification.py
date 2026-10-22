"""Audited acceptance of a repaired CPU verifier; never reruns the GPU task."""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023');P=N.parent/'rebaseline_20261017'
sys.path.insert(0,str(N/'pipeline'))
import runner
from task_entry import process_alive
from progress import atomic_json,now
runner.PLAN=N
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(config):
 assert os.environ.get('STY')
 d=N/'revisions/cpu_verification_repair';d.mkdir(exist_ok=False)
 oldpath=N/'queue_offline_r1/proposal.json';newpath=N/'queue_offline_verified/proposal.json'
 before=read(N/'status.json');old=read(oldpath);new=copy.deepcopy(old);s=copy.deepcopy(before)
 assert before['status']=='failed' and before['active']=='odr003_O_lightbox_r1' and len(before['tasks'])==3
 task=before['tasks'][0];assert task['status']=='failed' and task['runtime']['phase']=='verify'
 assert all(t['status']=='queued' and not t.get('runtime') for t in before['tasks'][1:])
 v=read(config);result=read(v['result']);record=read(v['record'])
 assert v['phase']=='verify' and v['command']==task['verify_command'] and result['exit_code']==0
 assert Path(v['log']+'.exitcode').read_text().strip()=='0' and not process_alive(record)
 assert (N/'logs/odr003_O_lightbox_r1.log.exitcode').read_text().strip()=='0'
 proof=read(N/'runs/odr003_O_lightbox_r1/proof.json');assert proof['status']=='passed'
 assert proof['report_sha256']==sha(N/'runs/odr003_O_lightbox_r1/report.json')
 assert proof['independently_recomputed_metric_rows']==512 and proof['counts']['observed']==0
 verifier=str(N/'implementation/verify_exploration.py');newhash=sha(verifier)
 assert proof['verifier_sha256']==newhash
 for i,t in enumerate(new['tasks']):
  assert verifier in t['metadata']['asset_sha256']
  t['metadata']['asset_sha256'][verifier]=newhash
  s['tasks'][i]['metadata']['asset_sha256'][verifier]=newhash
  assert runner.asset_hash_error(t) is None,t['id']
 runner.validate_manifest(new)
 current=s['tasks'][0]
 current.setdefault('history',[]).append(copy.deepcopy(current['runtime']))
 current['runtime']={k:v[k] for k in ['phase','session','log','config','record','result','command','launched_at','tail_offset','tail_pending','launch_attempted']}
 current['runtime'].update(exit_code=0,wrapper_exit_code=0,observed_live=True)
 current['status']='accepted';current['accepted_at']=now();current.pop('error',None)
 s.update(manifest_sha256=runner.manifest_hash(new),status='ready_after_independent_verification_repair',active=None,error=None,current={},completed=[current['id']])
 s.setdefault('notes',[]).append('Original O main succeeded and was not rerun. Independent CPU verifier dtype was corrected; failed verification history retained; all512 saved predictions independently accepted.')
 s.setdefault('manifest_revisions',[]).append(str(d/'audit.json'))
 runner.initialize_state(s,new)
 m=before['manager'];assert process_alive(m)
 assert str(N/'implementation/launch_plan.py').encode() in Path('/proc',str(m['pid']),'cmdline').read_bytes()
 atomic_json(d/'before.json',before);atomic_json(d/'old_manifest.json',old)
 atomic_json(d/'audit.json',dict(status='prepared',operation='repaired_CPU_verifier_only_accept_existing_GPU_output',old_manifest_sha256=runner.manifest_hash(old),new_manifest_sha256=runner.manifest_hash(new),
  main_forward_repeated=False,repaired_verification_config=str(config),proof_sha256=sha(N/'runs/odr003_O_lightbox_r1/proof.json'),new_verifier_sha256=newhash,
  permitted_definition_change='Only verifier file SHA in asset maps; all commands, scientific configs, input IDs and output files unchanged',failure_history_preserved=True))
 os.kill(m['pid'],signal.SIGTERM);end=time.monotonic()+15
 while process_alive(m) and time.monotonic()<end:time.sleep(.2)
 assert not process_alive(m)
 with (P/'pipeline.lock').open('a') as legacy_lock, (N/'pipeline.lock').open('a') as lock:
  fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stopped=read(N/'status.json');assert stopped['tasks']==before['tasks']
  newpath.parent.mkdir(exist_ok=False);atomic_json(newpath,new);atomic_json(d/'after_state.json',s);atomic_json(N/'status.json',s)
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 for k in ('STY','WINDOW','PYTHONPATH','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
 cmd=['bash',str(runner.WRAPPER),'spnv2_odr003_manager_verified',str(newpath.parent/'manager.log'),runner.PYTHON,'-B','-s','-u',str(N/'implementation/launch_plan.py'),'--manifest',str(newpath)]
 p=subprocess.run(cmd,cwd=N,env=env,capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
 atomic_json(d/'launch.json',dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
 end=time.monotonic()+25
 while True:
  state=read(N/'status.json')
  if state.get('manager',{}).get('pid')!=m['pid'] and process_alive(state.get('manager')):break
  assert time.monotonic()<end
  time.sleep(.2)
 assert state['tasks'][0]['status']=='accepted'
 atomic_json(d/'accepted.json',dict(status='passed',manager=state['manager'],manifest=str(newpath),manifest_sha256=runner.manifest_hash(new),original_main_not_rerun=True,failed_verification_history_preserved=True))
 print(json.dumps(read(d/'accepted.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verification-config',required=True);a=p.parse_args();main(a.verification_config)
