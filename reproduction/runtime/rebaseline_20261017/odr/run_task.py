"""One parent-queue task: reviewed GPU gate, exact child logging, independent CPU verify."""
import argparse,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from common import *
from evidence import emit_event

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();t=job(a.task)
 if not a.execute:print(json.dumps(t,indent=2));return
 assert os.environ.get('STY'),'Execute via own run_in_screen_guarded.sh'
 verify_implementation();accepted=accepted_source(t['phi']);verify_code(t['line']);assert sha(t['config'])==t['config_sha256']
 acceptance=json.loads((O/'reports/integration_cpu_acceptance.json').read_text());assert acceptance['status']=='passed' and acceptance['implementation_manifest_sha256']==sha(O/'implementation_manifest.json')
 fixed_barrier(t)
 lock=(O/'gpu_serial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 report=O/'reports'/(t['id']+'.json');assert not report.exists(),'Accepted output exists; no automatic rerun'
 env=environ();env.update(PYTHONHASHSEED=str(t['seed']),SPNV2_PILOT_EVAL_LIMIT='0',CUDA_VISIBLE_DEVICES='0')
 verify_env=env.copy();verify_env['CUDA_VISIBLE_DEVICES']=''
 verify_command=[PYTHON,'-B','-s','-u',str(O/'verify_result.py'),'--task',t['id']]
 if not t.get('reuse_of'):
  ready,reason=gpu_available();assert ready,reason
  assert not Path(t['output']).exists(),'Preserve failed/previous output; review rather than overwrite'
  log=O/'logs'/(t['id']+'.log');record=O/'reports'/(t['id']+'.process.json');assert not log.exists() and not record.exists()
  emit_event('task_start',task=t['id'],domain=t['domain'],source=accepted['final'],config=t['config'],N=t['N'],B=t['B'],physical_batch=1,batch=0,batches=t['N'],samples_seen=0,update_attempts=0,optimizer_updates=0)
  with log.open('x') as stream:
   child=subprocess.Popen(t['command'],cwd=O,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
   starttime=Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
   atomic(record,dict(task=t['id'],pid=child.pid,starttime=starttime,screen=os.environ['STY'],command=t['command'],config_sha256=sha(t['config']),source_sha256=accepted['final_sha256'],implementation_manifest_sha256=sha(O/'implementation_manifest.json'),gpu_gate_spec=parent_gpu_spec(),started_at=time.time()))
   for line in child.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
   rc=child.wait()
  log.with_name(log.name+'.exitcode').write_text(str(rc)+'\n')
  assert rc==0,f'ODR/evaluation exited {rc}; adaptation retained, block remaining queue'
 verification_log=O/'logs'/(t['id']+'.verify.log');assert not verification_log.exists()
 verification=subprocess.run(verify_command,cwd=O,env=verify_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 verification_log.write_text(verification.stdout);print(verification.stdout,end='');verification_log.with_name(verification_log.name+'.exitcode').write_text(str(verification.returncode)+'\n')
 assert verification.returncode==0,'Independent CPU verification failed; block queue'
 proof=json.loads(report.read_text());assert proof['status']=='passed'
 emit_event('task_complete',task=t['id'],domain=t['domain'],source=accepted['final'],output=str(report),message='Adaptation/evaluation and independent CPU verification accepted',samples_seen=t['N'],batch=t['N'],batches=t['N'],update_attempts=t['N']//t['B'],optimizer_updates=t['N']//t['B'])
if __name__=='__main__':
 try:main()
 except BaseException as error:
  emit_event('error',error=repr(error),message='ODR task failed; preserve evidence and block parent queue');raise
