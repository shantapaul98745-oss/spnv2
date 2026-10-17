"""Screen-only engineering gate, retaining events and exact child identity."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--record',required=True);p.add_argument('--log',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
if not os.environ.get('STY'):raise RuntimeError('Use guarded screen')
plan=Path(__file__).resolve().parents[1];record=Path(a.record)
if record.exists():raise FileExistsError(record)
command=a.command[1:] if a.command[:1]==['--'] else a.command
def atomic(path,obj):
 tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
pid=child.pid;starttime=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
current=dict(task=a.phase,phase=a.phase,process_state='running; PID/starttime recorded',pid=pid,starttime=starttime,screen_id=os.environ['STY'],started_at=time.time(),log=a.log,source=str(plan/'source'),config=str(plan/'configs/phi3_bn_ehs.yaml'),command=command)
state=dict(phase=a.phase,pid=pid,starttime=starttime,command=command,screen=os.environ['STY'],log=a.log,started_at=time.time())
atomic(record,state);warnings=[]
def publish(status,error=None):
 stage=dict(status=status,active=a.phase,current=current,notes=['CPU final acceptance passed13/13; immutable source snapshot frozen.','Engineering GPU gate; formal source training has not started.'],error=error)
 if warnings:stage['notes'].extend(warnings)
 try:atomic(plan/'stage.json',stage)
 except OSError as e:
  warnings.append('Stage write warning: '+str(e));print(warnings[-1],file=sys.stderr,flush=True)
def forward(sig,frame):
 if child.poll() is None:
  actual=Path(f'/proc/{pid}/stat')
  if actual.exists() and actual.read_text().rsplit(') ',1)[1].split()[19]==starttime:child.send_signal(sig)
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,forward)
publish('running')
for line in child.stdout:
 print(line,end='',flush=True)
 if line.startswith('SPNV2_EVENT '):
  try:
   event=json.loads(line[len('SPNV2_EVENT '):]);current.update(event);current['phase']=a.phase
   current['last_event_at']=time.time()
   if event.get('event')=='checkpoint':current['checkpoint']=event.get('path')
   if event.get('event')=='resume':current['resume']=event.get('path')
   publish('running')
  except (ValueError,TypeError) as e:print('EVENT_PARSE_WARNING '+str(e),file=sys.stderr,flush=True)
rc=child.wait();state.update(exitcode=rc,ended_at=time.time(),progress_warnings=warnings)
atomic(record,state);current.update(exit_code=rc,process_state='exited; pending artifact verification')
publish('awaiting_verification' if rc==0 and not warnings else 'failed',None if rc==0 else f'Gate exited {rc}; see retained log')
sys.exit(rc)
