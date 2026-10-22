"""One authorized Table2 pilot manager cutover. Never rerun this script."""
import copy, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
P=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D=P/'revisions/manager_table2_pilots_cutover_20261022'
Q=P/'queue_proposals/table2_pilots_20261022'
OLD=P/'queue_observed_stage_20261018.json'
NEW=Q/'proposal.json'
OLD_SHA='a6f82b1efc62ebb6223e1cfec4919e44de35424b53709230f39b0655ea3e27a5'
NEW_SHA='976cf1bf4caf473d47b861e01b69a216399da0e492b6bcb9fe282134365e5adc'
IDS=['pilot_table2_h','pilot_table2_e','pilot_table2_eh']
sys.path.insert(0,str(P/'pipeline'))
import runner
from revision import extend_state
from progress import atomic_json, now
from task_entry import process_alive, process_identity
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
 return h.hexdigest()
def argv(pid):return Path('/proc',str(pid),'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
def owners():
 inode=(P/'pipeline.lock').stat().st_ino
 return [int(x.split()[4]) for x in Path('/proc/locks').read_text().splitlines() if len(x.split())>=6 and x.split()[1]=='FLOCK' and x.split()[5].split(':')[-1]==str(inode)]
def old_records_unchanged(before,after):
 assert len(after['tasks'])==52
 assert [t['id'] for t in after['tasks'][:49]]==before['completed']
 for a,b in zip(before['tasks'],after['tasks'][:49]):
  for k in ['status','runtime','history','error','accepted_at','started_at','event_error','progress_warning']:
   assert a.get(k)==b.get(k),(a['id'],k)
 assert all(t['status']=='accepted' for t in after['tasks'][:49])
 assert set(after['completed']).issuperset(before['completed'])
def main():
 assert os.environ.get('STY'),'Independent controller screen required'
 assert Path(__file__).resolve()==D/'activate.py'
 assert not (D/'intent.json').exists(),'One-time action already attempted'
 old,new=read(OLD),read(NEW)
 assert runner.manifest_hash(old)==OLD_SHA and runner.manifest_hash(new)==NEW_SHA
 assert sha(NEW)=='2f8eedf7f0f5121401c108b38a2dfa970af2d51da21893022e37f71a4bfec3e3'
 assert sha(Q/'verified.json')=='3ac714bcaaaa3461d1f5951035248f10daf3e0c79c66f76f3e3bd45427a5d0af'
 proof=read(Q/'verified.json')
 assert proof['status']=='passed' and proof['old_tasks_preserved']==49 and proof['new_queued_pilots']==3
 assert (Q/'build.log.exitcode').read_text().strip()==(Q/'verify.log.exitcode').read_text().strip()=='0'
 assert new['tasks'][:49]==old['tasks'] and len(new['tasks'])==52
 assert [t['id'] for t in new['tasks'][49:]]==IDS
 runner.validate_manifest(new)
 asset_count=set()
 for t in new['tasks'][49:]:
  assert runner.asset_hash_error(t) is None,t['id']
  asset_count.update(t['metadata']['asset_sha256'])
  absent=[t['log'],t['log']+'.exitcode',t['log']+'.verify',t['metadata']['report'],t['metadata']['artifacts']]+t['expected_outputs']
  assert all(not Path(x).exists() for x in absent),t['id']
 assert len(asset_count)==56
 before=read(P/'status.json');m=copy.deepcopy(before['manager'])
 assert before['manifest_sha256']==OLD_SHA and before['status']=='phase_complete' and before.get('active') is None
 assert len(before['completed'])==49 and all(t['status']=='accepted' for t in before['tasks'])
 assert not before.get('error') and not before.get('downstream_blocked')
 assert m['pid']==1851843 and str(m['starttime'])=='526573814' and process_alive(m)
 ma=argv(m['pid']);assert str(P/'pipeline/runner.py') in ma and ma[ma.index('--manifest')+1]==str(OLD)
 assert owners()==[m['pid']]
 extend_state(before,old,new,caller_context={'operation':'authorized_cutover_preflight_only'})
 manager_log=Q/'manager_activation.log'
 assert not manager_log.exists() and not Path(str(manager_log)+'.exitcode').exists()
 atomic_json(D/'before.json',before)
 atomic_json(D/'intent.json',{'at':now(),'user_authorization':'2026-10-22 explicit user permits this manager-only switch to accepted52 candidate and H/E/EH pilots; original49 must not rerun; goal remains paused','old_manager':m,'new_manifest':str(NEW),'new_manifest_sha256':NEW_SHA,'new_asset_count':56,'old_queue_event_offset':(P/'events/queue.events.jsonl').stat().st_size,'no_active_worker':True})
 assert process_alive(m) and argv(m['pid'])==ma and owners()==[m['pid']]
 os.kill(m['pid'],signal.SIGTERM)
 deadline=time.monotonic()+15
 while process_alive(m) and time.monotonic()<deadline:time.sleep(.2)
 assert not process_alive(m),'Old manager still alive; preserve state, no duplicate launch'
 with (P/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stopped=read(P/'status.json')
  assert stopped['status']=='monitor_stopped' and stopped['manager']['pid']==m['pid'] and stopped.get('active') is None
  assert stopped['tasks']==before['tasks'] and stopped['completed']==before['completed']
  extend_state(stopped,old,new,caller_context={'operation':'authorized_after_stop_preflight_only'})
  atomic_json(D/'stopped.json',stopped)
 env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
 for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','STY','WINDOW']:env.pop(k,None)
 command=['bash',str(runner.WRAPPER),'rebaseline_queue_table2_pilots_20261022',str(manager_log),runner.PYTHON,'-B','-s','-u',str(P/'pipeline/runner.py'),'--manifest',str(NEW),'--extend-from',str(OLD)]
 launch=subprocess.run(command,cwd=str(P),env=env,capture_output=True,text=True,timeout=20)
 atomic_json(D/'launch.json',{'command':command,'returncode':launch.returncode,'stdout':launch.stdout,'stderr':launch.stderr})
 assert launch.returncode==0,launch.stderr+launch.stdout
 deadline=time.monotonic()+20
 while True:
  after=read(P/'status.json')
  if after.get('manifest_sha256')==NEW_SHA and process_alive(after.get('manager')) and after.get('active')=='pilot_table2_h' and process_alive(after.get('current')):
   break
  assert not after.get('error'),after.get('error')
  assert time.monotonic()<deadline,'No verified startup acknowledgement; preserve state, do not rerun controller'
  time.sleep(.2)
 old_records_unchanged(before,after)
 assert after['manager']['pid']!=m['pid'] and owners()==[after['manager']['pid']]
 assert not process_alive(m)
 assert after['tasks'][49]['status']=='running'
 assert all(t['status']=='queued' and not t.get('runtime') for t in after['tasks'][50:])
 worker=after['current'];record=read(after['tasks'][49]['runtime']['record'])
 assert process_alive(record) and argv(record['pid'])==after['tasks'][49]['runtime']['command']
 audit=read(after['manifest_revisions'][-1])
 assert audit['old_manifest_sha256']==OLD_SHA and audit['new_manifest_sha256']==NEW_SHA
 atomic_json(D/'after.json',after)
 atomic_json(D/'accepted.json',{'status':'passed','at':now(),'operation':'explicitly_authorized_manager_only_table2_pilots_append','old_manager':m,'new_manager':after['manager'],'worker':{k:record[k] for k in ['pid','starttime','screen_id','command']},'old49_definitions_preserved':True,'old49_execution_evidence_preserved':True,'old49_restarted':False,'task_count':52,'manifest':str(NEW),'manifest_sha256':NEW_SHA,'revision_audit':after['manifest_revisions'][-1],'active':'pilot_table2_h','remaining_pilots_queued':IDS[1:],'full_plan_completed':False,'normal_recovery':'Use only --manifest '+str(NEW)+'; never reuse this one-time --extend-from or this controller.'})
 print(json.dumps(read(D/'accepted.json')))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic_json(D/'failed.json',{'status':'failed','at':now(),'error':repr(e),'do_not_rerun':'Preserve evidence and inspect exact identities; never repeat the one-time controller blindly.'})
  raise

