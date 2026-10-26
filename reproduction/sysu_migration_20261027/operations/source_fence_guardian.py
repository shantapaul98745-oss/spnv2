"""Own two inherited diagnostic locks until an explicit safe rollback release."""
from pathlib import Path
import os,sys,json,hashlib,time,fcntl
M=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/migrations/sysu_20261027');D=M.parent.parent/'plans/g50p_risk_diagnostic_20261025';fds=[int(x) for x in sys.argv[1:]];assert len(fds)==2
for fd,p in zip(fds,[D/'pipeline.lock',D/'manager.lock']):
 s=os.fstat(fd);t=p.stat();assert (s.st_dev,s.st_ino)==(t.st_dev,t.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=Path('/proc')/str(os.getpid());s=(p/'stat').read_text();nonce=os.environ['SPNV2_SOURCE_FENCE_NONCE'];v=dict(schema='spnv2-source-fence-guardian-v1',nonce=nonce,pid=os.getpid(),starttime=int(s[s.rfind(')')+2:].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cmdline_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),holds=['diagnostic_pipeline','diagnostic_manager'],model_forwards=0,status='holding_source_dispatch_fence')
def save():
 t=M/'source_fence_guardian.tmp';t.write_text(json.dumps(v,indent=2)+'\n');t.replace(M/'source_fence_guardian.json')
save()
while True:
 p=M/'release_source_fence.json'
 if p.exists():
  r=json.loads(p.read_text())
  if r.get('fence_nonce')==nonce and r.get('target_dispatch_disabled') is True and r.get('ledger_reconciled') is True and r.get('explicit_rollback_authorized') is True:
   v.update(status='released_after_explicit_reconciled_rollback',released_at=time.time());save();break
 time.sleep(300)
