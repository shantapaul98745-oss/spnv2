from pathlib import Path
import json,time,os,fcntl,hashlib
M=Path('/home/ps/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/migrations/sysu_20261027');pub=b'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBDR8ER1taY8YULZ+CREswqYmpKmy1nNsa2TaV4Ir2gL spnv2-sysu-migration-20261027'
lock=(M/'transfer_key_cleanup.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 p=M/'asset_receive_status.json'
 if p.exists():
  s=json.loads(p.read_text())
  if s['phase']=='complete':break
  if s['phase']=='failed':
   (M/'transfer_key_cleanup_status.json').write_text(json.dumps(dict(phase='waiting_for_transfer_recovery',updated_at=time.time(),restricted_key_retained_for_retry=True),indent=2))
 time.sleep(60)
auth=Path('/home/ps/.ssh/authorized_keys')
with auth.open('r+b') as f:
 fcntl.flock(f,fcntl.LOCK_EX);old=f.read();lines=old.splitlines(keepends=True);drop=[x for x in lines if x.rstrip(b'\r\n').endswith(pub) and x.startswith(b'restrict,command=')]
 if len(drop)>1:raise RuntimeError('Ambiguous temporary key entries; do not remove other keys')
 new=b''.join(x for x in lines if x not in drop);f.seek(0);f.write(new);f.truncate();f.flush();os.fsync(f.fileno())
(M/'transfer_key_cleanup_status.json').write_text(json.dumps(dict(phase='completed',updated_at=time.time(),restricted_entries_removed=len(drop),other_authorized_keys_preserved=True,verified_files=s['verified_files'],expected_files=s['expected_files']),indent=2)+'\n')
