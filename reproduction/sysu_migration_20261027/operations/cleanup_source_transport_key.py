from pathlib import Path
import time,json,os,fcntl
M=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/migrations/sysu_20261027');l=(M/'source_transport_key_cleanup.lock').open('a+b');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
 ep=M/'asset_send_v2.exitcode';sp=M/'asset_send_v2_status.json'
 if ep.exists() and sp.exists():
  s=json.loads(sp.read_text())
  if ep.read_text().strip()=='0' and s['phase']=='complete' and s['verified_files']==s['expected_files']:break
 time.sleep(300)
k=M/'transfer_keys/migration_ed25519';removed=False
if k.exists():
 assert not k.is_symlink() and k.resolve()==k and k.stat().st_uid==os.getuid() and k.stat().st_mode&0o077==0
 k.unlink();removed=True
(M/'source_transport_key_cleanup.json').write_text(json.dumps(dict(status='completed',at=time.time(),private_transport_key_removed=removed,verified_files=s['verified_files'],model_forwards=0),indent=2)+'\n')
