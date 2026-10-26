"""Bounded checkpoint retention; no model imports or scientific state changes.
Call only under the store writer lock. Policies and deletion receipts are durable.
"""
from pathlib import Path
import hashlib,json,os,re,time
GEN=re.compile(r'generation-\d{20}$')
class StoragePolicyError(RuntimeError):pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');data=(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
 with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def validate(v):
 if v.get('schema')!='spnv2-storage-policy-v1' or v.get('retain_latest')!=2:raise StoragePolicyError('Explicit latest-two policy required')
 if not isinstance(v.get('estimated_peak_bytes'),int) or v['estimated_peak_bytes']<=0:raise StoragePolicyError('Positive pre-run disk estimate required')
 if not isinstance(v.get('protected_generations'),dict):raise StoragePolicyError('Explicit protected-generation registry required')
 for name,pin in v['protected_generations'].items():
  if not GEN.fullmatch(name) or not isinstance(pin,dict) or len(pin.get('state_sha256',''))!=64 or not pin.get('reason'):raise StoragePolicyError('Invalid protected generation')
 return v
def initialize(root,policy):
 root=Path(root);p=root/'storage_policy.json'
 if p.exists():
  existing=validate(json.loads(p.read_text()))
  if policy is not None and existing!=validate(policy):raise StoragePolicyError('Storage policy changed without a versioned amendment')
  return existing
 if policy is None:raise StoragePolicyError('A retention policy and disk estimate must be supplied before a new experiment')
 policy=validate(policy);atomic(p,policy);return policy
def policy(root):return validate(json.loads((Path(root)/'storage_policy.json').read_text()))
def usage(root):
 seen=set();logical=allocated=0
 for p in Path(root).rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  s=p.stat();logical+=s.st_size
  if (s.st_dev,s.st_ino) not in seen:allocated+=s.st_blocks*512;seen.add((s.st_dev,s.st_ino))
 return dict(logical_bytes=logical,allocated_bytes=allocated)
def event(root,v):
 p=Path(root)/'retention_journal.jsonl'
 with p.open('a') as f:f.write(json.dumps(dict(time_ns=time.time_ns(),**v),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def before_write(root,state_bytes):
 v=policy(root);u=usage(root)
 if u['logical_bytes']+state_bytes>v['estimated_peak_bytes']:
  event(root,dict(event='disk_estimate_exceeded',usage=u,new_state_bytes=state_bytes,estimated_peak_bytes=v['estimated_peak_bytes']))
  raise StoragePolicyError('Disk estimate exceeded; diagnose retention/references before continuing, do not auto-expand')
def pin(store,generation,reason):
 """Freeze/verification preparation must pin before publishing its reference."""
 with store._lock():
  v=policy(store.root);g=store.generations/f'generation-{generation:020d}'
  _,info=store._read_generation(generation,g,None,deserialize=False)
  v['protected_generations'][g.name]=dict(state_sha256=info.metadata['state_sha256'],reason=reason)
  validate(v);atomic(store.root/'storage_policy.json',v);event(store.root,dict(event='generation_pinned',generation=g.name,reason=reason))
def after_publish(store):
 v=policy(store.root);paths=store._paths();latest=paths[:2];keep={p.name for _,p in latest}|set(v['protected_generations'])
 # Both ordinary recovery points must be complete and checksum-valid before pruning.
 for n,p in latest:store._read_generation(n,p,None,deserialize=False)
 for name,pin in v['protected_generations'].items():
  p=store.generations/name
  if not p.is_dir():raise StoragePolicyError('Protected generation missing; no cleanup permitted')
  meta=json.loads((p/'metadata.json').read_text())
  if meta['state_sha256']!=pin['state_sha256']:raise StoragePolicyError('Protected identity changed; no cleanup permitted')
 deleted=[]
 for n,p in paths:
  if p.name in keep:continue
  if p.is_symlink() or p.resolve()!=p.absolute() or set(x.name for x in p.iterdir())!={'state.bin','metadata.json','metadata.sha256'}:raise StoragePolicyError('Unexpected checkpoint contents; retained for inspection')
  raw=(p/'metadata.json').read_bytes();meta=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=(p/'metadata.sha256').read_text().strip():raise StoragePolicyError('Unverified old metadata; retain failure evidence')
  fs=[p/name for name in ['state.bin','metadata.json','metadata.sha256']]
  if any(x.is_symlink() for x in fs) or (p/'state.bin').stat().st_size!=meta['state_bytes']:raise StoragePolicyError('Unverified old state; retain failure evidence')
  receipt=dict(event='delete_intent',path=str(p),generation=n,metadata=meta,files=[dict(path=str(x),size=x.stat().st_size,inode=x.stat().st_ino,nlink=x.stat().st_nlink) for x in fs],reason='superseded, unpinned ordinary recovery point; two newer complete versions verified')
  event(store.root,receipt)
  for x in fs:x.unlink()
  p.rmdir();deleted.append(n);event(store.root,dict(event='delete_completed',generation=n,path=str(p)))
 fd=os.open(store.generations,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 atomic(store.root/'storage_usage.json',dict(time_ns=time.time_ns(),**usage(store.root),estimated_peak_bytes=v['estimated_peak_bytes'],retained_generations=[n for n,_ in store._paths()],protected_generations=sorted(v['protected_generations']),deleted_generations=deleted))
 return deleted
