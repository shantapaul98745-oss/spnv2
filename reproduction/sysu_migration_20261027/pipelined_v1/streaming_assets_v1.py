"""Receipt-only streaming publication/readiness; never deserialize scientific state."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path, PurePosixPath

SCHEMA='spnv2-streaming-window-dependencies-v1'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def signature(path):
 s=Path(path).stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
def require(ok,message):
 if not ok: raise ValueError(message)
def relative(value):
 p=PurePosixPath(value)
 require(not p.is_absolute() and '..' not in p.parts and len(p.parts)>2 and p.parts[0]=='spnv2_runs','Invalid owned asset path')
 return str(p)
def bound(item):
 require(set(item)=={'path','sha256'} and digest(item['path'])==item['sha256'],'Streaming metadata binding changed')
 return read(item['path'])
def load_plan(index):
 spec=index.get('streaming_migration')
 require(isinstance(spec,dict) and set(spec)=={'code','dependencies'},'Missing explicit streaming migration bindings')
 require(digest(spec['code']['path'])==spec['code']['sha256']==digest(__file__),'Streaming code binding differs')
 plan=bound(spec['dependencies'])
 require(plan['schema']==SCHEMA and plan['frozen_order']==[r['run_id'] for r in index['tasks']], 'Streaming dependencies changed frozen order')
 manifest=bound(plan['asset_manifest']);rows=manifest['files']; lookup={relative(r['relative_path']):r for r in rows}
 require(len(lookup)==len(rows),'Duplicate asset paths')
 require(set(plan['windows'])==set(plan['frozen_order']),'Window dependency scope differs from frozen48')
 for paths in [plan['common_relative_paths'],plan['summary_relative_paths'],*(v[a] for v in plan['windows'].values() for a in ('run_pair','verify_pair'))]:
  require(isinstance(paths,list) and len(paths)==len(set(paths)) and set(paths)<=set(lookup),'Dependency escapes bound asset manifest')
 return plan,lookup

def checked_receipt(migration, row, *, canonical_root=None, staging_root=None, missing_ok=True):
 rel=relative(row['relative_path']); receipt_path=Path(migration)/'asset_receipts'/(hashlib.sha256(rel.encode()).hexdigest()+'.json')
 if not receipt_path.exists():
  if missing_ok:return None
  raise ValueError('Missing verified asset receipt: '+rel)
 receipt=read(receipt_path)
 require(receipt.get('status')=='verified' and receipt.get('relative_path')==rel and receipt.get('sha256')==row['sha256'] and receipt.get('bytes')==row['bytes'], 'Verified asset receipt binding differs: '+rel)
 path=Path(canonical_root if canonical_root is not None else staging_root)/rel
 if not path.exists():
  if missing_ok:return None
  raise ValueError('Receipt exists but required asset missing: '+rel)
 require(path.is_file() and not path.is_symlink() and path.absolute()==path.resolve() and signature(path)==receipt['file_stat'], 'Verified asset actual target stat differs: '+rel)
 return receipt

def required_paths(plan, task):
 action=task['action'];rid=task['run_id']
 if action=='summarize_diagnostic':paths=plan['summary_relative_paths']
 elif action=='verify_pilots':paths=[]
 elif action in ('run_pair','verify_pair'):
  if action=='verify_pair':rid=bound(task['run_task'])['run_id']
  require(rid in plan['windows'],'Task is outside frozen dependency list')
  paths=plan['windows'][rid][action]
 else:raise ValueError('Unsupported streaming action')
 return sorted(set(plan['common_relative_paths'])|set(paths))

def readiness(index, task):
 plan,lookup=load_plan(index);root=Path(plan['logical_target_root']);migration=root/'spnv2_runs/migrations/sysu_20261027'
 missing=[]
 for rel in required_paths(plan,task):
  if checked_receipt(migration,lookup[rel],canonical_root=root) is None:missing.append(rel)
 return dict(ready=not missing,missing=missing,action=task['action'],run_id=task['run_id'])

def publish_verified(manifest, migration, staging, target, *, only=None):
 """Host-side only; publish verified immutable inodes without moving receiver bases."""
 allowed=None if only is None else set(only);published=[];missing=[]
 for row in manifest['files']:
  rel=relative(row['relative_path'])
  if allowed is not None and rel not in allowed:continue
  proof=checked_receipt(migration,row,staging_root=staging)
  if proof is None:missing.append(rel);continue
  source=Path(staging)/rel;dest=Path(target)/rel
  require(Path(target).absolute()==Path(target).resolve() and dest.absolute()==dest.resolve(),'Canonical destination alias is forbidden: '+rel)
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   require(dest.is_file() and not dest.is_symlink() and os.path.samefile(source,dest) and signature(dest)==proof['file_stat'],'Refusing different canonical asset: '+rel)
  else:
   os.link(source,dest)
   fd=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
   require(signature(dest)==proof['file_stat'],'Published hardlink identity differs: '+rel)
  published.append(rel)
 return dict(published=published,missing=missing,no_content_changes=True,staging_paths_preserved=True)
