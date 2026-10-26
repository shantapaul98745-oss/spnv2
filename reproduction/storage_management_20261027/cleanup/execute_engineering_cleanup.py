from pathlib import Path
from contextlib import ExitStack
import json,hashlib,os,fcntl,time,datetime,collections,shutil
J=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai');P=J/'spnv2_runs/plans';A=J/'spnv2_runs/storage_audits/retention_20261027';N=P/'odr003_round2_20261023'
roots=[N,P/'g50p_risk_diagnostic_20261025'];v=json.loads((A/'inventory_engineering_reviewed.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(name,value):
 p=A/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)
def allocated():
 seen=set();n=0
 for root in roots:
  for base,dirs,files in os.walk(root,followlinks=False):
   for name in files:
    p=Path(base)/name;s=p.lstat();key=(s.st_dev,s.st_ino)
    if key not in seen:seen.add(key);n+=s.st_blocks*512
 return n
start=time.monotonic();result={'schema':'spnv2-storage-cleanup-v1','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inventory_sha256':digest(A/'inventory_engineering_reviewed.json'),'deleted_generations':0,'deleted_logical_bytes':0,'released_file_blocks_bytes':0,'kept_bad_or_changed_stores':[],'models_imported':False,'model_forwards':0,'historical_logs_proofs_budget_unchanged':True}
groups=collections.defaultdict(list)
for e in v['generation_entries']:
 if e['classification']=='deletable_after_safe_boundary_validation':groups[e['store']].append(e)
assert not v['reference_read_errors']
with ExitStack() as locks:
 for p in [P/'rebaseline_20261017/pipeline.lock',P/'odr003_exploratory_20261023/pipeline.lock',N/'pipeline.lock',P/'g50p_risk_diagnostic_20261025/pipeline.lock']:
  h=locks.enter_context(p.open('a+b'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
 result['allocated_before_bytes']=allocated();atomic('execution_status_engineering.json',dict(result,phase='validating_and_pruning'))
 journal=(A/'deletion_journal_engineering.jsonl').open('x')
 def record(item):journal.write(json.dumps(item)+'\n');journal.flush();os.fsync(journal.fileno())
 for store,items in sorted(groups.items()):
  store=Path(store);assert store.is_relative_to(N/'runs') and store.resolve()==store
  with (store/'.writer.lock').open('a+b') as writer:
   try:
    fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gens=sorted(p for p in (store/'checkpoints').glob('generation-*') if p.is_dir());keepers=gens[-2:]
    assert len(keepers)==2
    for g in keepers:
     raw=(g/'metadata.json').read_bytes();meta=json.loads(raw)
     assert hashlib.sha256(raw).hexdigest()==(g/'metadata.sha256').read_text().strip()
     assert (g/'state.bin').stat().st_size==meta['state_bytes'] and digest(g/'state.bin')==meta['state_sha256']
    # No unexpected published generations may have appeared since dependency review.
    expected={e['path'] for e in v['generation_entries'] if e['store']==str(store)}
    assert {str(g) for g in gens}==expected
    for e in items:
     g=Path(e['path']);assert g.parent==store/'checkpoints' and g not in keepers and g.resolve()==g
     assert {p.name for p in g.iterdir()}=={'state.bin','metadata.json','metadata.sha256'}
     for old in e['files']:
      p=g/old['name'];s=p.lstat();assert not p.is_symlink() and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(old['device'],old['inode'],old['size'],old['mtime_ns'])
     assert json.loads((g/'metadata.json').read_text())==e['existing_metadata']
   except Exception as ex:
    result['kept_bad_or_changed_stores'].append({'store':str(store),'reason':str(ex),'type':type(ex).__name__});continue
   record({'event':'retention_boundary_verified','store':str(store),'kept_latest_two':[str(g) for g in keepers]})
   for e in items:
    g=Path(e['path']);record({'event':'delete_intent','path':str(g),'bytes':e['bytes'],'purpose':'superseded intermediate checkpoint in independently accepted complete run','basis':e['reasons'],'existing_metadata':e['existing_metadata'],'files':e['files']})
    freed=0
    for old in e['files']:
     p=g/old['name'];s=p.lstat()
     if s.st_nlink==1:freed+=s.st_blocks*512
     p.unlink()
    g.rmdir();result['deleted_generations']+=1;result['deleted_logical_bytes']+=e['bytes'];result['released_file_blocks_bytes']+=freed
    record({'event':'delete_completed','path':str(g),'released_file_blocks_bytes':freed})
   fd=os.open(store/'checkpoints',os.O_DIRECTORY);os.fsync(fd);os.close(fd)
   atomic('execution_status_engineering.json',dict(result,phase='validating_and_pruning',last_completed_store=str(store),elapsed_seconds=time.monotonic()-start))
   print('PRUNED',store.name,store.parent.name,result['deleted_generations'],result['released_file_blocks_bytes'],flush=True)
 journal.close();result['allocated_after_bytes']=allocated()
result['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();result['elapsed_seconds']=time.monotonic()-start;result['phase']='completed';atomic('cleanup_result_engineering.json',result);atomic('execution_status_engineering.json',result);print(json.dumps(result),flush=True)
