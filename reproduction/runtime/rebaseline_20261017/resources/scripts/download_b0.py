"""Install only the missing official B0 asset; no existing cache overwrites."""
import ast,datetime,hashlib,json,os,re,traceback,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R.parent;RUNS=P.parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,obj):
 tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
assert os.environ.get('STY'),'Use own unique guarded screen'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('TORCH_HOME')==str(RUNS/'env/cache/torch')
origin=P/'source/core/nets/backbones/efficientnet.py';tree=ast.parse(origin.read_text());urls=None
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model_urls' for t in node.targets):urls=ast.literal_eval(node.value)
assert urls is not None
url=urls['efficientnet_b0'];assert url=='https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'
filename=url.rsplit('/',1)[1];prefix=re.search(r'-([a-f0-9]{8})\.pth$',filename).group(1)
cache=RUNS/'env/cache/torch/hub/checkpoints';target=cache/filename
b3paths=[RUNS/'cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth',cache/'efficientnet_b3_rwightman-cf984f9c.pth']
before={str(p):sha(p) for p in b3paths};report=dict(status='started',url=url,filename=filename,expected_filename_sha256_prefix=prefix,origin=str(origin),origin_sha256=sha(origin),target=str(target),started_at=datetime.datetime.now().astimezone().isoformat(),B3_before_sha256=before)
try:
 if target.exists():
  assert sha(target).startswith(prefix),'Existing B0 hash does not match official filename prefix; preserve and stop'
  report['download_performed']=False;report['installation']='existing file preserved'
 else:
  artifact=R/'cache'/filename
  assert not artifact.exists(),'Keep prior artifact; do not silently retry or overwrite'
  downloaded=0
  with urllib.request.urlopen(url,timeout=60) as response,artifact.open('xb') as stream:
   assert response.geturl()==url,'Unexpected redirect; exact official URL only'
   while True:
    chunk=response.read(1024*1024)
    if not chunk:break
    stream.write(chunk);downloaded+=len(chunk);assert downloaded<100*1024*1024,'Unexpectedly large B0 payload'
   stream.flush();os.fsync(stream.fileno())
  digest=sha(artifact);assert digest.startswith(prefix),'Official filename SHA prefix mismatch; do not install'
  import torch
  state=torch.load(artifact,map_location='cpu',weights_only=True)
  assert isinstance(state,dict) and state and all(torch.is_tensor(v) for v in state.values())
  assert all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in state.values())
  cache.mkdir(parents=True,exist_ok=True)
  # Same-filesystem hardlink creates the final entry atomically and never overwrites.
  os.link(artifact,target);assert sha(target)==digest
  report.update(download_performed=True,downloaded_bytes=downloaded,artifact=str(artifact),installation='atomic exclusive hardlink; existing entries never replaced',tensor_count=len(state))
 report.update(status='passed',full_sha256=sha(target),size_bytes=target.stat().st_size,filename_sha_prefix_verified=sha(target).startswith(prefix),completed_at=datetime.datetime.now().astimezone().isoformat())
except BaseException as error:
 report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),limitation='B0 initialization unavailable; no model statistics fabricated and no initialization changed')
 raise
finally:
 after={str(p):sha(p) for p in b3paths};report['B3_after_sha256']=after;report['B3_preserved_byte_identical']=before==after
 atomic(R/'reports/b0_download.json',report);print(json.dumps(report,indent=2),flush=True)
 assert before==after,'B3 cache integrity changed'
