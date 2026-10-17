"""Official hash migration verification; exclusive missing-B0 installation only."""
from pathlib import Path
import datetime,hashlib,html,json,os,re,traceback,urllib.request
R=Path(__file__).resolve().parents[1];P=R.parent;RUNS=P.parents[1]
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('TORCH_HOME')==str(RUNS/'env/cache/torch')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,obj):
 tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
FULL='7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934'
URL='https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth'
LEGACY='efficientnet_b0_rwightman-3dd342df.pth'
old=R/'cache'/LEGACY;new=R/'cache'/URL.rsplit('/',1)[1];target=RUNS/'env/cache/torch/hub/checkpoints'/LEGACY
b3=[RUNS/'cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth',RUNS/'env/cache/torch/hub/checkpoints/efficientnet_b3_rwightman-cf984f9c.pth']
before={str(p):sha(p) for p in b3};old_reports={f.name:sha(f) for f in sorted((R/'reports').glob('*.json'))}
report=dict(status='started',started_at=datetime.datetime.now().astimezone().isoformat(),new_official_url=URL,expected_official_prefix='7f5810bc',expected_full_sha256=FULL,legacy_loader_url='https://download.pytorch.org/models/'+LEGACY,legacy_loader_filename=LEGACY,legacy_target=str(target),B3_before_sha256=before,historical_reports_before_sha256=old_reports)
try:
 assert not target.exists(),'Target already exists; never replace cache entry'
 assert old.is_file() and sha(old)==FULL,'Historical quarantine payload differs'
 evidence=R/'evidence/b0_identity_20261017';evidence.mkdir(parents=True,exist_ok=False)
 urls={'torchvision_020_efficientnet':'https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/efficientnet.html','official_pr_7898':'https://github.com/pytorch/vision/pull/7898','official_issue_7744':'https://github.com/pytorch/vision/issues/7744'}
 evidence_report={};texts={}
 for name,url in urls.items():
  request=urllib.request.Request(url,headers={'User-Agent':'SPNv2-provenance/1.0'})
  with urllib.request.urlopen(request,timeout=60) as response:
   raw=response.read();assert response.geturl()==url,'Unexpected evidence redirect'
  file=evidence/(name+'.html');file.write_bytes(raw);text=html.unescape(re.sub('<[^>]+>','',raw.decode('utf-8')));texts[name]=text
  evidence_report[name]=dict(url=url,path=str(file),sha256=sha(file))
 doc=texts['torchvision_020_efficientnet'];b0=doc[doc.index('class EfficientNet_B0_Weights'):doc.index('class EfficientNet_B1_Weights')];b3doc=doc[doc.index('class EfficientNet_B3_Weights'):doc.index('class EfficientNet_B4_Weights')]
 assert 'IMAGENET1K_V1' in b0 and URL in b0 and 'efficientnet_b3_rwightman-b3899882.pth' in b3doc
 pr=texts['official_pr_7898'];issue=texts['official_issue_7744']
 assert 'some hash were wrong' in pr and 're-uploaded the proper weights' in pr and '7898' in pr
 assert FULL in issue and '3dd342df' in issue and 'IMAGENET1K_V1' in issue
 report['official_evidence']=evidence_report
 report['migration_reason']='Official torchvision PR7898 corrects bad hash URLs after hash checking became enforced; issue7744 records this exact B0 full SHA and identifies IMAGENET1K_V1. Torchvision0.20 declares the corrected B0 IMAGENET1K_V1 URL.'
 assert not new.exists(),'Use new payload file; do not overwrite prior artifact'
 received=0
 with urllib.request.urlopen(URL,timeout=60) as response,new.open('xb') as stream:
  assert response.geturl()==URL
  while True:
   chunk=response.read(1024*1024)
   if not chunk:break
   stream.write(chunk);received+=len(chunk);assert received<100*1024*1024
  stream.flush();os.fsync(stream.fileno())
 digest=sha(new);report.update(new_payload=str(new),received_bytes=received,new_full_sha256=digest,old_full_sha256=sha(old))
 assert digest.startswith('7f5810bc') and digest==FULL==sha(old),'Official prefix/full SHA or quarantine identity mismatch'
 import torch
 a=torch.load(old,map_location='cpu',weights_only=True);b=torch.load(new,map_location='cpu',weights_only=True)
 assert a.keys()==b.keys() and a
 comparisons=[]
 for name,value in a.items():
  other=b[name];assert torch.is_tensor(value) and torch.is_tensor(other) and value.dtype==other.dtype and value.shape==other.shape and torch.equal(value,other),name
  assert not value.is_floating_point() or bool(torch.isfinite(value).all())
  comparisons.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype),numel=value.numel(),exactly_equal=True))
 report.update(identity='same official EfficientNet_B0_Weights.IMAGENET1K_V1',all_tensors_exactly_equal=True,tensor_count=len(comparisons),tensor_comparisons=comparisons,byte_identical=True)
 assert not target.exists()
 # Exclusive atomic hardlink; frozen loader sees its required legacy filename.
 os.link(new,target);assert sha(target)==FULL
 report.update(status='passed',installed=True,installation='exclusive atomic new hardlink under frozen loader legacy filename; official corrected payload, no initialization substitution',installed_full_sha256=sha(target),completed_at=datetime.datetime.now().astimezone().isoformat())
except BaseException as error:
 report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),installed=target.exists(),limitation='Identity not accepted; retain prior failures and do not change initialization')
 raise
finally:
 after={str(p):sha(p) for p in b3};report['B3_after_sha256']=after;report['B3_preserved_byte_identical']=before==after
 report['historical_reports_preserved']=all(sha(R/'reports'/name)==value for name,value in old_reports.items())
 atomic(R/'reports/b0_official_v1_success_20261017.json',report)
 print(json.dumps({k:v for k,v in report.items() if k!='tensor_comparisons'},indent=2),flush=True)
 assert before==after and report['historical_reports_preserved']
