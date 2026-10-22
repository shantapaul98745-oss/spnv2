"""Independent readback of actual full-state GPU test artifacts, CPU only."""
import os,json,hashlib
from pathlib import Path
import torch,numpy as np
import test_real_model as test
test.torch=torch;test.np=np
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 torch.set_num_threads(2)
 path=N/'tests/real_correctness_20261023_b/report.json';r=json.loads(path.read_text())
 assert r['status']=='passed' and len(r['checks'])==82 and all(x['equal'] for x in r['checks'])
 assert (N/'logs/real_correctness_b.log.exitcode').read_text().strip()=='0'
 def load(key):
  a=r['artifacts'][key];assert sha(a['path'])==a['sha256']
  return test.safe_unpack(torch.load(a['path'],map_location='cpu',weights_only=True))
 def check(a,b):
  result=test.compare_tree(a,b);assert result['equal'],result
 a=load('original_A_reference');b=load('all_accepted')
 check(a['records'],[{k:v for k,v in x.items() if k!='accepted'} for x in b['records']])
 check(a['losses'],[{k:v for k,v in x.items() if k!='index'} for x in b['losses']])
 check(a['state'],b['state']);del a,b
 a=load('mixed_continuous');b=load('mixed_resumed');check(a,b);del a,b
 assert r['total_backbone_forward_counts']=={'adaptation':52,'isolated_score':64}
 assert not torch.cuda.is_initialized()
 out=dict(status='passed',strict_equality_required=True,source_sha256=r['source_sha256'],report_path=str(path),report_sha256=sha(path),
  all_accept_actual_artifacts_equal=True,mixed_resume_actual_artifacts_equal=True,checks=82,
  original_failure_preserved=str(N/'tests/real_correctness_20261023_a/report.json'),
  pure_operator_warmup=r['operator_warmup'],verifier_sha256=sha(__file__),scientific_accuracy_computed=False)
 with (N/'preflight/real_correctness_verified.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
