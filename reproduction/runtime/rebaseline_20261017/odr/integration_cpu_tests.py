"""CPU-only mocked regression suite; no real ODR/source weights/images/GPU execution."""
import argparse,copy,csv,contextlib,io,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from common import *
from evidence import *
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
import numpy as np,torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();D=O/'tests'/a.run_id;D.mkdir(parents=True,exist_ok=False);results=[]
def check(name,fn):
 start=time.time();details=fn();results.append(dict(name=name,status='passed',seconds=time.time()-start,details=details));print('PASS',name,json.dumps(details),flush=True)
def rejected(fn):
 try:fn()
 except (AssertionError,ValueError,KeyError,TypeError,RuntimeError,FileNotFoundError):return
 raise AssertionError('Corrupt evidence unexpectedly accepted')

def adapter_and_manifest():
 env=environ();env['CUDA_VISIBLE_DEVICES']=''
 run=subprocess.run([PYTHON,'-B','-s','-u',str(O/'cpu_tests.py')],cwd=O,env=env,capture_output=True,text=True)
 (D/'actual_adapter_math.log').write_text(run.stdout+run.stderr);(D/'actual_adapter_math.log.exitcode').write_text(str(run.returncode)+'\n');print(run.stdout,end='');print(run.stderr,end='')
 assert run.returncode==0
 proof=json.loads((O/'reports/math_cpu_acceptance_v2.json').read_text());assert proof['status']=='passed' and proof['passed']==15
 m=json.loads((O/'manifest.json').read_text());assert len(m['tasks'])==400 and m['unique_executions']==370
 fixed=[t['id'] for t in m['tasks'] if t['group']=='fixed'];assert len(fixed)==40
 for t in m['tasks']:
  assert t['queue_command'][-1]=='--execute' and 'run_task.py' in t['queue_command'][-4]
  assert t['verify_command'][-3].endswith('verify_result.py')
  assert t['depends_on_fixed']==(fixed if t['group']!='fixed' else [])
 return dict(actual_adapter_tests=15,requested=400,unique=370,fixed_barrier=40,source_snapshot_and_config_hashes_verified=True)

def desktop_gate():
 runner=parent_runner();spec=parent_gpu_spec();assert spec=={'index':0,'min_free_mb':14500,'max_used_mb':1200}
 def probe(processes,used=800,free=15584):
  def fake(cmd,**kwargs):return NS(stdout=f'{used}, {free}\n' if '--query-gpu=memory.used,memory.free' in cmd else processes,returncode=0)
  with patch.object(runner.subprocess,'run',side_effect=fake):return gpu_available()
 assert probe('7890, /usr/libexec/gnome-remote-desktop-daemon, 128\n')[0]
 for content in ['7890, /usr/libexec/gnome-remote-desktop-daemon, 257\n','7890, /usr/libexec/gnome-remote-desktop-daemon, N/A\n','7890, /usr/bin/python, 128\n','999, /usr/bin/python, 1\n']:
  assert probe(content)[0] is False
 assert probe('',used=1201,free=15183)[0] is False and probe('',used=1000,free=14000)[0] is False
 assert environ()['TORCH_HOME']==str(RUNS/'env/cache/torch')
 return dict(actual_parent_function_used=True,bounded_desktop_daemon_128MB_accepted=True,over_256MB_unknown_process_and_memory_failures_rejected=True,no_real_gpu_query=True)

def fixed40_gate():
 import common
 root=D/'mock_fixed_barrier';(root/'reports').mkdir(parents=True)
 tasks=[dict(id='mock_fixed_'+str(i),group='fixed',phi=3 if i<30 else 6) for i in range(40)]
 (root/'manifest.json').write_text(json.dumps(dict(tasks=tasks)));(root/'implementation_manifest.json').write_text('{"mock_fixture":true}\n')
 digest=sha(root/'implementation_manifest.json')
 def proof(t):return dict(status='passed',task_id=t['id'],source_sha256='mock_source_'+str(t['phi']),protocol=PROTOCOL,implementation_manifest_sha256=digest,mocked_CPU_fixture=True)
 for t in tasks[:-1]:(root/'reports'/(t['id']+'.json')).write_text(json.dumps(proof(t)))
 with patch.object(common,'O',root),patch.object(common,'accepted_source',side_effect=lambda phi:dict(final_sha256='mock_source_'+str(phi))):
  rejected(lambda:common.fixed_barrier(dict(group='Nscan')))
  path=root/'reports'/(tasks[-1]['id']+'.json');path.write_text(json.dumps(proof(tasks[-1])));common.fixed_barrier(dict(group='Bscan'))
  bad=proof(tasks[-1]);bad['source_sha256']='wrong';path.write_text(json.dumps(bad));rejected(lambda:common.fixed_barrier(dict(group='Bscan')))
 return dict(missing_40th_fixed_rejected=True,all_40_same_source_protocol_implementation_required=True,mocked_files_only=True)

def data_corruption():
 root=D/'input';root.mkdir();images=root/'images';images.mkdir();csvpath=root/'test.csv';camera=root/'camera.json';points=root/'points.mat'
 camera.write_bytes(b'camera');points.write_bytes(b'geometry');csvpath.write_text('a.jpg,1\nb.jpg,2\n')
 for name in ['a.jpg','b.jpg']:(images/name).write_bytes(b'mockimage')
 def collect():return input_evidence(csvpath,images,camera,points,'mock')
 base=collect();assert verify_input_evidence(base);anchor_input_evidence(base,csvpath,images,camera,points,'mock')
 bad=copy.deepcopy(base);bad['image_root']=str(root/'other_images');rejected(lambda:anchor_input_evidence(bad,csvpath,images,camera,points,'mock'))
 csvpath.write_text('a.jpg,9\nb.jpg,2\n');rejected(lambda:verify_input_evidence(base));csvpath.write_text('a.jpg,1\nb.jpg,2\n')
 camera.write_bytes(b'corrupt');rejected(lambda:verify_input_evidence(base));camera.write_bytes(b'camera')
 points.write_bytes(b'corrupt');rejected(lambda:verify_input_evidence(base));points.write_bytes(b'geometry')
 image=images/'a.jpg';stat=image.stat();os.utime(image,ns=(stat.st_atime_ns,stat.st_mtime_ns+1));rejected(lambda:verify_input_evidence(base));os.utime(image,ns=(stat.st_atime_ns,stat.st_mtime_ns))
 corrupt=copy.deepcopy(base);corrupt['image_metadata_sha256']='0'*64;rejected(lambda:verify_input_evidence(corrupt))
 assert verify_input_evidence(base)
 return dict(CSV_label_content_change_rejected=True,camera_and_keypoint_content_changes_rejected=True,image_mtime_change_rejected=True,metadata_hash_corruption_rejected=True,configured_input_paths_anchored=True,raw_image_content_hashed=False)

class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=nn.Sequential(nn.GroupNorm(1,2));self.heads=nn.Linear(2,1)
 def forward(self,x):return self.heads(self.backbone(x).mean((2,3)))
def state_corruption():
 torch.manual_seed(2021);model=Tiny();model.eval();model.requires_grad_(False)
 for norm in model.backbone.modules():
  if isinstance(norm,nn.GroupNorm):norm.requires_grad_(True)
 before=copy.deepcopy(model.state_dict());optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0)
 for i in range(3):
  model(torch.randn(1,2,2,2)).square().mean().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
 payload=optimizer.state_dict();validate_optimizer(payload,model,3)
 bad=copy.deepcopy(payload);next(iter(bad['state'].values()))['exp_avg'].fill_(float('nan'));rejected(lambda:validate_optimizer(bad,model,3))
 bad=copy.deepcopy(payload);next(iter(bad['state'].values()))['exp_avg']=torch.zeros(999);rejected(lambda:validate_optimizer(bad,model,3))
 bad=copy.deepcopy(payload);next(iter(bad['state'].values()))['step']=torch.tensor(2.);rejected(lambda:validate_optimizer(bad,model,3))
 state=np.random.RandomState(2021).get_state();rng=dict(python=random.Random(2021).getstate(),numpy=dict(kind=state[0],keys=state[1].tolist(),pos=state[2],has_gauss=state[3],cached_gaussian=state[4]),torch_cpu=torch.Generator().manual_seed(2021).get_state(),torch_cuda=[torch.zeros(16,dtype=torch.uint8)])
 validate_rng(rng)
 for mutate in [lambda r:r['numpy'].update(cached_gaussian=float('nan')),lambda r:r.update(torch_cuda=[torch.tensor([float('nan')])]),lambda r:r.update(torch_cpu=torch.zeros(2)),lambda r:r.pop('python'),lambda r:r['numpy'].update(keys=[1])]:
  bad=copy.deepcopy(rng);mutate(bad);rejected(lambda:validate_rng(bad))
 validate_weights(model.state_dict(),before)
 bad=copy.deepcopy(model.state_dict());bad['heads.weight'].fill_(float('nan'));rejected(lambda:validate_weights(bad,before))
 bad=copy.deepcopy(model.state_dict());bad['heads.weight']=bad['heads.weight'].double();rejected(lambda:validate_weights(bad,before))
 from audit_core import check_scope
 check_scope(model,before,'A')
 with torch.no_grad():model.heads.bias.add_(1)
 rejected(lambda:check_scope(model,before,'A'))
 return dict(nonfinite_optimizer_rejected=True,moment_shape_and_step_rejected=True,nonfinite_or_incomplete_RNG_rejected=True,weight_dtype_nan_and_head_mutation_rejected=True,CUDA_RNG_payload_mocked_not_restored=True)

def standard_events():
 captured=io.StringIO()
 with contextlib.redirect_stdout(captured):
  emit_event('update',samples_seen=4,batch=4,batches=1024,update_attempts=1,optimizer_updates=1,loss=.12,N=1024,B=4,physical_batch=1,actual_batch_size=1)
  emit_event('checkpoint',path='/mock/adapted.pth',checkpoint='/mock/adapted.pth')
  emit_event('evaluation_progress',batch=200,batches=2791,n_samples=200,score=None)
 lines=captured.getvalue().splitlines();assert all(x.startswith('SPNV2_EVENT ') for x in lines)
 rows=[json.loads(x[len('SPNV2_EVENT '):]) for x in lines];assert all(x['phase']=='odr' for x in rows) and rows[0]['loss']==.12 and rows[1]['path']=='/mock/adapted.pth'
 env=environ();env['CUDA_VISIBLE_DEVICES']=''
 child=subprocess.run([PYTHON,'-B','-s',str(O/'audited_entry.py'),'--task','__invalid_cpu_test_task__'],cwd=O,env=env,capture_output=True,text=True)
 (D/'expected_invalid_entry.log').write_text(child.stdout+child.stderr);assert child.returncode!=0
 events=[json.loads(x[len('SPNV2_EVENT '):]) for x in child.stdout.splitlines() if x.startswith('SPNV2_EVENT ')];assert events[-1]['event']=='error' and events[-1]['phase']=='odr'
 child=subprocess.run([PYTHON,'-B','-s',str(O/'run_task.py'),'--task','fixed_phi3_A_lightbox_s2021_N1024_B4'],cwd=O,env=env,capture_output=True,text=True);assert child.returncode==0 and json.loads(child.stdout)['group']=='fixed'
 assert 'event_callback=progress' in (O/'audited_entry.py').read_text()
 # Import-only check: verifier actively hides CUDA before any torch/model import.
 os.environ['CUDA_VISIBLE_DEVICES']='0';__import__('verify_result');assert os.environ['CUDA_VISIBLE_DEVICES']==''
 return dict(standard_update_checkpoint_evaluation_events=True,invalid_entry_error_event=True,runner_default_dry_list=True,CPU_verifier_hides_CUDA=True,no_real_ODR_invoked=True)

def zero_H_physical_metrics():
 from scipy.io import savemat
 from verify_heads import verify
 folder=D/'zeroH';folder.mkdir();csvpath=folder/'test.csv';names=['a.jpg','b.jpg','c.jpg'];csvpath.write_text(''.join(n+',label\n' for n in names));n=3
 H=['heat_eR','heat_eT','heat_eR_HIL','heat_eT_norm_HIL','heat_pose'];E=['effi_iou','effi_eR','effi_eT','effi_eR_HIL','effi_eT_norm_HIL','effi_pose']
 per={k:np.full(n,np.nan) for k in H+['final_pose']};per.update({k:np.full(n,.03) for k in E});per.update(sample_id=np.array(names,dtype=object),n_samples=n,h_reject=np.ones(n,dtype=bool),final_pose_valid=np.zeros(n,dtype=bool))
 mean={k:np.nan for k in H+['final_pose']};mean.update({k:.03 for k in E});mean.update(n_samples=n,h_reject=n)
 pred=dict(sample_id=np.array(names,dtype=object),heat_q=np.full((n,4),np.nan),heat_t=np.full((n,3),np.nan),effi_R=np.tile(np.eye(3),(n,1,1)),effi_t=np.zeros((n,3)),bbox=np.zeros((n,4)))
 def write():
  for name,payload in [('per_sample_performance',per),('mean_performance',mean),('predictions_pose',pred)]:savemat(folder/(name+'.mat'),payload)
 write();proof=verify(folder,csvpath,['heatmap','efficientpose']);assert proof['final_pose'] is None and proof['n_reject']==3 and proof['metric_denominators']['final_pose']==0 and proof['metric_denominators']['effi_eT']==3
 removed=per.pop('effi_eT');write();rejected(lambda:verify(folder,csvpath,['heatmap','efficientpose']));per['effi_eT']=removed
 pred['heat_q']=np.zeros((n,4));write();rejected(lambda:verify(folder,csvpath,['heatmap','efficientpose']))
 return dict(zero_accepted_H_preserves_null_score_and_NaN=True,E_full_denominator_preserved=True,missing_physical_metric_rejected=True,zero_filled_missing_prediction_rejected=True)

try:
 check('real_adapter_math_and_parent_task_manifest',adapter_and_manifest)
 check('same_parent_bounded_desktop_GPU_gate',desktop_gate)
 check('fixed40_before_any_scan',fixed40_gate)
 check('data_and_geometry_corruption_rejected',data_corruption)
 check('optimizer_RNG_weights_scope_corruption_rejected',state_corruption)
 check('standard_events_CPU_env_and_dry_CLI',standard_events)
 check('complete_physical_metrics_zero_H_no_zero_fill',zero_H_physical_metrics)
 report=dict(status='passed',implementation_manifest_sha256=sha(O/'implementation_manifest.json'),scope='CPU engineering regression; all data/models/RNG-CUDA/metrics are mocked; no real ODR/GPU or scientific accuracy results',tests=results,passed=len(results),math_tests=15,run_id=a.run_id,real_GPU_queries=False,real_ODR_executed=False)
 atomic(O/'reports'/('integration_cpu_acceptance_'+a.run_id+'.json'),report);atomic(O/'reports/integration_cpu_acceptance.json',report);print(json.dumps(report),flush=True)
except BaseException as error:
 report=dict(status='failed',tests=results,error=repr(error),traceback=traceback.format_exc(),run_id=a.run_id,real_GPU_queries=False,real_ODR_executed=False)
 atomic(O/'reports'/('integration_cpu_acceptance_'+a.run_id+'.json'),report);raise
