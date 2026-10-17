"""Finite CPU regressions on explicitly mock fixtures; not scientific results."""
import argparse,copy,os,random,time,traceback,ast
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from PIL import Image
from core import *
from records import inspect_record
from run_analysis import matrix_rows
from cases import render_case,project_matrix,selections
from projection import project_keypoints,quat2dcm
from materials import audit_prisma
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();D=A/'tests'/a.run_id;D.mkdir(exist_ok=False);checks=[]
def check(name,fn):
 begin=time.time();details=fn();checks.append(dict(name=name,status='passed',seconds=time.time()-begin,details=details));print('PASS',name,json.dumps(details),flush=True)
def reject(fn):
 try:fn()
 except (IntegrityError,EvidenceMissing,AssertionError,KeyError,ValueError):return
 raise AssertionError('Invalid evidence was accepted')

def fixture(name,valid,scores):
 folder=D/name;folder.mkdir();n=len(valid);ids=['mock_'+str(i)+'.png' for i in range(n)];csvpath=folder/'labels.csv';labels=[]
 for id in ids:labels.append([id,0,0,1,1,1,0,0,0,0,0,5,.5,.5]+[.5,.5]*11)
 with csvpath.open('w') as f:csv.writer(f).writerows(labels)
 valid=np.array(valid,bool);scores=np.array(scores,float)
 H=['heat_eR','heat_eT','heat_eR_HIL','heat_eT_norm_HIL','heat_pose'];E=['effi_iou','effi_eR','effi_eT','effi_eR_HIL','effi_eT_norm_HIL','effi_pose']
 per={k:np.full(n,np.nan) for k in H};per.update({k:np.full(n,.1) for k in E});per['effi_eR'][:]=.1*180/np.pi;per['effi_pose'][:]=.2
 for metric in H:per[metric][valid]=.1
 per['heat_eR_HIL'][valid]=scores[valid]-.1;per['heat_eR'][valid]=(scores[valid]-.1)*180/np.pi;per['heat_pose'][valid]=scores[valid]
 per.update(final_pose=np.where(valid,scores,np.nan),sample_id=np.array(ids,dtype=object),n_samples=n,h_reject=~valid,final_pose_valid=valid)
 mean={k:float(np.mean(per[k][valid])) if valid.any() else np.nan for k in H+['final_pose']};mean.update({k:float(np.mean(per[k])) for k in E});mean.update(n_samples=n,h_reject=int((~valid).sum()))
 pred=dict(sample_id=np.array(ids,dtype=object),heat_q=np.full((n,4),np.nan),heat_t=np.full((n,3),np.nan),effi_R=np.tile(np.eye(3),(n,1,1)),effi_t=np.tile([0,0,5.],(n,1)),bbox=np.tile([0,0,32,24.],(n,1)))
 pred['heat_q'][valid]=[1,0,0,0];pred['heat_t'][valid]=[0,0,5]
 for file,value in [('mean_performance',mean),('per_sample_performance',per),('predictions_pose',pred)]:savemat(folder/(file+'.mat'),value)
 return load_results(folder,csvpath,['heatmap','efficientpose'])
before=fixture('mock_source',[1,1,0,0],[1,2,0,0]);after=fixture('mock_odr',[1,0,1,0],[.7,0,1.8,0])

def matrix_and_intersection():
 stats,rows=paired(before,after,'mock_sha','mock_sha','mock_sha')
 assert stats['n_common']==stats['accepted_to_rejected']==stats['rejected_to_accepted']==stats['both_rejected']==1
 assert np.isclose(stats['delta_mean'],-.3) and np.isclose(stats['before_common_mean'],1) and np.isclose(stats['after_common_mean'],.7)
 assert np.isclose(before['summary']['final_pose'],1.5) and np.isclose(after['summary']['final_pose'],1.25) and not np.isclose(stats['delta_mean'],after['summary']['final_pose']-before['summary']['final_pose'])
 assert all(r['delta'] is None for r in rows if r['state']!='both_accepted')
 assert head('final_pose',['heatmap'])=='hH' and head('final_pose',['efficientpose'])=='hE' and head('final_pose',['heatmap','efficientpose'])=='mixed_hR_plus_eT'
 assert final_formula(['heatmap'])=='hR_HIL+hT_norm_HIL' and final_formula(['efficientpose'])=='eR_HIL+eT_norm_HIL'
 assert stats['metric_distributions']['heat_eR']['n_common']==1 and stats['metric_distributions']['effi_eT']['n_common']==4
 changed=copy.copy(after);changed['ids']=list(reversed(after['ids']));reject(lambda:paired(before,changed,'mock_sha','mock_sha','mock_sha'))
 reject(lambda:paired(before,after,'mock_sha','other_sha','mock_sha'))
 return dict(common_intersection=1,independent_transitions_each=1,paired_mean_delta=-.3,accepted_mean_difference_not_called_improvement=True,shuffled_ID_and_wrong_source_rejected=True,head_specific_denominators=True)
def zero_valid_and_missing():
 zero=fixture('mock_zero_H',[0,0,0,0],[0,0,0,0]);stats,rows=paired(zero,zero,'x','x','x')
 assert zero['summary']['final_pose'] is None and zero['summary']['metric_denominators']['final_pose']==0 and stats['delta_mean'] is None and stats['n_common']==0
 assert stats['both_rejected']==4 and np.isnan(zero['per']['final_pose']).all()
 reject(lambda:load_results(D/'missing',D/'mock_source/labels.csv',['heatmap','efficientpose']))
 folder=D/'mock_zero_H';per=loadmat(folder/'per_sample_performance.mat',simplify_cells=True);per.pop('effi_eT');savemat(folder/'per_sample_performance.mat',per);reject(lambda:load_results(folder,folder/'labels.csv',['heatmap','efficientpose']))
 return dict(zero_valid_null_not_zero=True,no_Inf_substitution=True,missing_MAT_and_physical_metric_rejected=True)
def hash_pin_and_pending():
 final=D/'mock_final.pth';final.write_bytes(b'MOCK_NOT_MODEL');cfg=D/'mock_cfg.yaml';cfg.write_text('mock: true');proof=D/'mock_source_acceptance.json'
 write(proof,dict(status='passed',cfg=str(cfg),final=str(final),final_sha256=sha(final),code_sha256=read(PLAN/'source_manifest.json')['core_tools_sha256'],mocked_fixture=True));pins={};pin_source(cfg,final,proof,pins)
 final.write_bytes(b'WRONG_HASH_MOCK');reject(lambda:pin_source(cfg,final,proof,pins))
 row=dict(id='mock_pending',stage='source',cohort='new',source_config='mock_pending',source_final=str(D/'absent_final'),source_report=str(D/'absent_report'),pending_report=str(D/'absent_pending'),folder=str(D/'absent_results'),heads=['heatmap','efficientpose'],domain='mock',reason='')
 pending=inspect_record(row,{}, {},dict(loaded={},catalog_records=[]));matrix=matrix_rows([pending]);assert pending['status']=='未完成' and all(x['accepted_mean'] is None and x['valid_denominator'] is None for x in matrix)
 waiting=dict(row,source_final=str(final));assert inspect_record(waiting,{}, {},dict(loaded={},catalog_records=[]))['status']=='待验收'
 final.write_bytes(b'MOCK_NOT_MODEL');write(proof,dict(status='passed',cfg=str(cfg),final=str(final),final_sha256=sha(final),code_sha256=read(PLAN/'source_manifest.json')['core_tools_sha256'],mocked_fixture=True))
 raw=D/'mock_eval_raw.json';accepted=D/'mock_eval_accepted.json';write(raw,dict(status='awaiting_verification',source_sha256=sha(final),source=str(final),csv=before['csv'],domain='mock',results=before['folder'],mocked_fixture=True));write(accepted,dict(status='passed',source_sha256=sha(final),source_report=str(raw),metrics=before['summary'],mocked_fixture=True))
 accepted_row=dict(row,source_final=str(final),source_report=str(proof),source_config_path=str(cfg),accepted_report=str(accepted),csv=before['csv'],heads=['heatmap','efficientpose']);accepted_record=inspect_record(accepted_row,{}, {},dict(loaded={},catalog_records=[]));assert accepted_record['status']=='已验收'
 return dict(changed_final_SHA_rejected=True,missing_new_source_stays_pending_null=True,no_historical_value_borrowed=True,uncompleted_awaiting_accepted_states_distinct=True)
def alias_and_seed_math():
 canonical=dict(id='mock_fixed',phi=3,line='A',domain='mock',seed=2021,N=4,B=4,source_final='mock_final');alias=dict(canonical,id='mock_alias',reuse_of='mock_fixed')
 path=D/'mock_canonical_proof.json';proof=dict(status='passed',source_sha256='mock_source',protocol={'mock':True},implementation_manifest_sha256='mock_impl');write(path,proof)
 advertised=dict(proof,reuse_of='mock_fixed',canonical_report=str(path),canonical_report_sha256=sha(path));assert alias_identity(alias,advertised,canonical,proof,path)=='mock_fixed'
 bad=dict(advertised,canonical_report_sha256='0'*64);reject(lambda:alias_identity(alias,bad,canonical,proof,path))
 bad=dict(alias,line='B2');reject(lambda:alias_identity(bad,advertised,canonical,proof,path))
 base=dict(stage='odr',cohort='new',status='已验收',source_config='mock_cfg',source_sha256='mock_source',line='A',domain='mock',N=4,B=4,seed=2021,canonical_id='mock_fixed',is_alias=False,loaded=before)
 group=seed_statistics([base,dict(base,is_alias=True)]);assert group[0]['canonical_experiments']==1 and group[0]['metrics']['final_pose']['mean'] is None
 reject(lambda:seed_statistics([base,dict(base,canonical_id='another_independent_same_seed')]))
 records=[]
 for seed,value in zip(range(2021,2026),[1.,2.,3.,4.,5.]):
  loaded=copy.copy(before);loaded['summary']=copy.deepcopy(before['summary']);loaded['summary']['final_pose']=value
  records.append(dict(base,seed=seed,canonical_id='mock_seed_'+str(seed),loaded=loaded))
 stats=seed_statistics(records)[0];assert stats['five_observation_seeds_complete'] and np.isclose(stats['metrics']['final_pose']['mean'],3.) and np.isclose(stats['metrics']['final_pose']['std_ddof1'],np.sqrt(2.5))
 separate=seed_statistics(records+[dict(r,line='B2',canonical_id=r['canonical_id']+'_B2') for r in records]);assert len(separate)==2
 historical=dict(base,cohort='historical',canonical_id='history');assert len(seed_statistics([historical]))==0
 return dict(alias_canonical_SHA_and_line_checked=True,alias_not_extra_experiment=True,independent_duplicate_seed_rejected=True,five_observation_seed_mean=3.,sample_std=float(np.sqrt(2.5)),A_B2_separate=True,historical_excluded=True)
def projection_png():
 images=D/'mock_images';images.mkdir()
 for id in before['ids']:Image.new('RGB',(32,24),(60,60,60)).save(images/id)
 camera=D/'mock_camera.json';write(camera,dict(Nu=32,Nv=24,cameraMatrix=[[20,0,16],[0,20,12],[0,0,1]],distCoeffs=[0,0,0,0,0]))
 points=np.vstack([np.linspace(-.5,.5,11),np.linspace(-.3,.3,11),np.zeros(11)]);geometry=D/'mock_points.mat';savemat(geometry,dict(tango3Dpoints=points))
 K=np.array([[20,0,16],[0,20,12],[0,0,1]],float);q=np.array([np.cos(np.pi/4),0,0,np.sin(np.pi/4)]);dist=np.zeros(5)
 h=project_keypoints(q,np.array([0,0,5]),K,dist,points.T);e=project_matrix(quat2dcm(q),np.array([0,0,5]),K,dist,points.T);assert np.allclose(h,e,atol=1e-6)
 evidence=D/'mock_input_evidence.json';write(evidence,dict(csv=dict(path=before['csv'],sha256=sha(before['csv'])),raw_image_content_hashed=False,geometry=dict(camera=dict(path=str(camera),sha256=sha(camera)),keypoints=dict(path=str(geometry),sha256=sha(geometry)))))
 verify_geometry_binding(evidence,sha(evidence),before['csv']);reject(lambda:verify_geometry_binding(evidence,'wrong_sha',before['csv']))
 output=D/'MOCK_pose_projection.png';item=render_case(before,after,before['ids'][0],output,camera,geometry,images);Image.open(output).verify();assert output.stat().st_size>10000 and item['predictions_drawn'][0]
 _,rows=paired(before,after,'x','x','x');selected=selections(rows);assert not any(name=='largest_degradation' for name,row in selected)
 return dict(mock_PNG=str(output),pose_convention_matches_frozen_quaternion_projection=True,all_improved_does_not_fake_degradation=True,not_real_prediction_results=True)
def schema_and_material_scope():
 for f in A.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 status=audit_prisma();assert status['status']=='受限' and status['complete'] is False and status['verified_limitation'] and status['conditional_prisma_status']=='verified_limitation'
 reject(lambda:own('/tmp/other_project'))
 return dict(PRISMA_missing_stays_limited=True,no_network_calls=True,outside_project_paths_rejected=True,all_scripts_parse=True)
try:
 check('strict_intersection_and_independent_failure_transitions',matrix_and_intersection)
 check('zero_H_missing_evidence_and_physical_metric',zero_valid_and_missing)
 check('fixed_source_SHA_and_missing_new_source',hash_pin_and_pending)
 check('canonical_alias_and_five_observation_seed_statistics',alias_and_seed_math)
 check('CPU_mock_pose_projection_PNG',projection_png)
 check('schema_and_PRISMA_scope',schema_and_material_scope)
 proof=dict(status='passed',scope='Finite CPU engineering/mock mathematics only; not formal source/ODR results',checks=checks,passed=len(checks),run_id=a.run_id,real_GPU_executed=False,analysis_code_sha256={f.name:sha(f) for f in sorted(A.glob('*.py'))})
 write(A/'reports'/('cpu_acceptance_'+a.run_id+'.json'),proof);write(A/'reports/cpu_acceptance.json',proof);print(json.dumps(proof,ensure_ascii=False))
except BaseException as error:
 write(A/'reports'/('cpu_acceptance_'+a.run_id+'.json'),dict(status='failed',checks=checks,error=repr(error),traceback=traceback.format_exc(),mock_only=True));raise
