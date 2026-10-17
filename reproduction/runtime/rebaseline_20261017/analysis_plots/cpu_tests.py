"""Small deterministic MOCK figures and hostile-evidence CPU regressions."""
import argparse,copy,statistics,traceback
from PIL import Image
import common
from common import *
from render import curve,distribution,parameters,benchmark
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();env_guard();out=D/'tests'/a.run_id;out.mkdir(exist_ok=False);checks=[]
def reject(fn):
 try:fn()
 except (IntegrityError,FileNotFoundError,KeyError,ValueError):return
 raise AssertionError('Bad evidence accepted')
def check(name,fn):
 result=fn();checks.append(dict(name=name,status='passed',details=result));print('PASS',name,json.dumps(result),flush=True)
bundle=gate_snapshot(A/'runs/actual_snapshot_20261017_d')
def empty_and_mock_stats():
 actual=expected_curves(bundle);validate_seed_interface(bundle,actual);assert len(actual)==56 and sum(c['expected_points'] for c in actual)==504 and all(c['status']=='pending' for c in actual)
 mock=copy.deepcopy(bundle)
 for row in mock['matrix']:
  r=mock['records'][row['task_id']]
  if r.get('group') in ['Bscan','Nscan']:
   row.update(status='已验收',accepted_mean=str(.02*(r['B'] if r['group']=='Bscan' else r['N'])+.001*(r['seed']-2020)),source_sha256='MOCK_SOURCE_SHA',valid_denominator='4')
 plots=expected_curves(mock);assert all(c['status']=='ready' for c in plots)
 chosen=plots[0];v=chosen['points'][0];assert np.isclose(v['std_ddof1'],statistics.stdev([.001*i for i in range(1,6)]))
 curve(chosen,out/'MOCK_Bscan.png',True);n=next(c for c in plots if c['kind']=='Nscan');curve(n,out/'MOCK_Nscan.png',True)
 changed=copy.deepcopy(mock);target=next(r for r in changed['matrix'] if changed['records'][r['task_id']].get('group')=='Bscan');target['accepted_mean']='';assert any(c['status']=='pending' for c in expected_curves(changed))
 changed=copy.deepcopy(mock);target=next(r for r in changed['matrix'] if changed['records'][r['task_id']].get('group')=='Bscan');target['source_sha256']='WRONG_SOURCE';reject(lambda:expected_curves(changed))
 curve_bad=copy.deepcopy(chosen);curve_bad['status']='pending';reject(lambda:curve(curve_bad,out/'FORBIDDEN_real_partial.png'))
 stats=latency_math(list(range(1,31)));assert np.isclose(stats['mean_ms'],15.5) and np.isclose(stats['population_std_ms'],np.sqrt((30**2-1)/12)) and np.isclose(stats['images_per_second'],1000/15.5)
 reject(lambda:latency_math([float('nan')]*30));reject(lambda:latency_math([0]*30))
 return dict(bench_math_tiny_MOCK_only=True,expected_series=56,expected_points=504,all_empty_stay_pending=True,MOCK_sample_SD_exact=True,source_contamination_rejected=True,incomplete_real_curves_refused=True)
def transitions_and_layout():
 item=dict(config='MOCK_phi3',line='B2',domain='MOCK',source_sha256='MOCK_SOURCE_SHA',N=1024,B=4,seed=2021,n_total=7,n_common=4,states=dict(both_accepted=4,accepted_to_rejected=1,rejected_to_accepted=1,both_rejected=1),bin_edges=[-.5,0,.5],bin_counts=[2,2]);distribution(item,out/'MOCK_distribution.png',True)
 zero=dict(item,n_common=0,n_total=3,states=dict(both_accepted=0,accepted_to_rejected=1,rejected_to_accepted=1,both_rejected=1),bin_edges=[],bin_counts=[]);distribution(zero,out/'MOCK_zero_valid.png',True)
 param=parameter_evidence();parameters(param,out/'MOCK_parameters_layout.png',True)
 bench=bench_evidence();assert bench['status']!='accepted' and not bench['measurements']
 mockbench=dict(status='MOCK',device=dict(name='MOCK DEVICE'),measurements=[dict(name=r['config'],statistics=dict(mean_ms=10+i,population_std_ms=.5,images_per_second=1000/(10+i)),memory_bytes=dict(peak_allocated=(i+1)*2**20,peak_reserved=(i+2)*2**20)) for i,r in enumerate(param['rows'])]);benchmark(mockbench,out/'MOCK_GPU_benchmark_layout.png',True)
 for p in out.glob('MOCK*.png'):Image.open(p).verify();im=Image.open(p);assert im.width>=1000 and im.height>=650
 ids=['MOCK_a','MOCK_b','MOCK_c','MOCK_d'];source=[1,2,float('nan'),float('nan')];after=[.7,float('nan'),1.8,float('nan')];va=[True,True,False,False];vb=[True,False,True,False];data=[dict(sample_id=id,before_score=str(x) if valid_a else '',after_score=str(y) if valid_b else '',delta=str(y-x) if valid_a and valid_b else '',state='both_accepted' if valid_a and valid_b else 'accepted_to_rejected' if valid_a else 'rejected_to_accepted' if valid_b else 'both_rejected') for id,x,y,valid_a,valid_b in zip(ids,source,after,va,vb)]
 audit_pair_numeric(data,ids,source,va,ids,after,vb);reject(lambda:audit_pair_numeric(data,list(reversed(ids)),source,va,ids,after,vb));bad=copy.deepcopy(data);bad[1]['delta']='0';reject(lambda:audit_pair_numeric(bad,ids,source,va,ids,after,vb))
 return dict(MAT_vs_plot_paired_values_and_ID_order_verified=True,MOCK_figures=6,real_benchmark_missing_stays_pending=True,zero_valid_no_fake_histogram=True,all_output_names_MOCK=True,not_formal_scientific_results=True)
def hostile_evidence():
 root=out/'mock_analysis';(root/'runs/mock').mkdir(parents=True);snapshot=root/'runs/mock';(root/'implementation_manifest.json').write_text('{"implementation":{}}')
 catalog=copy.deepcopy(bundle['catalog']);matrix=copy.deepcopy(bundle['matrix']);record=catalog['records'][0];folder=out/'MOCK_mat';folder.mkdir();final=out/'MOCK_final';final.write_bytes(b'MOCK');proof=out/'MOCK_proof.json';proof.write_text('{}');csvp=out/'MOCK.csv';csvp.write_text('MOCK_id\n');record.update(folder=str(folder),source_final=str(final),accepted_report=str(proof));mats={}
 for name in ['mean_performance','per_sample_performance','predictions_pose']:
  q=folder/(name+'.mat');q.write_bytes(b'MOCK_NOT_REAL_MAT');mats[name]=sha(q)
 
 for cell in matrix:
  if cell['task_id']==record['id']:cell['status']='已验收'
 inp=[dict(cohort='new',task_id=record['id'],acceptance_report_sha256=sha(proof),csv=str(csvp),csv_current_sha256=sha(csvp),source_sha256=sha(final),mat_sha256=mats)]
 for name,value in [('catalog.json',catalog),('input_evidence.json',inp),('paired_summary.json',[]),('five_observation_seed_summary.json',[])]:write(snapshot/name,value)
 import csv
 with (snapshot/'matrix_new.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
 write(snapshot/'artifact_manifest.json',dict(files_sha256={p.name:sha(p) for p in snapshot.iterdir()}));write(snapshot/'snapshot_acceptance.json',dict(status='passed',artifact_manifest_sha256=sha(snapshot/'artifact_manifest.json')))
 old=common.A;common.A=root
 try:
  gate_snapshot(snapshot);csvp.write_text('CORRUPT');reject(lambda:gate_snapshot(snapshot));csvp.write_text('MOCK_id\n');final.write_bytes(b'WRONG');reject(lambda:gate_snapshot(snapshot));final.write_bytes(b'MOCK');q=folder/'predictions_pose.mat';q.write_bytes(b'CORRUPT');reject(lambda:gate_snapshot(snapshot))
 finally:common.A=old
 reject(lambda:gate_snapshot(snapshot))
 return dict(CSV_corruption_rejected=True,source_SHA_corruption_rejected=True,MAT_corruption_rejected=True,mock_snapshot_disallowed_in_real_CLI=True)
try:
 check('pending_scan_points_and_five_stream_math',empty_and_mock_stats);check('MOCK_layout_failure_transitions_and_zero_valid',transitions_and_layout);check('hostile_CSV_MAT_source_and_mock_boundary',hostile_evidence)
 proof=dict(status='passed',run_id=a.run_id,scope='CPU/mock engineering only; not real scan curves or adaptation results',checks=checks,passed=len(checks),files_sha256={p.name:sha(p) for p in out.glob('MOCK*.png')},implementation_sha256={p.name:sha(p) for p in D.glob('*.py')},no_GPU_execution=True);write(D/'reports'/('cpu_acceptance_'+a.run_id+'.json'),proof);print(json.dumps(proof,ensure_ascii=False))
except BaseException as e:write(D/'reports'/('cpu_acceptance_'+a.run_id+'.json'),dict(status='failed',error=repr(e),checks=checks,traceback=traceback.format_exc()));raise
