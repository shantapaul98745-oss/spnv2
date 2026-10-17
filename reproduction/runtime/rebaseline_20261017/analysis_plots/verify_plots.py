"""Independent CPU figure acceptance: numeric inputs, source/MAT/CSV SHA and point counts."""
import argparse,statistics,subprocess
from PIL import Image
from common import *
def verify(run_id):
 env_guard();out=D/'runs'/run_id;manifest=read(out/'artifact_manifest.json')
 for name,value in manifest['files_sha256'].items():need(sha(out/name)==value,'Plot output changed: '+name)
 for name,value in manifest['implementation_sha256'].items():need(sha(D/name)==value,'Plot implementation changed: '+name)
 inputs=read(out/'input_manifest.json')
 for name,value in inputs['inputs'].items():need(sha(scope(name))==value,'Plot input changed: '+name)
 bundle=gate_snapshot(inputs['snapshot'],deep=True);index={(r['task_id'],r['metric']):r for r in bundle['matrix']};curves=read(out/'scan_points.json');point_count=0;validate_seed_interface(bundle,curves)
 need(len(curves)==56,'Wrong expected metric-series count')
 for c in curves:
  expected=[1,2,4,8,16,32] if c['kind']=='Bscan' else [2**i for i in range(2,14)];need([v['x'] for v in c['points']]==expected,'Wrong scan x or expected point count')
  for point in c['points']:
   point_count+=1;records=[bundle['records'][t] for t in point['task_ids']];need(sorted(r['seed'] for r in records)==list(range(2021,2026)),'Wrong seed set')
   expectedrecords=[r for r in bundle['records'].values() if r.get('group')==c['kind'] and r['source_config']==c['config'] and r['domain']==c['domain'] and r['line']==c['line'] and (r['B'] if c['kind']=='Bscan' else r['N'])==point['x']]
   need({r['id'] for r in records}=={r['id'] for r in expectedrecords},'Cross-protocol point contamination')
   cells=[index[(r['id'],c['metric'])] for r in records];accepted=[v for v in cells if v['status']=='已验收'];valid=[float(v['accepted_mean']) for v in accepted if finite(v['accepted_mean'])];complete=len(accepted)==len(valid)==5
   need(point['canonical_ids']==[r['canonical_id'] for r in records] and len(set(point['canonical_ids']))==5,'Alias inflated independent seed count')
   need(not accepted or {v['source_sha256'] for v in accepted}=={c['source_sha256']},'Curve source hash not exact accepted final')
   need(point['valid_seed_count']==len(valid) and (point['status']=='ready')==complete,'Point readiness mismatch')
   if complete:need(np.isclose(point['mean'],statistics.mean(valid),rtol=1e-12) and np.isclose(point['std_ddof1'],statistics.stdev(valid),rtol=1e-12),'Point mean/SD mismatch')
   else:need(point['mean'] is None and point['std_ddof1'] is None,'Missing seeds fabricated into numeric mean/SD')
  need((c['status']=='ready')==all(p['status']=='ready' for p in c['points']),'Incomplete curve advertised ready')
 need(point_count==504,'Wrong expected504 scan point count')
 bench=bench_evidence();need(bench==read(out/'benchmark_inputs.json'),'GPU benchmark input differs from independent acceptance')
 paired=read(out/'fixed_seed2021_distributions.json')
 for item in paired:
  p=scope(item['paired_csv']);need(sha(p)==item['paired_csv_sha256'],'Paired CSV changed');data=rows(p);delta=[]
  from scipy.io import loadmat
  after_record=bundle['records'][item['task_id']];before_record=next(r for r in bundle['records'].values() if r['stage']=='source' and r['source_config']==after_record['source_config'] and r['domain']==after_record['domain'])
  before=loadmat(scope(Path(before_record['folder'])/'per_sample_performance.mat'),simplify_cells=True);after=loadmat(scope(Path(after_record['folder'])/'per_sample_performance.mat'),simplify_cells=True)
  audit_pair_numeric(data,list(np.atleast_1d(before['sample_id'])),np.asarray(before['final_pose']).reshape(-1),np.asarray(before['final_pose_valid']).reshape(-1).astype(bool),list(np.atleast_1d(after['sample_id'])),np.asarray(after['final_pose']).reshape(-1),np.asarray(after['final_pose_valid']).reshape(-1).astype(bool))
  for row in data:
   if row['state']=='both_accepted':
    need(finite(row['before_score']) and finite(row['after_score']) and np.isclose(float(row['delta']),float(row['after_score'])-float(row['before_score'])),'Paired delta not actual same-ID change');delta.append(float(row['delta']))
   else:need(row['delta']=='','Rejected delta must stay missing')
  need(len(data)==item['n_total'] and len(delta)==item['n_common'],'Paired denominator mismatch')
  for state,count in item['states'].items():need(sum(r['state']==state for r in data)==count,'Transition count mismatch')
  if delta:need(np.histogram(delta,bins=item['bin_edges'])[0].tolist()==item['bin_counts'] and sum(item['bin_counts'])==len(delta),'Histogram count mismatch')
 param=parameter_evidence();need(param==read(out/'parameter_inputs.json'),'Table6 parameter input mismatch')
 reference=read(D/'reference/paper_reference.json');need(sha(scope(reference['pdf']))==reference['pdf_sha256'],'Paper PDF changed')
 for name,value in reference['page_evidence_sha256'].items():need(sha(D/'reference'/name)==value,'Paper visual evidence changed')
 for page in [8,9]:
  text=subprocess.check_output(['pdftotext','-f',str(page),'-l',str(page),'-layout',reference['pdf'],'-'],text=True)
  for r in reference['metric_rows']:
   if r['pdf_page']==page:need((str(int(r['value'])) if r['unit']=='count' else format(r['value'],'.3f')) in text,'Reference numeric token absent from cited paper page')
 plots=read(out/'plots.json');expected_plot_count=sum(c['status']=='ready' for c in curves)+len(paired)+1+int(bench['status']=='accepted');need(len(plots)==expected_plot_count,'Missing/extra expected real figure')
 for fig in plots:
  p=out/fig['png'];need(sha(p)==fig['png_sha256'],'PNG changed');im=Image.open(p);im.verify();im=Image.open(p);need(im.width>=1000 and im.height>=650,'PNG undersized')
  if fig['kind'] in ['Bscan','Nscan']:need(fig['data'] in curves and fig['data']['status']=='ready','Incomplete/unbound scan image')
 pending=read(out/'pending.json');need(pending['pending_scan_points']==sum(p['status']!='ready' for c in curves for p in c['points']),'Pending point count mismatch')
 proof=dict(status='passed',scope='real accepted evidence plots; no model execution',scan_metric_series=56,expected_scan_points=504,pending_scan_points=pending['pending_scan_points'],accepted_fixed_distributions=len(paired),real_PNGs=len(plots),Table6='parameters_only',GPU_bench=bench['status'],CSV_MAT_SHA_rechecked=True,new_accepted_MAT_records=sum(r['cohort']=='new' for r in read(Path(bundle['snapshot'])/'input_evidence.json')),MAT_mean_check_scope='Only genuinely accepted new records; no new MAT means claimed when count0',source_final_SHA_rechecked=True,reference_PDF_SHA_and_page_tokens_rechecked=True,artifact_manifest_sha256=sha(out/'artifact_manifest.json'))
 write(out/'plot_acceptance.json',proof);return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();print(json.dumps(verify(a.run_id),ensure_ascii=False))
