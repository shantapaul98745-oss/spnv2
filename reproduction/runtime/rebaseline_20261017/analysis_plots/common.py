"""Read-only verified evidence boundary for scientific figures."""
from pathlib import Path
import csv,json,hashlib,os,sys,math
import numpy as np
D=Path(__file__).resolve().parent;P=D.parent;A=P/'analysis';BASE=P.parents[2];DATA=Path('/home/DATA/Downloads/DataSet/speedplus')
METRICS=['heat_pose','effi_pose','final_pose','heat_eR','heat_eT','effi_eR','effi_eT']
HASH_CACHE={}
class IntegrityError(RuntimeError):pass
def need(ok,message):
 if not ok:raise IntegrityError(message)
def sha(p):
 p=Path(p);st=p.stat();key=(str(p.resolve()),st.st_size,st.st_mtime_ns)
 if key not in HASH_CACHE:
  with p.open('rb') as f:HASH_CACHE[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return HASH_CACHE[key]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def finite(x):return x is not None and x!='' and math.isfinite(float(x))
def env_guard():need(os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Unique guarded CPU screen and hidden CUDA required')
def scope(p):
 p=Path(p).resolve();need(p.is_relative_to(BASE) or p.is_relative_to(DATA),'Outside own project/data');return p

def gate_snapshot(snapshot,deep=False):
 snapshot=Path(snapshot).resolve();need(snapshot.parent==A/'runs','Real figures accept only analysis/runs snapshots, never mock fixtures')
 impl=read(A/'implementation_manifest.json')
 for name,item in impl['implementation'].items():need(sha(A/name)==item['sha256'],'Frozen analysis v1 implementation changed: '+name)
 accepted=read(snapshot/'snapshot_acceptance.json');manifest=snapshot/'artifact_manifest.json'
 need(accepted['status']=='passed' and accepted['artifact_manifest_sha256']==sha(manifest),'Snapshot independent acceptance missing/changed')
 for name,value in read(manifest)['files_sha256'].items():need(sha(snapshot/name)==value,'Snapshot artifact changed: '+name)
 catalog=read(snapshot/'catalog.json');records={r['id']:r for r in catalog['records']};matrix=rows(snapshot/'matrix_new.csv');evidence=read(snapshot/'input_evidence.json');index={(r['task_id'],r['metric']):r for r in matrix}
 need(len(records)==430 and len({r['task_id'] for r in matrix})==430,'Wrong snapshot record slots')
 need({r['task_id'] for r in matrix if r['status']=='已验收'}=={r['task_id'] for r in evidence if r['cohort']=='new'},'Accepted matrix/evidence coverage mismatch')
 for item in evidence:
  if item['cohort']!='new':continue
  r=records[item['task_id']];need(sha(scope(r['accepted_report']))==item['acceptance_report_sha256'],'Source/ODR acceptance report changed')
  need(sha(scope(item['csv']))==item['csv_current_sha256'],'CSV changed');need(sha(scope(r['source_final']))==item['source_sha256'],'Fixed final changed')
  for name,value in item['mat_sha256'].items():need(sha(scope(Path(r['folder'])/(name+'.mat')))==value,'MAT changed')
  if deep:
   import importlib.util
   spec=importlib.util.spec_from_file_location('readonly_analysis_matio',A/'matio.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
   actual=mod.verify(r['folder'],r['csv'],r['heads'])
   for metric,count in actual['metric_denominators'].items():
    cell=index[(r['id'],metric)];need(cell['status']=='已验收' and int(cell['valid_denominator'])==count,'MAT/CSV denominator mismatch')
    need((actual[metric] is None and cell['accepted_mean']=='') or (actual[metric] is not None and np.isclose(actual[metric],float(cell['accepted_mean']),rtol=1e-10)),'MAT/CSV mean mismatch')
 return dict(snapshot=str(snapshot),catalog=catalog,records=records,matrix=matrix,inputs={str(p):sha(p) for p in [manifest,snapshot/'snapshot_acceptance.json',snapshot/'matrix_new.csv',snapshot/'paired_summary.json',snapshot/'five_observation_seed_summary.json',snapshot/'input_evidence.json']})

def expected_curves(bundle):
 records=bundle['records'];matrix={(r['task_id'],r['metric']):r for r in bundle['matrix']};buckets={}
 for r in records.values():
  if r.get('group') not in ['Bscan','Nscan']:continue
  for metric in METRICS:
   row=matrix[(r['id'],metric)];key=(r['group'],r['source_config'],r['line'],r['domain'],metric,row['unit'],row['head']);x=r['B'] if r['group']=='Bscan' else r['N'];buckets.setdefault(key,{}).setdefault(x,[]).append((r,row))
 output=[]
 for key,points in sorted(buckets.items()):
  expected_x=[1,2,4,8,16,32] if key[0]=='Bscan' else [2**i for i in range(2,14)];need(sorted(points)==expected_x,'Unexpected scan point set')
  data=[];hashes=set()
  for x,items in sorted(points.items()):
   seeds=sorted(r['seed'] for r,v in items);need(seeds==list(range(2021,2026)),'Scan seed requests must be exactly2021-2025')
   accepted=[(r,v) for r,v in items if v['status']=='已验收'];values=[float(v['accepted_mean']) for r,v in accepted if finite(v['accepted_mean'])];source={v['source_sha256'] for r,v in accepted};need(len(source)<=1,'Mixed source final hashes at scan point');hashes|=source
   complete=len(accepted)==len(values)==5;data.append(dict(x=x,mean=float(np.mean(values)) if complete else None,std_ddof1=float(np.std(values,ddof=1)) if complete else None,accepted_seeds=[r['seed'] for r,v in accepted],valid_seed_count=len(values),canonical_ids=[r['canonical_id'] for r,v in items],task_ids=[r['id'] for r,v in items],denominators=[int(v['valid_denominator']) if v['valid_denominator'] else None for r,v in items],status='ready' if complete else 'pending'))
  need(len(hashes)<=1,'Source hash changed across curve');output.append(dict(kind=key[0],config=key[1],line=key[2],domain=key[3],metric=key[4],unit=key[5],head=key[6],source_sha256=next(iter(hashes)) if hashes else None,expected_points=len(expected_x),fixed_protocol={'N':1024} if key[0]=='Bscan' else {'B':4},points=data,status='ready' if all(v['status']=='ready' for v in data) else 'pending',seed_meaning='five ODR observation streams of one fixed source training; sample std ddof=1'))
 need(len(output)==56 and sum(v['expected_points'] for v in output)==504,'Expected56 metric-series/504 scan points')
 return output

def fixed_pairs(bundle):
 out=[];snapshot=Path(bundle['snapshot'])
 for item in read(snapshot/'paired_summary.json'):
  r=bundle['records'][item['task_id']]
  if r.get('group')!='fixed' or r['seed']!=2021:continue
  need(not r.get('is_alias') and r['canonical_id']==item['canonical_id'],'Alias in fixed distribution')
  p=snapshot/'paired'/(r['id']+'.csv');data=rows(p);states={name:sum(v['state']==name for v in data) for name in ['both_accepted','accepted_to_rejected','rejected_to_accepted','both_rejected']}
  deltas=[float(v['delta']) for v in data if v['state']=='both_accepted'];need(len(data)==item['n_total'] and len(deltas)==item['n_common'],'Paired distribution denominator mismatch')
  need(all(v['delta']=='' for v in data if v['state']!='both_accepted'),'Invalid comparisons filled with numeric delta')
  for name,value in states.items():need(value==item[name],'Paired transition count mismatch')
  need(not deltas or np.isclose(np.mean(deltas),item['delta_mean']),'Paired mean mismatch')
  edges=np.histogram_bin_edges(deltas,bins=30).tolist() if deltas else [];counts=np.histogram(deltas,bins=edges)[0].tolist() if deltas else []
  out.append(dict(task_id=r['id'],source_sha256=item['source_sha256'],config=r['source_config'],line=r['line'],domain=r['domain'],N=r['N'],B=r['B'],seed=2021,n_total=len(data),n_common=len(deltas),states=states,bin_edges=edges,bin_counts=counts,delta_mean=float(np.mean(deltas)) if deltas else None,paired_csv=str(p),paired_csv_sha256=sha(p),status='ready' if deltas else 'limited_zero_valid'))
 return out

def parameter_evidence():
 root=P/'resources/reports/table6_success_20261017';proof=root/'parameter_acceptance.json';accepted=read(proof);need(accepted['status']=='passed_parameters_only_all_four_models','Four-model parameter independent acceptance required')
 table=root/'table6_parameters.json';need(accepted['table_report_sha256']==sha(table),'Parameter table report changed');data=rows(root/'table6_parameters.csv');checks={r['config']:r['parameter_counts'] for r in accepted['checks']};need(len(data)==len(checks)==4,'Four actual models required')
 for r in data:
  counts=checks[r['config']]
  for key in ['whole_spnv2','efficientdet_backbone','efficientnet_encoder','bifpn','odr_updated_backbone_norm_affine','odr_heads']:need(int(r[key])==counts[key],'Parameter CSV/acceptance mismatch')
  need(int(r['whole_spnv2'])==int(r['efficientdet_backbone'])+sum(int(r[k]) for k in ['heatmap','efficientpose','segmentation']),'Parameter component sum mismatch');need(int(r['odr_heads'])==0,'ODR heads not frozen')
 return dict(status='parameters_only',rows=data,inputs={str(p):sha(p) for p in [proof,table,root/'table6_parameters.csv']},GPU_resources='pending; no inference/latency/peak-memory extrapolation')


def bench_evidence():
 root=P/'resources/bench';report=root/'reports/table6_forward_four_models_20261017.json';proof=root/'reports/table6_forward_four_models_20261017_verified.json';side=Path(str(report)+'.exitcode')
 if not report.exists() or not proof.exists() or not side.exists():return dict(status='pending',missing=[str(p) for p in [report,proof,side] if not p.exists()],measurements=[],inputs={},scope='whole EHS GPU forward only; never end-to-end or ODR adaptation time')
 value=read(report);accepted=read(proof)
 if value.get('status')!='passed' or accepted.get('status')!='passed':return dict(status='pending_independent_acceptance',measurements=[],inputs={str(p):sha(p) for p in [report,proof,side]},scope='No GPU results accepted for plotting yet')
 need(value['schema']=='spnv2-table6-forward-benchmark-v1' and accepted['schema']=='spnv2-table6-benchmark-cpu-verification-v1','Unknown benchmark schema');need(value['exitcode']==0 and side.read_text().strip()=='0','Benchmark failed exitcode')
 need(sha(root/'verify_benchmark.py')==accepted['verifier_sha256'],'Benchmark verifier code changed')
 need(Path(accepted['report']).resolve()==report and accepted['report_sha256']==sha(report) and accepted['exitcode_file_sha256']==sha(side),'Benchmark independent acceptance SHA mismatch')
 names=['table4_phi0_bn','phi3_bn_ehs','table4_phi3_gn','phi6_gn_ehs'];models={m['name']:m for m in accepted['models']};sources={m['name']:m for m in value['sources']};need([m['name'] for m in value['measurements']]==names and set(models)==set(sources)==set(names),'Wrong four model set')
 evidence={str(p):sha(p) for p in [report,proof,side]};items=[]
 for item in value['measurements']:
  name=item['name'];path=scope(item['path']);raw=read(path);source=sources[name];cpu=models[name];need(item['sha256']==sha(path)==cpu['raw_measurement_sha256'],'Benchmark raw SHA mismatch');evidence[str(path)]=sha(path)
  need(cpu['status']=='passed' and raw['name']==name and raw['source_final_sha256']==cpu['source_final_sha256']==source['final_sha256'],'Benchmark source mismatch')
  for field,hashfield in [('cfg','cfg_sha256'),('source_report','source_report_sha256'),('final','final_sha256'),('parameter_inventory','parameter_inventory_sha256')]:
   q=scope(source[field]);need(sha(q)==source[hashfield],'Benchmark source/cfg/parameter proof changed');evidence[str(q)]=sha(q)
  for domain,entry in source['domain_proofs'].items():q=scope(entry['path']);need(sha(q)==entry['sha256'],'Benchmark domain proof changed');evidence[str(q)]=sha(q)
  need(set(source['domain_proofs'])=={'synthetic','lightbox','sunlamp'},'Three-domain acceptance missing')
  need(raw['input_shape']==[1,3,512,768] and raw['input_kind']=='fixed_zero_tensor_cuda_resident' and raw['warmup_count']==10 and raw['forward_count']==40,'Wrong GPU benchmark workload')
  need(raw['dtype']=='torch.float32' and raw['eval_mode'] and raw['no_grad'] and raw['all_EHS_heads']==['heatmap','efficientpose','segmentation'] and raw['autocast'] is False,'Wrong forward modes/heads')
  need(raw['cudnn_benchmark'] is False and raw['deterministic_algorithms'] is True and raw['TF32'] is False and raw['timing_method']=='CUDA events with synchronize after every forward','Wrong timing/precision protocol')
  need(raw['strict_reload'] and raw['weights_unchanged'] and raw['finite_output_verified'] and raw['model_digest_before']==raw['model_digest_after'],'Benchmark changed model or outputs invalid')
  x=raw['latencies_ms'];need(len(x)==30 and all(finite(v) and float(v)>0 for v in x),'30 positive finite latency samples required');s=raw['statistics'];mean=float(np.mean(x));expected=dict(sample_count=30,mean_ms=mean,median_ms=float(np.median(x)),p05_ms=float(np.quantile(x,.05)),p90_ms=float(np.quantile(x,.90)),p95_ms=float(np.quantile(x,.95)),min_ms=min(x),max_ms=max(x),population_std_ms=float(np.std(x)),images_per_second=1000/mean)
  need(all(np.isclose(s[k],v,rtol=1e-12,atol=1e-12) for k,v in expected.items()) and s['percentile_method']=='linear interpolation at (n-1)*q','Benchmark statistics do not match raw samples');need(s==cpu['statistics'] and raw['parameter_counts']==cpu['parameter_counts']==source['parameter_counts'] and raw['memory_bytes']==cpu['memory_bytes'],'Benchmark CPU numeric acceptance mismatch')
  memory=raw['memory_bytes'];need(all(type(memory[k]) is int and memory[k]>=0 for k in ['baseline_allocated','baseline_reserved','peak_allocated','peak_reserved']) and memory['peak_allocated']>=memory['baseline_allocated']>0 and memory['peak_reserved']>=memory['baseline_reserved'] and memory['peak_reserved']>=memory['peak_allocated'],'Benchmark memory scope invalid')
  items.append(dict(name=name,source_sha256=raw['source_final_sha256'],statistics=s,memory_bytes=memory,input_shape=raw['input_shape'],raw_path=str(path),raw_sha256=sha(path)))
 return dict(status='accepted',measurements=items,device=value['device'],inputs=evidence,scope='Whole EHS float32 GPU forward, batch1, resident zero tensor, 10 warmup +30 measured; excludes preprocessing/loading/H2D/PnP; not end-to-end or ODR time')


def validate_seed_interface(bundle,curves):
 groups=read(Path(bundle['snapshot'])/'five_observation_seed_summary.json')
 index={(g['source_config'],g['source_sha256'],g['line'],g['domain'],g['N'],g['B']):g for g in groups}
 for c in curves:
  for point in c['points']:
   if point['status']!='ready':continue
   key=(c['config'],c['source_sha256'],c['line'],c['domain'],1024 if c['kind']=='Bscan' else point['x'],point['x'] if c['kind']=='Bscan' else 4);need(key in index,'Accepted five-stream group interface missing');g=index[key];v=g['metrics'][c['metric']]
   need(g['five_observation_seeds_complete'] and g['seeds']==list(range(2021,2026)) and np.isclose(v['mean'],point['mean']) and np.isclose(v['std_ddof1'],point['std_ddof1']),'Five-observation interface mean/SD mismatch')

def latency_math(values):
 need(len(values)==30 and all(finite(v) and float(v)>0 for v in values),'30 positive finite measured samples required')
 return dict(mean_ms=float(np.mean(values)),population_std_ms=float(np.std(values)),images_per_second=1000/float(np.mean(values)))


def audit_pair_numeric(data,before_ids,before_score,before_valid,after_ids,after_score,after_valid):
 need(before_ids==after_ids==[r['sample_id'] for r in data],'Plot paired stable IDs differ from actual MAT')
 for row,x,y,va,vb in zip(data,before_score,after_score,before_valid,after_valid):
  state='both_accepted' if va and vb else 'accepted_to_rejected' if va else 'rejected_to_accepted' if vb else 'both_rejected';need(row['state']==state,'Transition state differs from actual MAT')
  for field,value,valid in [('before_score',x,va),('after_score',y,vb)]:need((valid and finite(row[field]) and np.isclose(float(row[field]),value,rtol=1e-12)) or (not valid and row[field]==''),'Paired CSV score differs from actual MAT or missing value filled')
  need((va and vb and finite(row['delta']) and np.isclose(float(row['delta']),y-x,rtol=1e-12)) or (not (va and vb) and row['delta']==''),'Paired delta differs from actual MAT or missing value filled')
