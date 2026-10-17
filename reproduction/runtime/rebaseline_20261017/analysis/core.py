"""CPU evidence mathematics and immutable acceptance boundaries."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from scipy.io import loadmat
A=Path(__file__).resolve().parent;PLAN=A.parent;RUNS=PLAN.parents[1];BASE=RUNS.parent;DATA=Path('/home/DATA/Downloads/DataSet/speedplus')
class IntegrityError(RuntimeError):pass
class EvidenceMissing(RuntimeError):pass
HASHES={}
def sha(path):
 path=Path(path);st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
 if key not in HASHES:
  with path.open('rb') as f:HASHES[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return HASHES[key]
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def own(path):
 p=Path(path).resolve()
 if not any(p.is_relative_to(root.resolve()) for root in [BASE,DATA]):raise IntegrityError('Outside project/data evidence scope: '+str(p))
 return p
def required(path):
 p=own(path)
 if not p.is_file():raise EvidenceMissing(str(p))
 return p
def demand(condition,message):
 if not condition:raise IntegrityError(message)
def unit(metric):
 if 'eR_HIL' in metric:return 'rad'
 if 'eT_norm_HIL' in metric:return 'm/m'
 if metric=='final_pose':return 'score'
 if 'eR' in metric:return 'deg'
 if 'eT' in metric:return 'm'
 return 'dimensionless'
def metrics_for(heads):
 keys=[]
 for h in heads:keys+=['heat_eR','heat_eT','heat_eR_HIL','heat_eT_norm_HIL','heat_pose'] if h=='heatmap' else ['effi_iou','effi_eR','effi_eT','effi_eR_HIL','effi_eT_norm_HIL','effi_pose'] if h=='efficientpose' else []
 return keys+['final_pose']
def head(metric,heads=None):
 if metric.startswith('heat_'):return 'hH'
 if metric.startswith('effi_'):return 'hE'
 if heads==['heatmap']:return 'hH'
 if heads==['efficientpose']:return 'hE'
 return 'mixed_hR_plus_eT' if set(heads or [])=={'heatmap','efficientpose'} else 'unresolved'
def final_formula(heads):
 return 'hR_HIL+hT_norm_HIL' if heads==['heatmap'] else 'eR_HIL+eT_norm_HIL' if heads==['efficientpose'] else 'hR_HIL+eT_norm_HIL' if set(heads)=={'heatmap','efficientpose'} else 'unresolved'
def load_results(folder,csvpath,heads):
 from matio import verify
 folder=own(folder);csvpath=required(csvpath)
 paths={n:required(folder/(n+'.mat')) for n in ['mean_performance','per_sample_performance','predictions_pose']}
 try:summary=verify(folder,csvpath,heads)
 except (AssertionError,KeyError,ValueError) as e:raise IntegrityError('MAT physical-metric/ID/missing-prediction validation failed: '+str(folder)+' '+repr(e)) from e
 per=loadmat(paths['per_sample_performance'],simplify_cells=True);pred=loadmat(paths['predictions_pose'],simplify_cells=True)
 with csvpath.open() as f:ids=[r[0] for r in csv.reader(f)]
 demand(len(ids)==len(set(ids)),'Duplicate target sample ID')
 demand(list(np.atleast_1d(per['sample_id']))==list(np.atleast_1d(pred['sample_id']))==ids,'MAT/CSV IDs must match exact stable order; no positional guessing')
 return dict(summary=summary,ids=ids,per=per,pred=pred,mat_sha256={n:sha(p) for n,p in paths.items()},csv=str(csvpath),folder=str(folder),heads=heads)
def pin_source(config,final,proof,pins):
 required(final);r=read(required(proof));digest=sha(final)
 demand(r.get('cfg')==str(config),'Source acceptance configuration mismatch')
 demand(r['status']=='passed' and r['final']==str(final) and r['final_sha256']==digest,'Wrong/unaccepted source final hash/path')
 demand(r['code_sha256']==read(PLAN/'source_manifest.json')['core_tools_sha256'],'Source is not current frozen source')
 key=str(config)
 if key in pins:demand(pins[key]['sha256']==digest and pins[key]['final']==str(final),'Pinned final changed; target-dependent reselection forbidden')
 else:pins[key]=dict(final=str(final),sha256=digest,proof=str(proof),proof_sha256=sha(proof))
 return digest

def alias_identity(task,proof,canonical_task,canonical_proof,canonical_proof_path):
 demand(task.get('reuse_of')==canonical_task['id'] and proof.get('reuse_of')==canonical_task['id'],'Alias identity mismatch')
 demand(proof.get('canonical_report_sha256')==sha(required(canonical_proof_path)),'Alias canonical acceptance SHA missing/wrong')
 demand(Path(proof['canonical_report']).resolve()==Path(canonical_proof_path).resolve(),'Alias canonical path mismatch')
 for k in ['phi','line','domain','seed','N','B','source_final']:demand(task[k]==canonical_task[k],'Alias changes protocol field '+k)
 demand(proof['status']==canonical_proof['status']=='passed' and proof['source_sha256']==canonical_proof['source_sha256'] and proof['protocol']==canonical_proof['protocol'] and proof['implementation_manifest_sha256']==canonical_proof['implementation_manifest_sha256'],'Alias source/protocol/implementation mismatch')
 return canonical_task['id']

def paired(before,after,expected_source_sha,before_source_sha,after_source_sha):
 demand(expected_source_sha==before_source_sha==after_source_sha,'Paired comparison source SHA mismatch')
 demand(before['ids']==after['ids'],'Source/ODR stable IDs differ or are reordered; no implicit reorder')
 va=np.asarray(before['per']['final_pose_valid']).reshape(-1).astype(bool);vb=np.asarray(after['per']['final_pose_valid']).reshape(-1).astype(bool);common=va&vb
 x=np.asarray(before['per']['final_pose']).reshape(-1).astype(float);y=np.asarray(after['per']['final_pose']).reshape(-1).astype(float)
 demand(np.isfinite(x[va]).all() and np.isfinite(y[vb]).all() and np.isnan(x[~va]).all() and np.isnan(y[~vb]).all(),'Rejected predictions must remain NaN')
 delta=y[common]-x[common]
 stats=dict(n_total=len(x),n_common=int(common.sum()),accepted_to_rejected=int((va&~vb).sum()),rejected_to_accepted=int((~va&vb).sum()),both_rejected=int((~va&~vb).sum()),both_accepted=int(common.sum()),delta_mean=float(delta.mean()) if delta.size else None,delta_std_ddof1=float(delta.std(ddof=1)) if delta.size>1 else None,improved_count=int((delta<0).sum()),worsened_count=int((delta>0).sum()),unchanged_count=int((delta==0).sum()),quantiles={str(q):float(np.quantile(delta,q)) if delta.size else None for q in [0,.05,.25,.5,.75,.95,1]},before_common_mean=float(x[common].mean()) if delta.size else None,after_common_mean=float(y[common].mean()) if delta.size else None)
 distributions={}
 for metric in before['summary']['metric_denominators']:
  if metric not in after['per']:continue
  bx=np.asarray(before['per'][metric],dtype=float).reshape(-1);ay=np.asarray(after['per'][metric],dtype=float).reshape(-1);mask=np.isfinite(bx)&np.isfinite(ay);d=ay[mask]-bx[mask];higher=metric.endswith('iou')
  distributions[metric]=dict(unit=unit(metric),higher_is_better=higher,n_common=int(mask.sum()),delta_mean=float(d.mean()) if d.size else None,quantiles={str(q):float(np.quantile(d,q)) if d.size else None for q in [.05,.25,.5,.75,.95]},improved=int(((d>0) if higher else (d<0)).sum()),worsened=int(((d<0) if higher else (d>0)).sum()),unchanged=int((d==0).sum()))
 stats['metric_distributions']=distributions
 stats.update(improved_fraction=float((delta<0).mean()) if delta.size else None,worsened_fraction=float((delta>0).mean()) if delta.size else None)
 rows=[]
 for i,name in enumerate(before['ids']):rows.append(dict(sample_id=name,before_valid=bool(va[i]),after_valid=bool(vb[i]),before_score=float(x[i]) if va[i] else None,after_score=float(y[i]) if vb[i] else None,delta=float(y[i]-x[i]) if common[i] else None,state='both_accepted' if common[i] else 'accepted_to_rejected' if va[i] else 'rejected_to_accepted' if vb[i] else 'both_rejected'))
 return stats,rows

def seed_statistics(records):
 groups={}
 for r in records:
  if r['stage']!='odr' or r['cohort']!='new' or r['status']!='已验收':continue
  key=(r['source_config'],r['source_sha256'],r['line'],r['domain'],r['N'],r['B'])
  bucket=groups.setdefault(key,{})
  seed=r['seed']
  if seed in bucket:demand(bucket[seed]['canonical_id']==r['canonical_id'] and (r.get('is_alias') or bucket[seed].get('is_alias')),'Duplicate independent run/alias contamination in seed group')
  else:bucket[seed]=r
 output=[]
 for key,bucket in groups.items():
  seeds=sorted(bucket);complete=seeds==list(range(2021,2026));allkeys=sorted(set().union(*(set(r['loaded']['summary']['metric_denominators']) for r in bucket.values())))
  values={}
  for metric in allkeys:
   x=[r['loaded']['summary'][metric] for r in bucket.values()];valid=[v for v in x if v is not None]
   values[metric]=dict(mean=float(np.mean(valid)) if complete and len(valid)==5 else None,std_ddof1=float(np.std(valid,ddof=1)) if complete and len(valid)==5 else None,valid_seed_count=len(valid),unit=unit(metric),denominators=[bucket[s]['loaded']['summary']['metric_denominators'][metric] for s in seeds])
  output.append(dict(source_config=key[0],source_sha256=key[1],line=key[2],domain=key[3],N=key[4],B=key[5],seeds=seeds,canonical_experiments=len(bucket),five_observation_seeds_complete=complete,seed_meaning='同一source训练模型的五个ODR观测流种子，非五次source训练',heads=bucket[seeds[0]]['loaded']['heads'],metrics=values))
 return output


def verify_geometry_binding(path,expected_sha,expected_csv):
 demand(sha(required(path))==expected_sha,'Accepted ODR input evidence changed')
 evidence=read(path);demand(Path(evidence['csv']['path']).resolve()==Path(expected_csv).resolve() and evidence['csv']['sha256']==sha(required(expected_csv)),'Target CSV changed since accepted ODR')
 demand(evidence['raw_image_content_hashed'] is False,'Unexpected image-content hash claim')
 for geometry in evidence['geometry'].values():demand(sha(required(geometry['path']))==geometry['sha256'],'Camera/keypoint geometry changed since accepted ODR')
 return evidence
