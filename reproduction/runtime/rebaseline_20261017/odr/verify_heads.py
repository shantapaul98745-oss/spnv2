"""Validate H-only, E-only or H+E full-length exports; Table2 branch metrics."""
import json,csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def verify(path,csvpath,heads):
 path=Path(path)
 with open(csvpath) as f:ids=[r[0] for r in csv.reader(f)]
 n=len(ids);m=loadmat(path/'mean_performance.mat',simplify_cells=True);v=loadmat(path/'per_sample_performance.mat',simplify_cells=True);p=loadmat(path/'predictions_pose.mat',simplify_cells=True)
 required={'final_pose'}
 if 'heatmap' in heads:required.update({'heat_eR','heat_eT','heat_eR_HIL','heat_eT_norm_HIL','heat_pose'})
 if 'efficientpose' in heads:required.update({'effi_iou','effi_eR','effi_eT','effi_eR_HIL','effi_eT_norm_HIL','effi_pose'})
 assert required.issubset(v) and required.issubset(m),'Missing required physical metrics'
 assert int(m['n_samples'])==int(v['n_samples'])==n
 assert list(np.atleast_1d(v['sample_id']))==list(np.atleast_1d(p['sample_id']))==ids
 accepted=~np.asarray(v['h_reject'],dtype=bool).reshape(-1) if 'heatmap' in heads else np.ones(n,dtype=bool)
 assert len(accepted)==n,'Invalid accepted-pose mask'
 assert np.array_equal(np.asarray(v['final_pose_valid']).reshape(-1).astype(bool),accepted)
 result={}
 prefixes=['heat_' if h=='heatmap' else 'effi_' for h in heads]
 for k in v:
  if not any(k.startswith(x) for x in prefixes) and k!='final_pose':continue
  ar=np.asarray(v[k]).reshape(-1);mask=accepted if k.startswith('heat_') or k=='final_pose' else np.ones(n,dtype=bool)
  assert ar.size==n and np.isfinite(ar[mask]).all() and np.isnan(ar[~mask]).all(),k
  if mask.any():
   assert np.isclose(ar[mask].mean(dtype=np.float64),float(m[k]),rtol=2e-5,atol=1e-7),k
  else:
   assert np.isnan(float(m[k])),k
  result[k]=float(m[k]) if mask.any() else None
 rotation='heat_eR_HIL' if 'heatmap' in heads else 'effi_eR_HIL'
 translation='effi_eT_norm_HIL' if 'efficientpose' in heads else 'heat_eT_norm_HIL'
 assert np.allclose(np.asarray(v['final_pose'])[accepted],(np.asarray(v[rotation])+np.asarray(v[translation]))[accepted])
 shapes={'heatmap':{'heat_q':(n,4),'heat_t':(n,3)},'efficientpose':{'effi_R':(n,3,3),'effi_t':(n,3),'bbox':(n,4)}}
 for h in heads:
  mask=accepted if h=='heatmap' else np.ones(n,dtype=bool)
  for key,shape in shapes[h].items():
   ar=np.asarray(p[key]);assert ar.shape==shape and np.isfinite(ar[mask]).all() and np.isnan(ar[~mask]).all(),key
 result.update(metric_denominators={k:int((accepted if k.startswith('heat_') or k=='final_pose' else np.ones(n,dtype=bool)).sum()) for k in result},n_samples=n,n_reject=int((~accepted).sum()) if 'heatmap' in heads else None,coverage=float(accepted.mean()),heads=heads,directory=str(path))
 (path/'verified_heads_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return result
