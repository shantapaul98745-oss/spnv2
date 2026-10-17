"""PRISMA25 inventory restricted to configured own-project paths; no network access."""
from core import *

def audit_prisma(manifest_path=None,pins=None):
 candidates=[BASE/'materials/prisma25',BASE/'spnv2/prisma25',RUNS/'materials/prisma25',DATA/'prisma25',BASE/'data/prisma25']
 inventories=[]
 for path in candidates:
  if not path.exists():inventories.append(dict(path=str(path),exists=False));continue
  own(path);files=[f for f in path.rglob('*') if f.is_file()]
  inventories.append(dict(path=str(path),exists=True,images=[str(f) for f in files if f.suffix.lower() in ['.png','.jpg','.jpeg']],labels=[str(f) for f in files if f.suffix.lower() in ['.csv','.json'] and 'camera' not in f.name.lower()],camera=[str(f) for f in files if 'camera' in f.name.lower() and f.suffix=='.json']))
 result=dict(status='受限',material_complete=False,offline_hH_accepted=False,scope='仅项目自身材料路径；未联网、未访问其他项目',inventories=inventories,missing=['25张图像与标签逐ID对应的可验证清单','有效相机内参及畸变','已接受新source final的离线hH完整结果'],complete=False)
 result.update(verified_limitation=not any(x['exists'] for x in inventories),conditional_prisma_status='verified_limitation' if not any(x['exists'] for x in inventories) else 'pending_inventory',limitation_basis='五个指定项目内候选路径逐项核实；全部不存在' if not any(x['exists'] for x in inventories) else '存在候选目录，尚无完整逐ID材料清单；不可视为受限验收通过')
 if manifest_path is None:return result
 from PIL import Image
 m=read(required(manifest_path));demand(m['schema']=='project-prisma25-v1','Unknown PRISMA material schema')
 labels=read(required(m['labels']));camera=read(required(m['camera']));K=np.asarray(camera['cameraMatrix'],float);dist=np.asarray(camera['distCoeffs'],float)
 demand(K.shape==(3,3) and np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and dist.size==5 and np.isfinite(dist).all(),'Invalid PRISMA camera')
 ids=[r['sample_id'] for r in labels];demand(len(ids)==len(set(ids))==25,'PRISMA needs exactly25 unique labeled IDs')
 for row in labels:
  p=required(row['image']);Image.open(p).verify();q=np.asarray(row['quaternion'],float);t=np.asarray(row['translation'],float)
  demand(q.shape==(4,) and t.shape==(3,) and np.isfinite(q).all() and np.isfinite(t).all() and np.linalg.norm(q)>0,'Invalid PRISMA pose label')
  demand(row['image_sha256']==sha(p),'PRISMA image content hash mismatch')
 demand(m['labels_sha256']==sha(m['labels']) and m['camera_sha256']==sha(m['camera']),'PRISMA material SHA mismatch')
 result.update(status='待验收',verified_limitation=False,conditional_prisma_status='awaiting_offline_hH_acceptance',material_complete=True,missing=['新source离线hH独立验收结果'],material_manifest_sha256=sha(manifest_path))
 if m.get('accepted_hH_report'):
  proof=read(required(m['accepted_hH_report']));demand(proof['status']=='passed' and proof['source_sha256'] in {v['sha256'] for v in (pins or {}).values()},'PRISMA must use accepted pinned new source')
  loaded=load_results(proof['metrics']['directory'],m['csv'],['heatmap']);demand(loaded['ids']==ids and loaded['summary']==proof['metrics'],'PRISMA25 MAT/labels/acceptance mismatch')
  result.update(status='已验收',verified_limitation=False,conditional_prisma_status='accepted',offline_hH_accepted=True,missing=[],complete=True,source_sha256=proof['source_sha256'],metrics=loaded['summary'])
 return result
