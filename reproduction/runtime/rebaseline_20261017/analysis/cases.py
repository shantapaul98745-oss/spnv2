"""Illustrative CPU pose projections from actual accepted predictions only."""
from core import *
from projection import project_keypoints

def project_matrix(R,t,K,dist,points):
 # EfficientPose R is the same body attitude convention as quat2dcm(q).
 xyz=np.asarray(R).T@np.asarray(points).T+np.asarray(t).reshape(3,1);x=xyz[0]/xyz[2];y=xyz[1]/xyz[2];r2=x*x+y*y
 radial=1+dist[0]*r2+dist[1]*r2*r2+dist[4]*r2*r2*r2
 xx=x*radial+2*dist[2]*x*y+dist[3]*(r2+2*x*x);yy=y*radial+dist[2]*(r2+2*y*y)+2*dist[3]*x*y
 return np.vstack([K[0,0]*xx+K[0,2],K[1,1]*yy+K[1,2]])

def selections(rows):
 common=[r for r in rows if r['delta'] is not None];out=[]
 if common:
  negative=[r for r in common if r['delta']<0];positive=[r for r in common if r['delta']>0]
  if negative:out.append(('largest_improvement',min(negative,key=lambda r:(r['delta'],r['sample_id']))))
  if positive:out.append(('largest_degradation',max(positive,key=lambda r:(r['delta'],r['sample_id']))))
  med=np.median([r['delta'] for r in common]);out.append(('median_common_delta',min(common,key=lambda r:(abs(r['delta']-med),r['sample_id']))))
 for state in ['accepted_to_rejected','rejected_to_accepted','both_rejected']:
  candidates=[r for r in rows if r['state']==state]
  if candidates:out.append((state,min(candidates,key=lambda r:r['sample_id'])))
 return out

def render_case(before,after,sample,output,camera_path,keypoints_path,image_root):
 from PIL import Image
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 camera=read(required(camera_path));K=np.asarray(camera['cameraMatrix'],float);dist=np.asarray(camera['distCoeffs'],float);points=np.asarray(loadmat(required(keypoints_path))['tango3Dpoints'],float).T
 demand(before['ids']==after['ids'],'Case ID order mismatch');i=before['ids'].index(sample);imagepath=required(Path(image_root)/sample)
 if 'input_evidence' in after:
  evidence=after['input_evidence'];item=evidence['image_records'][i];stat=imagepath.stat()
  demand(item[1]==sample and Path(item[2]).resolve()==imagepath.resolve() and (item[3],item[4])==(stat.st_size,stat.st_mtime_ns),'Case image metadata differs from accepted ODR input')
  demand(sha(camera_path)==evidence['geometry']['camera']['sha256'] and sha(keypoints_path)==evidence['geometry']['keypoints']['sha256'],'Case geometry differs from accepted ODR input')
 image=np.asarray(Image.open(imagepath).convert('RGB'));h,w=image.shape[:2]
 sx=w/float(camera['Nu']);sy=h/float(camera['Nv']);fig,axes=plt.subplots(1,2,figsize=(12,4.5));present=[]
 for ax,record,title in zip(axes,[before,after],['Pinned source final','ODR adapted']):
  ax.imshow(image);pred=record['pred'];drawn=[]
  if all(k in pred and np.asarray(pred[k]).ndim>0 and np.asarray(pred[k]).shape[0]>i for k in ['heat_q','heat_t']) and np.isfinite(np.asarray(pred['heat_q'])[i]).all() and np.isfinite(np.asarray(pred['heat_t'])[i]).all():
   xy=project_keypoints(np.asarray(pred['heat_q'])[i],np.asarray(pred['heat_t'])[i],K,dist,points);ax.scatter(xy[0]*sx,xy[1]*sy,s=18,marker='o',facecolors='none',edgecolors='#e15759',label='hH pose projection');drawn.append('heatmap_pose_projection')
  if all(k in pred and np.asarray(pred[k]).ndim>0 and np.asarray(pred[k]).shape[0]>i for k in ['effi_R','effi_t']) and np.isfinite(np.asarray(pred['effi_R'])[i]).all() and np.isfinite(np.asarray(pred['effi_t'])[i]).all():
   xy=project_matrix(np.asarray(pred['effi_R'])[i],np.asarray(pred['effi_t'])[i],K,dist,points);ax.scatter(xy[0]*sx,xy[1]*sy,s=16,marker='x',c='#4e79a7',label='hE pose projection');drawn.append('efficientpose_pose_projection')
  with Path(record['csv']).open() as f:label=list(csv.reader(f))[i]
  if len(label)>=36:
   gt=np.asarray(label[14:],float).reshape(-1,2);ax.scatter(gt[:,0]*w,gt[:,1]*h,s=10,c='#59a14f',label='labeled 2D keypoint')
  ax.set_title(title);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
  if drawn:ax.legend(fontsize=7,loc='upper left')
  present.append(drawn)
 demand(any(present),'No finite actual pose predictions available; do not fabricate keypoints')
 fig.suptitle(sample+' | pose-derived projection; raw predicted 2D keypoints not exported');fig.tight_layout();fig.savefig(output,dpi=150);plt.close(fig)
 return dict(sample_id=sample,image=str(imagepath),image_sha256=sha(imagepath),camera_sha256=sha(camera_path),keypoints_sha256=sha(keypoints_path),predictions_drawn=present,png=str(output),png_sha256=sha(output),limitation='实际MAT不含原始预测2D关键点；图中为真实姿态预测投影，非补造热图检测')
