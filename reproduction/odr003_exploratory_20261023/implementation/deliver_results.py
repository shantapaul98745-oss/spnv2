"""Export the accepted exploratory result without model forward or reserved GT."""
import csv,hashlib,json,math,os,statistics,sys
from pathlib import Path
N=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
METHODS=('O','A','V','E50','E75','G50','G75')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with Path(p).open() as f:return [json.loads(x) for x in f]
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(x):return statistics.mean(x)
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,x))+' |' for x in data])
def fmt(x):return 'None' if x is None else f'{x:.9f}'
def main():
 assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 source=N/'summaries/base_20261023/summary.json';s=read(source);proof=read(source.with_name('proof.json'))
 assert proof['status']=='passed' and proof['summary_sha256']==sha(source) and proof['adaptation_runs']==36
 assert s['selection']['run_random_controls'] is False
 out=N/'delivery/base_20261023';out.mkdir(parents=True,exist_ok=False)
 data=[];run_reports={}
 for ref in s['inputs']:
  d=Path(ref['path']);r=read(d/'report.json');run_reports[(r['domain'],r['stream'],r['method'])]=(d,r)
  assert sha(d/'report.json')==ref['report_sha256'] and sha(d/'proof.json')==ref['proof_sha256']
  data.append(dict(domain=r['domain'],stream=r['stream'],method=r['method'],observed=r['counts']['observed'],accepted=r['counts']['accepted'],used=r['counts']['used'],updates=r['counts']['updates'],tail=r['counts']['tail'],developer_count=512,extended_valid=r['extended_valid'],original_rejected=r['original_rejected'],wall_seconds=r['wall_seconds'],scoring_seconds=r['timings']['scoring_seconds'],scoring_forward_seconds=r['timings']['scoring_forward_seconds'],scoring_geometry_seconds=r['timings']['scoring_geometry_seconds'],update_seconds=r['timings']['update_seconds'],evaluation_seconds=r['timings']['evaluation_seconds'],peak_allocated_GiB=r['peak_gpu_allocated_bytes']/2**30,peak_reserved_GiB=r['peak_gpu_reserved_bytes']/2**30,quality_forwards=r['forward_counts']['quality'],accepted_update_forwards=r['forward_counts']['accepted_update'],evaluation_forwards=r['forward_counts']['developer_evaluation'],report_path=str(d/'report.json'),proof_path=str(d/'proof.json')))
 with (out/'run_resources_and_coverage.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 compact=[];per_stream=[];original=[];resource=[]
 for domain,part in s['domains'].items():
  for m in METHODS:
   a=part['method_summaries'][m];v=a['metrics'];raw=[x for x in data if x['domain']==domain and x['method']==m]
   compact.append([domain,m,fmt(v['mean_error']['mean']),fmt(v['mean_rotation_deg']['mean']),fmt(v['mean_translation_m']['mean']),fmt(v['mean_positive_degradation']['mean']),fmt(v['worst_5pct_degradation']['mean']),fmt(v['negative_transfer_rate']['mean'])])
   resource.append([domain,m,f'{mean([x["wall_seconds"] for x in raw]):.2f}',f'{mean([x["scoring_seconds"] for x in raw]):.2f}',f'{mean([x["scoring_geometry_seconds"] for x in raw]):.2f}',f'{mean([x["used"] for x in raw]):.2f}',f'{max(x["peak_allocated_GiB"] for x in raw):.3f}',f'{max(x["peak_reserved_GiB"] for x in raw):.3f}'])
   for stream,p in part['streams'].items():
    r=p['methods'][m];v=r['metrics'];per_stream.append(dict(domain=domain,stream=stream,method=m,reference_count=p['reference_count'],tail_count=r['tail_count'],**v,coverage_pass=r['coverage']['pass']))
    directory=run_reports[(domain,'offline' if m=='O' else stream,m)][0]
    predictions=rows(directory/'predictions.jsonl');valid=[x for x in predictions if x['original_valid']]
    extension=[x for x in predictions if x['valid']]
    original.append([domain,stream,m,len(valid),sum(x['original_h_rejected'] for x in predictions),fmt(mean([x['original_error'] for x in valid])) if valid else 'None',len(extension),fmt(mean([x['error'] for x in extension])) if extension else 'None'])
 with (out/'paired_stream_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(per_stream[0]));w.writeheader();w.writerows(per_stream)
 with (out/'accepted_subset_vs_extended.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['domain','stream','method','original_valid','original_H_rejected','original_subset_hybrid_mean','extended_valid','extended_full_valid_hybrid_mean']);w.writerows(original)
 # Extreme examples are diagnostics selected after the complete exploration.
 cases=[]
 for domain,order in [('lightbox',False),('sunlamp',True)]:
  candidates=[]
  for stream,part in s['domains'][domain]['streams'].items():
   for r in part['methods']['G50']['per_sample']:
    if r['in_reference'] and r['valid']:candidates.append(dict(domain=domain,stream=stream,**r))
  candidates.sort(key=lambda x:x['delta_vs_a'],reverse=order);seen=set()
  for r in candidates:
   if r['sample_id'] in seen:continue
   seen.add(r['sample_id']);evidence={}
   for method in ('O','A','G50'):
    directory=run_reports[(domain,'offline' if method=='O' else r['stream'],method)][0]
    pr=next(x for x in rows(directory/'predictions.jsonl') if x['sample_id']==r['sample_id'])
    evidence[method]=dict(path=str(directory/'predictions.jsonl'),sha256=sha(directory/'predictions.jsonl'),record=pr)
   cases.append(dict(selection='largest_G50_minus_A_error' if order else 'smallest_G50_minus_A_error',**r,evidence=evidence))
   if len(seen)==2:break
 dump(out/'case_evidence.json',dict(selection='posthoc_extreme_diagnostics_not_representative_or_causal',cases=cases))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(2,3,figsize=(12,7))
 for row,domain in enumerate(('lightbox','sunlamp')):
  for col,key in enumerate(('mean_error','mean_positive_degradation','worst_5pct_degradation')):
   ax=axes[row,col]
   for j,method in enumerate(('A','V','E50','E75','G50','G75')):
    vals=s['domains'][domain]['method_summaries'][method]['metrics'][key]['values']
    ax.scatter([j]*3,vals,s=20,alpha=.75);ax.scatter(j,mean(vals),marker='_',s=250,c='black')
   ax.set_xticks(range(6),('A','V','E50','E75','G50','G75'));ax.set_title(domain+' / '+key.replace('_',' '));ax.grid(axis='y',alpha=.25)
 fig.tight_layout();fig.savefig(out/'paired_risk_plot.png',dpi=180);plt.close(fig)
 # Developer-only projection panels use saved predictions, not another forward.
 import cv2
 sys.path.insert(0,str(N.parent/'rebaseline_20261017/odr/source_A/core'))
 from utils.postprocess import project_keypoints
 from utils.utils import load_camera_intrinsics,load_tango_3d_keypoints
 from scipy.spatial.transform import Rotation
 receipt=read(N/'preflight/official_source_receipt.json');geometry=read(receipt['source_core_manifest_path'])
 camera=load_camera_intrinsics(geometry['camera_path']);points=load_tango_3d_keypoints(geometry['keypoint_path'])
 dev_labels={domain:{x['sample_id']:x for x in read(N/'data'/domain/'dev_labels.json')['items']} for domain in ('lightbox','sunlamp')}
 fig,axes=plt.subplots(len(cases),3,figsize=(12,3.4*len(cases)))
 for row,case in enumerate(cases):
  domain=case['domain'];sid=case['sample_id'];assert sid in dev_labels[domain]
  image=cv2.imread('/home/DATA/Downloads/DataSet/speedplus/'+domain+'/images_768x512_RGB/'+sid.split('/')[-1]);image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  gt=np.asarray(dev_labels[domain][sid]['keypoints_normalized'])*[768,512]
  for col,method in enumerate(('O','A','G50')):
   ax=axes[row,col];pr=case['evidence'][method]['record'];ax.imshow(image);ax.scatter(gt[:,0],gt[:,1],s=20,c='lime',marker='+',label='GT keypoints')
   if pr['h_available']:
    xy=project_keypoints(np.asarray(pr['heat_q']),np.asarray(pr['heat_t']),camera['cameraMatrix'],camera['distCoeffs'],points).T*[768/1920,512/1200]
    ax.scatter(xy[:,0],xy[:,1],s=12,facecolors='none',edgecolors='cyan',label='H pose projection')
   if pr['e_available']:
    q=Rotation.from_matrix(np.asarray(pr['effi_R']).T).as_quat()[[3,0,1,2]]
    xy=project_keypoints(q,np.asarray(pr['effi_t']),camera['cameraMatrix'],camera['distCoeffs'],points).T*[768/1920,512/1200]
    ax.scatter(xy[:,0],xy[:,1],s=10,c='magenta',marker='x',label='E pose projection')
   ax.set_xlim(0,768);ax.set_ylim(512,0);ax.set_title(f'{method} {sid} / s{case["stream"]}\nE={pr["error"]:.5f}');ax.axis('off')
   if row==0 and col==0:ax.legend(fontsize=7,loc='upper left')
 fig.tight_layout();fig.savefig(out/'projection_cases.png',dpi=160);plt.close(fig)
 text=['# 003几何门控：首轮探索结果','',
 '36次基础ODR、两域Offline开发评价和共享校准全部通过独立验收。三条观测流等权；本报告只使用已冻结开发GT，没有正式确认或预留集/全域评价。',
 '', '结论：没有一档共享G阈值同时满足两域的预设风险方向，因此不执行条件性18次随机对照。G50在lightbox平均正退化降低21.93%、最严重5%退化均值降低30.19%；sunlamp平均正退化降低2.68%，但最严重5%退化均值增加3.96%。G75在sunlamp平均正退化增加10.21%、严重退化增加15.51%。两档G平均精度与覆盖约束均通过，未通过的是风险方向条件。',
 '', '## 口径与有效分母','',
 '每域开发512张；固定参考I_ref由O/A可用预测确定，lightbox各流512、sunlamp各流511；最大5%取ceil(0.05*n)=26个D（包含零），D=max(E方法−EO,0)。混合E=hH旋转HIL项（rad）+hE归一化平移HIL项（m/m），为无量纲；另列旋转度数和平移米。',
 'sunlamp的O仅511/512有有限H预测，适应方法为512/512；O在程序中相对A的coverage=False不属于方法筛选失败。该无效ID未进入任何方法的固定风险参考集，未用E替代H。原接受子集与完整有限预测另表保存，不能混为论文全域口径。',
 '', '## 三流均值对照','',table(['域','方法','混合E','hH旋转(deg)','hE平移(m)','平均正退化D','最大5% D','负迁移率'],compact),
 '', '![三条配对流与均值](paired_risk_plot.png)','',
 '全部设置均报告，没有挑最好分支替代原A。数值是探索性描述，不作显著性或有效性结论。熵对照sunlamp平均误差有下降，但正退化与严重退化仍高于A，说明平均误差与风险目标不同。',
 'G50在lightbox三流的平均正退化和尾部退化均下降，但负迁移比例从54.818%到55.078%（增加0.260个百分点）；损害幅度下降不等于受损样本比例下降。sunlamp的G50尾部退化相对A逐流差为+0.044335、−0.001498、−0.019276，均值+0.007854，描述性标准差0.032820。不能将后两流改善代替跨流汇总。',
 'V六次都接受全部1024图像，更新量与A相同，因此V与A的微小数值差不能解释为合法性过滤收益。',
 '', '## 原接受子集与扩展完整有限预测','',
 '完整逐流表见 accepted_subset_vs_extended.csv，明确原有效数、H拒绝数、各自混合均值及扩展有效数。风险与精度筛选使用上文固定I_ref，而非每组自行缩小分母。',
 '', '## 实际资源开销','',table(['域','方法','端到端秒均值','评分秒均值','几何秒均值','实际更新用图均值','最大allocated GiB','最大reserved GiB'],resource),
 '', '端到端包含资产预检、加载、适应、checkpoint与开发评价。评分分项包括事务隔离；forward/几何是嵌套分项，不与总耗时重复相加。A/R无质量评分前向；E仅S评分，V为HE/PnP，G为完整几何。计时如实保留首次编译、序列顺序和峰值缓存影响，没有正式资源统计推断。逐次数据见run_resources_and_coverage.csv。',
 '', '## 改善与退化案例','',
 '从完整探索后按G50−A误差差选取两个lightbox改善和两个sunlamp退化的极端诊断样例；它们不代表总体，也不证明单张更新的因果作用。case_evidence.json保存ID、流、各方法逐样本预测记录及文件SHA，图像只来自开发集。',
 '', '![仅开发集的GT与H/E姿态投影](projection_cases.png)','',
 '## 限制与下一步边界','',
 '三流不足以正式确认。开发GT参与了规则选择；预留图像允许无标签参与适应，隔离的是GT选型。图像相似分组不能保证采集独立。官方权重结果与自己的独立训练结果分开；开发子集成绩不能直接当成论文全域差距。没有执行R，无法排除少更新的解释，也不能声称几何质量具有独立贡献。',
 '本轮没有达到预设的两域风险改善方向。到此停止新增实验，后续消融/规则修改/正式确认须另定方案。原150轮训练和完整扫描继续作为可选扩展未完成记录，不自动恢复。',
 '', '## 证据索引','',f'源summary SHA256: {sha(source)}',f'独立proof SHA256: {sha(source.with_name("proof.json"))}',f'官方source SHA256: {s["source_sha256"]}',
 '完整指标与逐样本证据：summaries/base_20261023/summary.json、proof.json以及38份runs/*/report.json、proof.json、predictions.jsonl/risk_rows.jsonl。无效值保留None。']
 (out/'REPORT.md').write_text('\n'.join(text)+'\n')
 manifest={x.name:dict(sha256=sha(x),bytes=x.stat().st_size) for x in out.iterdir() if x.is_file()}
 dump(out/'delivery_manifest.json',dict(status='complete',source_summary_sha256=sha(source),source_proof_sha256=sha(source.with_name('proof.json')),base_runs=36,random_runs=0,formal_confirmation=False,selection=s['selection'],files=manifest))
 print(json.dumps(dict(status='complete',directory=str(out),files=len(manifest),model_forward_passes=0,reserved_GT_read=False)))
if __name__=='__main__':main()
