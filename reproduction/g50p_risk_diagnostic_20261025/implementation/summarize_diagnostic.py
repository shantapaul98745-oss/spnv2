"""Read accepted paired results only; no model import or extra inference."""
import json,pathlib,hashlib,math,collections
from budget import atomic_json
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def checked(b):
 if sha(b['path'])!=b['sha256']:raise ValueError('binding changed: '+b['path'])
 return read(b['path'])
def weighted_ranking(scores,labels,weights):
 rows=[(s,y,w) for s,y,w in zip(scores,labels,weights) if s is not None and y is not None]
 pos=sum(w for s,y,w in rows if y);neg=sum(w for s,y,w in rows if not y)
 if not pos or not neg:return dict(AUROC=None,AUPRC=None,positive_weight=pos,negative_weight=neg,event_base_rate=pos/(pos+neg) if pos+neg else None,score_direction='higher means harmful',finite_windows=len(rows),missing_windows=len(scores)-len(rows),reason='missing or single class')
 auc=sum(w*v*(1 if s>t else .5 if s==t else 0) for s,y,w in rows if y for t,z,v in rows if not z)/(pos*neg)
 groups=collections.defaultdict(lambda:[0.,0.])
 for s,y,w in rows:groups[s][0 if y else 1]+=w
 tp=fp=ap=0.
 for score,(a,b) in sorted(groups.items(),reverse=True):tp+=a;fp+=b;ap+=(a/pos)*tp/(tp+fp)
 return dict(AUROC=auc,AUPRC=ap,event_base_rate=pos/(pos+neg),positive_weight=pos,negative_weight=neg,score_direction='higher means harmful',finite_windows=len(rows),missing_windows=len(scores)-len(rows))
def score_signals(target):
 names={'max_H_rotation_change':'h_rotation_change','max_HE_rotation_disagreement_increase':'he_disagreement_increase','max_all11_residual_increase':'reprojection_increase'}
 rows=target['protection']['rows'];result={}
 for name,key in names.items():
  values=[r['rules']['joint_geometry_worsening']['subconditions'][key]['value'] for r in rows]
  result[name]=max(values) if values and all(v is not None and math.isfinite(v) for v in values) else None
 return result
def execute(task_path,*,admission,launch_record,conflict_check,budget):
 task=read(task_path);idx=read(task['index_path']);frozen=checked(idx['frozen_inputs']);rec=checked(task['execution_receipts']);rows=[]
 counts={}
 for key,run in frozen['runs'].items():counts[key]=checked(run['original_report'])['counts']
 selection_counts=collections.Counter((w['domain']+':'+str(w['stream_seed']),w['original_decision']) for w in frozen['windows'])
 for entry in rec['runs']:
  budget.check_time('diagnostic metadata summary')
  rid=entry['run_id']
  if rid not in rec['accepted']:continue
  report=checked(entry['report']);proof=checked(rec['accepted'][rid]['proof']);rt=checked(entry['task'])
  if proof['status']!='passed' or proof['report_sha256']!=entry['report']['sha256']:raise ValueError('unaccepted pair in summary')
  sel=rt['selection'];key=sel['domain']+':'+str(sel['stream_seed']);decision=sel['original_decision'];n=counts[key]['commits' if decision=='commit' else 'rollbacks']
  changes=report['branch_results']['paired_summary'];images=report['branch_results']['per_image_changes']
  labels={k:None if changes[k] is None else changes[k]>0 for k in ['mean_error','tail_D']};labels['coverage']=any(x['accept_new_missing'] for x in images)
  rows.append(dict(run_id=rid,window_id=sel['window_id'],stream=key,original_decision=decision,weight=n/selection_counts[(key,decision)],population_in_stratum=n,selected_in_stratum=selection_counts[(key,decision)],paired_changes=changes,branch_summaries={b:report['branch_results'][b]['summary'] for b in ['accept','rollback']},new_missing_ids=[x['sample_id'] for x in images if x['accept_new_missing']],new_valid_ids=[x['sample_id'] for x in images if x['rollback_new_missing']],harm_labels=labels,continuous_signals=score_signals(report['original_target']),forward_categories=report['forward_categories'],attempt_costs=report['attempt_costs'],report=entry['report'],proof=rec['accepted'][rid]['proof']))
 completed={x['window_id'] for x in rows};missing=[dict(selection=w,reason='no independent accepted paired result; see actual attempts') for w in frozen['windows'] if w['window_id'] not in completed]
 diagnostics={}
 for key in frozen['runs']:
  subset=[x for x in rows if x['stream']==key];obj={}
  for metric in ['mean_error','tail_D','coverage']:
   raw=collections.Counter();weighted=collections.Counter()
   for x in subset:
    y=x['harm_labels'][metric];reject=x['original_decision']=='rollback';label='unknown' if y is None else 'true_reject' if reject and y else 'false_reject' if reject else 'missed_harm' if y else 'accepted_nonharm'
    raw[label]+=1;weighted[label]+=x['weight']
   obj[metric]=dict(raw_confusion=dict(raw),poststratified_confusion=dict(weighted),unobserved_selected_windows=8-len(subset),ranking={signal:weighted_ranking([x['continuous_signals'][signal] for x in subset],[x['harm_labels'][metric] for x in subset],[x['weight'] for x in subset]) for signal in ['max_H_rotation_change','max_HE_rotation_disagreement_increase','max_all11_residual_increase']})
  diagnostics[key]=obj
 out=pathlib.Path(task['output_dir']);out.mkdir(parents=True,exist_ok=True)
 result=dict(schema='g50p-risk-diagnostic-summary-v1',status='complete' if len(rows)==48 else 'partial',completed_windows=len(rows),maximum_windows=48,new_complete_ODR=0,A_report=str(out.parent/'A/report.json'),pairs=rows,not_completed=missing,by_stream=diagnostics,execution_receipts=task['execution_receipts'],budget=budget.snapshot(),queue_wait_seconds=rec['queue_wait_seconds'],interpretation='Fixed-window developer-panel diagnosis; mean error, tail7 and missingness separate. Poststratified descriptive counts use original window populations, not independent training repeats; partial strata are not renormalized. No fitted combined score or threshold search.',formal_efficacy_claim=False)
 sp=out/'summary.json';atomic_json(sp,result)
 lines=['# G50-P更新风险判据诊断 v1.0', '',f'独立验收窗口：{len(rows)}/48。新增完整ODR：0。','', 'A关联与失效模式见 ../A/REPORT.md；全部分支逐图结果、缺失、成本、父检查点和哈希见summary.json及对应pair proof。','', '|窗口|原决策|接受减回滚 mean_error|接受减回滚 tail7_D|新增缺失|','|---|---|---|---|---|']
 lines += [f"|{x['window_id']}|{x['original_decision']}|{x['paired_changes']['mean_error']}|{x['paired_changes']['tail_D']}|{len(x['new_missing_ids'])}|" for x in rows]
 lines += ['', '正值表示接受更新更差；None保留固定128分母失效。误拒绝与漏检按平均误差、尾部和覆盖分别统计，不能选择最有利标签。48窗口和128图像不是独立训练重复，不作正式显著性或泛化结论。', '', '## 逐流判据诊断', '', '|流|目标|拒绝且有害|拒绝但未见该目标损害|接受且有害（漏检）|接受且未见损害|未知|', '|---|---|---|---|---|---|---|']
 for key,obj in diagnostics.items():
  for metric,diag in obj.items():
   c=diag['raw_confusion'];lines.append(f"|{key}|{metric}|{c.get('true_reject',0)}|{c.get('false_reject',0)}|{c.get('missed_harm',0)}|{c.get('accepted_nonharm',0)}|{c.get('unknown',0)}|")
 lines += ['', '拒绝但面板未见损害的候选提供了误拒绝/过度拒绝的证据；接受且面板恶化的候选提供了漏检证据。三类目标分别判断，面板平均或尾部改善不抵消新增缺失。这里的未见损害包含零变化，不等价于已证明安全。', '', '原始计数、按原窗口群体比例还原的描述性权重、事件基率、预声明单信号AUROC/AUPRC和缺失率详见summary.json。固定位置的48个窗口并非随机独立训练重复，不能将后分层称为无偏总体估计。连续窗口信号只有所有所需图片都可计算时才取最大值，否则None；有限子集排序能力不能冒称全窗口识别率。', '', '结论限于这些固定窗口和已使用开发面板的更新影响。单次更新的配对变化可诊断判据失效，但不能单独证明跨流稳定性、长期轨迹收益或因果泛化优势。若正负事件不足、缺失或流间不一致，应报告证据不足，不再补有利窗口或调整规则。', '',f'未完成窗口数：{len(missing)}；不补有利样本，不改种子、规则或预算。', '', '运行中budget.snapshot含当前任务全额预占，不能当最终实际耗时。最终已结算有效时间/未知成本/前向上界以manager在实际汇总退出后写的final_manifest.json和预算日志为准；等待单列。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 proof=dict(schema='g50p-risk-diagnostic-summary-proof-v1',status='passed',completed_windows=len(rows),summary_sha256=sha(sp),model_forwards_added=0,all_completed_pairs_independently_accepted=True,complete_scope=len(rows)==48,execution_receipts=task['execution_receipts'])
 pp=out/'proof.json';atomic_json(pp,proof)
 return dict(status='diagnostic_delivered',exit_code=0,summary=str(sp),proof=str(pp),proof_sha256=sha(pp),model_forwards=0)
