"""CPU-only final evidence matrix; never launches training/evaluation/ODR."""
import argparse,collections,csv,datetime,os,sys,traceback
from pathlib import Path
from core import *
from catalog import discover
from records import inspect_record,historical
from materials import audit_prisma
from cases import selections,render_case

def save_csv(path,rows,fields=None):
 fields=fields or sorted(set().union(*(set(r) for r in rows))) if rows else (fields or ['status'])
 with Path(path).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for row in rows:writer.writerow({k:(json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v) for k,v in row.items()})
def matrix_rows(records):
 output=[]
 for r in records:
  summary=r.get('loaded',{}).get('summary',{})
  for metric in metrics_for(r['heads']):
   output.append(dict(cohort=r['cohort'],source_config=r['source_config'],source_sha256=r.get('source_sha256'),domain=r['domain'],stage=r['stage'],line=r.get('line'),N=r.get('N'),B=r.get('B'),observation_seed=r.get('seed'),task_id=r['id'],canonical_id=r.get('canonical_id'),is_alias=r.get('is_alias',False),status=r['status'],head=head(metric,r['heads']),final_metric_formula=final_formula(r['heads']) if metric=='final_pose' else None,metric=metric,unit=unit(metric),accepted_mean=summary.get(metric),valid_denominator=summary.get('metric_denominators',{}).get(metric),n_samples=summary.get('n_samples'),rejections=summary.get('n_reject'),expected_total=r.get('expected_total'),reason=r['reason']))
 return output

def build(run_id,include_history=True,cases=False,case_limit=12,prisma_manifest=None):
 output=A/'runs'/run_id;output.mkdir(exist_ok=False);report={}
 try:
  catalog=discover();write(output/'catalog.json',catalog);pins_path=A/'source_pins.json';pins=read(pins_path) if pins_path.exists() else {};cache=dict(catalog_records=catalog['records'],loaded={});records=[];issues=[]
  for row in catalog['records']:
   try:r=inspect_record(row,pins,catalog['protocol'],cache)
   except EvidenceMissing as error:
    r=dict(row,status='待验收',reason='独立验收引用材料缺失: '+str(error),source_sha256=pins.get(row['source_config_path'],{}).get('sha256'));issues.append(dict(id=row['id'],status='待验收',reason=r['reason']))
   records.append(r)
  # Pin only after every existing accepted proof passed integrity checks.
  write(pins_path,pins);write(output/'source_pins_snapshot.json',pins)
  histories,history_limits=historical() if include_history else ([],[])
  matrix=matrix_rows(records);history_matrix=matrix_rows(histories)
  input_manifest=[]
  for evidence in records+histories:
   if evidence['status']!='已验收':continue
   loaded=evidence['loaded'];input_manifest.append(dict(cohort=evidence['cohort'],task_id=evidence['id'],canonical_id=evidence['canonical_id'],is_alias=evidence.get('is_alias',False),source_sha256=evidence['source_sha256'],acceptance_report_sha256=evidence['acceptance_sha256'],mat_sha256=loaded['mat_sha256'],csv=loaded['csv'],csv_current_sha256=sha(loaded['csv']),sample_ID_sequence_sha256=hashlib.sha256(json.dumps(loaded['ids'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))
  write(output/'input_evidence.json',input_manifest)
  save_csv(output/'matrix_new.csv',matrix);save_csv(output/'matrix_historical.csv',history_matrix)
  save_csv(output/'accepted_means.csv',[r for r in matrix if r['status']=='已验收'])
  sources={(r['source_config'],r['domain']):r for r in records if r['stage']=='source' and r['status']=='已验收'}
  pairs=[];pair_details={};seen=set()
  for r in records:
   if r['stage']!='odr' or r['status']!='已验收' or r.get('is_alias'):continue
   demand(r['canonical_id'] not in seen,'Independent canonical experiment duplicated');seen.add(r['canonical_id']);before=sources.get((r['source_config'],r['domain']))
   if before is None:issues.append(dict(id=r['id'],status='待验收',reason='对应新source全域验收缺失，不能借历史值配对'));continue
   stats,rows=paired(before['loaded'],r['loaded'],r['source_sha256'],before['source_sha256'],r['source_sha256'])
   item=dict(task_id=r['id'],canonical_id=r['canonical_id'],source_config=r['source_config'],source_sha256=r['source_sha256'],line=r['line'],domain=r['domain'],N=r['N'],B=r['B'],seed=r['seed'],**stats)
   pairs.append(item);pair_details[r['id']]=(before,r,rows)
   paired_dir=output/'paired';paired_dir.mkdir(exist_ok=True);save_csv(paired_dir/(r['id']+'.csv'),rows)
  save_csv(output/'paired_summary.csv',pairs);write(output/'paired_summary.json',pairs)
  groups=seed_statistics(records);write(output/'five_observation_seed_summary.json',groups)
  flat=[]
  for group in groups:
   for metric,data in group['metrics'].items():flat.append({k:v for k,v in group.items() if k!='metrics'}|dict(metric=metric,head=head(metric,group['heads']),**data))
  save_csv(output/'five_observation_seed_summary.csv',flat)
  case_report=dict(status='未完成',exported=[],limitations=['新source→ODR已接受配对尚不可用；不从历史模型借预测'])
  if cases and pair_details:
   pools=[]
   for task_id,(before,after,rows) in sorted(pair_details.items()):
    if after['group']=='fixed' and after['seed']==2021:pools.append((before,after,selections(rows)))
   exported=[];limits=[];case_dir=output/'cases';case_dir.mkdir()
   # Fixed final, seed2021 and lexicographic experiment order are predeclared;
   # sample choice illustrates paired error/state classes, never chooses source weights.
   for rank in range(6):
    for before,after,choices in pools:
     if len(exported)>=case_limit:break
     if rank>=len(choices):continue
     category,row=choices[rank];png=case_dir/(after['id']+'_'+category+'.png')
     try:
      geometry=after['loaded']['input_evidence']['geometry']
      item=render_case(before['loaded'],after['loaded'],row['sample_id'],png,geometry['camera']['path'],geometry['keypoints']['path'],DATA/after['domain']/'images_768x512_RGB')
      item.update(task_id=after['id'],source_sha256=after['source_sha256'],category=category,selection_basis=row,source_policy='固定final，不按target成绩选source；fixed ODR seed2021，配置ID字典序轮转覆盖')
      exported.append(item)
     except (EvidenceMissing,IntegrityError,KeyError,ValueError) as error:limits.append(dict(task_id=after['id'],sample_id=row['sample_id'],reason=repr(error)))
   case_report=dict(status='已验收' if exported and not limits else '受限',exported=exported,limitations=limits,raw_predicted_2D_keypoints_available=False,projection_only=True)
  elif pair_details:case_report=dict(status='待验收',exported=[],limitations=['需显式--cases执行CPU投影导出；不生成虚假案例'])
  write(output/'case_evidence.json',case_report)
  prisma=audit_prisma(prisma_manifest,pins);write(output/'prisma25_materials.json',prisma)
  states=dict(collections.Counter(r['status'] for r in records));allaccepted=len(records)==430 and all(r['status']=='已验收' for r in records)
  core_complete=allaccepted and len(seen)==370 and len(pairs)==370 and len(groups)==74 and all(g['five_observation_seeds_complete'] for g in groups) and case_report['status']=='已验收' and not issues
  complete=core_complete and prisma['complete']
  report=dict(status='已验收' if complete else '未完成',complete=complete,core_scope_complete=core_complete,scientific_scope_complete=complete,conditional_prisma_status=prisma['conditional_prisma_status'],verified_limitation=prisma['verified_limitation'],new_record_states=states,expected_source_evaluations=30,expected_ODR_requests=400,accepted_new_records=states.get('已验收',0),canonical_ODR_experiments=len(seen),paired_intersection_experiments=len(pairs),aliases_accepted=sum(r['status']=='已验收' and r.get('is_alias',False) for r in records),history_records_separate=len(histories),history_limits=history_limits,issues=issues,case_status=case_report['status'],prisma_status=prisma['status'],source_pins=pins,catalog_inputs=catalog['inputs'],engineering_acceptance=dict(ODR_CPU=read(PLAN/'odr/reports/integration_cpu_acceptance.json')['status'],ODR_CPU_report_sha256=sha(PLAN/'odr/reports/integration_cpu_acceptance.json')),clarifications=['各自accepted均值与严格同ID双有效交集配对均值分表；接受子集变化不称逐图改善','拒绝/缺失保留NaN与null，不填0或Inf；失败状态转移独立计数','final指标按真实分支定义：仅H为hR+hT，仅E为eR+eT，双头才为hR+eT；CSV保存具体公式','五种子仅ODR观测流随机性，不是五次source训练；A/B1/B2和历史source分开','PRISMA25或任何必要证据缺失时complete必须false'],created_at=datetime.datetime.now().astimezone().isoformat())
  write(output/'status.json',report)
  display=lambda value:'—' if value is None else str(value)
  pin_lines=['## 固定source final SHA256','','| 配置 | final SHA256 |','|---|---|']+[f'| {Path(k).stem} | {v["sha256"]} |' for k,v in pins.items()] if pins else ['## 固定source final SHA256','','尚无已验收新source final；不借历史权重或指标。']
  lines=['# SPNv2最终证据矩阵（实时材料快照）','',f'状态：{report["status"]}；核心范围完成：{str(core_complete).lower()}；含PRISMA科学范围完成：{str(complete).lower()}。',f'新证据状态：{states}；历史记录单独保存 {len(histories)} 项。','',*['- '+x for x in report['clarifications']],'',*pin_lines,'','## 新source完整矩阵','','| 配置 | 域 | 阶段 | 线 | N | B | seed | head | 指标 | 单位 | accepted均值 | 有效分母 | 拒绝数 | 实际总数 | 状态 |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for row in matrix:lines.append('| '+' | '.join(display(row.get(k)) for k in ['source_config','domain','stage','line','N','B','observation_seed','head','metric','unit','accepted_mean','valid_denominator','rejections','n_samples','status'])+' |')
  lines+=['','## 严格配对与五观测流种子','',f'已接受且完成严格ID交集比较：{len(pairs)} 个独立canonical实验；source/ODR各自accepted均值见accepted_means.csv，交集均值、分位数和失败转移见paired_summary.csv与逐图paired目录。',f'五观测流分组：{len(groups)}；未齐五种子的组不出五种子均值/标准差。','','## 历史结果（独立栏，禁止并入新source）','','| 历史配置 | 域 | 阶段 | seed | 指标 | 均值 | 分母 | 拒绝 |','|---|---|---|---|---|---|---|---|']
  for row in history_matrix:lines.append('| '+' | '.join(display(row.get(k)) for k in ['source_config','domain','stage','observation_seed','metric','accepted_mean','valid_denominator','rejections'])+' |')
  lines+=['','## 案例与PRISMA25','',f'姿态投影案例：{case_report["status"]}；原始预测2D关键点不在MAT中，不能声称已导出。',f'PRISMA25材料/离线hH：{prisma["status"]}；条件状态：{prisma["conditional_prisma_status"]}；核实受限：{prisma["verified_limitation"]}。材料缺项见prisma25_materials.json。','']
  (output/'证据汇总.md').write_text('\n'.join(lines))
  write(output/'artifact_manifest.json',dict(status=report['status'],complete=complete,files_sha256={str(f.relative_to(output)):sha(f) for f in sorted(output.rglob('*')) if f.is_file()},analysis_code_sha256={f.name:sha(f) for f in sorted(A.glob('*.py'))},read_only_inputs=True,no_GPU_execution=True))
  return report
 except BaseException as error:
  report=dict(status='受限',complete=False,error=repr(error),traceback=traceback.format_exc(),reason='证据缺损/哈希或ID不符时拒绝汇总为已接受结果')
  write(output/'failed_status.json',report);raise
if __name__=='__main__':
 assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run in unique guarded CPU screen'
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--cases',action='store_true');p.add_argument('--case-limit',type=int,default=12);p.add_argument('--no-history',action='store_true');p.add_argument('--prisma-manifest');a=p.parse_args()
 print(json.dumps(build(a.run_id,not a.no_history,a.cases,a.case_limit,a.prisma_manifest),ensure_ascii=False,allow_nan=False))
