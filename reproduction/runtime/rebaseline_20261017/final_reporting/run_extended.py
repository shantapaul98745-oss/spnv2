"""Supplementary raw-numeric evidence; never overwrite original accepted-subset results."""
import argparse,collections,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_reporting.evidence import *

def merge(target,other):
 for key,value in other.items():need(key not in target or target[key]==value,'Input changed during collection: '+key);target[key]=value

def build(snapshot,identifier,allow_incomplete=False):
 guard();run_id(identifier);snapshot,catalog,base_status,inputs=snapshot_catalog(snapshot,allow_incomplete)
 out=D/'runs'/identifier;out.mkdir(parents=True,exist_ok=False);records={r['id']:r for r in catalog['records']};need(len(records)==430,'Duplicate matrix record identity')
 results={};matrix=[]
 for row in catalog['records']:
  result=inspect_record(row,records);results[row['id']]=result
  shared=dict(task_id=row['id'],stage=row['stage'],source_config=row['source_config'],domain=row['domain'],line=row.get('line'),N=row.get('N'),B=row.get('B'),seed=row.get('seed'),
   canonical_id=row.get('canonical_id',row['id']),is_alias=row.get('is_alias',False),status=result['status'],source_sha256=result.get('source_sha256'))
  if result['status']!='accepted':matrix.append(dict(shared,protocol=None,metric=None,unit=None,mean=None,count=None,total=row['expected_total'],missing=None,denominator_scope=None));continue
  merge(inputs,result['inputs'])
  for label,protocol in result['protocols'].items():
   if not protocol['applicable']:continue
   for metric,value in protocol['metrics'].items():matrix.append(dict(shared,protocol=label,metric=metric,**value))
 fields=['task_id','stage','source_config','domain','line','N','B','seed','canonical_id','is_alias','status','source_sha256','protocol','metric','unit','mean','count','total','missing','coverage','denominator_scope','mask']
 csv_write(out/'extended_matrix.csv',matrix,fields)
 write(out/'catalog.json',catalog);write(out/'records.json',results)
 pairs=[];pair_dir=out/'paired';pair_dir.mkdir();source_rows={(r['source_config'],r['domain']):r for r in records.values() if r['stage']=='source'};before_cache={}
 for row in records.values():
  if row['stage']!='odr' or row.get('is_alias') or results[row['id']]['status']!='accepted':continue
  before_row=source_rows[row['source_config'],row['domain']];before=results[before_row['id']];after=results[row['id']]
  if before['status']!='accepted':continue
  if before_row['id'] not in before_cache:before_cache[before_row['id']]=list(per_rows(before))
  per=list(same_id_pairs(before,after,before_cache[before_row['id']]))
  need(len(per)==row['expected_total'],'Paired full-domain count mismatch')
  path=pair_dir/(row['id']+'.csv');csv_write(path,per,['index','sample_id','before_score','after_score','delta','numeric_state','original_acceptance_state'])
  numeric=[p for p in per if p['delta'] is not None];deltas=[p['delta'] for p in numeric]
  pairs.append(dict(task_id=row['id'],source_sha256=after['source_sha256'],N=row['N'],B=row['B'],seed=row['seed'],line=row['line'],domain=row['domain'],source_config=row['source_config'],
   total=len(per),common_numeric=len(numeric),numeric_transitions=dict(collections.Counter(p['numeric_state'] for p in per)),original_acceptance_transitions=dict(collections.Counter(p['original_acceptance_state'] for p in per)),
   before_intersection_mean=statistics.mean(p['before_score'] for p in numeric) if numeric else None,after_intersection_mean=statistics.mean(p['after_score'] for p in numeric) if numeric else None,
   mean_delta=statistics.mean(deltas) if deltas else None,improved=sum(x<0 for x in deltas),degraded=sum(x>0 for x in deltas),unchanged=sum(x==0 for x in deltas),
   paired_csv=str(path),paired_csv_sha256=sha(path),scope='Same-ID intersection of raw-numeric primary scores; original geometric-rejection transitions reported separately'))
 groups=seed_groups(catalog['records'],results)
 accepted=sum(r['status']=='accepted' for r in results.values());aliases=sum(r.get('is_alias',False) and results[r['id']]['status']=='accepted' for r in records.values())
 complete=accepted==430 and aliases==30 and len(pairs)==370 and len(groups)==74 and all(g['all_five_experiments_accepted'] for g in groups)
 limited=[dict(task_id=key,missing=value['primary']['metrics']['pose']['missing'],total=value['primary']['metrics']['pose']['total']) for key,value in results.items() if value['status']=='accepted' and value['primary']['metrics']['pose']['missing']]
 status=dict(schema='spnv2-final-extended-evidence-v1',status='complete_with_missing_predictions' if complete and limited else 'complete' if complete else 'pending',
  executed_scope_complete=complete,accepted_records=accepted,expected_records=430,accepted_source_records=sum(r['stage']=='source' and results[r['id']]['status']=='accepted' for r in records.values()),
  canonical_ODR_pairs=len(pairs),aliases_accepted=aliases,five_observation_groups=len(groups),numeric_coverage_limits=limited,
  zero_valid_metric_behavior='None, never zero or Inf; absence of applicable H distinguished from missing H',
  original_accepted_subset_preserved=True,source_training_seed=2021,seed_meaning='Five ODR observation streams of each fixed source final, not five source trainings',
  scope='Supplementary numerical H before geometry rejection; all_samples only for actual N/N metric coverage; original accepted-subset reports remain separate')
 write(out/'paired_summary.json',pairs);write(out/'five_observation_seed_summary.json',groups);write(out/'status.json',status)
 write(out/'input_manifest.json',dict(snapshot=str(snapshot),inputs=inputs,base_core_complete=base_status['core_scope_complete']))
 lines=['# 扩展评价证据汇总','',f'正式记录：{accepted}/430；独立ODR配对：{len(pairs)}/370；别名：{aliases}/30。',
  f'运行证据齐全：{complete}；原接受子集保持不变。','',
  'H几何拒绝前仍有限的姿态单独统计；每个指标只有N/N才标全部样本。没有数值的预测保留缺失，不用E替代H，不填零。',
  '逐图改善只比较同ID双方均有数值的交集；原几何拒绝状态转移另列。五个种子表示观测流，不表示五次source训练。','',
  f'主指标仍缺数值的记录数：{len(limited)}。完整逐项分母见extended_matrix.csv；没有训练或评价的条目保持pending。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 write(out/'artifact_manifest.json',dict(schema='spnv2-final-extended-artifacts-v1',files_sha256={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()},
  implementation_sha256={str(D/name):sha(D/name) for name in ['evidence.py','run_extended.py','verify_extended.py']},no_GPU_execution=True))
 if not allow_incomplete:need(complete,'Supplementary execution scope incomplete; preserve output and block downstream')
 return status

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--run-id',required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
 print(json.dumps(build(a.snapshot,a.run_id,a.allow_incomplete),ensure_ascii=False))
