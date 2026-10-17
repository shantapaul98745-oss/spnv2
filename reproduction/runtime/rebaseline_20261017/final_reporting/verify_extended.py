"""Independent CPU reduction of every linked supplemental per-image result."""
import argparse,collections,csv,itertools,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_reporting.evidence import P,D,UNITS,SERIES,need,sha,read,digest,write,scoped,guard,run_id,snapshot_catalog,inspect_record,per_rows

PROTOCOLS=(('original_accepted_H','extended_H','heatmap',True),('original_E','original_E','efficientpose',False),
 ('original_H_E_intersection','extended_hybrid','both',True),('extended_H','extended_H','heatmap',False),('extended_hybrid','extended_hybrid','both',False))

def equivalent(actual,expected,label):
 if isinstance(expected,dict):
  need(isinstance(actual,dict) and set(actual)==set(expected),'Object keys differ: '+label)
  for key,value in expected.items():equivalent(actual[key],value,label+'/'+str(key))
 elif isinstance(expected,list):
  need(isinstance(actual,list) and len(actual)==len(expected),'List length differs: '+label)
  for index,(a,b) in enumerate(zip(actual,expected)):equivalent(a,b,label+'/'+str(index))
 elif type(expected) is float:
  need(type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=2e-12,abs_tol=2e-12),'Numeric reduction differs: '+label)
 else:need(type(actual) is type(expected) and actual==expected,'Value/type differs: '+label)

def reduce_samples(row,result):
 """Rebuild denominators independently; existing proof separately checks pose math."""
 h='heatmap' in row['heads'];e='efficientpose' in row['heads'];total=row['expected_total']
 with scoped(row['csv']).open(newline='') as stream:ids=[r[0] for r in csv.reader(stream)]
 need(len(ids)==total and len(set(ids))==total,'Actual complete CSV/unique ID count changed')
 buckets={(label,metric):[] for label,key,head,subset in PROTOCOLS for metric in UNITS}
 counters=collections.Counter();mode='H_rotation_E_translation' if h and e else 'H_rotation_H_translation' if h else 'E_only_original'
 count=0
 for i,item in enumerate(per_rows(result)):
  need(i<total and type(item['index']) is int and item['index']==i and item['sample_id']==ids[i],'Per-image order/ID/count mismatch')
  need(type(item['original_accepted']) is bool,'Original accepted flag must be Boolean')
  accepted=item['original_accepted'];H=item['extended_H'];E=item['original_E'];hybrid=item['extended_hybrid']
  need((H is not None)==h and (E is not None)==e and (hybrid is not None)==(h and e),'Absent head has fabricated per-image metrics')
  flags=dict(original_accepted=accepted,H_numeric_rotation=h and H['eR'] is not None,H_numeric_translation=h and H['eT'] is not None,
   H_numeric_pose=h and H['pose'] is not None,H_E_intersection=h and e and hybrid['pose'] is not None,
   geometry_rejected_numeric_H=h and not accepted and H['pose'] is not None)
  for flag,value in flags.items():
   need(type(item[flag]) is bool and item[flag] is value,'Per-image availability flag mismatch: '+flag);counters[flag]+=int(value)
  need(item['original_reject'] is (not accepted) if h else item['original_reject'] is None,'Original reject/accepted flags disagree')
  if h and accepted:need(all(x is not None for x in H.values()),'Accepted H has missing numerical prediction')
  if e:need(all(x is not None for x in E.values()),'Original E has missing numerical prediction')
  if accepted and flags['H_E_intersection']:counters['original_accepted_H_E_intersection']+=1
  for label,key,head,subset in PROTOCOLS:
   applicable=(h and e) if head=='both' else h if head=='heatmap' else e
   if not applicable or (subset and not accepted):continue
   for metric,value in item[key].items():
    if value is not None:
     need(value>=0,'Negative physical/error metric');buckets[label,metric].append(value)
  count+=1
 need(count==total,'Per-image evidence truncated')
 protocols={}
 for label,key,head,subset in PROTOCOLS:
  applicable=(h and e) if head=='both' else h if head=='heatmap' else e
  if not applicable:protocols[label]=dict(applicable=False,reason='required_head_not_present',metrics=None);continue
  metrics={}
  for metric,unit in UNITS.items():
   values=buckets[label,metric];n=len(values)
   metrics[metric]=dict(mean=math.fsum(values)/n if n else None,count=n,total=total,missing=total-n,coverage=n/total,
    denominator_scope='original_accepted_subset' if subset else 'all_samples' if n==total else 'numeric_pose_available_subset',mask=label+':'+metric,unit=unit)
  protocols[label]=dict(applicable=True,metrics=metrics)
 counts={key:counters[key] for key in ['original_accepted','H_numeric_rotation','H_numeric_translation','H_numeric_pose','H_E_intersection','geometry_rejected_numeric_H','original_accepted_H_E_intersection']}
 counts.update(total=total,missing_H_rotation=total-counters['H_numeric_rotation'] if h else None,missing_H_translation=total-counters['H_numeric_translation'] if h else None,heatmap_applicable=h,efficientpose_applicable=e)
 equivalent(result['protocols'],protocols,'protocols');equivalent(result['counts'],counts,'counts');need(result['primary_mode']==mode,'Primary H/E mode changed')
 primary=protocols['extended_hybrid' if h and e else 'extended_H' if h else 'original_E'];equivalent(result['primary'],primary,'primary')
 raw=read(result['report_path']);need(raw['sample_id_sha256']==digest(ids),'Actual CSV/extended ID hash mismatch')
 return protocols

def primary_key(result):
 return {'H_rotation_E_translation':'extended_hybrid','H_rotation_H_translation':'extended_H','E_only_original':'original_E'}[result['primary_mode']]

def verify_pairs(row,before,after,path,cached_before=None):
 need(before['source_sha256']==after['source_sha256'],'Before/after belong to different source training')
 values=[];numeric=collections.Counter();original=collections.Counter();count=0;before_values=[];after_values=[]
 left=list(per_rows(before)) if cached_before is None else cached_before
 sentinel=object()
 with scoped(path).open(newline='') as f:
  actual=csv.DictReader(f);need(actual.fieldnames==['index','sample_id','before_score','after_score','delta','numeric_state','original_acceptance_state'],'Unexpected paired CSV columns')
  for index,(b,a,csv_row) in enumerate(itertools.zip_longest(left,per_rows(after),actual,fillvalue=sentinel)):
   need(b is not sentinel and a is not sentinel and csv_row is not sentinel,'Paired CSV or per-image stream length differs')
   need(b['index']==a['index']==index and b['sample_id']==a['sample_id'],'Paired IDs/order differ')
   x=b[primary_key(before)]['pose'];y=a[primary_key(after)]['pose'];delta=None if x is None or y is None else y-x
   ns='both_numeric' if x is not None and y is not None else 'numeric_to_missing' if x is not None else 'missing_to_numeric' if y is not None else 'both_missing'
   bs=b['original_accepted'];aa=a['original_accepted'];os='both_accepted' if bs and aa else 'accepted_to_rejected' if bs else 'rejected_to_accepted' if aa else 'both_rejected'
   expected=dict(index=index,sample_id=b['sample_id'],before_score=x,after_score=y,delta=delta,numeric_state=ns,original_acceptance_state=os)
   need(csv_row=={k:'' if v is None else str(v) for k,v in expected.items()},'Paired CSV altered numerical or failure-state evidence')
   numeric[ns]+=1;original[os]+=1;count+=1
   if delta is not None:values.append(delta);before_values.append(x);after_values.append(y)
 need(count==row['expected_total'],'Paired domain denominator incomplete')
 n=len(values)
 return dict(task_id=row['id'],source_sha256=after['source_sha256'],N=row['N'],B=row['B'],seed=row['seed'],line=row['line'],domain=row['domain'],source_config=row['source_config'],
  total=count,common_numeric=n,numeric_transitions=dict(numeric),original_acceptance_transitions=dict(original),
  before_intersection_mean=math.fsum(before_values)/n if n else None,after_intersection_mean=math.fsum(after_values)/n if n else None,mean_delta=math.fsum(values)/n if n else None,
  improved=sum(v<0 for v in values),degraded=sum(v>0 for v in values),unchanged=sum(v==0 for v in values),paired_csv=str(path),paired_csv_sha256=sha(path),
  scope='Same-ID intersection of raw-numeric primary scores; original geometric-rejection transitions reported separately')

def verify_seed_groups(catalog,results,actual):
 groups=collections.defaultdict(list)
 for r in catalog:
  if r['stage']=='odr' and not r.get('is_alias'):groups[tuple(r[k] for k in ['source_config','line','domain','N','B'])].append(r)
 rebuilt=[]
 for key,items in sorted(groups.items()):
  items=sorted(items,key=lambda x:x['seed']);need([x['seed'] for x in items]==[2021,2022,2023,2024,2025],'Five observation seeds missing/duplicated')
  ready=[results[x['id']] for x in items];all_ready=all(x['status']=='accepted' for x in ready);hashes={x['source_sha256'] for x in ready if x['status']=='accepted'};need(len(hashes)<=1,'Five streams mix source trainings')
  metrics={}
  for protocol,metric in SERIES:
   series=[x['protocols'][protocol]['metrics'][metric] if x['status']=='accepted' else None for x in ready]
   values=[v['mean'] if v else None for v in series];denoms=[v['count'] if v else None for v in series];finite=all(v is not None for v in values)
   metrics[protocol+':'+metric]=dict(mean=float(np.mean(values,dtype=np.float64)) if finite else None,std_ddof1=float(np.std(values,dtype=np.float64,ddof=1)) if finite else None,
    unit=UNITS[metric],seed_means=values,denominators=denoms,status='ready' if finite else 'pending',all_samples=finite and all(n==r['expected_total'] for n,r in zip(denoms,items)))
  rebuilt.append(dict(source_config=key[0],line=key[1],domain=key[2],N=key[3],B=key[4],source_sha256=next(iter(hashes)) if hashes else None,task_ids=[r['id'] for r in items],
   seeds=[2021,2022,2023,2024,2025],metrics=metrics,all_five_experiments_accepted=all_ready,seed_meaning='Five ODR observation streams of one source training, not five source trainings'))
 need(len(rebuilt)==74,'Expected74 canonical five-stream groups');equivalent(actual,rebuilt,'five_stream_reduction');return rebuilt

def verify(identifier,allow_incomplete=False):
 guard();run_id(identifier);out=D/'runs'/identifier;manifest_path=out/'artifact_manifest.json';manifest=read(manifest_path)
 need(manifest['schema']=='spnv2-final-extended-artifacts-v1' and manifest['no_GPU_execution'] is True,'Unknown supplemental artifact scope')
 expected_files=set(manifest['files_sha256'])
 actual_files={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}-{'artifact_manifest.json','extended_acceptance.json'}
 need(actual_files==expected_files,'Supplemental artifact set differs')
 for name,h in manifest['files_sha256'].items():need(sha(out/name)==h,'Supplemental artifact changed: '+name)
 for path,h in manifest['implementation_sha256'].items():need(sha(path)==h,'Supplemental implementation drift')
 im=read(out/'input_manifest.json')
 for path,h in im['inputs'].items():need(sha(path)==h,'Supplemental original input changed')
 snapshot,catalog,base_status,inputs=snapshot_catalog(im['snapshot'],allow_incomplete);equivalent(read(out/'catalog.json'),catalog,'actual_catalog')
 catalog_rows=catalog['records'];records={r['id']:r for r in catalog_rows};need(len(records)==430,'Duplicate catalog identifiers')
 results={r['id']:inspect_record(r,records) for r in catalog_rows};equivalent(read(out/'records.json'),results,'independent_reinspected_links')
 for result in results.values():
  if result['status']=='accepted':
   for path,h in result['inputs'].items():need(path not in inputs or inputs[path]==h,'Input changed during reinspection');inputs[path]=h
 equivalent(im,dict(snapshot=str(snapshot),inputs=inputs,base_core_complete=base_status['core_scope_complete']),'input_manifest')
 matrix=[]
 for row in catalog_rows:
  result=results[row['id']]
  shared=dict(task_id=row['id'],stage=row['stage'],source_config=row['source_config'],domain=row['domain'],line=row.get('line'),N=row.get('N'),B=row.get('B'),seed=row.get('seed'),
   canonical_id=row.get('canonical_id',row['id']),is_alias=row.get('is_alias',False),status=result['status'],source_sha256=result.get('source_sha256'))
  if result['status']!='accepted':matrix.append(dict(shared,protocol=None,metric=None,unit=None,mean=None,count=None,total=row['expected_total'],missing=None,denominator_scope=None));continue
  reduce_samples(row,result);protocols=result['protocols']
  for label,protocol in protocols.items():
   if protocol['applicable']:
    for metric,value in protocol['metrics'].items():matrix.append(dict(shared,protocol=label,metric=metric,**value))
 with (out/'extended_matrix.csv').open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;actual=list(reader)
 need(fields==['task_id','stage','source_config','domain','line','N','B','seed','canonical_id','is_alias','status','source_sha256','protocol','metric','unit','mean','count','total','missing','coverage','denominator_scope','mask'],'Supplemental matrix column/units contract changed')
 need(actual==[{k:'' if item.get(k) is None else str(item[k]) for k in fields} for item in matrix],'Supplemental matrix/count/units differ')
 source_rows={(r['source_config'],r['domain']):r for r in catalog_rows if r['stage']=='source'};pairs=[];cache={}
 for row in catalog_rows:
  if row['stage']!='odr' or row.get('is_alias') or results[row['id']]['status']!='accepted':continue
  before=results[source_rows[row['source_config'],row['domain']]['id']]
  if before['status']!='accepted':continue
  if before['id'] not in cache:cache[before['id']]=list(per_rows(before))
  pairs.append(verify_pairs(row,before,results[row['id']],out/'paired'/(row['id']+'.csv'),cache[before['id']]))
 equivalent(read(out/'paired_summary.json'),pairs,'paired_summary')
 groups=verify_seed_groups(catalog_rows,results,read(out/'five_observation_seed_summary.json'))
 accepted=sum(r['status']=='accepted' for r in results.values());aliases=sum(r.get('is_alias',False) and results[r['id']]['status']=='accepted' for r in catalog_rows)
 complete=accepted==430 and aliases==30 and len(pairs)==370 and all(x['all_five_experiments_accepted'] for x in groups)
 limits=[dict(task_id=key,missing=value['primary']['metrics']['pose']['missing'],total=value['primary']['metrics']['pose']['total']) for key,value in results.items() if value['status']=='accepted' and value['primary']['metrics']['pose']['missing']]
 status=read(out/'status.json');need(status['schema']=='spnv2-final-extended-evidence-v1' and status['executed_scope_complete'] is complete,'Supplemental completeness overstated')
 for key,value in dict(accepted_records=accepted,expected_records=430,accepted_source_records=sum(r['stage']=='source' and results[r['id']]['status']=='accepted' for r in catalog_rows),canonical_ODR_pairs=len(pairs),aliases_accepted=aliases,five_observation_groups=len(groups),numeric_coverage_limits=limits).items():equivalent(status[key],value,key)
 need(status['status']==('complete_with_missing_predictions' if complete and limits else 'complete' if complete else 'pending'),'Incorrect completed/pending scientific status')
 need(status['original_accepted_subset_preserved'] is True and status['source_training_seed']==2021,'Original protocol/seed changed')
 if not allow_incomplete:need(complete,'Final supplemental verification requires all430 actual experiments, pairs and aliases')
 proof=dict(schema='spnv2-final-extended-acceptance-v1',status='passed',run_id=identifier,executed_scope_complete=complete,
  artifact_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),inputs=inputs,verified_numeric_limitations=limits,
  accepted_records=accepted,accepted_source_records=status['accepted_source_records'],canonical_ODR_pairs=len(pairs),aliases_accepted=aliases,five_observation_groups=len(groups),
  independent_reductions=['actual CSV IDs','430 evidence bindings','per-image means/counts/units','370 same-ID intersections and failure transitions','74 five-observation-stream groups; sample std ddof1'],engineering_allow_incomplete=allow_incomplete)
 write(out/'extended_acceptance.json',proof);return proof

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
 print(json.dumps(verify(args.run_id,args.allow_incomplete),ensure_ascii=False))
