"""Read-only evidence links and explicit denominators for final supplemental reports."""
import csv,hashlib,json,math,os,re,statistics
from pathlib import Path
P=Path(__file__).resolve().parents[1];D=P/'final_reporting';BASE=P.parents[2];DATA=Path('/home/DATA/Downloads/DataSet/speedplus').resolve()
UNITS={'eR':'deg','eT':'m','eR_HIL':'rad','eT_norm_HIL':'m/m','pose':'score'}
SERIES=[('original_E','eT'),('original_E','eR'),('original_E','pose'),('extended_H','eT'),('extended_H','eR'),('extended_H','pose'),('extended_hybrid','pose')]
_hashes={}
def need(value,message):
 if not value:raise ValueError(message)
def scoped(path):
 path=Path(path).resolve();need(path.is_relative_to(BASE) or path.is_relative_to(DATA),'Path outside own project/data: '+str(path));return path
def sha(path):
 path=scoped(path);st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
 if key not in _hashes:
  with path.open('rb') as f:_hashes[key]=hashlib.file_digest(f,'sha256').hexdigest()
 return _hashes[key]
def read(path):return json.loads(scoped(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
 path=scoped(path);path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
 if path.exists():need(path.read_text()==data,'Refuse overwriting different evidence: '+str(path))
 else:
  with path.open('x') as f:f.write(data)
def csv_write(path,rows,fields):
 with scoped(path).open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def guard():need(os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Use unique guarded CPU screen with hidden CUDA')
def run_id(value):need(isinstance(value,str) and re.fullmatch('[A-Za-z0-9_-]+',value),'Unsafe run ID');return value
def primary_field(mode):return {'H_rotation_E_translation':'extended_hybrid','H_rotation_H_translation':'extended_H','E_only_original':'original_E'}[mode]

def context_code_maps(context):
 output=[]
 for field,root in [('observer_code_sha256',P/'extended_evaluation'),('wrapper_code_sha256',P/'evaluation_entrypoints')]:
  mapping=context[field];need(mapping,'Empty context code release map');resolved={}
  for name,h in mapping.items():
   need(not Path(name).is_absolute(),'Context code release names must be relative')
   path=scoped(root/name);need(path.is_relative_to(root.resolve()),'Context code path escapes release');resolved[str(path)]=h
  output.append(resolved)
 return output

def context_assets(context):
 items=[context['configuration'],context['source_training_acceptance'],context['base_entry'],context['frozen_source_manifest'],context['observer_preparation_manifest'],*context['weights'].values()]
 if 'line_manifest' in context:items.append(context['line_manifest'])
 return items

def wrapper_link_binding(wrapper,link,base_path):
 need(scoped(wrapper['base_report'])==scoped(base_path),'Wrapper belongs to another base acceptance')
 expected={k:v for k,v in link.items() if k!='entrypoint_verification'}
 need(wrapper['extension']==expected,'Base extension pointer differs from same-forward wrapper proof')
 need(scoped(wrapper['report'])==scoped(link['report']['path']) and wrapper['report_sha256']==link['report']['sha256'],'Wrapper/report pointer differs')
 need(scoped(wrapper['independent_proof'])==scoped(link['verification']['path']) and wrapper['independent_proof_sha256']==link['verification']['sha256'],'Wrapper/independent proof pointer differs')
def finite_or_none(value):return value is None or (type(value) in (int,float) and math.isfinite(value))
def check_metric(value,total,original):
 need(type(value['count']) is int and 0<=value['count']<=total and value['total']==total and value['missing']==total-value['count'],'Invalid metric denominator')
 need(finite_or_none(value['mean']) and (value['mean'] is None)==(value['count']==0),'Missing metric assigned a number')
 need(math.isclose(value['coverage'],value['count']/total,rel_tol=1e-15,abs_tol=1e-15),'Coverage mismatch')
 scope='original_accepted_subset' if original else ('all_samples' if value['count']==total else 'numeric_pose_available_subset')
 need(value['denominator_scope']==scope,'Wrong all-samples/accepted-subset claim')

def snapshot_catalog(snapshot,allow_incomplete=False):
 snapshot=scoped(snapshot);need(snapshot.parent==(P/'analysis/runs').resolve(),'Use an actual analysis snapshot')
 manifest=read(snapshot/'artifact_manifest.json');proof=read(snapshot/'snapshot_acceptance.json');status=read(snapshot/'status.json');catalog=read(snapshot/'catalog.json')
 need(proof['status']=='passed' and proof['artifact_manifest_sha256']==sha(snapshot/'artifact_manifest.json'),'Analysis snapshot lacks independent artifact acceptance')
 for name,h in manifest['files_sha256'].items():need(sha(snapshot/name)==h,'Base snapshot artifact changed: '+name)
 need(len(catalog['records'])==430 and sum(r['stage']=='source' for r in catalog['records'])==30,'Full430-record catalog required')
 if not allow_incomplete:need(status['core_scope_complete'] and status['accepted_new_records']==430,'Final supplemental report requires all base results accepted')
 return snapshot,catalog,status,{str(snapshot/'artifact_manifest.json'):sha(snapshot/'artifact_manifest.json'),str(snapshot/'snapshot_acceptance.json'):sha(snapshot/'snapshot_acceptance.json')}

def checked_link(row,link,source_sha):
 need(link['schema']=='spnv2-evaluation-extension-link-v1','Unknown extension link')
 for key in ['report','verification']:
  need(sha(link[key]['path'])==link[key]['sha256'],'Extension link changed: '+key)
 report=read(link['report']['path']);proof=read(link['verification']['path'])
 need(report['schema']=='spnv2-extended-evaluation-v1' and proof['schema']=='spnv2-extended-evaluation-cpu-verification-v1' and proof['status']=='passed','Extension independent acceptance missing')
 need(proof['report_sha256']==link['report']['sha256'] and proof['report_path']==link['report']['path'],'Extension proof/report binding mismatch')
 need(report['sample_count']==proof['sample_count']==row['expected_total'] and report['domain']==row['domain'] and report['heads']==row['heads'],'Extension actual domain/head/total mismatch')
 need(report['context_sha256']==proof['context_sha256']==link['context_sha256']==digest(report['context']),'Extension context hash mismatch')
 context=report['context']
 need(context['schema']=='spnv2-evaluation-entrypoint-context-v1' and context['heads']==row['heads'],'Unknown captured context/head protocol')
 expected_applicability='applicable' if 'heatmap' in row['heads'] else 'not_applicable_no_heatmap'
 need(link['applicability']==expected_applicability,'Absent H head misrepresented as missing prediction')
 need(link['observer_status']==('complete' if 'heatmap' in row['heads'] else 'not_applicable'),'Wrong observer completion/applicability')
 need(scoped(context['configuration']['path'])==scoped(row['config']) and context['configuration']['sha256']==sha(row['config']),'Captured actual config mismatch')
 need(scoped(context['weights']['source_final']['path'])==scoped(row['source_final']) and context['weights']['source_final']['sha256']==source_sha,'Extension belongs to another source final')
 for item in context_assets(context):
  need(sha(item['path'])==item['sha256'],'Captured immutable input changed')
 need(scoped(context['source_training_acceptance']['path'])==scoped(row['source_report']),'Source training proof differs from catalog')
 need(context['wrapper_protocol_sha256']==sha(P/'evaluation_entrypoints/protocol_manifest.json'),'Wrapper release changed')
 for mapping in [*context_code_maps(context),report['observer_code_sha256'],proof['observer_code_sha256']]:
  need(mapping and all(sha(path)==h for path,h in mapping.items()),'Observer/wrapper code drift')
 for item in report['inputs'].values():need(sha(item['path'])==item['sha256'],'Extended CSV/capture/MAT input changed')
 need(scoped(report['inputs']['csv']['path'])==scoped(row['csv']) and scoped(report['inputs']['cfg']['path'])==scoped(row['config']),'Extension actual CSV/config differs from catalog')
 need(report['per_sample_sha256']==proof['per_sample_sha256']==sha(report['per_sample_path']),'Extended per-ID evidence changed')
 need(report['counts']==proof['counts'] and report['primary_mode']==proof['primary_mode'] and report['sample_id_sha256']==proof['sample_id_sha256'],'Independent extension totals/mode/IDs disagree')
 need(report['original_accepted_subset_unchanged'] is True,'Original accepted subset was modified')
 for label,protocol in report['protocols'].items():
  if not protocol['applicable']:need(protocol['metrics'] is None,'Absent head has fabricated metrics');continue
  need(set(protocol['metrics'])==set(UNITS),'Unexpected extended metric set')
  for metric,value in protocol['metrics'].items():
   need(value['unit']==UNITS[metric],'Wrong physical/unit scale');check_metric(value,row['expected_total'],label.startswith('original_accepted') or label=='original_H_E_intersection')
 return report

def inspect_record(row,records):
 base_path=Path(row['accepted_report'])
 if not base_path.exists():return dict(id=row['id'],status='pending',reason='base_evaluation_not_accepted')
 base=read(base_path)
 if base.get('status')!='passed':return dict(id=row['id'],status='pending',reason='base_evaluation_awaiting_verification')
 train=read(row['source_report']);need(train['status']=='passed' and train['final']==row['source_final'] and train['final_sha256']==sha(row['source_final']),'Independent source final changed')
 source_sha=train['final_sha256'];need(base['source_sha256']==source_sha,'Base evaluation source differs from fixed final')
 if row['stage']=='source':
  if 'extension' not in base:return dict(id=row['id'],status='pending',reason='same_forward_extension_missing')
  link=base['extension'];wrapper=read(link['entrypoint_verification']['path'])
  need(sha(link['entrypoint_verification']['path'])==link['entrypoint_verification']['sha256'] and wrapper['schema']=='spnv2-source-evaluation-extension-entrypoint-verification-v1' and wrapper['status']=='passed' and wrapper['source_sha256']==source_sha,'Source wrapper proof changed')
  need(wrapper['base_fields_sha256']==digest({k:v for k,v in base.items() if k!='extension'}),'Original source accepted fields drifted')
  wrapper_link_binding(wrapper,link,base_path)
  need(sha(wrapper['raw_report'])==wrapper['raw_report_sha256'],'Original evaluation raw report changed')
  if 'pending_report' in row:need(scoped(wrapper['raw_report'])==scoped(row['pending_report']),'Wrong raw source evaluation report')
  effective=row
 else:
  path=P/'evaluation_entrypoints/reports'/(row['id']+'.json')
  if not path.exists():return dict(id=row['id'],status='pending',reason='ODR_extension_not_accepted')
  wrapper=read(path);need(wrapper['schema']=='spnv2-odr-evaluation-extension-entrypoint-verification-v1' and wrapper['status']=='passed' and wrapper['task_id']==row['id'] and wrapper['source_sha256']==source_sha and wrapper['base_report_sha256']==sha(base_path),'ODR wrapper/base binding mismatch')
  link=wrapper['extension'];effective=records[row['canonical_id']]
  wrapper_link_binding(wrapper,link,base_path)
  if row.get('is_alias'):
   canonical_path=P/'evaluation_entrypoints/reports'/(row['canonical_id']+'.json');canonical=read(canonical_path)
   need(wrapper['canonical_task']==row['canonical_id'] and wrapper['canonical_proof']=={'path':str(canonical_path),'sha256':sha(canonical_path)},'Alias canonical extension proof changed')
   need(canonical['extension']==link,'Alias invented an independent extension')
   need(canonical['schema']=='spnv2-odr-evaluation-extension-entrypoint-verification-v1' and canonical['status']=='passed' and canonical['task_id']==row['canonical_id'] and canonical['source_sha256']==source_sha,'Alias canonical proof is not accepted/bound to source')
 report=checked_link(effective,link,source_sha)
 inputs={str(base_path):sha(base_path),row['source_report']:sha(row['source_report']),row['source_final']:source_sha}
 for item in [link['report'],link['verification']]:inputs[item['path']]=item['sha256']
 if row['stage']=='source':inputs[link['entrypoint_verification']['path']]=link['entrypoint_verification']['sha256']
 else:inputs[str(path)]=sha(path)
 for item in report['inputs'].values():inputs[item['path']]=item['sha256']
 context=report['context']
 for item in context_assets(context):inputs[item['path']]=item['sha256']
 for mapping in context_code_maps(context):inputs.update(mapping)
 inputs[str(P/'evaluation_entrypoints/protocol_manifest.json')]=context['wrapper_protocol_sha256']
 if row.get('is_alias'):inputs[str(canonical_path)]=sha(canonical_path)
 inputs[report['per_sample_path']]=report['per_sample_sha256']
 return dict(id=row['id'],status='accepted',source_sha256=source_sha,report_path=link['report']['path'],
  extension=link,primary_mode=report['primary_mode'],counts=report['counts'],protocols=report['protocols'],
  primary=report['extended_primary'],per_sample_path=report['per_sample_path'],inputs=inputs)

def per_rows(record):
 with scoped(record['per_sample_path']).open() as stream:
  for line in stream:
   row=json.loads(line)
   for key in ['extended_H','original_E','extended_hybrid']:
    if row[key] is not None:need(set(row[key])==set(UNITS) and all(finite_or_none(x) for x in row[key].values()),'Nonfinite or missing per-ID metric fields')
   yield row

def same_id_pairs(before,after,before_data=None):
 need(before['source_sha256']==after['source_sha256'],'Paired source final differs')
 left=list(per_rows(before)) if before_data is None else before_data;right=per_rows(after)
 for index,b in enumerate(left):
  try:a=next(right)
  except StopIteration:raise ValueError('Paired after stream truncated')
  need(b['index']==a['index']==index and b['sample_id']==a['sample_id'],'Same-ID pair order mismatch')
  x=b[primary_field(before['primary_mode'])]['pose'];y=a[primary_field(after['primary_mode'])]['pose']
  valid_x,valid_y=x is not None,y is not None
  state='both_numeric' if valid_x and valid_y else 'numeric_to_missing' if valid_x else 'missing_to_numeric' if valid_y else 'both_missing'
  original='both_accepted' if b['original_accepted'] and a['original_accepted'] else 'accepted_to_rejected' if b['original_accepted'] else 'rejected_to_accepted' if a['original_accepted'] else 'both_rejected'
  yield dict(index=index,sample_id=b['sample_id'],before_score=x,after_score=y,delta=y-x if valid_x and valid_y else None,numeric_state=state,original_acceptance_state=original)
 try:next(right)
 except StopIteration:return
 raise ValueError('Paired after stream has extra IDs')

def seed_groups(catalog,accepted):
 groups={}
 for row in catalog:
  if row['stage']!='odr' or row.get('is_alias'):continue
  key=tuple(row[k] for k in ['source_config','line','domain','N','B']);groups.setdefault(key,[]).append(row)
 output=[]
 for key,rows in sorted(groups.items()):
  rows=sorted(rows,key=lambda r:r['seed']);need([r['seed'] for r in rows]==list(range(2021,2026)),'Expected exactly five observation streams per protocol')
  ready=[accepted.get(r['id']) for r in rows];hashes={r['source_sha256'] for r in ready if r and r['status']=='accepted'};need(len(hashes)<=1,'Different source models mixed into five-seed summary')
  series={}
  for protocol,metric in SERIES:
   values=[];denominators=[]
   for result in ready:
    item=result['protocols'][protocol]['metrics'][metric] if result and result['status']=='accepted' else None
    values.append(item['mean'] if item else None);denominators.append(item['count'] if item else None)
   complete=all(x is not None for x in values)
   series[protocol+':'+metric]=dict(mean=statistics.mean(values) if complete else None,std_ddof1=statistics.stdev(values) if complete else None,
    unit=UNITS[metric],seed_means=values,denominators=denominators,status='ready' if complete else 'pending',
    all_samples=complete and all(d==r['expected_total'] for d,r in zip(denominators,rows)))
  output.append(dict(source_config=key[0],line=key[1],domain=key[2],N=key[3],B=key[4],source_sha256=next(iter(hashes)) if hashes else None,
   task_ids=[r['id'] for r in rows],seeds=list(range(2021,2026)),metrics=series,
   all_five_experiments_accepted=all(r and r['status']=='accepted' for r in ready),
   seed_meaning='Five ODR observation streams of one source training, not five source trainings'))
 need(len(output)==74,'Expected74 independently grouped ODR protocols')
 return output
