"""Load accepted new evidence and separately label historical own-run references."""
from core import *

def bind_source(row,pins):return pin_source(Path(row['source_config_path']),Path(row['source_final']),Path(row['source_report']),pins)
def inspect_record(row,pins,protocol,cache):
 r=dict(row);r.update(status='未完成',reason='尚无已验收结果',source_sha256=None,canonical_id=r.get('canonical_id',r['id']),is_alias=r.get('is_alias',False))
 if not Path(r['source_report']).is_file():
  if Path(r['source_final']).is_file():r.update(status='待验收',reason='final文件存在但source训练验收缺失')
  return r
 digest=bind_source(r,pins);r['source_sha256']=digest
 if not Path(r['accepted_report']).is_file():
  if Path(r['pending_report']).is_file() or Path(r['folder']).exists():r.update(status='待验收',reason='存在运行/结果材料，尚无独立验收')
  return r
 proof=read(required(r['accepted_report']));demand(proof.get('status')=='passed','Existing acceptance is not passed: '+r['accepted_report']);demand(proof['source_sha256']==digest,'Accepted source/ODR final SHA mismatch')
 if r['stage']=='source':
  raw=read(required(proof['source_report']));demand(raw['source_sha256']==digest and raw['source']==r['source_final'],'Evaluation does not use pinned source final')
  demand(Path(raw['csv']).resolve()==Path(r['csv']).resolve() and raw['domain']==r['domain'],'Evaluation CSV/domain mismatch')
  r['folder']=raw['results'];loaded=load_results(r['folder'],r['csv'],r['heads'])
  demand(proof['metrics']==loaded['summary'],'Source accepted metrics differ from 3MAT revalidation')
 else:
  demand(proof.get('task_id')==r['id'] and proof['protocol']==protocol,'ODR task/protocol mismatch')
  demand(proof['implementation_manifest_sha256']==sha(PLAN/'odr/implementation_manifest.json'),'ODR implementation not current accepted line')
  task=r['task']
  if r['is_alias']:
   target=next(x for x in cache['catalog_records'] if x['id']==task['reuse_of']);targetproof=read(required(target['accepted_report']));alias_identity(task,proof,target['task'],targetproof,target['accepted_report']);canonical=inspect_record(target,pins,protocol,cache)
   demand(canonical['status']=='已验收','Alias canonical not independently accepted')
   r.update(status='已验收',reason='已验真canonical复用，不新增独立实验',loaded=canonical['loaded'],canonical_report_sha256=sha(target['accepted_report']),acceptance_sha256=sha(r['accepted_report']));return r
  if r['id'] in cache['loaded']:loaded=cache['loaded'][r['id']]
  else:
   auditpath=Path(r['folder']).parent/'adaptation_audit.json';demand(sha(required(auditpath))==proof['adaptation_audit_sha256'],'Adaptation audit changed after acceptance')
   audit=read(auditpath);demand(audit['source_sha256']==digest and audit['task']==task and audit['N_seen']==r['N'] and audit['optimizer_steps']==r['N']//r['B'],'ODR source/task/counters mismatch')
   loaded=load_results(r['folder'],r['csv'],r['heads']);demand(proof['metrics']==loaded['summary'],'ODR accepted metrics differ from revalidated 3MAT')
   inputpath=Path(r['folder']).parent/'input_evidence.json';loaded['input_evidence']=verify_geometry_binding(inputpath,proof['input_evidence_sha256'],r['csv'])
   demand(audit['input_evidence_sha256']==proof['input_evidence_sha256'],'ODR input audit/proof SHA mismatch')
   cache['loaded'][r['id']]=loaded
 r.update(status='已验收',reason='source final及独立验收与3MAT证据一致',loaded=loaded,acceptance_sha256=sha(r['accepted_report']))
 return r

def historical():
 # The known own historical five-seed ledger is an explicit separate cohort.
 ledger=RUNS/'reports/odr_seeds_20261005/results.json'
 if not ledger.is_file():return [],[dict(status='受限',reason='历史验收ledger缺失，未并入新结果')]
 rows=[];limits=[];digests={}
 for i,entry in enumerate(read(ledger)):
  phi=entry['phi'];norm='BN' if phi==3 else 'GN';final=RUNS/f'frozen/phi{phi}_{norm}_single_gpu_localstyle_seed2021_epoch20/model_final.pth.tar'
  try:
   if phi not in digests:
    manifest=read(required(final.parent/'manifest.json'));demand(sha(required(final))==manifest['files']['model_final.pth.tar']['sha256'],'Historical frozen final SHA mismatch');digests[phi]=sha(final)
   folder=entry['directory'];domain=entry['domain'];csvpath=DATA/domain/'labels'/('validation.csv' if domain=='synthetic' else 'test.csv');loaded=load_results(folder,csvpath,['heatmap','efficientpose'])
   demand(loaded['summary']['n_samples']==entry['n_samples'] and loaded['summary']['n_reject']==entry['n_reject'],'Historical ledger denominator mismatch')
   for metric in loaded['summary']['metric_denominators']:
    if metric in entry:demand(loaded['summary'][metric]==entry[metric],'Historical ledger metric changed')
   rows.append(dict(id='historical_'+str(i),stage='source' if entry['stage']=='offline' else 'odr',cohort='historical',source_config=f'historical_phi{phi}_{norm}',source_final=str(final),source_sha256=digests[phi],domain=domain,heads=['heatmap','efficientpose'],line='A' if entry['stage']=='odr' else None,seed=entry.get('seed'),N=(1024 if phi==3 else 4096) if entry['stage']=='odr' else None,B=(4 if phi==3 else 1) if entry['stage']=='odr' else None,status='已验收',reason='旧ledger/冻结final/3MAT核对，仅历史单列；旧φ3 seed2021缺适应观测清单，不作为新线验收，不把差异全部归因累积',loaded=loaded,is_alias=False,canonical_id='historical_'+str(i),acceptance_sha256=sha(ledger)))
  except EvidenceMissing as error:limits.append(dict(id='historical_'+str(i),status='受限',reason=str(error)))
 return rows,limits
