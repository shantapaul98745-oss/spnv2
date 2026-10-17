"""Read-only independent acceptance of generated matrix/status artifact consistency."""
import argparse,os
from core import *
def verify(run_id):
 output=A/'runs'/run_id;manifest=read(required(output/'artifact_manifest.json'));status=read(required(output/'status.json'))
 for name,value in manifest['files_sha256'].items():demand(sha(required(output/name))==value,'Analysis output artifact changed: '+name)
 with (output/'matrix_new.csv').open() as f:rows=list(csv.DictReader(f))
 with (output/'matrix_historical.csv').open() as f:history=list(csv.DictReader(f))
 ids={r['task_id'] for r in rows};demand(len(ids)==430,'Expected30 source evaluations and400 ODR requests')
 demand(all(r['cohort']=='new' for r in rows) and all(r['cohort']=='historical' for r in history),'Historical results mixed into new matrix')
 for r in rows:
  if r['status']!='已验收':demand(all(r[k]=='' for k in ['accepted_mean','valid_denominator','rejections','n_samples']),'Unaccepted record contains borrowed/fabricated values')
  if r['metric']=='final_pose':
   expected={'hH':'hR_HIL+hT_norm_HIL','hE':'eR_HIL+eT_norm_HIL','mixed_hR_plus_eT':'hR_HIL+eT_norm_HIL'}
   demand(r['head'] in expected and r['final_metric_formula']==expected[r['head']],'Final metric branch/formula mislabeled')
  if r['status']=='已验收' and r['valid_denominator']=='0':demand(r['accepted_mean']=='','Zero-valid metric must have null mean')
 pairs=read(output/'paired_summary.json');groups=read(output/'five_observation_seed_summary.json');cases=read(output/'case_evidence.json');prisma=read(output/'prisma25_materials.json')
 demand(len({p['canonical_id'] for p in pairs})==len(pairs),'Alias counted as independent paired experiment')
 for g in groups:
  demand(g['canonical_experiments']==len(set(g['seeds'])),'Duplicate observation seed')
  if not g['five_observation_seeds_complete']:demand(all(x['mean'] is None and x['std_ddof1'] is None for x in g['metrics'].values()),'Incomplete five-seed group advertised aggregate')
 if any(r['status']!='已验收' for r in rows) or not prisma['complete'] or cases['status']!='已验收':demand(status['complete'] is False,'Missing necessary evidence but complete flag true')
 if any(r['status']!='已验收' for r in rows) or cases['status']!='已验收':demand(status['core_scope_complete'] is False,'Missing core evidence but core complete true')
 demand(status['scientific_scope_complete']==(status['core_scope_complete'] and prisma['complete']),'Core/PRISMA scope boundary wrong')
 demand(status['verified_limitation']==prisma['verified_limitation'] and status['conditional_prisma_status']==prisma['conditional_prisma_status'],'PRISMA limitation state mismatch')
 # Verify Markdown renders distinct A/B1/B2, N/B and metric columns.
 text=(output/'证据汇总.md').read_text();section=text.split('## 新source完整矩阵',1)[1].split('## 严格配对',1)[0];lines=[s for s in section.splitlines() if s.startswith('|')]
 demand(all(len(s.split('|'))==17 for s in lines),'Chinese matrix Markdown column count mismatch')
 result=dict(status='passed',run_id=run_id,matrix_record_slots=len(ids),new_accepted=status['accepted_new_records'],history_records_separate=status['history_records_separate'],core_complete=status['core_scope_complete'],scientific_complete=status['scientific_scope_complete'],conditional_prisma_status=prisma['conditional_prisma_status'],unaccepted_values_null=True,historical_separation=True,canonical_dedup=True,markdown_columns_verified=True,artifact_manifest_sha256=sha(output/'artifact_manifest.json'))
 write(output/'snapshot_acceptance.json',result);return result
if __name__=='__main__':
 assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();print(json.dumps(verify(a.run_id),ensure_ascii=False))
