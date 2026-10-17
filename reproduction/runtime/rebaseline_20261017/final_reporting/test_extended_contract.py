"""Clearly marked CPU fixtures plus current REAL pending catalog; never GPU."""
import argparse,copy,csv,json,os,sys,uuid
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_reporting.evidence import P,D,UNITS,sha,read,digest,write,csv_write,guard,seed_groups,same_id_pairs
from final_reporting.verify_extended import equivalent,reduce_samples,verify_pairs,verify_seed_groups,verify
from final_reporting.run_extended import build

class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.identifier='cpu_extended_'+uuid.uuid4().hex[:12];cls.root=D/'tests'/cls.identifier;cls.root.mkdir(parents=True)
  (cls.root/'TEST_ONLY.md').write_text('MOCK numerical rows for CPU contract tests; not experiment results.\n')
 def reject(self,fn):
  with self.assertRaises((ValueError,KeyError,TypeError)):fn()
 def fixture(self,heads=('heatmap','efficientpose'),missing=False):
  root=self.root/uuid.uuid4().hex[:8];root.mkdir();ids=['MOCK_img_a','MOCK_img_b','MOCK_img_c']
  with (root/'labels.csv').open('w',newline='') as f:csv.writer(f).writerows([[x] for x in ids])
  h='heatmap' in heads;e='efficientpose' in heads;rows=[]
  for i,ident in enumerate(ids):
   accepted=(i==0) if h else True
   H={m:(None if missing or i==2 else float(i+1)) for m in UNITS} if h else None
   if missing:accepted=False if h else True
   E={m:float(i+2) for m in UNITS} if e else None
   hybrid={m:(H[m] if m in ['eR','eR_HIL'] else E[m] if m in ['eT','eT_norm_HIL'] else None if H['eR_HIL'] is None else H['eR_HIL']+E['eT_norm_HIL']) for m in UNITS} if h and e else None
   rows.append(dict(index=i,sample_id=ident,original_accepted=accepted,original_reject=not accepted if h else None,
    H_numeric_rotation=h and H['eR'] is not None,H_numeric_translation=h and H['eT'] is not None,H_numeric_pose=h and H['pose'] is not None,
    H_E_intersection=h and e and hybrid['pose'] is not None,geometry_rejected_numeric_H=h and not accepted and H['pose'] is not None,
    extended_H=H,original_E=E,extended_hybrid=hybrid))
  per=root/'per.jsonl';per.write_text(''.join(json.dumps(x)+'\n' for x in rows));report=root/'report.json';write(report,dict(sample_id_sha256=digest(ids)))
  # Deliberately independent fixture aggregate, not the production reducer.
  protocols={}
  for label,key,applicable,subset in [('original_accepted_H','extended_H',h,True),('original_E','original_E',e,False),('original_H_E_intersection','extended_hybrid',h and e,True),('extended_H','extended_H',h,False),('extended_hybrid','extended_hybrid',h and e,False)]:
   if not applicable:protocols[label]=dict(applicable=False,reason='required_head_not_present',metrics=None);continue
   metrics={}
   for m,u in UNITS.items():
    vals=[r[key][m] for r in rows if (not subset or r['original_accepted']) and r[key][m] is not None];n=len(vals)
    metrics[m]=dict(mean=sum(vals)/n if n else None,count=n,total=3,missing=3-n,coverage=n/3,denominator_scope='original_accepted_subset' if subset else 'all_samples' if n==3 else 'numeric_pose_available_subset',mask=label+':'+m,unit=u)
   protocols[label]=dict(applicable=True,metrics=metrics)
  counts={k:sum(r[k] for r in rows) for k in ['original_accepted','H_numeric_rotation','H_numeric_translation','H_numeric_pose','H_E_intersection','geometry_rejected_numeric_H']}
  counts.update(original_accepted_H_E_intersection=sum(r['original_accepted'] and r['H_E_intersection'] for r in rows),total=3,
   missing_H_rotation=3-counts['H_numeric_rotation'] if h else None,missing_H_translation=3-counts['H_numeric_translation'] if h else None,heatmap_applicable=h,efficientpose_applicable=e)
  mode='H_rotation_E_translation' if h and e else 'H_rotation_H_translation' if h else 'E_only_original';primary=protocols['extended_hybrid' if h and e else 'extended_H' if h else 'original_E']
  return dict(id='MOCK',heads=list(heads),expected_total=3,csv=str(root/'labels.csv'),N=4,B=1,seed=2021,line='A',domain='MOCK',source_config='MOCK'),dict(id='MOCK',status='accepted',source_sha256='TEST_ONLY_SHA',per_sample_path=str(per),report_path=str(report),protocols=protocols,counts=counts,primary_mode=mode,primary=primary),rows

 def test_01_full_and_original_denominators(self):
  row,result,_=self.fixture();reduce_samples(row,result)
  self.assertEqual(result['protocols']['original_accepted_H']['metrics']['pose']['count'],1)
  self.assertEqual(result['protocols']['extended_H']['metrics']['pose']['count'],2)
  self.assertEqual(result['protocols']['original_E']['metrics']['pose']['count'],3)
 def test_02_no_H_is_not_missing_H(self):
  row,result,_=self.fixture(('efficientpose',));reduce_samples(row,result);self.assertIsNone(result['counts']['missing_H_rotation'])
 def test_03_zero_valid_stays_none(self):
  row,result,_=self.fixture(('heatmap',),True);reduce_samples(row,result);self.assertIsNone(result['primary']['metrics']['pose']['mean'])
  result['primary']['metrics']['pose']['mean']=0.0;self.reject(lambda:reduce_samples(row,result))
 def test_04_reordered_ids_rejected(self):
  row,result,rows=self.fixture();rows[0],rows[1]=rows[1],rows[0];Path(result['per_sample_path']).write_text(''.join(json.dumps(r)+'\n' for r in rows));self.reject(lambda:reduce_samples(row,result))
 def test_05_truncated_data_rejected(self):
  row,result,rows=self.fixture();Path(result['per_sample_path']).write_text(json.dumps(rows[0])+'\n');self.reject(lambda:reduce_samples(row,result))
 def test_06_mask_unit_and_mean_tamper(self):
  row,result,_=self.fixture()
  for key,value in [('count',3),('unit','rad'),('mean',9.0),('denominator_scope','all_samples')]:
   altered=copy.deepcopy(result);altered['protocols']['extended_H']['metrics']['pose'][key]=value;self.reject(lambda:reduce_samples(row,altered))
 def test_07_numeric_flags_recomputed(self):
  row,result,rows=self.fixture();rows[1]['geometry_rejected_numeric_H']=False;Path(result['per_sample_path']).write_text(''.join(json.dumps(r)+'\n' for r in rows));self.reject(lambda:reduce_samples(row,result))
 def test_08_pairs_numeric_and_geometric_separate(self):
  row,before,rows=self.fixture();_,after,_=self.fixture();after['source_sha256']=before['source_sha256'];p=self.root/'pairs.csv'
  rows=list(same_id_pairs(before,after));csv_write(p,rows,['index','sample_id','before_score','after_score','delta','numeric_state','original_acceptance_state'])
  s=verify_pairs(row,before,after,p);self.assertEqual(s['common_numeric'],2);self.assertEqual(s['original_acceptance_transitions']['both_rejected'],2)
  altered=p.read_text().replace('both_missing','both_numeric');p.write_text(altered);self.reject(lambda:verify_pairs(row,before,after,p))
 def test_09_cross_source_pair_rejected(self):
  _,a,_=self.fixture();_,b,_=self.fixture();b['source_sha256']='ANOTHER_SOURCE';self.reject(lambda:list(same_id_pairs(a,b)))
 def test_10_five_stream_zero_valid_not_unrun(self):
  catalog=[];results={}
  for group in range(74):
   for seed in range(2021,2026):
    key=f'MOCK_{group}_{seed}';row=dict(id=key,stage='odr',source_config='MOCK',line='A',domain='MOCK',N=group+1,B=1,seed=seed,expected_total=3);catalog.append(row)
    metric=dict(mean=None,count=0);results[key]=dict(status='accepted',source_sha256='ONE_FIXED_SOURCE',protocols={x:dict(metrics={m:metric for m in UNITS}) for x in ['original_E','extended_H','extended_hybrid']})
  summary=seed_groups(catalog,results);verify_seed_groups(catalog,results,summary)
  self.assertTrue(all(g['all_five_experiments_accepted'] for g in summary));self.assertIsNone(summary[0]['metrics']['extended_H:pose']['mean'])
  bad=copy.deepcopy(summary);bad[0]['metrics']['extended_H:pose']['mean']=0.0;self.reject(lambda:verify_seed_groups(catalog,results,bad))
 def test_11_none_zero_and_boolean_types(self):
  self.reject(lambda:equivalent(0.0,None,'missing'));self.reject(lambda:equivalent(True,1,'count'));equivalent(1.0000000000001,1.0,'rounding')
 def test_12_real_unfinished_catalog_stays_pending(self):
  snapshot=P/'analysis/runs/actual_snapshot_20261017_d';status=build(snapshot,self.identifier,True);proof=verify(self.identifier,True)
  self.assertFalse(status['executed_scope_complete']);self.assertFalse(proof['executed_scope_complete']);self.assertEqual(proof['accepted_records'],0)
  self.reject(lambda:verify(self.identifier,False))

if __name__=='__main__':
 guard();parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contract);r=unittest.TextTestRunner(verbosity=2).run(suite)
 write(args.report,dict(schema='spnv2-final-extended-cpu-contract-v1',status='passed' if r.wasSuccessful() else 'failed',test_count=r.testsRun,failures=len(r.failures),errors=len(r.errors),fixtures='MOCK only; actual current catalog independently remains pending',no_GPU=True))
 sys.exit(0 if r.wasSuccessful() else 1)
