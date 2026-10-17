"""Read-only compatibility of actual completed MOCK entrypoint fixtures with final-reporting APIs."""
from pathlib import Path
import json,sys,os,hashlib,copy,argparse
P=Path(__file__).resolve().parents[2];R=P/'entrypoints_review';E=P/'evaluation_entrypoints';F=E/'tests/regression_20261017_d/MOCK_entrypoints'
sys.path.insert(0,str(P))
from final_reporting import evidence as ev
from final_reporting import verify_extended as ve
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def need(x,m):
 if not x:raise ValueError(m)
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,FileNotFoundError):return
 raise ValueError('Negative compatibility case was accepted')
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();ev.run_id(a.run_id)
codes=[P/'final_reporting/evidence.py',P/'final_reporting/verify_extended.py',P/'final_reporting/run_extended.py'];pins={str(x):sha(x) for x in codes};impl=sha(E/'implementation_manifest.json');protocol=sha(E/'protocol_manifest.json')
checked={};source_modes=[];odr_count=0;alias_count=0

def row_for(proof,report):
 c=report['context'];return dict(id=proof.get('task_id',proof.get('identifier')),stage='source' if c['kind']=='source' else 'odr',expected_total=report['sample_count'],domain=report['domain'],heads=report['heads'],config=c['configuration']['path'],source_final=c['weights']['source_final']['path'],source_report=c['source_training_acceptance']['path'],csv=report['inputs']['csv']['path'],accepted_report=proof['base_report'])
def check(proof,*,source=False):
 link=proof['extension'];report=read(link['report']['path']);row=row_for(proof,report)
 result=ev.checked_link(row,link,proof['source_sha256']);need(result==report,'checked_link result changed');ev.wrapper_link_binding(proof,link,proof['base_report'])
 resolved=ev.context_code_maps(report['context']);need(set(resolved[0])=={str(P/'extended_evaluation'/k) for k in report['context']['observer_code_sha256']},'Observer relative map resolved incorrectly');need(set(resolved[1])=={str(E/k) for k in report['context']['wrapper_code_sha256']},'Wrapper relative map resolved incorrectly')
 reduced=dict(id=row['id'],status='accepted',source_sha256=proof['source_sha256'],report_path=report['report_path'],per_sample_path=report['per_sample_path'],protocols=report['protocols'],counts=report['counts'],primary_mode=report['primary_mode'],primary=report['extended_primary'])
 ve.reduce_samples(row,reduced)
 if source:
  accepted=read(row['accepted_report']);need(accepted['extension']['entrypoint_verification']['sha256']==sha(proof['_path']),'Fixture source wrapper pointer changed')
  # Real original source verify_eval output + wrapper pointer; no fabricated accepted state.
  inspected=ev.inspect_record(row,{row['id']:row});need(inspected['status']=='accepted','Source inspect_record rejected actual MOCK proof')
 for f in [proof['_path'],row['accepted_report'],link['report']['path'],link['verification']['path'],report['per_sample_path']]:checked[str(f)]=sha(f)
 for obj in report['inputs'].values():checked[obj['path']]=obj['sha256']
 for mapping in resolved:checked.update(mapping)
 return row,report,reduced

for path in sorted((F/'reports/source').glob('*.json')):
 proof=read(path);proof['_path']=str(path);row,report,reduced=check(proof,source=True)
 source_modes.append(dict(identifier=proof['identifier'],primary_mode=report['primary_mode'],heads=report['heads'],total=report['sample_count'],numeric_H=report['counts']['H_numeric_pose'],original_accepted=report['counts']['original_accepted'],pose_mean=report['extended_primary']['metrics']['pose']['mean'],applicability=proof['extension']['applicability'],checked_link=True,inspect_record=True,independent_reduce_samples=True))
 bad=copy.deepcopy(row);bad['expected_total']+=1;reject(lambda:ev.checked_link(bad,proof['extension'],proof['source_sha256']))
 bad=copy.deepcopy(proof['extension']);bad['verification']['sha256']='0'*64;reject(lambda:ev.checked_link(row,bad,proof['source_sha256']))
 reject(lambda:ev.checked_link(row,proof['extension'],'0'*64))
need(len(source_modes)==5,'Expected exactly5 source modes')
for path in sorted((F/'reports').glob('MOCK*.json')):
 proof=read(path);proof['_path']=str(path)
 if proof.get('canonical_task'):
  c=proof['canonical_proof'];canonical=read(c['path']);need(sha(c['path'])==c['sha256'] and canonical['task_id']==proof['canonical_task'] and canonical['extension']==proof['extension'] and canonical['status']=='passed','Alias canonical chain differs');need(not (F/'runs/odr'/proof['task_id']/'capture').exists(),'Alias fabricated capture');canonical['_path']=c['path'];check(canonical);alias_count+=1
 else:check(proof);odr_count+=1
 need(sha(proof['base_report'])==proof['base_report_sha256'],'Original ODR base proof changed')
need(odr_count==41 and alias_count==1,'Expected40fixed+1scan+1alias MOCK')
first=read(F/'reports/MOCK_fixed0.json');second=read(F/'reports/MOCK_fixed1.json')
# Both are independently passed real tiny MOCK links with the same actual cfg/domain/source.
need(first['source_sha256']==second['source_sha256'],'Fixture source must match')
need(read(first['extension']['report']['path'])['context']['configuration']==read(second['extension']['report']['path'])['context']['configuration'],'Fixture configuration must match')
reject(lambda:ev.wrapper_link_binding(first,second['extension'],first['base_report']))
reject(lambda:ev.wrapper_link_binding(first,first['extension'],second['base_report']))
for path,h in checked.items():need(sha(path)==h,'Read-only fixture changed during compatibility check')
need(pins=={str(x):sha(x) for x in codes},'Root implementation changed during check; rerun against stable version')
need(sha(E/'implementation_manifest.json')==impl and sha(E/'protocol_manifest.json')==protocol,'Frozen entrypoint manifest changed')
report=dict(schema='spnv2-entrypoints-final-reporting-MOCK-compatibility-v1',status='passed',run_id=a.run_id,source_modes=source_modes,canonical_ODR_MOCK_proofs=odr_count,alias_MOCK_proofs=alias_count,negative_contract_checks=15,negative_legal_same_cfg_source_link_crosswires=2,wrapper_base_report_and_exact_extension_binding_verified=True,relative_context_maps_verified=True,all_checked_fixture_inputs_unchanged=True,implementation_sha256=pins,entrypoint_implementation_manifest_sha256=impl,entrypoint_protocol_manifest_sha256=protocol,inputs_sha256=checked,scope='Engineering compatibility only: actual completed tiny MOCK entrypoint proof chains, root checked_link/source inspect_record/independent reduce_samples; no production result claimed',real_GPU_executed=False,real_adaptation_executed=False,formal_scientific_results=False)
out=R/'reports'/(a.run_id+'.json');ev.write(out,report);print(json.dumps({k:v for k,v in report.items() if k!='inputs_sha256'},ensure_ascii=False))
