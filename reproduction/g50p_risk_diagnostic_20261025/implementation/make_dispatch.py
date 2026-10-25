"""Assemble immutable local payload after code validation; no model or GT IO."""
import json,pathlib,hashlib,math
LOCAL=pathlib.Path(__file__).resolve().parent.parent
SERVER=pathlib.PurePosixPath('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
 return dict(path=str(SERVER/p.relative_to(LOCAL).as_posix()),sha256=sha(p))
def main():
 frozen=json.loads((LOCAL/'frozen_inputs.json').read_text(encoding='utf8'))
 fb=dict(path=str(SERVER/'frozen_inputs.json'),sha256=sha(LOCAL/'frozen_inputs.json'))
 codes={p.name:sha(p) for p in sorted((LOCAL/'implementation').glob('*.py'))}
 initial=dump(LOCAL/'validation/initial_cost.json',dict(schema='g50p-diagnostic-initial-cost-v1',status='passed',physical_forwards_upper=4748,active_seconds_upper=360,measured_test_seconds=69.357,observed_process_interval_upper_seconds=314,classification='Conservative inclusive charge for new local necessary TinyEHS/backbone correctness tests including repeated successful test invocations; pure records mock tests zero real model forwards. Not SPNv2 scientific results.',failure_dispatch_unknown_upper=2,source_test_files={k:v for k,v in codes.items() if k in ['test_pair_runner.py','test_verify_pair.py']},time_margin_for_pure_contract_checks_seconds=46))
 tasks=[];by={x['window_id']:x for x in frozen['windows']}
 for i,wid in enumerate(frozen['execution_order']):
  w=by[wid];key=w['domain']+':'+str(w['stream_seed']);rid=f"g50p_diag_v1_{w['domain']}_s{w['stream_seed']}_w{w['candidate_number']:04d}"
  task=dict(schema='g50p-risk-diagnostic-task-v1',action='run_pair',run_id=rid,output_dir=str(SERVER/'pairs'/rid),implementation_sha256=codes,index_binding=fb,selection=w,panel=frozen['panels'][key],producer=frozen['producer'],reference_trace_proof_path=str(SERVER/'inputs/extraction_proof.json'))
  b=dump(LOCAL/'tasks'/(rid+'.json'),task);tasks.append(dict(run_id=rid,task_path=b['path'],task_sha256=b['sha256'],pilot=i<2))
 common=dict(schema='g50p-risk-diagnostic-task-v1',implementation_sha256=codes,frozen_inputs=fb)
 section=dict(path=str(SERVER/'inputs/innovation_2_20.md'),sha256=sha(LOCAL/'inputs/innovation_2_20.md'))
 pre=[]
 for action,rid,extra in [('extract_a','g50p_diag_extract_A',dict(output_dir=str(SERVER/'inputs'),literature_section=section)),('analyze_a','g50p_diag_analyze_A',dict(output_dir=str(SERVER/'A'),analysis_manifest=dict(path=str(SERVER/'inputs/analysis_manifest.json')),extraction_proof_path=str(SERVER/'inputs/extraction_proof.json')))]:
  t=dict(common,action=action,run_id=rid,**extra);b=dump(LOCAL/'tasks'/(rid+'.json'),t);pre.append(dict(action=action,run_id=rid,task_path=b['path'],task_sha256=b['sha256']))
 old=pathlib.PurePosixPath(frozen['producer']['implementation_dir'])
 idx=dict(schema='g50p-risk-diagnostic-index-v1',diagnostic_id=frozen['diagnostic_id'],frozen_inputs=fb,protocol=frozen['protocol'],implementation_sha256=codes,host_modules={n:dict(path=str(old/(n+'.py')),sha256=frozen['producer']['implementation_sha256'][n+'.py']) for n in ['host_gate','host_snapshot']},tasks=tasks,preprocessing=pre,complete_ODR_runs=0,a_execution_max_seconds=1800,initial_cost_receipt=initial,pilot_admission_template=dict(common,action='verify_pilots',run_id='g50p_diag_verify_pilots',proof_path=str(SERVER/'preflight/pilot_proof.json')),summary_template=dict(common,action='summarize_diagnostic',run_id='g50p_diag_summary',output_dir=str(SERVER/'delivery')))
 b=dump(LOCAL/'index.json',idx)
 print(json.dumps(dict(index=b,tasks=len(tasks),pilot_ids=frozen['execution_order'][:2],code_files=len(codes),initial_forward_upper=4748,initial_active_upper=360)))
if __name__=='__main__':main()
