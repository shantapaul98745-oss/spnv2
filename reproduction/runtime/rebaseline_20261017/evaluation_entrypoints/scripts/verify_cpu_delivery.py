"""Independent static CPU acceptance of completed MOCK integration evidence and immutable protocol."""
from pathlib import Path
import sys,os,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import entry_contract as e
assert os.environ.get('STY') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();m=e.verify_protocol();cpu=e.read(e.E/'reports'/('cpu_acceptance_'+a.run_id+'.json'));e.need(cpu['status']=='passed' and cpu['passed']==6 and cpu['protocol_manifest_sha256']==e.sha(e.E/'protocol_manifest.json'),'CPU report incomplete/stale');e.need(cpu['wrapper_code_sha256']==m['wrapper_code_sha256'] and cpu['formal_scientific_results'] is False and cpu['real_GPU_executed'] is False,'CPU report code/scope mismatch')
root=e.E/'tests'/a.run_id/'MOCK_entrypoints';source=list((root/'reports/source').glob('*.json'));odr=list((root/'reports').glob('MOCK*.json'));e.need(len(source)==5 and len(odr)==42,'Expected5 MOCK source modes and40 fixed+1scan+1alias')
for path in source+odr:
 proof=e.read(path);e.need(proof['status']=='passed' and proof['wrapper_protocol_sha256']==e.sha(e.E/'protocol_manifest.json'),'Wrapper proof protocol/status mismatch');link=proof['extension'];e.need(link['schema']=='spnv2-evaluation-extension-link-v1','Wrong public link schema')
 for field in ['report','verification']:e.need(e.sha(link[field]['path'])==link[field]['sha256'],'Extension link SHA changed')
 api=e.read(link['verification']['path']);e.need(api['status']=='passed' and api['report_sha256']==link['report']['sha256'] and api['context_sha256']==link['context_sha256'],'Independent API proof link mismatch')
 if path in source:
  accepted=e.read(proof['base_report']);e.need(e.digest({k:v for k,v in accepted.items() if k!='extension'})==proof['base_fields_sha256'],'Source original fields changed');ptr=accepted['extension'];e.need(ptr['entrypoint_verification']==dict(path=str(path),sha256=e.sha(path)),'Source entrypoint pointer changed')
 else:
  e.need(e.sha(proof['base_report'])==proof['base_report_sha256'],'ODR base report drift')
  if proof.get('canonical_task'):
   cp=proof['canonical_proof'];e.need(e.sha(cp['path'])==cp['sha256'] and e.read(cp['path'])['extension']==link,'Alias canonical extension changed');e.need(not (root/'runs/odr'/proof['task_id']/'capture').exists(),'Alias created own capture')
log=e.E/'logs'/('cpu_'+a.run_id.replace('regression_','')+'.log');e.need('command_exit_code=0' in log.read_text(),'CPU integration log exit nonzero/missing')
result=dict(schema='spnv2-evaluation-entrypoints-independent-cpu-acceptance-v1',status='passed',run_id=a.run_id,protocol_manifest_sha256=e.sha(e.E/'protocol_manifest.json'),CPU_report_sha256=e.sha(e.E/'reports'/('cpu_acceptance_'+a.run_id+'.json')),source_mode_proofs=5,ODR_proofs=42,canonical_ODR_MOCK_observations=41,alias_no_capture=True,all_public_links_rehashed=True,source_original_fields_and_pointer_verified=True,ODR_base_report_bytes_SHA_verified=True,no_GPU_or_model_execution=True,formal_scientific_results=False)
e.stable_write(e.E/'reports'/('independent_cpu_acceptance_'+a.run_id+'.json'),result);print(json.dumps(result))
