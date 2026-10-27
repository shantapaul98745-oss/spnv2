"""One explicitly versioned defining-domain correction; no model replay here."""
import sys,json,copy,hashlib,time,uuid,os
from pathlib import Path
from contextlib import ExitStack
import prepare_runtime as b
import namespace_runner as ns
import dispatch_config as dc
U,J,D,M=b.U,b.J,b.D,b.M
REV=D/'revisions/numeric_acceptance_20261027';V2=REV/'v2_geometry';R=U/'spnv2/reproduction/numeric_acceptance_20261027'
def main(commit):
 with ExitStack() as stack:
  for name in ns.LOCK_PLANS:stack.enter_context(b.lock(U/'spnv2_runs/plans'/name/'pipeline.lock'))
  stack.enter_context(b.lock(D/'manager.lock'))
  if b.sha(D/'index.json')!='dac81296bd3e715e7487701b3d6779a0865fc0547b47b9ca9acad0185ea700fc':raise ValueError('Expected exact v1 numeric baseline')
  state=b.read(D/'status.json');active=state.get('active');rid='g50p_diag_v1_sunlamp_s2021_w0023'
  if state['status']!='blocked' or active['nonce']!='0c6be39dec624522acd9aea2de04d124' or active['exit_code']!=1 or len(state['accepted'])!=27 or len(state['main'])!=27:raise ValueError('Unexpected active science; refuse metadata switch')
  if (REV/'manager.log.exitcode').read_text().strip()!='1':raise ValueError('Old manager exit not proven')
  failure=D/'pairs'/rid/'failure.json';f=b.read(failure)
  if f['trace_difference'].get('numeric_policy_failure',{}).get('path')!='arrival.rotation_disagreement':raise ValueError('Not the specifically diagnosed angle domain stop')
  import numeric_compare as numeric
  numeric.configure(b.read(REV/'policy_v2.json'))
  if not numeric.equal(f['trace_difference']['actual'],f['trace_difference']['expected']):raise ValueError('Other differences violate unchanged acceptance bounds')
  journal=(D/'budget/journal.jsonl').read_bytes();events=[json.loads(x) for x in journal.splitlines()];started={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_started'};closed={e['payload']['attempt_id'] for e in events if e['kind']=='attempt_closed'}
  if started-closed:raise ValueError('Unclosed computation prevents continuation')
  before=b.read(D/'index.json');b.immutable_bytes(V2/'before_index.json',(D/'index.json').read_bytes());b.immutable_bytes(V2/'before_state.json',(D/'status.json').read_bytes());b.immutable_bytes(V2/'archived_failure.json',failure.read_bytes())
  rev=b.checked(before['numeric_acceptance']) if hasattr(b,'checked') else b.read(b.mapped(before['numeric_acceptance']['path']))
  rev=copy.deepcopy(rev);rev.update(published_code_commit=commit,policy=b.bind(REV/'policy_v2.json'),comparator=b.bind(R/'numeric_compare.py'),supersedes_revision=before['numeric_acceptance'],supersedes_index=b.bind(V2/'before_index.json'),geometry_domain_correction='Compare the defining dimensionless cosine before acos; unchanged raw float bounds, all raw poses and discrete decisions still checked',geometry_math_and_GT_unchanged=True)
  revpath=b.immutable(REV/'revision_v2.json',rev);index=copy.deepcopy(before);index['numeric_acceptance']=b.bind(revpath);b.immutable(V2/'index_candidate.json',index);b.atomic(D/'index.json',b.raw(index));new_index=b.bind(D/'index.json')
  state.update(index=new_index,status='between_tasks',active=None,manager=None,reason='derived_angle_defining_domain_correction',next_resource_check=0.)
  state.setdefault('technical_revisions',[]).append({'name':'numeric_geometry_defining_domain','revision':index['numeric_acceptance'],'old_failed_attempt_and_costs_preserved':True,'accepted_reused':27});b.atomic(D/'status.json',b.raw(state));failure.unlink()
  control=M/'runtime_control';handoff=b.read(control/'budget_handoff_numeric_20261027.json');handoff['diagnostic_index']=new_index;hp=b.immutable(control/'budget_handoff_numeric_v2.json',handoff)
  template=b.read(control/'template_numeric_20261027.json');template.update(diagnostic_index=new_index,budget_handoff_receipt=b.bind(hp),numeric_acceptance=index['numeric_acceptance']);tp=b.immutable(control/'template_numeric_v2.json',template)
  dep={'schema':dc.DEPLOYMENT_SCHEMA,'migration_id':ns.MIGRATION,'template':b.bind(tp),'index':new_index,'controller':index['numeric_acceptance'],'source_fence':template['source_fence_receipt']};dp=b.immutable(control/'deployment_numeric_v2.json',dep)
  row=next(x for x in index['tasks'] if x['run_id']==rid);task=b.mapped(row['task_path']);config=dc.compose_single_task(b.bind(dp),b.bind(task),resolve=b.mapped);version=dc.write_config_version(config,resolve=b.mapped)
  nonce=uuid.uuid4().hex;proof=control/'config_versions'/('numeric_v2_validation_'+nonce+'.json');validator=R/'prepare_runtime.py';request=b.immutable(V2/'validation_request.json',{'schema':'spnv2-numeric-acceptance-validation-request-v1','nonce':nonce,'validator':b.bind(validator),'config':version,'task':b.bind(task),'output':b.logical(proof)})
  python=config['paths']['logical_environment']+'/bin/python';base=ns.build_bwrap([],config,scientific=True,config_sha256=version['sha256'],config_path=version['path'],_manager=True);va=base[:base.index(python)]+[python,'-B','-s','-u',b.logical(validator),'--validate-request',b.logical(request),'--request-sha256',b.sha(request)]
  if (D/'budget/journal.jsonl').read_bytes()!=journal:raise ValueError('Cost mutation during metadata revision')
  result={'index':new_index,'revision':index['numeric_acceptance'],'policy':rev['policy'],'config':version,'deployment':b.bind(dp),'validation_nonce':nonce,'validation_output':b.logical(proof),'validation_argv':va,'manager_argv':ns.build_manager_bwrap(config,b.bind(dp),version),'screen_socket_dir':config['screen_socket_dir'],'accepted_reused':27,'model_forwards':0,'published_code_commit':commit};b.immutable(V2/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('validation_argv','manager_argv')}))
if __name__=='__main__':
 if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise SystemExit('Explicit published commit required')
 main(sys.argv[1])
