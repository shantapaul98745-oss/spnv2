"""Prepare exact metadata candidate/spec after the old idle controller lock is free."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, os, time, uuid
from pathlib import Path
U=Path('/home/ps/Downloads/UsrSourceCode/LuXinghai');J=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
M=U/'spnv2_runs/migrations/sysu_20261027';R=U/'spnv2/reproduction/sysu_migration_20261027'
def raw(o):return (json.dumps(o,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def binding(p,logical=False):return dict(path=str(J/Path(p).relative_to(U)) if logical else str(p),sha256=sha(p))
def publish(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.pending-'+uuid.uuid4().hex)
 with tmp.open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
 os.replace(tmp,p)
def main():
 a=argparse.ArgumentParser();a.add_argument('--old-spec',required=True);a.add_argument('--published-code-commit',required=True);a.add_argument('--prepare',action='store_true');args=a.parse_args()
 assert len(args.published_code_commit)==40 and all(c in '0123456789abcdef' for c in args.published_code_commit),'Exact published commit required'
 if not args.prepare:raise SystemExit('Requires explicit --prepare after safe idle-controller stop')
 lock=(M/'deployment_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (U/'spnv2_runs/plans/g50p_risk_diagnostic_20261025/index.json').exists(),'Already installed runtime needs separate safe-boundary reconciliation'
 previous=read(M/'runtime_candidate/index_candidate.json');prevhash=sha(M/'runtime_candidate/index_candidate.json');state=read(M/'source_handoff_snapshot_20261027/reconciled_state.json')
 auditpath=M/'pipeline_optimization_20261027/dependencies_audit.json';audit=read(auditpath)
 assert audit['schema']=='spnv2-streaming-window-dependencies-v1' and audit['model_forwards']==0
 order=[r['run_id'] for r in previous['tasks']]
 assert len(order)==48 and set(audit['windows'])==set(order)-set(state['accepted'])
 plan=dict(schema=audit['schema'],audit_basis=binding(auditpath,True),asset_manifest=binding(M/'asset_transfer_manifest.json',True),
  logical_target_root=str(J),frozen_order=order,common_relative_paths=audit['common_relative_paths'],summary_relative_paths=audit['summary_relative_paths'],
  windows={rid:dict(run_pair=audit['windows'].get(rid,{}).get('run_pair',[]),verify_pair=audit['windows'].get(rid,{}).get('verify_pair',[])) for rid in order},
  source_originals_retained=True,accepted_main_replay_forbidden=True,summary_uses_original_metadata_only=True)
 dep=M/'pipeline_optimization_20261027/window_dependencies.json'
 hist=M/'runtime_candidate/history'/('streaming_before_'+prevhash[:12]);hist.mkdir(parents=True,exist_ok=True)
 for source in [M/'runtime_candidate/index_candidate.json',M/'runtime_candidate/build_receipt.json',Path(args.old_spec)]:
  target=hist/source.name
  assert not target.exists() or target.read_bytes()==source.read_bytes()
  if not target.exists():publish(target,source.read_bytes())
 publish(dep,raw(plan));index=copy.deepcopy(previous)
 index['implementation_sha256']['manager.py']=sha(R/'manager_sysu.py')
 for name,item in index['migration_extensions'].items():
  physical=U/Path(item['path']).relative_to(J);item['sha256']=sha(physical)
 index['streaming_migration']=dict(code=binding(R/'streaming_assets_v1.py',True),dependencies=binding(dep,True))
 rebound=[]
 for row in index['tasks']:
  if row['run_id'] in state['main']:continue
  path=M/'runtime_candidate/tasks'/index['sysu_task_revision']/Path(row['task_path']).name
  old_raw=path.read_bytes();task=read(path);original=copy.deepcopy(task);task['implementation_sha256']=copy.deepcopy(index['implementation_sha256'])
  assert {k:v for k,v in task.items() if k!='implementation_sha256'}=={k:v for k,v in original.items() if k!='implementation_sha256'}
  oldtask=hist/'tasks'/path.name;publish(oldtask,old_raw);publish(path,raw(task));row['task_sha256']=sha(path);rebound.append(row['run_id'])
 for name in ('pilot_admission_template','summary_template'):index[name]['implementation_sha256']=copy.deepcopy(index['implementation_sha256'])
 publish(M/'runtime_candidate/index_candidate.json',raw(index))
 build=read(M/'runtime_candidate/build_receipt.json');build.update(candidate_index_sha256=sha(M/'runtime_candidate/index_candidate.json'),new_manager_sha256=index['implementation_sha256']['manager.py'],previous_candidate_index_sha256=prevhash,unstarted_tasks_rebound_only_for_metadata_manager=rebound,streaming_revision=True)
 publish(M/'runtime_candidate/build_receipt.json',raw(build))
 spec=read(args.old_spec);spec['nonce']=uuid.uuid4().hex;spec['manager_nonce']=uuid.uuid4().hex
 spec['controller']=binding(R/'deployment_controller.py')
 spec['inputs']['streaming_dependencies']=binding(dep)
 for item in spec['inputs'].values():item['sha256']=sha(item['path'])
 spec['previous_controller_spec']=binding(args.old_spec);spec['streaming_revision_authorized']=True;spec['created_at']=time.time()
 spec['published_code_commit']=args.published_code_commit
 spec['publication_notes']=['User-authorized pipelined migration: first independent w16 verification after its immutable dependency closure verifies; no all276-file admission requirement.', 'Frozen-order remaining tasks each wait for their receiptverified canonical dependencies before original resource and budget admission.', 'Background immutable archive transfer continues without moving staging/bases; source originals/proofs retained; no scientific replay, tolerance or budget change.']
 spec['ready_condition']='verified_w16_dependency_closure_then_frozen_order_per_task_dependencies'
 spec['deferred_archive_is_not_scientific_prerequisite']=True
 out=M/'deployment_specs'/('launch_'+spec['nonce']+'.json');publish(out,raw(spec))
 receipt=dict(schema='spnv2-stream-candidate-build-v1',candidate_index=binding(M/'runtime_candidate/index_candidate.json'),dependencies=binding(dep),launch_spec=binding(out),controller=spec['controller'],nonce=spec['nonce'],manager_nonce=spec['manager_nonce'],history=str(hist),unstarted_mains=len(rebound),preserved_main_count=len(state['main']),preserved_accepted_count=len(state['accepted']),model_forwards=0)
 publish(M/'pipeline_optimization_20261027/build_receipt.json',raw(receipt));print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__':main()
