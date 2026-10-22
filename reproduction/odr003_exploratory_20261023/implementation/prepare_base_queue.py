"""Prepare, never activate, the append-only3->40 ODR003 base-stage candidate.

Run only in a unique CPU guarded screen after the independent3-task phase is
independently accepted. No manager signals, runtime writes or experiments here.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

B=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
P=B/'spnv2_runs/plans/rebaseline_20261017'
N=B/'spnv2_runs/plans/odr003_exploratory_20261023'
Q=N/'queue_base'
TASK_DIR=Q/'tasks'
OLD_DEFAULT=N/'queue_offline_verified/proposal.json'
PY='/home/ps/anaconda3/envs/SPNv2/bin/python'
METHODS=('A','V','E50','E75','G50','G75')
DOMAINS=('lightbox','sunlamp')
SEEDS=(2021,2022,2023)
CODE=('run_exploration.py','data_boundary.py','prediction.py','state_adapter.py','gate_core.py','risk_metrics.py')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def reject(value):raise ValueError('Nonfinite JSON constant: '+value)
    obj=json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)
    json.dumps(obj,allow_nan=False)
    return obj


def bind(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())


def checked_file(spec):
    require(set(spec)=={'path','sha256'},'Invalid asset binding')
    path=Path(spec['path']).resolve()
    require(sha(path)==spec['sha256'],'Bound file changed: '+str(path))
    return path


def task_json_path(task):
    argv=task['command']
    require('--task-json' in argv,'Offline task does not use bound runner entrypoint')
    return Path(argv[argv.index('--task-json')+1]).resolve()


def check_ready(old_path):
    sys.path.insert(0,str(N/'pipeline'))
    import runner
    from revision import extend_state
    require(Path(runner.__file__).resolve().parent==N/'pipeline','Builder imported a different plan runner')
    runner.PLAN=N
    old_path=Path(old_path).resolve()
    require(old_path.is_relative_to(N),'Starting manifest must be under this independent003 plan')
    old=read(old_path);before=read(N/'status.json')
    require(len(old['tasks'])==3 and old['plan_id']==N.name,'Only the independent3-task calibration stage can be extended here')
    require(before['manifest_sha256']==runner.manifest_hash(old),'Status is not bound to the specified3-task manifest')
    require(before['status']=='phase_complete' and before['active'] is None and not before.get('error'),'The3-task phase must be complete without an active worker/error')
    ids=[t['id'] for t in old['tasks']]
    require(len(before['tasks'])==3 and [t['id'] for t in before['tasks']]==ids,'Runtime task coverage differs from3')
    require(len(before['completed'])==3 and set(before['completed'])==set(ids),'All3 accepted IDs required')
    require(all(t['status']=='accepted' for t in before['tasks']),'Every existing task must be independently accepted')
    runner.validate_manifest(old)
    return runner,extend_state,old,before


def main(old_path):
    require(bool(os.environ.get('STY')),'Use a unique guarded CPU screen')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Candidate construction must hide CUDA')
    require(not Q.exists() and not TASK_DIR.exists(),'Candidate/task directory already exists; refuse overwrite')
    runner,extend_state,old,before=check_ready(old_path)
    implementation=N/'implementation'
    code={name:sha(implementation/name) for name in CODE}
    official=read(N/'preflight/official_source_receipt.json')
    require(official['status']=='passed' and official['source_offline_identity_verified'] is True and official['strict_model_load_performed'] is True,'Official asset receipt is not accepted')
    geometry=read(official['source_core_manifest_path'])
    require(sha(official['source_core_manifest_path'])==official['source_core_manifest_sha256'],'Frozen source-core manifest changed')
    engineering=read(N/'preflight/real_correctness_verified.json')
    require(engineering['status']=='passed' and engineering['strict_equality_required'] is True and engineering['source_sha256']==official['source_sha256'],'Real engineering evidence missing or incompatible')
    require(sha(engineering['report_path'])==engineering['report_sha256'],'Engineering report changed')
    prepared_path=N/'preflight/data_boundary_prepared.json';verified_path=N/'preflight/data_boundary_verified.json'
    prepared=read(prepared_path);verified=read(verified_path)
    require(prepared['status']=='prepared' and verified['status']=='passed' and verified['prepared_proof_sha256']==sha(prepared_path),'Data boundary evidence differs')
    bundle_path=N/'calibration/bundle.json';proof_path=N/'calibration/proof.json'
    bundle=read(bundle_path);proof=read(proof_path)
    require(bundle['status']=='passed' and proof['status']=='passed' and proof['bundle_sha256']==sha(bundle_path),'Calibration is not independently accepted')
    require(proof['recomputed'] is True and proof['developer_records']==1024,'Calibration verification count/recomputation missing')
    require(bundle['source_sha256']==proof['source_sha256']==official['source_sha256'],'Calibration source differs')
    expected_dev={d:prepared['artifacts'][d+'/developer']['sha256'] for d in DOMAINS}
    require(bundle['developer_manifest_sha256']==expected_dev,'Calibration developer boundaries changed')
    # Read the exact accepted O runs from the currently bound3-task manifest;
    # preserve their filenames if an explicitly reviewed recovery changed them.
    offline_entries=[];offline_domains=set();offline_assets={}
    for task in old['tasks'][:2]:
        path=task_json_path(task);spec=read(path);domain=spec['domain'];directory=Path(spec['output_dir']).resolve()
        require(spec['method']=='O' and spec['stream']=='offline' and domain in DOMAINS and domain not in offline_domains,'Expected one O task per domain')
        offline_domains.add(domain)
        require(directory.is_relative_to(N/'runs'),'Offline outputs outside this experiment')
        r=read(directory/'report.json');p=read(directory/'proof.json')
        require(r['status']==p['status']=='passed','Offline execution/independent proof is not accepted')
        require(p['report_sha256']==sha(directory/'report.json') and p['task_sha256']==sha(path),'Offline proof/task/report mismatch')
        require(r['task_sha256']==sha(path) and r['bindings']==spec['bindings'] and r['code_sha256']==code,'Offline protocol/code does not match the current frozen modules')
        require(r['source_sha256']==official['source_sha256'] and r['domain']==domain and r['method']=='O','Offline source/domain mismatch')
        require(Path(p['main_exitcode_path']).resolve()==Path(task['log']+'.exitcode').resolve() and Path(p['main_exitcode_path']).read_text().strip()=='0','Offline main did not exit successfully')
        for output in r['outputs'].values():
            require(sha(output['path'])==output['sha256'],'Offline output changed')
            offline_assets[output['path']]=output['sha256']
        entry=dict(run_id=spec['run_id'],method='O',domain=domain,stream='offline',directory=str(directory),
            task_json=bind(path),main_exitcode=str(Path(task['log']+'.exitcode').resolve()))
        offline_entries.append(entry)
        offline_assets[str(directory/'report.json')]=sha(directory/'report.json')
        offline_assets[str(directory/'proof.json')]=sha(directory/'proof.json')
    require(offline_domains==set(DOMAINS),'Both accepted Offline domains required')
    observed_inputs=[dict(path=e['directory'],report_sha256=sha(Path(e['directory'])/'report.json'),proof_sha256=sha(Path(e['directory'])/'proof.json')) for e in offline_entries]
    require(bundle['inputs']==observed_inputs,'Calibration inputs do not match the two accepted Offline tasks')
    calibration_task=old['tasks'][2]
    require('calibrate' in calibration_task['command'] and str(bundle_path) in calibration_task['command'],'Final3-stage task is not this calibration')
    require(Path(calibration_task['log']+'.exitcode').read_text().strip()=='0','Calibration command did not exit0')
    require(Path(calibration_task['log']+'.verify.exitcode').read_text().strip()=='0','Calibration verification did not exit0')
    shared={key:bind(path) for key,path in {
        'official_receipt':N/'preflight/official_source_receipt.json','source':Path(official['source_path']),
        'config':Path(official['config_path']),'source_core_manifest':Path(official['source_core_manifest_path']),
        'data_prepared':prepared_path,'data_verified':verified_path,
        'camera':Path(geometry['camera_path']),'keypoints':Path(geometry['keypoint_path'])}.items()}
    common_assets={v['path']:v['sha256'] for v in shared.values() if Path(v['path']).is_relative_to(B)}
    common_assets.update({str(Path(geometry['root'])/k):v for k,v in geometry['files'].items()})
    common_assets.update({str(implementation/name):value for name,value in code.items()})
    for path in [implementation/'verify_exploration.py',implementation/'phase_tools.py',implementation/'verify_summary.py',implementation/'launch_plan.py',Path(__file__).resolve(),N/'preflight/real_correctness_verified.json',bundle_path,proof_path]:
        common_assets[str(path)]=sha(path)
    for name in ('runner.py','task_entry.py','progress.py','revision.py','pending_observers.py'):
        path=N/'pipeline'/name
        require(sha(path)==sha(P/'pipeline'/name),'Copied pipeline differs from the already reviewed implementation')
        common_assets[str(path)]=sha(path)
    template=old['tasks'][0]
    require(isinstance(template['gpu'],dict) and template['verify_gpu'] is False,'Unexpected GPU/verification template')
    new=copy.deepcopy(old)
    new['scope']='independent003_original3_preserved_plus36_base_adaptations_and_developer_summary'
    new['backlog']=['After accepted36-run exploration summary, at most18 matched-random controls only if the frozen selection rule qualifies a shared G setting; they are not included in this40-task stage.',
        'Formal confirmation and single-signal ablations require a later decision. Historical remaining150 source epochs and fullB/N sweeps remain deferred optional and not completed. Goal mode remains paused.']
    # All scientific prerequisites are checked before creating either directory.
    Q.mkdir(parents=True,exist_ok=False);TASK_DIR.mkdir(parents=True,exist_ok=False)
    previous=calibration_task['id'];entries=list(offline_entries);task_files=[]
    for domain in DOMAINS:
        for seed in SEEDS:
            for method in METHODS:
                ident=f'odr003_{method}_{domain}_s{seed}_N1024_B4';directory=N/'runs'/ident;log=N/'logs'/(ident+'.log')
                require(not directory.exists(),'Run directory already exists: '+str(directory))
                for suffix in ('','.exitcode','.verify','.verify.exitcode'):
                    require(not Path(str(log)+suffix).exists(),'Existing run log/output: '+str(log)+suffix)
                bindings=copy.deepcopy(shared)
                for key,artifact in [('developer_manifest',domain+'/developer'),('developer_labels',domain+'/developer_labels'),('stream_manifest',domain+'/stream_'+str(seed))]:
                    x=prepared['artifacts'][artifact];bindings[key]={'path':x['path'],'sha256':x['sha256']};checked_file(bindings[key])
                if method.startswith(('E','G')):
                    bindings.update(calibration=bind(bundle_path),calibration_proof=bind(proof_path))
                spec=dict(schema='odr003-task-v1',run_id=ident,method=method,domain=domain,stream=str(seed),seed=seed,
                    output_dir=str(directory),bindings=bindings,code_sha256=code,checkpoint_every_updates=64)
                path=TASK_DIR/(ident+'.json');write(path,spec);task_files.append(bind(path))
                assets=copy.deepcopy(common_assets)
                assets.update({v['path']:v['sha256'] for v in bindings.values() if Path(v['path']).is_relative_to(B)})
                assets[str(path)]=sha(path)
                env=copy.deepcopy(template['env']);env['PYTHONHASHSEED']=str(seed)
                new['tasks'].append(dict(id=ident,kind='odr003_base_developer_exploration',
                    command=[PY,'-B','-s','-u',str(implementation/'run_exploration.py'),'--task-json',str(path)],
                    cwd=str(N),requires=[previous],session=ident+'_20261023',log=str(log),env=env,
                    gpu=copy.deepcopy(template['gpu']),verify_gpu=False,verify_env=copy.deepcopy(template['verify_env']),
                    verify_command=[PY,'-B','-s','-u',str(implementation/'verify_exploration.py'),'--task-json',str(path),'--main-exitcode',str(log)+'.exitcode','--proof',str(directory/'proof.json')],
                    expected_outputs=[str(directory/'proof.json')],metadata=dict(source=official['source_path'],asset_sha256=assets,output=str(directory),
                        scope='one_fresh_source_N1024_B4_adaptation_then_developer512_only')))
                entries.append(dict(run_id=ident,method=method,domain=domain,stream=str(seed),directory=str(directory),task_json=bind(path),main_exitcode=str(log)+'.exitcode'))
                previous=ident
    index=TASK_DIR/'summary_index.json'
    summary_dir=N/'summaries/base_20261023'
    require(not summary_dir.exists(),'Summary output directory exists; refuse overwrite')
    index_doc=dict(schema='odr003-base-summary-index-v1',run_directories=[e['directory'] for e in entries],runs=entries,
        source_sha256=official['source_sha256'],code_sha256=code,
        bindings={key:shared[key] for key in ('official_receipt','source','config','source_core_manifest','data_prepared','data_verified')},
        developer_manifests={d:{'path':prepared['artifacts'][d+'/developer']['path'],'sha256':prepared['artifacts'][d+'/developer']['sha256']} for d in DOMAINS},
        calibration=bind(bundle_path),calibration_proof=bind(proof_path),
        summary_implementation_sha256={name:sha(implementation/name) for name in ('phase_tools.py','verify_summary.py','verify_exploration.py','risk_metrics.py')},
        expected_runs=38,expected_adaptation_runs=36,random_controls_included=False,formal_confirmation=False)
    write(index,index_doc)
    ident='odr003_base_summary';log=N/'logs'/(ident+'.log');summary=summary_dir/'summary.json';summary_proof=summary_dir/'proof.json'
    for suffix in ('','.exitcode','.verify','.verify.exitcode'):
        require(not Path(str(log)+suffix).exists(),'Summary log already exists')
    summary_assets={**common_assets,**offline_assets,str(index):sha(index)}
    summary_assets.update({x['path']:x['sha256'] for x in task_files})
    for e in offline_entries:summary_assets[e['task_json']['path']]=e['task_json']['sha256']
    cpu_env=copy.deepcopy(calibration_task['env']);cpu_env['CUDA_VISIBLE_DEVICES']=''
    new['tasks'].append(dict(id=ident,kind='odr003_base_developer_summary',
        command=[PY,'-B','-s','-u',str(implementation/'phase_tools.py'),'summarize','--index',str(index),'--out',str(summary)],
        cwd=str(N),requires=[previous],session=ident+'_20261023',log=str(log),env=cpu_env,gpu=False,verify_gpu=False,
        verify_env=copy.deepcopy(calibration_task.get('verify_env',{'CUDA_VISIBLE_DEVICES':''})),
        verify_command=[PY,'-B','-s','-u',str(implementation/'verify_summary.py'),'--index',str(index),'--summary',str(summary),'--main-exitcode',str(log)+'.exitcode','--proof',str(summary_proof)],
        expected_outputs=[str(summary_proof)],metadata=dict(source=official['source_path'],asset_sha256=summary_assets,output=str(summary_dir))))
    require(new['tasks'][:3]==old['tasks'] and len(new['tasks'])==40,'Candidate does not preserve exact3 prefix')
    runner.validate_manifest(new)
    amended,audit=extend_state(before,old,new,caller_context=dict(operation='odr003_base_candidate_CPU_only',source=official['source_sha256']))
    require(amended['tasks']==before['tasks'] and amended['completed']==before['completed'],'Revision preview altered old execution records')
    for task in new['tasks'][3:]:
        error=runner.asset_hash_error(task)
        require(error is None,str(error))
    # The running manager can refresh timestamps during CPU preparation. Its
    # completed task evidence, identity and phase must nevertheless stay intact.
    after=read(N/'status.json')
    for key in ('tasks','completed','status','active','manifest_sha256'):
        require(after[key]==before[key],'Production state changed during candidate preparation: '+key)
    write(Q/'proposal.json',new)
    write(Q/'build_proof.json',dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),
        old_manifest_path=str(Path(old_path).resolve()),old_manifest_sha256=runner.manifest_hash(old),old_count=3,new_count=40,
        old_prefix_preserved=True,old_execution_records_preserved=True,new_manifest_sha256=runner.manifest_hash(new),
        proposal_file_sha256=sha(Q/'proposal.json'),source_sha256=official['source_sha256'],calibration=bind(bundle_path),calibration_proof=bind(proof_path),
        task_jsons=task_files,summary_index=bind(index),new_task_ids=[x['id'] for x in new['tasks'][3:]],
        adaptation_runs_added=36,summary_tasks_added=1,random_runs_added=0,scientific_runs_started=0,
        manager_or_runtime_modified=False,revision_preview=audit,builder_sha256=sha(__file__),
        activation_entrypoint=str(N/'implementation/launch_plan.py'),
        activation_note='Main model must validate candidate and perform the already authorized manager-only switch through launch_plan.py, which retains oldP->newN dual locks. This builder performs no switch.',
        historical_plan_preserved=str(P),historical52_not_in_new_queue=True))
    print(json.dumps(dict(status='passed',candidate=str(Q/'proposal.json'),tasks=40,activated=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--from-manifest',default=str(OLD_DEFAULT))
    main(parser.parse_args().from_manifest)
