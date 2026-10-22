"""Append only two Offline developer evaluations and frozen shared calibration.

Build/verification is CPU-only. This does not stop or launch the manager. The
36 adaptation task definitions are bound only after calibration actually exists.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

B=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
P=B/'spnv2_runs/plans/rebaseline_20261017'
N=B/'spnv2_runs/plans/odr003_exploratory_20261023'
Q=N/'queue_calibration_20261023'
OLD=P/'queue_proposals/table2_pilots_20261022/proposal.json'
PY='/home/ps/anaconda3/envs/SPNv2/bin/python'
sys.path.insert(0,str(P/'pipeline'))
import runner
from revision import extend_state

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    assert os.environ.get('STY'),'Build inside a unique guarded screen'
    assert not Q.exists(),'No overwrite of a built candidate'
    official=read(N/'preflight/official_source_receipt.json');assert official['status']=='passed'
    engineering=read(N/'preflight/real_correctness_verified.json')
    assert engineering['status']=='passed' and engineering['strict_equality_required'] is True
    assert engineering['source_sha256']==official['source_sha256']
    assert sha(engineering['report_path'])==engineering['report_sha256']
    prepared=read(N/'preflight/data_boundary_prepared.json')
    old=read(OLD);before=read(P/'status.json')
    assert len(old['tasks'])==52 and len(before['completed'])==52 and before['active'] is None
    assert before['status']=='phase_complete'
    new=copy.deepcopy(old)
    new['scope']='original52_preserved_plus_003_official_offline_and_shared_calibration'
    new['backlog']=['Confirmed ODR003 exploration: after calibration, bind and append 36 base runs, then at most18 conditional random controls. Formal confirmation deferred.',
        'Historical remaining150 source epochs and full B/N sweeps deferred optional; not completed. Goal remains paused.']
    Q.mkdir(parents=True)
    required=('run_exploration.py','data_boundary.py','prediction.py','state_adapter.py','gate_core.py','risk_metrics.py')
    code={name:sha(N/'implementation'/name) for name in required}
    geometry=read(N/'preflight/source_core_manifest.json')
    shared={key:bind(path) for key,path in {
        'official_receipt':N/'preflight/official_source_receipt.json','source':Path(official['source_path']),
        'config':Path(official['config_path']),'source_core_manifest':N/'preflight/source_core_manifest.json',
        'data_prepared':N/'preflight/data_boundary_prepared.json','data_verified':N/'preflight/data_boundary_verified.json',
        'camera':Path(geometry['camera_path']),'keypoints':Path(geometry['keypoint_path'])}.items()}
    env=dict(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED='2021',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',
        TORCH_HOME=str(B/'spnv2_runs/env/cache/torch'),CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    # Manager asset policy only permits project-owned paths. External dataset
    # geometry remains hash-bound and checked by both task runner and verifier.
    common_assets={v['path']:v['sha256'] for v in shared.values() if Path(v['path']).is_relative_to(B)}
    common_assets.update({str(Path(geometry['root'])/k):v for k,v in geometry['files'].items()})
    common_assets.update({str(N/'implementation'/name):value for name,value in code.items()})
    for name in ('verify_exploration.py','phase_tools.py'):
        common_assets[str(N/'implementation'/name)]=sha(N/'implementation'/name)
    common_assets[str(N/'preflight/real_correctness_verified.json')]=sha(N/'preflight/real_correctness_verified.json')
    previous='pilot_table2_eh';run_dirs=[]
    for domain in ('lightbox','sunlamp'):
        ident='odr003_O_'+domain;run=N/'runs'/ident;run_dirs.append(str(run));log=N/'logs'/(ident+'.log')
        bindings=copy.deepcopy(shared)
        for key,rel in [('developer_manifest','developer_images.json'),('developer_labels','dev_labels.json')]:
            bindings[key]=bind(N/'data'/domain/rel)
        task=dict(schema='odr003-task-v1',run_id=ident,method='O',domain=domain,stream='offline',seed=2021,
                  output_dir=str(run),bindings=bindings,code_sha256=code,checkpoint_every_updates=64)
        path=Q/(ident+'.json');write(path,task)
        assets=copy.deepcopy(common_assets);assets.update({x['path']:x['sha256'] for x in bindings.values() if Path(x['path']).is_relative_to(B)});assets[str(path)]=sha(path)
        new['tasks'].append(dict(id=ident,kind='official_source_developer_evaluation',
            command=[PY,'-B','-s','-u',str(N/'implementation/run_exploration.py'),'--task-json',str(path)],
            cwd=str(N),requires=[previous],session=ident+'_20261023',log=str(log),env=env,
            gpu=dict(index=0,min_free_mb=14500,max_used_mb=1200),verify_gpu=False,
            verify_env=dict(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCH_HOME=env['TORCH_HOME']),
            verify_command=[PY,'-B','-s','-u',str(N/'implementation/verify_exploration.py'),'--task-json',str(path),'--main-exitcode',str(log)+'.exitcode','--proof',str(run/'proof.json')],
            expected_outputs=[str(run/'proof.json')],metadata=dict(source=official['source_path'],asset_sha256=assets,output=str(run))))
        previous=ident
    ident='odr003_shared_calibration';log=N/'logs'/(ident+'.log');out=N/'calibration/bundle.json';proof=N/'calibration/proof.json'
    new['tasks'].append(dict(id=ident,kind='frozen_source_developer_calibration',
        command=[PY,'-B','-s','-u',str(N/'implementation/phase_tools.py'),'calibrate','--runs',*run_dirs,'--out',str(out)],
        cwd=str(N),requires=[previous],session=ident+'_20261023',log=str(log),env={**env,'CUDA_VISIBLE_DEVICES':''},gpu=False,verify_gpu=False,
        verify_env=dict(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        verify_command=[PY,'-B','-s','-u',str(N/'implementation/phase_tools.py'),'verify-calibration','--bundle',str(out),'--main-exitcode',str(log)+'.exitcode','--proof',str(proof)],
        expected_outputs=[str(proof)],metadata=dict(asset_sha256=common_assets,source=official['source_path'])))
    runner.validate_manifest(new)
    _,audit=extend_state(before,old,new,caller_context=dict(operation='003_calibration_candidate_CPU_only'))
    assert new['tasks'][:52]==old['tasks'] and len(new['tasks'])==55
    for task in new['tasks'][52:]:
        assert runner.asset_hash_error(task) is None,task['id']
        for p in [task['log'],task['log']+'.exitcode',*task['expected_outputs']]:assert not Path(p).exists(),p
    write(Q/'proposal.json',new)
    write(Q/'build_proof.json',dict(status='passed',old_count=52,new_count=55,old_prefix_preserved=True,
        old_manifest_path=str(OLD),old_manifest_sha256=runner.manifest_hash(old),new_manifest_sha256=runner.manifest_hash(new),
        proposal_file_sha256=sha(Q/'proposal.json'),new_task_ids=[x['id'] for x in new['tasks'][52:]],
        scientific_adaptation_runs_started=0,source_sha256=official['source_sha256'],revision_preview=audit))
    print(json.dumps(dict(status='passed',candidate=str(Q/'proposal.json'),tasks=55,activated=False)))

if __name__=='__main__':main()
