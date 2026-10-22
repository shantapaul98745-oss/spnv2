"""Independent CPU acceptance of the38-run developer-only base-stage summary.

Reopens actual upstream proofs/reports/risk rows, validates the complete frozen
matrix, and recomputes risk_metrics. No model forward, target GT or queue state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

PLAN=Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_exploratory_20261023')
DATA_ROOT=Path('/home/DATA/Downloads/DataSet/speedplus')
DOMAINS=('lightbox','sunlamp')
STREAMS=('2021','2022','2023')
METHODS=('A','V','E50','E75','G50','G75')
RESOURCE_FIELDS=('run_id','method','domain','stream','counts','timings','wall_seconds','forward_counts',
    'peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','extended_valid','original_rejected')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def finite_json(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite JSON numeric value')
    elif isinstance(value,dict):
        for item in value.values():finite_json(item)
    elif isinstance(value,list):
        for item in value:finite_json(item)


def read(path):
    def reject(value):raise ValueError('Nonfinite JSON constant: '+value)
    result=json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)
    finite_json(result)
    return result


def rows(path):
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()),'Blank risk evidence line')
            value=json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite risk constant')))
            require(isinstance(value,dict),'Risk row is not an object');finite_json(value);result.append(value)
    return result


def binding(spec):
    require(isinstance(spec,dict) and {'path','sha256'}<=spec.keys(),'Invalid file binding')
    path=Path(spec['path']).resolve()
    require(path.is_file() and sha(path)==spec['sha256'],'Bound bytes missing or changed: '+str(path))
    if 'bytes' in spec:require(path.stat().st_size==spec['bytes'],'Bound size changed')
    return path


def deny_labels(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(args[0])).resolve()
    if path.name in {'dev_labels.json','test.json'} or (path.is_relative_to(DATA_ROOT) and ('labels' in path.parts or path.suffix.lower()=='.csv')):
        raise PermissionError('Summary verifier must not read any target GT labels')


def verify(index_path,summary_path,exitcode_path,proof_path):
    require(bool(os.environ.get('STY')),'Use a unique guarded screen')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Summary verifier is CPU-only')
    require(Path(exitcode_path).read_text().strip()=='0','Summary producer did not exit0')
    require(not Path(proof_path).exists(),'Do not overwrite an existing acceptance proof')
    sys.addaudithook(deny_labels)
    index_path=Path(index_path).resolve();summary_path=Path(summary_path).resolve();proof_path=Path(proof_path).resolve()
    require(index_path.is_relative_to(PLAN/'queue_base'),'Unexpected frozen index path')
    require(summary_path==PLAN/'summaries/base_20261023/summary.json' and proof_path==summary_path.with_name('proof.json'),'Unexpected base summary/proof path')
    spec=read(index_path)
    require(spec['schema']=='odr003-base-summary-index-v1','Wrong base index schema')
    require(spec['expected_runs']==38 and spec['expected_adaptation_runs']==36 and spec['random_controls_included'] is False and spec['formal_confirmation'] is False,'Unconfirmed matrix/scope')
    require(len(spec['runs'])==38 and spec['run_directories']==[r['directory'] for r in spec['runs']],'Index run list mismatch')
    require(len(set(spec['run_directories']))==38,'Duplicate indexed run directory')
    implementation=Path(__file__).resolve().parent
    for name,expected in spec['code_sha256'].items():require(sha(implementation/name)==expected,'Run implementation changed: '+name)
    require(set(spec['summary_implementation_sha256'])=={'phase_tools.py','verify_summary.py','verify_exploration.py','risk_metrics.py'},'Summary verifier code coverage')
    for name,expected in spec['summary_implementation_sha256'].items():require(sha(implementation/name)==expected,'Summary code changed: '+name)
    static_paths={key:binding(value) for key,value in spec['bindings'].items()}
    official=read(static_paths['official_receipt']);prepared=read(static_paths['data_prepared']);verified=read(static_paths['data_verified'])
    require(official['status']=='passed' and official['source_offline_identity_verified'] is True and official['strict_model_load_performed'] is True,'Official source not accepted')
    require(official['source_sha256']==spec['source_sha256']==spec['bindings']['source']['sha256'],'Summary source identity mismatch')
    require(prepared['status']=='prepared' and verified['status']=='passed' and verified['prepared_proof_sha256']==spec['bindings']['data_prepared']['sha256'],'Data boundary receipt mismatch')
    bundle=read(binding(spec['calibration']));calibration_proof=read(binding(spec['calibration_proof']))
    require(bundle['status']=='passed' and calibration_proof['status']=='passed' and calibration_proof['recomputed'] is True and calibration_proof['developer_records']==1024,'Calibration not independently accepted')
    require(calibration_proof['bundle_sha256']==spec['calibration']['sha256'] and bundle['source_sha256']==calibration_proof['source_sha256']==spec['source_sha256'],'Calibration bytes/source differ')
    developer={}
    for domain in DOMAINS:
        expected=prepared['artifacts'][domain+'/developer']
        path=binding(spec['developer_manifests'][domain])
        require(str(path)==str(Path(expected['path']).resolve()) and spec['developer_manifests'][domain]['sha256']==expected['sha256'],'Developer boundary differs from prepared evidence')
        manifest=read(path)
        require(manifest['contains_gt'] is False,'Developer image manifest contains labels')
        ids=[r['sample_id'] for r in manifest['items']]
        require(len(ids)==512 and len(set(ids))==512 and all(x.startswith(domain+'/') for x in ids),'Wrong developer denominator/domain')
        developer[domain]=ids
    expected_keys={(d,'offline','O') for d in DOMAINS}|{(d,s,m) for d in DOMAINS for s in STREAMS for m in METHODS}
    actual_keys={(r['domain'],r['stream'],r['method']) for r in spec['runs']}
    require(actual_keys==expected_keys and len(actual_keys)==38,'Missing or extra base run')
    all_rows=[];inputs=[];resources=[];validated=[];offline_inputs=[]
    for entry in spec['runs']:
        directory=Path(entry['directory']).resolve()
        require(directory.is_relative_to(PLAN/'runs') and directory.name==entry['run_id'],'Run output identity/path mismatch')
        require(not (directory/'failure.json').exists() and not (directory/'proof.json.failure.json').exists(),'Run retains a failure artifact')
        task_path=binding(entry['task_json']);task=read(task_path)
        require(task['output_dir']==str(directory),'Task points to different run')
        for key in ('run_id','method','domain','stream'):require(task[key]==entry[key],'Index/task identity mismatch')
        require(task['code_sha256']==spec['code_sha256'],'Mixed run implementation identity')
        report_path=directory/'report.json';upstream_proof_path=directory/'proof.json'
        report=read(report_path);proof=read(upstream_proof_path)
        require(report['status']=='passed' and proof['status']=='passed' and proof['main_exitcode']==0,'Run not independently accepted')
        require(proof['report_sha256']==sha(report_path) and proof['task_sha256']==entry['task_json']['sha256']==report['task_sha256'],'Upstream report/task bytes differ')
        require(Path(proof['main_exitcode_path']).resolve()==Path(entry['main_exitcode']).resolve() and Path(entry['main_exitcode']).read_text().strip()=='0','Main exit evidence differs')
        require(proof['verifier_sha256']==spec['summary_implementation_sha256']['verify_exploration.py'],'Upstream verifier identity differs')
        for key in ('run_id','method','domain','stream','bindings','code_sha256'):
            require(report[key]==task[key]==proof[key],'Proof/report/task disagree: '+key)
        require(report['source_sha256']==proof['source_sha256']==spec['source_sha256'],'Mixed model source')
        require(proof['outputs']==report['outputs'] and proof['counts']==report['counts'],'Proof output/count bindings differ')
        require(proof['developer_count']==512 and proof['independently_recomputed_metric_rows']==512,'Missing independent full developer evidence')
        require(proof['developer_only_labels_read'] is True and proof['reserved_GT_read'] is False and proof['full_source_GT_read'] is False,'Upstream GT boundary failed')
        require(proof['verifier_forward_passes']==0 and proof['cuda_initialized'] is False,'Unexpected independent-verifier execution profile')
        method,domain,stream=entry['method'],entry['domain'],entry['stream']
        dev_binding=task['bindings']['developer_manifest']
        require(dev_binding==spec['developer_manifests'][domain],'Run developer boundary differs')
        if method.startswith(('E','G')):
            require(task['bindings']['calibration']==spec['calibration'] and task['bindings']['calibration_proof']==spec['calibration_proof'],'Mixed gate calibration')
        outputs={key:binding(value) for key,value in report['outputs'].items()}
        require(outputs['risk_rows']==directory/'risk_rows.jsonl','Unexpected risk evidence file')
        risk=rows(outputs['risk_rows'])
        require(len(risk)==(1536 if method=='O' else 512),'Missing full risk rows')
        require(proof['risk_rows']==len(risk),'Independent risk row count differs')
        for row in risk:
            require(row['method']==method and row['domain']==domain and 'mask' not in row,'Risk row method/domain or unexpected R mask')
            require(row['stream'] in STREAMS if method=='O' else row['stream']==stream,'Risk row stream differs')
        all_rows.extend(risk)
        input_record=dict(path=str(directory),report_sha256=sha(report_path),proof_sha256=sha(upstream_proof_path))
        inputs.append(input_record)
        if method=='O':offline_inputs.append(input_record)
        resources.append({key:report[key] for key in RESOURCE_FIELDS})
        validated.append(dict(run_id=entry['run_id'],task_sha256=entry['task_json']['sha256'],**input_record))
    require(bundle['inputs']==offline_inputs,'Calibration did not use the exact accepted Offline runs summarized here')
    require(len(all_rows)==21504,'Full paired risk matrix count differs')
    from risk_metrics import evaluate_exploration
    recomputed=evaluate_exploration(all_rows,developer)
    summary=read(summary_path)
    for key in ('schema_version','protocol','domains','selection'):
        require(summary[key]==recomputed[key],'Independently recomputed summary differs: '+key)
    require(summary['status']=='passed' and summary['formal_confirmation'] is False and summary['completed_adaptation_runs']==36,'Producer completion/scope differs')
    require(summary['source_sha256']==spec['source_sha256'] and summary['inputs']==inputs and summary['resources']==resources,'Producer input/resource identity differs')
    markdown=summary_path.with_suffix('.md')
    require(markdown.is_file() and markdown.stat().st_size>0,'Human-readable summary missing')
    # Recheck mutable upstream metadata after all computation, without reopening GT.
    for item in validated:
        directory=Path(item['path'])
        require(sha(directory/'report.json')==item['report_sha256'] and sha(directory/'proof.json')==item['proof_sha256'],'Upstream evidence changed during summary acceptance')
    require(sha(implementation/'risk_metrics.py')==spec['summary_implementation_sha256']['risk_metrics.py'],'Metric code changed during acceptance')
    result=dict(schema='odr003-independent-base-summary-proof-v1',status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        index_path=str(index_path),index_sha256=sha(index_path),summary_path=str(summary_path),summary_sha256=sha(summary_path),
        markdown_path=str(markdown),markdown_sha256=sha(markdown),main_exitcode_path=str(Path(exitcode_path).resolve()),main_exitcode=0,
        source_sha256=spec['source_sha256'],run_count=38,adaptation_runs=36,offline_runs=2,
        developer_count_per_domain=512,risk_record_count=len(all_rows),independent_unit='three_observation_streams',
        core_fields_recomputed=['protocol','domains','selection'],selection=recomputed['selection'],
        inputs=validated,calibration=spec['calibration'],calibration_proof=spec['calibration_proof'],
        target_GT_files_read=False,model_loaded=False,forward_passes=0,random_runs_executed_by_verifier=0,
        formal_confirmation=False,verifier_path=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),
        next_action='Only a qualifying shared G setting permits separately prepared at-most18 matched-random tasks; no automatic launches here')
    with proof_path.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status='passed',proof=str(proof_path),runs=38,adaptation_runs=36,selected_method=recomputed['selection']['selected_method'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',required=True);parser.add_argument('--summary',required=True)
    parser.add_argument('--main-exitcode',required=True);parser.add_argument('--proof',required=True)
    args=parser.parse_args()
    try:verify(args.index,args.summary,args.main_exitcode,args.proof)
    except BaseException as error:
        failure=Path(args.proof).with_name(Path(args.proof).name+'.failure.json')
        if not failure.exists():
            failure.parent.mkdir(parents=True,exist_ok=True)
            with failure.open('x',encoding='utf-8') as f:
                json.dump(dict(status='failed',error=repr(error),traceback=traceback.format_exc()),f,indent=2);f.write('\n')
        raise
