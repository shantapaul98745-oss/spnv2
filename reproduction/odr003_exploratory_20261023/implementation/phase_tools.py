"""Bounded CPU phases: shared source calibration and developer-only summaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return [json.loads(line) for line in f]
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def accepted_run(directory):
    d=Path(directory);r=read(d/'report.json');proof=read(d/'proof.json')
    if r.get('status')!='passed' or proof.get('status')!='passed':
        raise ValueError('Unaccepted run '+str(d))
    # Independent verifier pins the exact producer report.
    pinned=proof.get('report_sha256')
    if pinned!=sha(d/'report.json'):raise ValueError('Independent proof/report mismatch')
    for item in r['outputs'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Output content changed')
    return r,dict(path=str(d),report_sha256=sha(d/'report.json'),proof_sha256=sha(d/'proof.json'))


def calibrate(directories,out):
    from gate_core import fit_calibration,decide
    records=[];sources=set();domains=set();dev={};inputs=[]
    for directory in directories:
        r,proof=accepted_run(directory)
        if r['method']!='O' or r['domain'] in domains:raise ValueError('One Offline O per domain required')
        domain=r['domain'];domains.add(domain);sources.add(r['source_sha256']);inputs.append(proof)
        dev[domain]=r['bindings']['developer_manifest']['sha256']
        data=rows(r['outputs']['predictions']['path'])
        ids=read(r['bindings']['developer_manifest']['path'])['items']
        if [x['sample_id'] for x in data]!=[x['sample_id'] for x in ids] or len(data)!=512:
            raise ValueError('Wrong developer prediction denominator/order')
        records.extend(data)
    if domains!={'lightbox','sunlamp'} or len(sources)!=1:raise ValueError('Source/domain identity differs')
    calibration=fit_calibration(records)
    counts={d:{m:sum(decide(m,r,calibration)[0] for r in records if r['domain']==d)
               for m in ('G50','G75','E50','E75')} for d in sorted(domains)}
    result=dict(schema='odr003-calibration-bundle-v1',status='passed',source_sha256=sources.pop(),
                developer_manifest_sha256=dev,inputs=inputs,calibration=calibration,
                source_developer_acceptance_counts=counts,developer_count_per_domain=512,
                target_labels_used_by_calibration=False,scoring_state='fixed_official_source',
                scope='shared_frozen_mapping_and_thresholds_no_future_stream_ranking')
    write(out,result)


def summarize(index,out):
    from risk_metrics import evaluate_exploration
    spec=read(index);allrows=[];identities=set();proofs=[];seen=set();developer={};resources=[]
    for directory in spec['run_directories']:
        r,proof=accepted_run(directory)
        key=(r['domain'],r['stream'],r['method'],r.get('mask',''))
        if key in seen:raise ValueError('Duplicate run')
        seen.add(key);identities.add(r['source_sha256']);proofs.append(proof)
        allrows.extend(rows(r['outputs']['risk_rows']['path']))
        ids=[x['sample_id'] for x in read(r['bindings']['developer_manifest']['path'])['items']]
        if r['domain'] in developer and ids!=developer[r['domain']]:raise ValueError('Developer lists differ')
        developer[r['domain']]=ids
        resources.append({k:r[k] for k in ['run_id','method','domain','stream','counts','timings','wall_seconds',
            'forward_counts','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','extended_valid','original_rejected']})
    if len(identities)!=1:raise ValueError('Cannot mix source weights')
    result=evaluate_exploration(allrows,developer)
    result.update(status='passed',source_sha256=identities.pop(),inputs=proofs,resources=resources,
                  formal_confirmation=False,completed_adaptation_runs=sum(k[2]!='O' for k in seen))
    write(out,result)
    lines=['# 003首轮开发集探索结果','',
           '仅探索结果；未运行正式确认或预留集/全域成绩。所有比较使用相同官方source和固定开发边界。','',
           '| 域 | 方法 | 平均误差 | 平均正退化 | 最严重5%退化 | 覆盖约束 |',
           '|---|---|---:|---:|---:|---|']
    def display(v):return 'None' if v is None else f'{v:.10g}'
    for d,domain in result['domains'].items():
        for m,s in domain['method_summaries'].items():
            values=[display(s['metrics'][key]['mean']) for key in ('mean_error','mean_positive_degradation','worst_5pct_degradation')]
            lines.append('| '+' | '.join([d,m,*values,str(s['coverage_pass'])])+' |')
    lines.extend(['','共同G设置筛选：'+str(result['selection']['selected_method']),
                  '是否满足追加随机预算对照条件：'+str(result['selection']['run_random_controls']),
                  '结果不支持正式有效性声明；所有设置与失败/缺失均保留。'])
    Path(out).with_suffix('.md').write_text('\n'.join(lines)+'\n')


def verify_calibration(bundle,exitfile,proofpath):
    from gate_core import fit_calibration
    if Path(exitfile).read_text().strip()!='0':raise ValueError('Calibration phase did not exit successfully')
    b=read(bundle);records=[]
    for spec in b['inputs']:
        r,current=accepted_run(spec['path'])
        if spec!=current or r['method']!='O' or r['source_sha256']!=b['source_sha256']:raise ValueError('Calibration inputs differ')
        records.extend(rows(r['outputs']['predictions']['path']))
    if fit_calibration(records)!=b['calibration']:raise ValueError('Calibration recomputation mismatch')
    write(proofpath,dict(status='passed',bundle_path=str(bundle),bundle_sha256=sha(bundle),
                        source_sha256=b['source_sha256'],developer_records=len(records),
                        recomputed=True,scope='CPU source-prediction calibration verification'))


def main():
    assert os.environ.get('STY'),'Use a unique guarded screen'
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('calibrate');p.add_argument('--runs',nargs=2,required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('summarize');p.add_argument('--index',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('verify-calibration');p.add_argument('--bundle',required=True);p.add_argument('--main-exitcode',required=True);p.add_argument('--proof',required=True)
    a=parser.parse_args()
    if a.command=='calibrate':calibrate(a.runs,a.out)
    elif a.command=='summarize':summarize(a.index,a.out)
    else:verify_calibration(a.bundle,a.main_exitcode,a.proof)
    print(json.dumps(dict(status='passed',phase=a.command)))

if __name__=='__main__':main()
