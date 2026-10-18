"""Accept saved old/new source comparison and verify the original phi6 continued."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys
import yaml

Q=Path(__file__).resolve().parent
P=Q.parent/'rebaseline_20261017'
B=P.parents[2]
sys.path.insert(0,str(Q))
import verify_comparison as vc
import pause_controller as pc

def read(p): return json.loads(Path(p).read_text())
def bind(p): return dict(path=str(p),sha256=vc.sha(p))
def canonical_cfg(p):
    value=yaml.safe_load(Path(p).read_text())
    for key in ['EXP_NAME','OUTPUT_DIR','LOG_DIR']:value.pop(key)
    value['TEST'].pop('MODEL_FILE')
    return value

def main():
    vc.require(os.environ.get('STY'),'Use independent CPU screen')
    vc.require(not (Q/'reports/source_old_new_verified.json').exists(),'Preserve accepted comparison')
    m=read(Q/'manifest.json');old=vc.collect(m)
    vc.require(read(Q/'reports/comparison_verified.json')['status']=='passed','Old summary not verified')
    expected={'synthetic':11994,'lightbox':6740,'sunlamp':2791}
    current_final='7692e3cb2ac9c1890ff3881aa42f70d0cc81175740e4c8795466e7eda01e011a'
    new_source=read(P/'reports/source_phi3_verified.json')
    vc.require(new_source['status']=='passed' and new_source['final_sha256']==current_final,'New source not accepted')
    vc.require(vc.sha(new_source['final'])==current_final,'New final changed')
    models=[];binding=[]
    for item in old:
        domain=item['domain'];task=next(t for t in m['tasks'] if t['domain_id']==domain)
        new_path=P/'reports'/('eval_phi3_'+domain+'_verified.json');base=read(new_path)
        vc.require(base['status']=='passed' and base['source_sha256']==current_final,'New evaluation binding failed')
        link=base['extension']
        for k in ['report','verification','entrypoint_verification']:
            vc.require(vc.sha(link[k]['path'])==link[k]['sha256'],'New linked proof changed')
        report=read(link['report']['path'])
        for k in ['verification','entrypoint_verification']:
            vc.require(read(link[k]['path'])['status']=='passed','New proof did not pass')
        vc.require(report['domain']==domain and report['context']['weights']['source_final']['sha256']==current_final,'New capture binding mismatch')
        new_cfg=P/'configs'/('phi3_bn_ehs_eval_'+domain+'.yaml')
        vc.require(report['context']['configuration']['sha256']==vc.sha(new_cfg),'New configuration changed')
        vc.require(canonical_cfg(task['cfg'])==canonical_cfg(new_cfg),'Old/new inference configuration differs')
        vc.require(report['sample_id_sha256']==item['sample_id_sha256'],'Ordered sample identity differs')
        old_ext=read(item['extension']['report']['path'])
        for key in ['observer_code_sha256','wrapper_code_sha256','wrapper_protocol_sha256','frozen_source_manifest']:
            vc.require(old_ext['context'][key]==report['context'][key],'Inference/observer protocol differs: '+key)
        metric=vc.full_sample_metrics(report,expected[domain])
        models.append(dict(domain=domain,n=expected[domain],old={k:item['metrics'][k]['mean'] for k in ['eT','eR','pose']},
                           new={k:metric[k]['mean'] for k in ['eT','eR','pose']},old_counts=item['counts'],new_counts=report['counts'],
                           old_accepted_score=item['accepted_subset']['metrics']['final_pose']['mean'],
                           new_accepted_score=base['summary']['metrics']['final_pose']['mean']))
        binding.extend([bind(task['verified_report']),bind(new_path),*link.values()])
    paper={(r['domain'],r['metric']):r['value'] for r in vc.paper_rows(read(Q/'reference/paper_reference.json'))}
    mapping={'eT':'effi_eT','eR':'heat_eR','pose':'final_pose'}
    labels={'eT':'E平移误差（m）','eR':'H旋转误差（°）','pose':'综合分数'}
    rows=[]
    for item in models:
        for key,refkey in mapping.items():
            ref=paper[item['domain'],refkey];o=item['old'][key];n=item['new'][key]
            rows.append(dict(domain=item['domain'],metric=key,paper=ref,old=o,new=n,
                             old_minus_paper=o-ref,old_vs_paper_percent=(o/ref-1)*100,
                             new_minus_paper=n-ref,new_vs_paper_percent=(n/ref-1)*100,
                             new_minus_old=n-o,new_vs_old_percent=(n/o-1)*100))
    history=[]
    for item in models:
        if item['domain']=='synthetic':continue
        path=B/'spnv2_runs/reports/analysis_20261008'/('phi3_'+item['domain']+'_offline_denominator.json')
        h=read(path);value=item['old']['pose'];old_h=h['all_sample_score']
        history.append(dict(domain=item['domain'],historical_score=old_h,reevaluated_score=value,
                            absolute_change=value-old_h,relative_percent=(value/old_h-1)*100,evidence=bind(path)))
    recovery=read(Q/'recovery.json');before=read(Q/'pause_intent.json');after=pc.check_production()
    vc.require(recovery['status']=='resumed' and after['worker']['state'] not in {'T','t'},'Phi6 has not resumed')
    vc.require(after['last_event']['samples_seen']>before['last_event']['samples_seen'],'Phi6 samples did not advance')
    vc.require(after['last_event']['timestamp']>dt.datetime.fromisoformat(recovery['at']).timestamp(),'No post-resume training event')
    for name in ['controller.log','watchdog.log','summary_cpu.log']:
        vc.require((Q/'logs'/(name+'.exitcode')).read_text().strip()=='0','Controller/summary exit failed')
    status=read(Q/'status.json');vc.require(status['training_resumed'] and len(status['accepted'])==3 and not status['error'],'Evaluation did not complete')
    phase_evidence=[]
    for task in m['tasks']:
        for phase in ['main','verify']:
            result=Q/'events'/(task['id']+'.'+phase+'.result.json')
            rec=read(result);vc.require(rec['exit_code']==0,'Phase exit failed')
            phase_evidence.append(bind(result))
    duration=(dt.datetime.fromisoformat(recovery['at'])-dt.datetime.fromisoformat(before['at'])).total_seconds()
    result=dict(status='passed',verified_at=vc.now(),scope='fixed old/new phi3 source versus paper Table4; no training or ODR',
                rows=rows,models=models,historical_comparison=history,evidence=binding,phase_exits=phase_evidence,
                manifest=bind(Q/'manifest.json'),paper_reference=bind(Q/'reference/paper_reference.json'),
                phi6_resume=dict(before=before,after=after,recovery=recovery,paused_seconds=duration),
                limitations=['Matched inference and sample denominator, but source training is not a strict paired experiment.',
                             'Physical BN batch4 is not paper four-GPU SyncBN.',
                             'Paper values use published rounding; synthetic is validation, not hidden test.'])
    lines=['# SPNv2 论文、旧模型与累积模型统一对比','',f'验收时间：{vc.now()}（北京时间）','',
           '旧模型：物理batch4、累积1，固定历史20轮final，本次完整重新评价；新模型：物理batch4、累积4，复用已通过验收的20轮final及同入口结果。两者推理代码、后处理、数据顺序、评价设置和同次前向observer逐项核对一致。', '',
           '论文参考：Table4 φ3-BN EHS source。E平移、H旋转以及混合分数均统计完整样本；分数越小越好。synthetic指11994张validation。', '',
           '| 数据域 | 指标 | 论文source | 旧模型，不累积 | 新模型，累积4 |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['domain']} | {labels[r['metric']]} | {r['paper']:.6f} | {r['old']:.6f} | {r['new']:.6f} |")
    lines += ['', '## 相对论文的差距', '', '绝对差＝本地−论文；括号为相对差。正数表示误差更大。', '',
              '| 数据域 | 指标 | 旧模型差距 | 新模型差距 |','|---|---|---:|---:|']
    for r in rows:lines.append(f"| {r['domain']} | {labels[r['metric']]} | {r['old_minus_paper']:+.6f}（{r['old_vs_paper_percent']:+.2f}%） | {r['new_minus_paper']:+.6f}（{r['new_vs_paper_percent']:+.2f}%） |")
    lines += ['', '## 新模型相对旧模型', '', '| 数据域 | 指标 | 绝对变化 | 相对变化 |','|---|---|---:|---:|']
    for r in rows:lines.append(f"| {r['domain']} | {labels[r['metric']]} | {r['new_minus_old']:+.6f} | {r['new_vs_old_percent']:+.2f}% |")
    lines += ['', '## 分母与原接受子集', '', '拒绝前仍有限的H预测参与全部样本统计。缺失未填零，也未用E旋转代替H；接受子集分数另表保存。', '',
              '| 域 | 完整分母（两模型均覆盖） | 旧拒绝数 | 新拒绝数 | 旧接受子集分数 | 新接受子集分数 |',
              '|---|---:|---:|---:|---:|---:|']
    for x in models:lines.append(f"| {x['domain']} | {x['n']} | {x['n']-x['old_counts']['original_accepted']} | {x['n']-x['new_counts']['original_accepted']} | {x['old_accepted_score']:.9f} | {x['new_accepted_score']:.9f} |")
    lines += ['', '## 与旧历史补评比较', '', '| 域 | 历史拼接分数 | 本次完整保存分数 | 绝对变化 | 相对变化 |','|---|---:|---:|---:|---:|']
    for h in history:lines.append(f"| {h['domain']} | {h['historical_score']:.9f} | {h['reevaluated_score']:.9f} | {h['absolute_change']:+.9f} | {h['relative_percent']:+.4f}% |")
    lines += ['', '这次实际差异很小，历史补评没有造成此前与论文的大幅差距。本次完整结果仍有可观的目标域误差；旧历史计算与本次推理数值路径并非逐位相同，不能要求完全相等。', '',
              '新模型在lightbox三项均改善；sunlamp旋转略改善，但平移变差，综合分数略变差；synthetic三项略变差。梯度累积没有带来所有域的一致提升，也没有消除与论文的差距。训练不是只改累积的严格配对实验，不能据此把变化全部归因于累积。物理BN batch4与论文四卡SyncBN的训练条件仍有差异。', '',
              '## φ6恢复', '', f"原训练PID2584192/start527383895于{recovery['at']}继续；暂停约{duration:.0f}秒。读取到恢复后的训练事件，samples_seen从{before['last_event']['samples_seen']}增长到{after['last_event']['samples_seen']}。原管理器和49项生产队列保持，不重载、不回退、不重新初始化。", '',
              '完整10配置190轮及后续ODR/消融/扫描/案例/材料核验/最终验收继续按原计划；本次仅完成用户追加的旧固定模型补评与对照。', '',
              '机器可读证据及全部未舍入数值：reports/source_old_new_verified.json；逐样本原始结果：evaluations/；扩展逐样本文件与独立proof见各reports/*_verified.json链接。', '']
    vc.atomic_json(Q/'reports/source_old_new_verified.json',result)
    vc.atomic_write(Q/'SOURCE_COMPARISON.md','\n'.join(lines))
    print(json.dumps(dict(status='passed',rows=rows,history=history,pause_seconds=duration,phi6_samples=after['last_event']['samples_seen']),ensure_ascii=False))

if __name__=='__main__':main()
