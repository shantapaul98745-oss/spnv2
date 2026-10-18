"""Verify fixed-weight full-domain evaluations and produce a paper Table 4 comparison."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

Q = Path(__file__).resolve().parent
P = Q.parent / 'rebaseline_20261017'
sys.path.insert(0, str(Q / 'pipeline'))
from progress import atomic_json, atomic_write, now


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_assets(manifest):
    for path, expected in manifest['shared_assets'].items():
        require(sha(path) == expected, 'Pinned input changed: ' + path)


def full_sample_metrics(report, n):
    require(report['sample_count'] == n and report['counts']['total'] == n,
            'Unexpected sample count')
    primary = report['extended_primary']
    require(primary['applicable'], 'Hybrid result is not applicable')
    selected = {}
    for key in ('eT', 'eR', 'pose'):
        item = primary['metrics'][key]
        require(item['total'] == n and item['count'] == n and item['missing'] == 0
                and item['denominator_scope'] == 'all_samples',
                key + ': incomplete H/E coverage; preserve missing predictions, do not substitute or fill zero')
        require(isinstance(item['mean'], (float, int)) and math.isfinite(item['mean']),
                key + ': nonfinite mean')
        selected[key] = dict(item)
    return selected


def verify_task(manifest, task_id):
    task = next(t for t in manifest['tasks'] if t['id'] == task_id)
    model = next(m for m in manifest['models'] if m['id'] == task['model_id'])
    domain = next(d for d in manifest['domains'] if d['id'] == task['domain_id'])
    require(not Path(task['verified_report']).exists(), 'Comparison acceptance already exists')
    verify_assets(manifest)
    require(sha(model['final']) == model['final_sha256'], 'Pinned final changed')
    subprocess.run(task['base_verify_command'], check=True)
    base = read(task['base_verified_report'])
    require(base['status'] == 'passed' and base['source_sha256'] == model['final_sha256'],
            'Base proof did not accept the pinned final')
    link = base['extension']
    for key in ('report', 'verification', 'entrypoint_verification'):
        require(sha(link[key]['path']) == link[key]['sha256'], 'Linked proof hash mismatch: ' + key)
    report = read(link['report']['path'])
    proof = read(link['verification']['path'])
    entry = read(link['entrypoint_verification']['path'])
    require(proof['status'] == entry['status'] == 'passed', 'Independent proof failed')
    require(report['domain'] == domain['id'], 'Domain mismatch')
    require(report['context']['weights']['source_final']['sha256'] == model['final_sha256'],
            'Observer captured a different weight')
    require(report['context']['configuration']['sha256'] == sha(task['cfg']), 'Configuration mismatch')
    metrics = full_sample_metrics(report, domain['expected_samples'])
    verify_assets(manifest)
    require(sha(model['final']) == model['final_sha256'], 'Final changed during evaluation')
    result = dict(schema='spnv2-unified-source-full-samples-verified-v1', status='passed',
                  verified_at=now(), task_id=task_id, model_id=model['id'], domain=domain['id'],
                  final=model['final'], final_sha256=model['final_sha256'],
                  manifest_sha256=sha(Q / 'manifest.json'), cfg_sha256=sha(task['cfg']),
                  csv_sha256=sha(domain['csv']), sample_id_sha256=report['sample_id_sha256'],
                  counts=report['counts'], metrics=metrics,
                  accepted_subset=base['summary'], base_report=str(task['base_verified_report']),
                  base_report_sha256=sha(task['base_verified_report']), extension=link)
    atomic_json(task['verified_report'], result)
    print(json.dumps({'event': 'full_samples_accepted', 'task_id': task_id,
                      'n': domain['expected_samples'], 'metrics': metrics}, ensure_ascii=False), flush=True)


def collect(manifest):
    verify_assets(manifest)
    records = []
    domain_ids = {}
    for task in manifest['tasks']:
        result = read(task['verified_report'])
        require(result['status'] == 'passed' and result['manifest_sha256'] == sha(Q / 'manifest.json'),
                'Unaccepted or differently bound result: ' + task['id'])
        require(result['task_id'] == task['id'], 'Task identity mismatch')
        require(sha(result['base_report']) == result['base_report_sha256'], 'Base proof changed')
        for value in result['extension'].values():
            if isinstance(value, dict) and 'path' in value and 'sha256' in value:
                require(sha(value['path']) == value['sha256'], 'Extension changed')
        key = result['domain']
        identity = (result['csv_sha256'], result['sample_id_sha256'])
        require(key not in domain_ids or domain_ids[key] == identity,
                'Models were not evaluated on the identical ordered sample list')
        domain_ids[key] = identity
        records.append(result)
    expected = {(m['id'], d['id']) for m in manifest['models'] for d in manifest['domains']}
    require(len(records) == len(expected) and {(r['model_id'], r['domain']) for r in records} == expected,
            'Expected exactly the bound models times domains')
    return records


def paper_rows(value):
    if isinstance(value, list):
        for item in value:
            yield from paper_rows(item)
    elif isinstance(value, dict):
        if value.get('table') == 'Table4' and value.get('config') == 'phi3_bn_ehs':
            yield value
        else:
            for item in value.values():
                yield from paper_rows(item)


def build_rows(manifest, records, reference):
    keymap = {'eT': 'effi_eT', 'eR': 'heat_eR', 'pose': 'final_pose'}
    paper = {(r['domain'], r['metric']): r['value'] for r in paper_rows(reference)}
    require(len(paper) == 9, 'Expected nine paper Table4 source reference values')
    rows = []
    for r in records:
        for key, metric in keymap.items():
            actual = r['metrics'][key]['mean']
            ref = paper[(r['domain'], metric)]
            rows.append(dict(model_id=r['model_id'], domain=r['domain'], metric=key,
                             unit=r['metrics'][key]['unit'], count=r['metrics'][key]['count'],
                             paper=ref, value=actual, delta=actual-ref,
                             relative_percent=(actual/ref-1)*100))
    return rows


def render(manifest, records, rows):
    models = manifest['models']
    labels = {'eT': 'E平移误差（m）', 'eR': 'H旋转误差（°）', 'pose': '综合分数'}
    lines = ['# SPNv2 旧模型完整保存补评与论文 source 对比', '',
             '完成时间（北京时间）：' + now(), '',
             '固定旧模型 final，使用已验收的同次推理入口重新评价三个完整数据域，保存每张图像的预测与拒绝标记。没有重新训练或执行 ODR。', '',
             '论文参考为 Table 4 的 φ3-BN EHS source；数值越小越好。全部结果逐项通过退出、严格重载、原评价、同次前向扩展和全样本覆盖验收。', '',
             '| 数据域 | 指标 | 论文 | ' + ' | '.join(m['label'] for m in models) + ' |',
             '|---|---|---:|' + '---:|' * len(models)]
    indexed = {(r['model_id'], r['domain'], r['metric']): r for r in rows}
    for d in manifest['domains']:
        for key in labels:
            values = [indexed[(m['id'], d['id'], key)] for m in models]
            lines.append('| ' + d['id'] + ' | ' + labels[key] + ' | '
                         + format(values[0]['paper'], '.6f') + ' | '
                         + ' | '.join(format(r['value'], '.6f') for r in values) + ' |')
    lines += ['', '## 相对论文的差距', '',
              '绝对差＝本次值−论文值；相对差＝绝对差÷论文值。正数表示误差更大，负数表示误差更小。论文值按发表时的舍入精度引用。', '',
              '| 模型 | 数据域 | 指标 | 绝对差 | 相对差 |', '|---|---|---|---:|---:|']
    names = {m['id']: m['label'] for m in models}
    for r in rows:
        lines.append(f"| {names[r['model_id']]} | {r['domain']} | {labels[r['metric']]} | {r['delta']:+.6f} | {r['relative_percent']:+.2f}% |")
    lines += ['', '## 全样本覆盖与几何拒绝', '',
              '拒绝前仍存在的有限 H 姿态保留计分；接受子集另存，未填零、未用 E 姿态替代 H。', '',
              '| 模型 | 数据域 | 全样本分母 | 原接受数量 | 原拒绝数量 |', '|---|---|---:|---:|---:|']
    for r in records:
        c = r['counts']
        lines.append(f"| {names[r['model_id']]} | {r['domain']} | {c['total']} | {c['original_accepted']} | {c['total']-c['original_accepted']} |")
    lines += ['', '## 结论的边界', '',
              '本次仅补评旧模型。与已完成的新累积模型比较时，训练并非只改变梯度累积的严格配对实验，不能把全部差异归因于梯度累积；单卡物理 batch4 的 BN 不等于论文四卡 SyncBN。', '',
              '原独立复现仍按既定 10 配置190轮计划执行；本表仅是用户新增的固定权重对照评价，不替代任何正式训练结果。', '',
              '逐样本预测、权重哈希和原/扩展验收链接见本目录对应 reports 与 evaluations；comparison.json 与 comparison.csv 保存未舍入数值。', '']
    return '\n'.join(lines)


def summarize(manifest):
    require(not (Q / 'reports/comparison.json').exists(), 'Preserve existing summary')
    records = collect(manifest)
    reference = read(Q / 'reference/paper_reference.json')
    rows = build_rows(manifest, records, reference)
    atomic_json(Q / 'reports/comparison.json', dict(schema='spnv2-old-source-comparison-v1',
                status='awaiting_verification', created_at=now(), records=records, rows=rows,
                manifest_sha256=sha(Q / 'manifest.json')))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write(Q / 'reports/comparison.csv', stream.getvalue())
    atomic_write(Q / 'COMPARISON.md', render(manifest, records, rows))


def verify_summary(manifest):
    summary = read(Q / 'reports/comparison.json')
    records = collect(manifest)
    rows = build_rows(manifest, records, read(Q / 'reference/paper_reference.json'))
    require(summary['records'] == records and summary['rows'] == rows, 'Summary does not match verified evidence')
    require(len(rows) == len(manifest['tasks']) * 3, 'Unexpected model/domain/metric rows')
    atomic_json(Q / 'reports/comparison_verified.json', dict(status='passed', verified_at=now(),
                task_count=len(manifest['tasks']), comparison_rows=len(rows), manifest_sha256=sha(Q / 'manifest.json'),
                files={str(p): sha(p) for p in [Q/'reports/comparison.json', Q/'reports/comparison.csv', Q/'COMPARISON.md']}))


def main():
    require(os.environ.get('STY'), 'Use a unique guarded screen')
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--task-id')
    action.add_argument('--summarize', action='store_true')
    action.add_argument('--verify-summary', action='store_true')
    args = parser.parse_args()
    manifest = read(Q / 'manifest.json')
    if args.task_id:
        verify_task(manifest, args.task_id)
    elif args.summarize:
        summarize(manifest)
    else:
        verify_summary(manifest)


if __name__ == '__main__':
    main()
