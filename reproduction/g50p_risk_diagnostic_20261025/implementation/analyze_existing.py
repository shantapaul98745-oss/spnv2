"""Experiment A: inspect saved deterministic records, never run a model.

Only stdlib is imported.  Continuous signal directions, label definitions and
initial-error strata must be bound in the diagnostic manifest before analysis.
Missing signals/records stay unavailable; this module never fits score cutoffs,
changes an experiment, reads images/GT, or attributes a final error to a single
candidate update.  The input exporter owns checkpoint integrity verification.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


SOURCE_SHA = 'd8f9c281d23adaeb77d03023aee8d40ae1cb9e18dd503539c4719bbe719514cc'
DOMAINS = ('lightbox', 'sunlamp')
SEEDS = (2021, 2022, 2023)
RULES = ('new_unavailable', 'joint_geometry_worsening', 'h_rotation_change',
         'he_disagreement_increase', 'reprojection_increase')
STATIC_DIRECTIONS={'entropy':'higher','confidence':'lower','reprojection':'higher',
                   'rotation_disagreement':'higher','translation_disagreement':'higher'}
STATIC_TARGETS={'A','G50','P','RW','RP'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite saved JSON cannot be diagnosed as a finite signal')
    elif isinstance(value, dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value, list):
        for item in value: finite_tree(item)


def load_json(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    finite_tree(value)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_binding(binding, base):
    require(isinstance(binding, dict) and set(binding) >= {'path', 'sha256'}, 'Missing input binding')
    path = Path(binding['path'])
    if not path.is_absolute(): path = Path(base)/path
    path = path.resolve(strict=True)
    require(path.name not in {'dev_labels.json','test.json'} and 'labels' not in path.parts,
            'Diagnostic A accepts saved predictions/metrics, never a raw GT label file')
    require(path.is_file() and sha(path) == binding['sha256'], 'Saved diagnostic input SHA differs: '+str(path))
    return path


def quantile(values, q):
    ordered = sorted(values)
    require(bool(ordered) and 0 <= q <= 1, 'Invalid finite quantile population')
    at = (len(ordered)-1)*q
    low, high = math.floor(at), math.ceil(at)
    return ordered[low]+(ordered[high]-ordered[low])*(at-low)


def distribution(values):
    finite = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in finite), 'Nonfinite diagnostic distribution')
    return dict(total=len(values), available=len(finite), missing=len(values)-len(finite),
                minimum=min(finite) if finite else None, q10=quantile(finite,.1) if finite else None,
                median=quantile(finite,.5) if finite else None, q90=quantile(finite,.9) if finite else None,
                maximum=max(finite) if finite else None,
                mean=math.fsum(finite)/len(finite) if finite else None)


def ranking_metrics(scores, labels):
    """Tie-aware AUROC and non-interpolated AP; never choose a score threshold."""
    require(len(scores) == len(labels), 'Score/label lengths differ')
    pairs = [(float(s), y) for s,y in zip(scores,labels) if s is not None and y is not None]
    require(all(math.isfinite(s) and type(y) is bool for s,y in pairs), 'Invalid ranking input')
    positives = sum(y for _,y in pairs); negatives=len(pairs)-positives
    result = dict(population=len(scores), available=len(pairs), missing=len(scores)-len(pairs),
                  missing_score=sum(s is None for s in scores),missing_label=sum(y is None for y in labels),
                  positive=positives, negative=negatives,
                  prevalence=positives/len(pairs) if pairs else None,
                  AUROC=None, AUPRC=None, AUPRC_definition='noninterpolated average precision; tie groups together',
                  score_threshold_search=False, direction_flipped_after_results=False)
    if not positives or not negatives:
        result['reason']='both classes required; missing/one-class stratum is not evidence'
        return result
    ordered=sorted(pairs,key=lambda x:x[0]); rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum += ((i+1+j)/2)*sum(y for _,y in ordered[i:j]); i=j
    result['AUROC']=(rank_sum-positives*(positives+1)/2)/(positives*negatives)
    ordered.sort(key=lambda x:x[0],reverse=True);tp=0;seen=0;ap=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        increment=sum(y for _,y in ordered[i:j]);tp+=increment;seen+=j-i
        ap+=(increment/positives)*(tp/seen);i=j
    result['AUPRC']=ap;result['reason']=None
    return result


def decision_runs(windows):
    groups=[]
    for row in windows:
        number=row['number'];decision=row['decision']
        require(decision in ('commit','rollback'), 'Unknown saved window decision')
        if groups and groups[-1]['decision']==decision and groups[-1]['last']+1==number:
            groups[-1]['last']=number;groups[-1]['count']+=1
        else:groups.append(dict(first=number,last=number,count=1,decision=decision))
    return groups


def summary_audit(summary):
    require(summary['source_sha256']==SOURCE_SHA, 'Wrong official source')
    rows=[];candidates=0;rollbacks=0
    for domain in DOMAINS:
        for seed in SEEDS:
            key=f'P:{domain}:{seed}';resource=summary['resources_by_run'][key]
            proof=resource['window_diagnostics'];counts=resource['counts'];windows=proof['window_decisions']
            require([w['number'] for w in windows]==list(range(1,len(windows)+1)), 'Window numbers not contiguous')
            require(len(windows)==counts['candidates'] and sum(w['decision']=='rollback' for w in windows)==counts['rollbacks'],
                    'Saved summary counts disagree')
            candidates+=counts['candidates'];rollbacks+=counts['rollbacks']
            checked=sum(w['checked_images'] for w in windows)
            rule_stats={name:dict(comparable_image_pairs=proof['rule_comparable_image_counts'].get(name,0),
                triggered_image_pairs=proof['rule_triggered_image_counts'].get(name,0),
                triggered_windows=proof['rule_triggered_window_counts'].get(name,0),
                all_checked_image_pairs=checked) for name in RULES}
            for value in rule_stats.values():
                denom=value['comparable_image_pairs']
                value['computability_fraction']=denom/checked if checked else None
                value['trigger_fraction_given_comparable']=value['triggered_image_pairs']/denom if denom else None
            rows.append(dict(run_key=key,counts=counts,checked_image_pairs=checked,
                  decision_runs=decision_runs(windows),rules=rule_stats,pool_totals=proof['pool_diagnostics'],
                  amplitude_unavailable_from_summary=True,
                  point_or_source_ID_unavailable_from_summary=True))
    require((candidates,rollbacks)==(793,117), 'This A diagnostic is bound to the saved 793 candidates /117 vetoes')
    return dict(candidates=candidates,rollbacks=rollbacks,commits=candidates-rollbacks,
                rollback_fraction=rollbacks/candidates,runs=rows)


def native_margin(condition):
    value,limit=condition.get('compared_value'),condition.get('compared_threshold')
    if value is None or limit is None:return None
    return limit-value if condition.get('operator','').startswith('abs_normalized_quaternion_dot') else value-limit


def _pair_conditions(row):
    rules=row['rules'];joint=rules['joint_geometry_worsening']
    return {**{name:rules[name] for name in ('new_unavailable','joint_geometry_worsening')},
            **joint['subconditions']}


def trace_audit(doc, expected_resource):
    """Inspect recorded pairs/pool transitions, with no pose recomputation."""
    trace=doc['scientific_trace'];obs=[e for e in trace if e['kind']=='observation']
    windows=[e for e in trace if e['kind']=='window'];counts=doc['counters']
    require(len(obs)==1024 and len({e['sample_id'] for e in obs})==1024, 'Trace must be the frozen distinct1024 stream')
    require([e['observation_index'] for e in obs]==list(range(1024)), 'Observation order differs')
    require(counts==expected_resource['counts'] and len(windows)==counts['candidates'], 'Trace/summary counts differ')
    require(sum(e['accepted'] for e in obs)==counts['accepted'], 'Accepted IDs/count differ')
    byid={e['sample_id']:e for e in obs};rule_records=defaultdict(list)
    source_combinations=Counter();source_mentions=Counter();dedup_removed=0
    point_set_events=[]
    checks=defaultdict(list);triggers=defaultdict(list);trigger_reasons=defaultdict(Counter)
    pool_slots={'uniform':[],'fragile':[]};active_intervals={};intervals=[];replacement_events=[]
    accepted_signals=defaultdict(list);rejected_signals=defaultdict(list)
    for e in obs:
        for field in ('entropy','confidence','reprojection','rotation_disagreement','translation_disagreement'):
            (accepted_signals if e['accepted'] else rejected_signals)[field].append(e['arrival'].get(field))
    for window in windows:
        number=window['number'];records=window['protection_records'];protection=window['protection']
        ids=[r['sample_id'] for r in records]
        require(len(ids)==len(set(ids))==protection['checked_image_count']<=12, 'Saved protection de-duplication changed')
        require(protection['rollback']==(window['decision']=='rollback'), 'Saved protection/decision differ')
        current_end=max(byid[sid]['observation_index'] for sid in window['sample_ids'])
        paired={row['sample_id']:row for row in protection['rows']}
        require(set(paired)==set(ids), 'Saved protection pair/list mismatch')
        raw_pairs={row['sample_id']:row for row in window.get('protection_pairs',[])}
        if raw_pairs:require(set(raw_pairs)==set(ids),'Raw pre/post protection IDs differ')
        raw_mentions=0
        for record in records:
            sid=record['sample_id'];require(byid[sid]['observation_index']<=current_end, 'Future protection input')
            sources=record['protection_sources'];names=tuple(sorted({s['pool'] for s in sources}))
            require(names and set(names)<={'current_window','uniform','fragile'}, 'Unexpected protection source')
            raw_mentions+=len(sources);source_combinations['+'.join(names)]+=1
            for source in sources:
                name=source['pool'];source_mentions[name]+=1
                if name=='current_window':require(sid in window['sample_ids'], 'Current-window source differs')
                else:require(pool_slots[name][source['slot']]==sid, 'Saved historical pool source differs from transitions')
            checks[sid].append(number)
            row=paired[sid]
            raw=raw_pairs.get(sid,{})
            pre,post=raw.get('pre',{}),raw.get('post',{})
            pre_used=pre.get('used_indices');post_used=post.get('used_indices')
            changed=None if pre_used is None or post_used is None else set(pre_used)!=set(post_used)
            if row.get('point_set_changed') is not None:
                require(changed is None or changed==row['point_set_changed'],'Saved point-set diagnostic differs')
            point_set_events.append(dict(window=number,sample_id=sid,decision=window['decision'],
                point_set_changed=changed,pre_used_indices=pre_used,post_used_indices=post_used,
                pre_effective_threshold=pre.get('pnp_selection_diagnostic',{}).get('effective_threshold'),
                post_effective_threshold=post.get('pnp_selection_diagnostic',{}).get('effective_threshold'),
                pre_h_available=pre.get('h_available'),post_h_available=post.get('h_available'),
                pre_original_h_rejected=pre.get('original_h_rejected'),post_original_h_rejected=post.get('original_h_rejected')))
            if row['rollback']:
                triggers[sid].append(number);trigger_reasons[sid].update(row['reasons'])
            for name,condition in _pair_conditions(row).items():
                rule_records[name].append(dict(window=number,sample_id=sid,decision=window['decision'],
                    arrival_G50_accepted=byid[sid]['accepted'],comparable=condition['comparable'],
                    triggered=condition['triggered'],value=condition.get('value'),threshold=condition.get('threshold'),
                    native_margin=native_margin(condition)))
        dedup_removed+=raw_mentions-len(ids)
        transition=window['pool_transition'];require(transition is not None, 'P must retain pool insertion transitions')
        for event in transition['pool_events']:
            if not event['selected']:continue
            pool=event['pool'];slot=event['slot'];sid=event['sample_id'];replaced=event['replaced_sample_id']
            slots=pool_slots[pool]
            if replaced is None:
                require(slot==len(slots) and len(slots)<4, 'Unexpected reservoir initial insertion')
                slots.append(sid)
            else:
                require(slots[slot]==replaced, 'Reservoir replacement previous ID differs')
                interval=active_intervals.pop((pool,slot));interval.update(end_after_window=number,
                    end_after_observation_index=current_end,right_censored=False,
                    residence_windows=number-interval['start_after_window'],
                    residence_observations=current_end-interval['start_after_observation_index'])
                intervals.append(interval);slots[slot]=sid
                replacement_events.append(dict(pool=pool,window=number,slot=slot,old=replaced,new=sid))
            active_intervals[(pool,slot)]=dict(pool=pool,slot=slot,sample_id=sid,start_after_window=number,
                start_after_observation_index=current_end)
    final_window=len(windows)
    for interval in active_intervals.values():
        interval.update(end_after_window=final_window,end_after_observation_index=1023,right_censored=True,
            residence_windows=final_window-interval['start_after_window'],
            residence_observations=1023-interval['start_after_observation_index'])
        intervals.append(interval)
    if doc.get('pools'):
        for pool in ('uniform','fragile'):
            require(pool_slots[pool]==doc['pools'][pool]['slots'], 'Reconstructed final pool differs')
    detailed_rules={}
    for name,records in rule_records.items():
        detailed_rules[name]=dict(total_pairs=len(records),comparable=sum(r['comparable'] for r in records),
            triggered=sum(r['triggered'] is True for r in records),
            values=distribution([r['value'] for r in records]),
            positive_direction_native_margins=distribution([r['native_margin'] for r in records]),
            accepted_vs_rejected={label:dict(count=sum(r['arrival_G50_accepted']==accepted for r in records),
               triggered=sum(r['arrival_G50_accepted']==accepted and r['triggered'] is True for r in records))
               for label,accepted in [('G50_accepted',True),('G50_rejected',False)]},
            records=records)
    recurring=[]
    for sid,numbers in triggers.items():
        event_windows=[dict(number=n,decision='rollback') for n in numbers]
        recurring.append(dict(sample_id=sid,arrival_G50_accepted=byid[sid]['accepted'],
            triggering_windows=numbers,trigger_count=len(numbers),reasons=dict(trigger_reasons[sid]),
            longest_consecutive_trigger_run=max(g['count'] for g in decision_runs(event_windows)),
            checked_windows=checks[sid],checked_count=len(checks[sid])))
    recurring.sort(key=lambda r:(-r['trigger_count'],r['sample_id']))
    return dict(status='saved_trace_available',observations=1024,candidates=len(windows),
        counts=counts,decision_runs=decision_runs(windows),rules=detailed_rules,
        sources=dict(raw_mentions=dict(source_mentions),source_combinations=dict(source_combinations),
                     deduplicated_mentions_removed=dedup_removed,unique_checked_images=len(checks)),
        per_image_checks=[dict(sample_id=sid,windows=numbers,count=len(numbers)) for sid,numbers in sorted(checks.items())],
        recurring_trigger_images=recurring,residence_intervals=intervals,replacement_events=replacement_events,
        point_set_events=point_set_events,
        final_reservoir_slots=pool_slots,
        gate_accepted_vs_rejected={label:{field:distribution(values) for field,values in signals.items()}
            for label,signals in [('accepted',accepted_signals),('rejected',rejected_signals)]},
        interpretation='Saved record associations and repeated veto witnesses; not a causal harm label for an update')


def _field(row,name):
    if name in row:return row[name]
    return row.get('prediction',{}).get(name)


def _initial_error_bins(rows,quantiles,reference_ids):
    require(quantiles==[0,.25,.5,.75,1], 'Only the predeclared four equal-count initial-error layers are authorized')
    byid={r['sample_id']:r for r in rows};reference=list(reference_ids)
    require(len(reference)==len(set(reference)) and set(reference)<=set(byid),'Frozen reference boundary differs')
    ordered=sorted((byid[sid] for sid in reference if byid[sid]['valid']),key=lambda r:(r['error'],r['sample_id']))
    require(len(ordered)==len(reference),'Frozen Iref has unavailable initial error; record before interpreting ranks')
    bins={r['sample_id']:None if not r['valid'] else 'outside_reference' for r in rows}
    for index,row in enumerate(ordered):bins[row['sample_id']]=min(3,index*4//len(ordered))
    layers=[]
    for group in range(4):
        selected=[r for r in ordered if bins[r['sample_id']]==group]
        layers.append(dict(stratum=group+1,count=len(selected),sample_ids=[r['sample_id'] for r in selected],
                           minimum_initial_error=min(r['error'] for r in selected) if selected else None,
                           maximum_initial_error=max(r['error'] for r in selected) if selected else None))
    return layers,bins


def static_diagnostic(summary, offline_rows, config, safe_point=lambda:None):
    if not config:
        return dict(status='not_evaluated',reason='Signal direction/target/strata not yet bound to confirmed diagnostic protocol')
    signals=config['signals'];targets=config['targets'];quantiles=config['initial_error_quantiles']
    require(signals and targets, 'Predeclared signals and targets required')
    require(len(signals)==5 and {x['field']:x['risk_direction'] for x in signals}==STATIC_DIRECTIONS,
            'Only the five predeclared individual signal directions are authorized')
    require(len(targets)==5 and {t['method'] for t in targets}==STATIC_TARGETS and
            all(t['kind']=='greater_than' and t['field']=='positive_degradation' and
                type(t['value']) in (int,float) and t['value']==0 for t in targets),
            'Only saved finite after-minus-Offline>0 outcomes are authorized')
    output={}
    for domain in DOMAINS:
        safe_point()
        if domain not in offline_rows:
            output[domain]=dict(status='missing_saved_offline_signals',model_forward_added=0);continue
        source=offline_rows[domain];ids=[r['sample_id'] for r in source]
        require(len(ids)==len(set(ids))==512,'Offline diagnostic uses the frozen dev512 once')
        original=summary['domains'][domain]['streams']['2021']['methods']['O']['per_sample']
        by_error={r['sample_id']:r for r in original}
        require(set(ids)==set(by_error),'Offline signal/initial error ID boundary differs')
        for row in source:
            error=row.get('metrics',{}).get('error',row.get('error'))
            expected=by_error[row['sample_id']]['error']
            require((error is None and expected is None) or (error is not None and expected is not None and
                    math.isclose(error,expected,rel_tol=1e-12,abs_tol=1e-12)), 'Offline signal/initial mixed error differs')
        aligned=[dict(by_error[r['sample_id']],saved_prediction=r) for r in source]
        domain_result=[];layer_evidence={}
        for seed in SEEDS:
            safe_point()
            stream=summary['domains'][domain]['streams'][str(seed)]
            reference=set(stream['reference_ids'])
            layers,bins=_initial_error_bins(aligned,quantiles,stream['reference_ids'])
            layer_evidence[str(seed)]=layers
            for target in targets:
                safe_point()
                method=target['method'];rows={r['sample_id']:r for r in stream['methods'][method]['per_sample']}
                require(target['kind']=='greater_than','Unsupported unconfirmed target; do not invent a label')
                labels={sid:None if sid not in reference or rows[sid].get(target['field']) is None else
                        rows[sid][target['field']]>target['value'] for sid in ids}
                newly_invalid=[sid for sid in ids if sid in reference and not rows[sid]['valid']]
                for signal in signals:
                    scores=[]
                    for row in aligned:
                        value=_field(row['saved_prediction'],signal['field'])
                        if value is not None:
                            require(type(value) in (int,float) and math.isfinite(value),'Malformed saved continuous signal')
                            value=float(value)*(1 if signal['risk_direction']=='higher' else -1)
                        scores.append(value)
                    strata=([('all',list(range(512)))]+[(f'initial_error_stratum_{i+1}',
                            [j for j,row in enumerate(aligned) if bins[row['sample_id']]==i]) for i in range(4)]+
                            [('initial_error_missing',[j for j,row in enumerate(aligned) if bins[row['sample_id']] is None]),
                             ('outside_fixed_reference',[j for j,row in enumerate(aligned) if bins[row['sample_id']]=='outside_reference'])])
                    for label,indices in strata:
                        domain_result.append(dict(seed=seed,target=target,signal=signal,stratum=label,
                            newly_invalid_reference_ids=newly_invalid,
                            metrics=ranking_metrics([scores[j] for j in indices],[labels[ids[j]] for j in indices])))
        output[domain]=dict(status='saved_signal_ranking_available',initial_error_layers_by_stream=layer_evidence,
            quantile_probabilities=quantiles,tie_policy='stable(E0,sample_id)rank within frozen Iref; equal-count four layers',
            results=domain_result,not_new_independent_images_or_streams=True)
    return dict(status='descriptive_static_association_only',signal_specifications=signals,
                target_specifications=targets,score_threshold_search=False,new_model_forward=0,
                domains=output,causal_update_harm_claim=False)


def analyze(manifest_path,safe_point=lambda:None):
    safe_point()
    path=Path(manifest_path).resolve(strict=True);manifest=load_json(path);base=path.parent
    summary_path=read_binding(manifest['summary'],base);summary=load_json(summary_path)
    evidence=[dict(role='manifest',path=str(path),sha256=sha(path)),dict(role='summary',path=str(summary_path),sha256=sha(summary_path))]
    aggregate=summary_audit(summary);missing=[];traces={}
    for item in manifest.get('traces',[]):
        safe_point()
        trace_path=read_binding(item,base);doc=load_json(trace_path);key=item['run_key']
        require(key in summary['resources_by_run'] and key.startswith('P:'),'Unexpected trace run slot')
        require(doc.get('source_sha256')==SOURCE_SHA and
                doc.get('protocol_sha256')==summary['protocol_sha256'],'Trace source/protocol differs')
        traces[key]=trace_audit(doc,summary['resources_by_run'][key])
        evidence.append(dict(role=key,path=str(trace_path),sha256=sha(trace_path)))
    for row in aggregate['runs']:
        if row['run_key'] not in traces:missing.append(dict(scope=row['run_key'],
            missing='complete saved protection pairs, IDs/source overlap and reservoir transitions',
            action='export existing complete scientific_trace/pools metadata only; no forward'))
    offline={}
    for item in manifest.get('offline_predictions',[]):
        safe_point()
        input_path=read_binding(item,base)
        if input_path.suffix=='.jsonl':rows=[json.loads(line) for line in input_path.read_text(encoding='utf8').splitlines() if line.strip()]
        else:
            document=load_json(input_path);rows=document['rows'] if isinstance(document,dict) else document
        finite_tree(rows);offline[item['domain']]=rows
        evidence.append(dict(role='offline_signals:'+item['domain'],path=str(input_path),sha256=sha(input_path)))
    for domain in DOMAINS:
        if domain not in offline:missing.append(dict(scope=domain,missing='saved strict-source Offline raw geometry/entropy signals',action='read existing prediction JSON/JSONL, never recompute'))
    if manifest.get('literature_section'):
        literature=read_binding(manifest['literature_section'],base)
        text=literature.read_text(encoding='utf8')
        require('2.20' in text and bool(text.strip()),'The bound source must contain original section2.20')
        evidence.append(dict(role='confirmed_literature_section_2_20',path=str(literature),sha256=sha(literature)))
    else:missing.append(dict(scope='protocol',missing='bound original section2.20',action='bind exact provided source section before interpreting its claim'))
    if not manifest.get('static_diagnostic'):
        missing.append(dict(scope='static_diagnostic',missing='predeclared signal directions, outcome labels and initial-error strata',
                            action='use the confirmed protocol; do not choose after viewing rankings'))
    return dict(schema='g50p-risk-diagnostic-experiment-A-v1',status='complete_saved_record_analysis' if not missing else 'partial_saved_record_analysis',
        model_forwards_added=0,GT_files_read=False,images_read=False,inputs=evidence,
        implementation_sha256=sha(__file__),source_sha256=SOURCE_SHA,
        original_protocol_sha256=summary['protocol_sha256'],summary_audit=aggregate,
        detailed_traces=traces,static_diagnostic=static_diagnostic(summary,offline,manifest.get('static_diagnostic'),safe_point),
        missing_inputs=missing,threshold_or_seed_search=False,new_experiment=False,
        interpretation_limits=['Final developer errors cannot identify the causal effect of one candidate update.',
          'Signal ranking describes saved associations; it is not prospective update-risk validation.',
          'Uncomputable signals, missing labels and one-class strata remain explicitly unavailable.',
          'Three observation streams and repeated protection checks are not independent large samples.'])


def render(report):
    a=report['summary_audit'];lines=['# 实验A：已保存记录的只读诊断','',
       f'本次只分析既有确定性记录：{a["candidates"]}个P候选、{a["rollbacks"]}次撤销；新增模型前向0，未读取图像或GT文件。',
       '', '候选最终误差与保护信号之间的关系仅为描述性关联，不能归因于某一次更新。','',
       '| 流 | 候选 | 提交 | 撤销 | 连续决策区间 |','| --- | --- | --- | --- | --- |']
    for row in a['runs']:
        c=row['counts'];spans='；'.join(f'{r["first"]}–{r["last"]} {r["decision"]}' for r in row['decision_runs'])
        lines.append(f'| {row["run_key"]} | {c["candidates"]} | {c["commits"]} | {c["rollbacks"]} | {spans} |')
    lines+=['','## 细节可用性','',f'已读取完整保护记录的流数：{len(report["detailed_traces"])}/6。',
            '每条流的规则可计算分母、子条件触发幅度、来源重叠、重复触发图片、驻留与替换记录均在JSON中保留；缺少原始字段时登记缺口，不作推理补齐。']
    for key,item in report['detailed_traces'].items():
        witnesses=item['recurring_trigger_images'][:5]
        lines+=['',f'### {key}',f'重复来源去重次数：{item["sources"]["deduplicated_mentions_removed"]}；实际检查不同图片：{item["sources"]["unique_checked_images"]}。']
        for image in witnesses:lines.append(f'- {image["sample_id"]}：触发{image["trigger_count"]}个窗口，最长连续{image["longest_consecutive_trigger_run"]}，到达时G50接受={image["arrival_G50_accepted"]}。')
    lines+=['','## 静态连续信号诊断','',
       '连续信号方向、结局标签和初始误差分层必须预先绑定；AUROC保留并列平均秩，AUPRC使用不插值平均精度，未搜索分数阈值。',
       f'状态：{report["static_diagnostic"]["status"]}。单类、信号缺失和非参考图片不填成阴性或零。','',
       '初始误差分层只用于解释：每流固定Iref内按(E0,ID)稳定排序四等人数层，不看after结果，不用于部署筛选。完整分层AUROC/AP、类比例、可计算分母和缺失率均保留于同目录JSON；下表保留全部未分层组合，不按成绩筛选。']
    for domain,item in report['static_diagnostic'].get('domains',{}).items():
        if item['status']!='saved_signal_ranking_available':continue
        lines+=['',f'### {domain}：Offline信号与整轮ODR后退化事件的关联',
                '| 流 | 结局方法 | 信号（预声明风险方向） | 有限N | 正类数 | 正类比例 | AUROC | AP | 信号缺失 | 标签缺失 |',
                '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        def number(value):return 'None' if value is None else f'{value:.6f}'
        for row in item['results']:
            if row['stratum']!='all':continue
            m=row['metrics'];signal=row['signal']
            lines.append(f'| {row["seed"]} | {row["target"]["method"]} | {signal["field"]} ({signal["risk_direction"]}) | {m["available"]} | {m["positive"]} | {number(m["prevalence"])} | {number(m["AUROC"])} | {number(m["AUPRC"])} | {m["missing_score"]} | {m["missing_label"]} |')
    lines+=['','## 缺口','']
    for item in report['missing_inputs']:lines.append(f'- {item["scope"]}：{item["missing"]}；{item["action"]}。')
    if not report['missing_inputs']:lines.append('本分析声明所需的已保存字段已齐全；这不表示更新风险判据已验证有效。')
    lines+=['','## 输入SHA256','', '| 角色 | 文件 | SHA256 |','| --- | --- | --- |']
    for item in report['inputs']:lines.append(f'| {item["role"]} | {item["path"]} | {item["sha256"]} |')
    return '\n'.join(lines)+'\n'


class AnalysisResourceWait(Exception):
    pass


def _publish(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_text(encoding='utf8')==text,'Existing immutable diagnostic output differs: '+str(path))
    else:
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w',encoding='utf8') as handle:
            handle.write(text);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        if os.name!='nt':
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)


def execute(task_path,*,admission,launch_record,conflict_check,budget):
    """Server read-only A entry; no method capable of dispatching a forward."""
    require(admission.get('allow_start') is True and admission.get('boundary_verified') is True,
            'Fresh shared-host admission is required before diagnostic A')
    started=time.perf_counter();task_path=Path(task_path).resolve(strict=True);task=load_json(task_path)
    require(launch_record.get('run_id')==task.get('run_id'),'Actual A launch/task identity differs')
    def safe():
        budget.check_time()
        reason=conflict_check()
        if reason:raise AnalysisResourceWait(str(reason))
    try:
        safe();frozen_path=read_binding(task['frozen_inputs'],task_path.parent)
        frozen=load_json(frozen_path)
        extraction_path=Path(task['extraction_proof_path']).resolve(strict=True)
        extraction=load_json(extraction_path)
        require(extraction.get('schema')=='g50p-risk-diagnostic-A-extraction-proof-v1' and
                extraction.get('status')=='passed' and extraction.get('model_forwards')==0,
                'Saved-record extraction did not pass its zero-forward contract')
        require(extraction.get('frozen_inputs')==task['frozen_inputs'], 'Extraction bound different frozen inputs')
        parents={run['run_key']:run['original_final_checkpoint']['state_sha256'] for run in frozen.get('runs',{}).values()}
        require(isinstance(parents,dict) and bool(parents) and
                extraction.get('parent_final_state_sha256')==parents,'Frozen parent checkpoint SHA map differs')
        outputs=extraction.get('outputs')
        require(isinstance(outputs,list) and bool(outputs),'Extraction output SHA bindings missing')
        for binding in outputs:
            safe();read_binding(binding,extraction_path.parent)
        manifest_binding=extraction['analysis_manifest'];manifest_path=read_binding(manifest_binding,extraction_path.parent)
        expected=Path(task['analysis_manifest']['path'])
        if not expected.is_absolute():expected=task_path.parent/expected
        require(manifest_path==expected.resolve(strict=True),'Actual extracted analysis manifest path differs')
        if task['analysis_manifest'].get('sha256'):
            require(manifest_binding['sha256']==task['analysis_manifest']['sha256'],'Task analysis manifest SHA differs')
        report=analyze(manifest_path,safe)
        require(report['status']=='complete_saved_record_analysis' and not report['missing_inputs'],
                'Diagnostic A remains incomplete; do not bypass missing records')
        safe();output=Path(task['output_dir']).resolve();output.mkdir(parents=True,exist_ok=True)
        report_path=output/'report.json';markdown_path=output/'REPORT.md';proof_path=output/'proof.json'
        _publish(report_path,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        _publish(markdown_path,render(report));safe()
        proof=dict(schema='g50p-risk-diagnostic-A-proof-v1',status='passed',model_forwards=0,
            report=dict(path=str(report_path),sha256=sha(report_path)),report_sha256=sha(report_path),
            markdown=dict(path=str(markdown_path),sha256=sha(markdown_path)),
            analysis_manifest=manifest_binding,frozen_inputs=task['frozen_inputs'],
            extraction_proof=dict(path=str(extraction_path),sha256=sha(extraction_path)),
            parent_final_state_sha256=parents,task_sha256=sha(task_path),implementation_sha256=sha(__file__),
            candidates=793,rollbacks=117,causal_update_harm_claim=False,new_model_forward=False)
        _publish(proof_path,json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        return dict(status='a_analyzed',exit_code=0,model_forwards=0,report=str(report_path),
            proof=str(proof_path),proof_sha256=sha(proof_path),execution_attempt_elapsed_seconds=time.perf_counter()-started)
    except AnalysisResourceWait as error:
        return dict(status='waiting_resources',exit_code=75,model_forwards=0,reason=str(error),
                    execution_attempt_elapsed_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=analyze(args.manifest);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for name,data in [('analysis.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),('ANALYSIS.md',render(report))]:
        target=output/name
        if target.exists():require(target.read_text(encoding='utf8')==data,'Refuse to rewrite a changed diagnostic artifact: '+str(target))
        else:target.write_text(data,encoding='utf8')
    print(json.dumps(dict(status=report['status'],candidates=793,rollbacks=117,model_forwards_added=0,
                         output=str(output.resolve()),analysis_sha256=sha(output/'analysis.json'))))


if __name__=='__main__':main()
