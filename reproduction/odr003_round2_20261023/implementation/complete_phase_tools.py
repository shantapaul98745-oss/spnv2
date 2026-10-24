"""Compile deterministic O/A/G50 baselines and the approved P/RW/RP matrix.

Only immutable metadata is read here. No model import, GT access, process launch,
or resource probe occurs while preparing the next stage.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from frozen_backend import sha256
from guarded_entry import atomic_json
from round2_protocol import build_baseline_matrix, build_complete_matrix, derive_controls, frozen_protocol


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def binding(path):
    return dict(path=str(path), sha256=sha256(path))


def checked(value):
    path = Path(value['path'])
    if sha256(path) != value['sha256']:
        raise ValueError('Bound metadata changed: '+str(path))
    return read(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Refusing to replace frozen artifact: '+str(path))
    else:
        atomic_json(path, value)
    return path


def accepted_engineering(plan, index):
    proof_path = Path(plan)/'preflight/engineering_proof.json'
    proof = read(proof_path)
    if (proof.get('status') != 'passed' or proof.get('allow_complete_matrix') is not True
            or proof.get('implementation_sha256') != index['implementation_sha256']
            or proof.get('protocol_sha256') != frozen_protocol()['protocol_sha256']):
        raise ValueError('Complete matrix requires a passed independent engineering admission')
    return binding(proof_path)


def run_name(index, definition):
    prefix = index.get('run_prefix', 'g50p2_det_20261024')
    if not prefix or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in prefix):
        raise ValueError('Current execution requires a safe unique run prefix')
    group, domain = definition['group'], definition['domain']
    suffix = 'offline_dev512' if group == 'O' else f"s{definition['stream_seed']}_N1024_B4"
    return f'{prefix}_{group.lower()}_{domain}_{suffix}'


def baseline_task(plan, index, definition, engineering):
    plan = Path(plan)
    seed = definition.get('stream_seed') or 2021
    original = read(plan/'baseline_bindings.json')['tasks'][f"{definition['domain']}_{seed}"]
    template = read(index['tasks'][0]['task_path'])
    run_id = run_name(index, definition)
    task = dict(schema='odr003-round2-task-v1', action='run_offline' if definition['group'] == 'O' else 'run',
                stage='offline' if definition['group'] == 'O' else 'complete', run_id=run_id,
                output_dir=str(plan/'runs'/run_id), definition=definition, protocol=frozen_protocol(),
                implementation_sha256=index['implementation_sha256'], baseline_task=original,
                frozen_modules=template['frozen_modules'], confirmation_sha256=sha256(plan/'confirmation.json'),
                engineering_admission=engineering)
    if index.get('confirmation'):
        checked(index['confirmation'])
        task['confirmation'] = index['confirmation']
    path = immutable(plan/'tasks'/(run_id+'.json'), task)
    return dict(run_id=run_id, definition=definition, task_path=str(path), task_sha256=sha256(path))


def prepare_baselines(plan, index):
    engineering = accepted_engineering(plan, index)
    rows = [baseline_task(plan, index, row, engineering) for row in build_baseline_matrix()]
    if len(rows) != 14 or len({row['run_id'] for row in rows}) != 14:
        raise ValueError('Deterministic comparison requires exactly two O and twelve A/G50 results')
    value = dict(schema='odr003-round2-complete-index-v2', phase='baselines', tasks=rows,
                 protocol_sha256=frozen_protocol()['protocol_sha256'], implementation_sha256=index['implementation_sha256'],
                 engineering_admission=engineering, skipped_controls=[], max_new_complete_odr=30,
                 max_offline_evaluations=2, fixed_reference_bindings=index.get('fixed_reference_bindings'))
    if index.get('confirmation'): value['confirmation'] = index['confirmation']
    path = immutable(Path(plan)/'complete/baseline_index.json', value)
    return path, value


def freeze_reusable_baselines(plan, index, baseline_index, accepted):
    """Freeze only fourteen independently accepted results from this backend.

    Historical results remain provenance. They cannot substitute for the newly
    authorized deterministic O/A/G50 measurements.
    """
    definitions = build_baseline_matrix()
    rows = baseline_index['tasks']
    if ([row['definition'] for row in rows] != definitions
            or baseline_index['implementation_sha256'] != index['implementation_sha256']):
        raise ValueError('Baseline matrix does not belong to this deterministic execution')
    by_group = {}
    proofs = []
    for row in rows:
        item = accepted.get(row['run_id'])
        if item is None:
            raise ValueError('All fourteen deterministic baselines require independent acceptance')
        task_binding = dict(path=row['task_path'], sha256=row['task_sha256'])
        task, report, proof = checked(task_binding), checked(item['report']), checked(item['proof'])
        if (task['definition'] != row['definition'] or report.get('definition') != row['definition']
                or task['protocol'] != frozen_protocol() or report.get('protocol_sha256') != frozen_protocol()['protocol_sha256']
                or report.get('source_sha256') != frozen_protocol()['source_sha256']
                or task['implementation_sha256'] != index['implementation_sha256']
                or proof.get('status') != 'passed' or proof.get('report_sha256') != item['report']['sha256']
                or proof.get('definition') != row['definition'] or proof.get('task_binding') != task_binding):
            raise ValueError('Deterministic baseline proof/source/task identity differs')
        prediction = report['outputs']['predictions']
        if sha256(prediction['path']) != prediction['sha256']:
            raise ValueError('Accepted deterministic baseline predictions changed')
        definition = row['definition']
        key = (definition['domain'], definition['group'], None if definition['group'] == 'O' else definition['stream_seed'])
        if key in by_group:
            raise ValueError('Duplicate deterministic baseline slot')
        by_group[key] = dict(task=task_binding, report=item['report'], proof=item['proof'], predictions=prediction)
        proofs.append(item['proof'])
    pairs = {f"{definition['domain']}:{definition['stream_seed']}": {
                group: by_group[(definition['domain'], group, None if group == 'O' else definition['stream_seed'])]
                for group in ('O', 'A', 'G50')}
             for definition in build_complete_matrix()[:6]}
    value = dict(schema='odr003-round2-reused-baselines-v2', pairs=pairs,
                 protocol_sha256=frozen_protocol()['protocol_sha256'], source_sha256=frozen_protocol()['source_sha256'],
                 implementation_sha256=index['implementation_sha256'], baseline_proofs=proofs,
                 baseline_index=binding(Path(plan)/'complete/baseline_index.json'),
                 historical_results_reused_as_new_backend=False, no_additional_forwards=True, no_reserved_GT=True)
    path = immutable(Path(plan)/'complete/baseline_reuse_deterministic.json', value)
    return path, value


def complete_task(plan, index, definition, engineering, *, paired=None):
    plan = Path(plan)
    domain, seed = definition['domain'], definition['stream_seed']
    original = read(plan/'baseline_bindings.json')['tasks'][f'{domain}_{seed}']
    # Engineering records bind the exact original modules, including their
    # pre-existing scientific semantics. Only domain/stream changes below.
    template = read(index['tasks'][0]['task_path'])
    run_id = run_name(index, definition)
    task = dict(schema='odr003-round2-task-v1', action='run', stage='complete',
                run_id=run_id, output_dir=str(plan/'runs'/run_id),
                definition=definition, protocol=frozen_protocol(),
                implementation_sha256=index['implementation_sha256'],
                baseline_task=original, frozen_modules=template['frozen_modules'],
                confirmation_sha256=sha256(plan/'confirmation.json'), engineering_admission=engineering)
    if index.get('confirmation'):
        checked(index['confirmation'])
        task['confirmation'] = index['confirmation']
    if paired is not None:
        report = checked(paired['report'])
        proof = checked(paired['proof'])
        if (proof.get('status') != 'passed' or proof.get('report_sha256') != paired['report']['sha256']
                or report['definition']['group'] != 'P'
                or report['definition']['domain'] != domain or report['definition']['stream_seed'] != seed
                or report['protocol_sha256'] != task['protocol']['protocol_sha256']):
            raise ValueError('Control pairing is not an independently accepted identical P stream')
        task.update(paired_P_report=paired['report'], paired_P_proof=paired['proof'],
                    control=derive_controls(domain, seed, report['counts']['candidates'], report['counts']['commits']))
    path = immutable(plan/'tasks'/(run_id+'.json'), task)
    return dict(run_id=run_id, definition=definition, task_path=str(path), task_sha256=sha256(path))


def prepare_P(plan, index, baseline_index=None, accepted=None):
    engineering = accepted_engineering(plan, index)
    if baseline_index is None or accepted is None:
        raise ValueError('Fourteen accepted deterministic baselines are required before P6')
    baseline_path, _ = freeze_reusable_baselines(plan, index, baseline_index, accepted)
    rows = list(baseline_index['tasks']) + [complete_task(plan, index, row, engineering) for row in build_complete_matrix()[:6]]
    value = dict(schema='odr003-round2-complete-index-v2', phase='P', tasks=rows,
                 protocol_sha256=frozen_protocol()['protocol_sha256'],
                 implementation_sha256=index['implementation_sha256'],
                 engineering_admission=engineering, reusable_baselines=binding(baseline_path),
                 baseline_index=binding(Path(plan)/'complete/baseline_index.json'),
                 skipped_controls=[], max_new_complete_odr=30, max_offline_evaluations=2,
                 fixed_reference_bindings=index.get('fixed_reference_bindings'))
    if index.get('confirmation'): value['confirmation'] = index['confirmation']
    path = immutable(Path(plan)/'complete/P_index.json', value)
    return path, value


def prepare_controls(plan, index, P_index, accepted):
    by_key = {}
    for row in P_index['tasks']:
        if row['definition']['group'] != 'P':
            continue
        item = accepted.get(row['run_id'])
        if item is None:
            raise ValueError('All six P runs require independent acceptance before controls')
        by_key[row['definition']['run_key']] = item
    rows, skipped = list(P_index['tasks']), []
    for definition in build_complete_matrix()[6:]:
        pair = by_key[definition['paired_P_dependency']]
        report = checked(pair['report'])
        control = derive_controls(definition['domain'], definition['stream_seed'],
                                  report['counts']['candidates'], report['counts']['commits'])
        if definition['group'] == 'RW' and not control['RW']['enabled']:
            skipped.append(dict(definition=definition, reason=control['RW']['skip_reason'],
                                paired_P_report=pair['report'], paired_P_proof=pair['proof']))
            continue
        rows.append(complete_task(plan, index, definition, P_index['engineering_admission'], paired=pair))
    if (sum(row['definition']['group'] != 'O' for row in rows) > 30
            or sum(row['definition']['group'] == 'O' for row in rows) != 2
            or len({row['run_id'] for row in rows}) != len(rows)):
        raise ValueError('Complete run budget or uniqueness violated')
    value = {**P_index, 'phase': 'controls', 'tasks': rows, 'skipped_controls': skipped,
             'P_index': binding(Path(plan)/'complete/P_index.json')}
    path = immutable(Path(plan)/'complete/all_index.json', value)
    return path, value


def verify_task(plan, index, science_row, receipt, *, queue_wait_seconds=0.):
    original_task = checked(dict(path=science_row['task_path'], sha256=science_row['task_sha256']))
    baselines = None
    if science_row['definition']['group'] in ('P', 'RW', 'RP'):
        pair_key = f"{science_row['definition']['domain']}:{science_row['definition']['stream_seed']}"
        baselines = checked(index['reusable_baselines'])['pairs'][pair_key]
    run_id = science_row['run_id']+'_verify'
    task = dict(schema='odr003-round2-task-v1', action='verify_complete', stage='verification',
                run_id=run_id, protocol=frozen_protocol(), implementation_sha256=index['implementation_sha256'],
                run_task=dict(path=science_row['task_path'], sha256=science_row['task_sha256']),
                execution_receipt=receipt, all_attempts=receipt['attempts'],
                proof_path=str(Path(original_task['output_dir'])/'proof.json'), reusable_baselines=baselines,
                queue_wait_seconds=queue_wait_seconds)
    references = index.get('fixed_reference_bindings')
    if references:
        task['fixed_reference_binding'] = references[f"{science_row['definition']['domain']}:{science_row['definition']['stream_seed']}"]
    path = Path(plan)/'tasks'/(run_id+'.json')
    if path.exists():
        previous = read(path)
        task['queue_wait_seconds'] = previous['queue_wait_seconds']
    immutable(path, task)
    return path, run_id


def summary_task(plan, index_path, index, state):
    accepted = state['complete_accepted']
    if set(accepted) != {row['run_id'] for row in index['tasks']}:
        raise ValueError('Summary requires every non-skipped complete run independently accepted')
    task_path = Path(plan)/'tasks/g50p2_round2_summary.json'
    if task_path.exists():
        previous = read(task_path)
        if (previous['complete_index'] != binding(index_path)
                or previous['run_proofs'] != [accepted[row['run_id']]['proof'] for row in index['tasks']]):
            raise ValueError('Previously frozen summary inputs changed')
        checked(previous['execution_receipts'])
        return task_path, previous['run_id']
    receipts_path = immutable(Path(plan)/'complete/execution_receipts.json',
        dict(schema='odr003-round2-complete-receipts-v1', runs=list(state['complete_main'].values()),
             accepted=accepted, all_attempts=state['attempts'],
             verification_attempts=[item['verification_attempt'] for item in accepted.values()],
             queue_wait_seconds=state['queue_wait_seconds']))
    task = dict(schema='odr003-round2-task-v1', action='summarize_round2', stage='verification',
                run_id='g50p2_round2_summary', protocol=frozen_protocol(),
                implementation_sha256=index['implementation_sha256'],
                complete_index=binding(index_path),
                run_proofs=[accepted[row['run_id']]['proof'] for row in index['tasks']],
                baseline_proofs=[accepted[row['run_id']]['proof'] for row in index['tasks']
                                 if row['definition']['group'] in ('O', 'A', 'G50')],
                baseline_reuse=index['reusable_baselines'],
                skipped_controls=index['skipped_controls'], execution_receipts=binding(receipts_path),
                summary_path=str(Path(plan)/'summaries/round2/summary.json'),
                proof_path=str(Path(plan)/'summaries/round2/proof.json'))
    path = immutable(task_path, task)
    return path, task['run_id']
