"""Compile the approved P/RW/RP matrix and its independent audit tasks.

Only immutable metadata is read here. No model import, GT access, process launch,
or resource probe occurs while preparing the next stage.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from frozen_backend import sha256
from guarded_entry import atomic_json
from round2_protocol import build_complete_matrix, derive_controls, frozen_protocol


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


def freeze_reusable_baselines(plan, index):
    """Retain exact accepted O/A/G50 metadata, never rerun those forwards."""
    admission = index.get('first_round_acceptance')
    if not admission:
        raise ValueError('First-round accepted summary/proof must be bound before automatic continuation')
    summary, proof = checked(admission['summary']), checked(admission['proof'])
    if summary.get('status') != 'completed' and summary.get('status') != 'passed':
        # The original summarizer uses accepted_exploratory_summary in some
        # versions. Its independent proof, not a guessed status spelling, binds it.
        if proof.get('status') != 'passed':
            raise ValueError('Original baseline summary lacks accepted evidence')
    if proof.get('status') != 'passed':
        raise ValueError('Original baseline summary proof did not pass')
    if (proof.get('summary_sha256') is not None
            and proof['summary_sha256'] != admission['summary']['sha256']):
        raise ValueError('Original summary proof hash mismatch')
    document_path = Path(plan)/'baseline_result_bindings.json'
    document = read(document_path)
    if (document.get('source_sha256') != frozen_protocol()['source_sha256']
            or document.get('measurement_repeated') is not False):
        raise ValueError('Original result binding document identity changed')
    inputs = {str(Path(row['path'])): row for row in summary['inputs']}
    pairs = {}
    for definition in build_complete_matrix()[:6]:
        domain, seed = definition['domain'], definition['stream_seed']
        pair = {}
        for group in ('O', 'A', 'G50'):
            candidates = [row for row in document['runs'] if row['method'] == group
                          and row['domain'] == domain
                          and str(row['stream']) == ('offline' if group == 'O' else str(seed))]
            if len(candidates) != 1:
                raise ValueError('Original accepted baseline is missing or ambiguous')
            source = candidates[0]
            directory = str(Path(source['report']['path']).parent)
            entry = inputs.get(directory)
            if (entry is None or source['report']['sha256'] != entry['report_sha256']
                    or source['proof']['sha256'] != entry['proof_sha256']):
                raise ValueError('Original binding differs from accepted summary')
            run_proof = checked(source['proof'])
            checked(source['report'])
            if (run_proof.get('status') != 'passed'
                    or run_proof.get('report_sha256') != source['report']['sha256']):
                raise ValueError('Original run lacks independent acceptance')
            pair[group] = {}
            for key in ('task', 'report', 'proof', 'predictions'):
                if sha256(source[key]['path']) != source[key]['sha256']:
                    raise ValueError('Original baseline artifact changed: '+key)
                pair[group][key] = source[key]
        pairs[f'{domain}:{seed}'] = pair
    value = dict(schema='odr003-round2-reused-baselines-v1',
                 first_round_acceptance=admission, pairs=pairs, no_forwards=True,
                 no_reserved_GT=True)
    path = immutable(Path(plan)/'complete/baseline_reuse.json', value)
    return path, value


def complete_task(plan, index, definition, engineering, *, paired=None):
    plan = Path(plan)
    domain, seed = definition['domain'], definition['stream_seed']
    original = read(plan/'baseline_bindings.json')['tasks'][f'{domain}_{seed}']
    # Engineering records bind the exact original modules, including their
    # pre-existing scientific semantics. Only domain/stream changes below.
    template = read(index['tasks'][0]['task_path'])
    run_id = f"g50p2_{definition['group'].lower()}_{domain}_s{seed}_N1024_B4"
    task = dict(schema='odr003-round2-task-v1', action='run', stage='complete',
                run_id=run_id, output_dir=str(plan/'runs'/run_id),
                definition=definition, protocol=frozen_protocol(),
                implementation_sha256=index['implementation_sha256'],
                baseline_task=original, frozen_modules=template['frozen_modules'],
                confirmation_sha256=sha256(plan/'confirmation.json'), engineering_admission=engineering)
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


def prepare_P(plan, index):
    engineering = accepted_engineering(plan, index)
    baseline_path, _ = freeze_reusable_baselines(plan, index)
    rows = [complete_task(plan, index, row, engineering) for row in build_complete_matrix()[:6]]
    value = dict(schema='odr003-round2-complete-index-v1', phase='P', tasks=rows,
                 protocol_sha256=frozen_protocol()['protocol_sha256'],
                 implementation_sha256=index['implementation_sha256'],
                 engineering_admission=engineering, reusable_baselines=binding(baseline_path),
                 skipped_controls=[], max_new_complete_odr=18)
    path = immutable(Path(plan)/'complete/P_index.json', value)
    return path, value


def prepare_controls(plan, index, P_index, accepted):
    by_key = {}
    for row in P_index['tasks']:
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
    if len(rows) > 18 or len({row['run_id'] for row in rows}) != len(rows):
        raise ValueError('Complete run budget or uniqueness violated')
    value = {**P_index, 'phase': 'controls', 'tasks': rows, 'skipped_controls': skipped,
             'P_index': binding(Path(plan)/'complete/P_index.json')}
    path = immutable(Path(plan)/'complete/all_index.json', value)
    return path, value


def verify_task(plan, index, science_row, receipt, *, queue_wait_seconds=0.):
    original_task = checked(dict(path=science_row['task_path'], sha256=science_row['task_sha256']))
    pair_key = f"{science_row['definition']['domain']}:{science_row['definition']['stream_seed']}"
    baselines = checked(index['reusable_baselines'])['pairs'][pair_key]
    run_id = science_row['run_id']+'_verify'
    task = dict(schema='odr003-round2-task-v1', action='verify_complete', stage='verification',
                run_id=run_id, protocol=frozen_protocol(), implementation_sha256=index['implementation_sha256'],
                run_task=dict(path=science_row['task_path'], sha256=science_row['task_sha256']),
                execution_receipt=receipt, all_attempts=receipt['attempts'],
                proof_path=str(Path(original_task['output_dir'])/'proof.json'), reusable_baselines=baselines,
                queue_wait_seconds=queue_wait_seconds)
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
                skipped_controls=index['skipped_controls'], execution_receipts=binding(receipts_path),
                summary_path=str(Path(plan)/'summaries/round2/summary.json'),
                proof_path=str(Path(plan)/'summaries/round2/proof.json'))
    path = immutable(task_path, task)
    return path, task['run_id']
