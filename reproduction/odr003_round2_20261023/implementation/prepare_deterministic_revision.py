"""Prepare the confirmed deterministic revision without touching live runtime.

Only immutable metadata is assembled. No model, GT, resource query, process,
old checkpoint migration or production-state mutation is performed here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath

from frozen_backend import sha256
from round2_protocol import frozen_protocol, build_engineering_matrix, build_baseline_matrix, build_complete_matrix
from prepare_round2 import write_identical_or_new, FROZEN_NAMES, SERVER_PLAN


REVISION = 'deterministic_20261024'
PREFIX = 'g50p2_det_20261024'


def prepare(root, candidate, inputs):
    root, candidate = Path(root), Path(candidate)
    protocol = frozen_protocol()
    if (protocol['max_new_complete_odr_runs'] != 30
            or len(build_baseline_matrix()) != 14 or len(build_complete_matrix()) != 18):
        raise ValueError('Only the explicitly approved deterministic matrix may be prepared')
    report_path = root/'local_validation/contract_report.json'
    contracts = json.loads(report_path.read_text(encoding='utf8'))
    code = {p.name:sha256(p) for p in sorted((root/'implementation').glob('*.py'))}
    if not {'run_round2.py','offline_baseline.py','deterministic_runtime.py','guarded_entry.py',
            'queue_round2.py','verify_engineering_phase.py','verify_complete.py','summarize_round2.py'} <= code.keys():
        raise ValueError('All deterministic execution and independent acceptance entries must exist')
    if contracts.get('status') != 'passed' or contracts['implementation_sha256'] != code:
        raise ValueError('Current deterministic implementation must pass its local contracts')
    confirmation = json.loads((root/'confirmation.json').read_text(encoding='utf8'))
    if (confirmation.get('maximum_complete_odr') != 30
            or confirmation.get('status') != 'confirmed_implementation_authorized'):
        raise ValueError('The expanded research protocol requires explicit confirmation')
    references = inputs['fixed_reference_bindings']
    if set(references) != {f'{d}:{s}' for d in ('lightbox','sunlamp') for s in (2021,2022,2023)}:
        raise ValueError('All six unchanged paired reference lists must be bound')
    baselines = json.loads((root/'baseline_bindings.json').read_text(encoding='utf8'))['tasks']
    entries = []
    for definition in build_engineering_matrix():
        domain = definition['domain']
        original = copy.deepcopy(baselines[domain+'_2021'])
        old_impl = PurePosixPath(original['output_dir']).parents[1]/'implementation'
        frozen = {name:dict(path=str(old_impl/(name+'.py')), sha256=original['code_sha256'][name+'.py'])
                  for name in FROZEN_NAMES}
        run_id = PREFIX+'_eng_'+domain+'_'+definition['trajectory'].lower()
        task = dict(schema='odr003-round2-task-v1', action='run', stage='engineering',
                    run_id=run_id, output_dir=str(SERVER_PLAN/'runs'/REVISION/run_id),
                    definition=definition, protocol=protocol, implementation_sha256=code,
                    baseline_task=original, frozen_modules=frozen,
                    execution_revision=REVISION, confirmation=inputs['confirmation'],
                    confirmation_sha256=sha256(root/'confirmation.json'))
        local = candidate/'tasks'/REVISION/(run_id+'.json')
        write_identical_or_new(local, task)
        entries.append(dict(run_id=run_id, definition=definition,
                            task_path=str(SERVER_PLAN/'tasks'/REVISION/local.name), task_sha256=sha256(local)))
    index = dict(schema='odr003-round2-engineering-index-v1', stage='prepared_not_started',
                 execution_revision=REVISION, run_prefix=PREFIX,
                 protocol_sha256=protocol['protocol_sha256'], implementation_sha256=code,
                 tasks=entries, logical_trajectories=8, maximum_logical_observations=1024,
                 max_complete_odr=30, max_offline_evaluations=2,
                 synthetic_contracts=dict(path=str(SERVER_PLAN/'revisions'/REVISION/'validation/contract_report.json'),
                                          sha256=inputs['synthetic_contracts_sha256']),
                 confirmation=inputs['confirmation'], fixed_reference_bindings=references,
                 prior_execution_history=inputs['prior_execution_history'],
                 execution_policy=inputs['execution_policy'],
                 resource_wait_interval_seconds=300, automatic_full_continuation=True,
                 actual_recomputed_work_separate=True, independent_verification_required=True,
                 shared_host_idle_gate_required=True,
                 next_stage='strict engineering -> O2/A6/G50six -> P6 -> RW/RPmax12 -> independent summary')
    write_identical_or_new(candidate/'engineering_index.json', index)
    write_identical_or_new(candidate/'matrix_templates.json', dict(status='inactive_templates_only',
                    protocol=protocol, baseline=build_baseline_matrix(), experimental=build_complete_matrix(),
                    max_new_complete_odr=30, max_offline_evaluations=2))
    return dict(status='prepared_not_started', tasks=8, implementation_files=len(code),
                engineering_index_sha256=sha256(candidate/'engineering_index.json'),
                no_model_forward=True, no_production_mutation=True)
