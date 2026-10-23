"""Prepare immutable engineering tasks only; never launch or touch old queues."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path, PurePosixPath

from frozen_backend import sha256, canonical_sha
from round2_protocol import frozen_protocol, build_engineering_matrix, build_complete_matrix

SERVER_PLAN = PurePosixPath('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/odr003_round2_20261023')
FROZEN_NAMES = ('state_adapter', 'data_boundary', 'gate_core', 'run_exploration', 'prediction')


def write_identical_or_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf8') != data:
            raise ValueError('Refusing to replace an existing frozen artifact: '+str(path))
        return
    with path.open('x', encoding='utf8') as f: f.write(data)


def prepare(root, baseline_path, *, execution_revision=None, origin_binding=None):
    root = Path(root).resolve()
    baseline_doc = json.loads(Path(baseline_path).read_text(encoding='utf8'))
    code = {p.name: sha256(p) for p in sorted((root/'implementation').glob('*.py'))}
    if not {'guarded_entry.py', 'run_round2.py', 'verify_round2.py', 'protected_engine.py',
            'reference_driver.py', 'host_snapshot.py'} <= code.keys():
        raise ValueError('The execution and independent verification entry points must exist before binding')
    protocol = frozen_protocol()
    synthetic_path = root/'local_validation/contract_report.json'
    synthetic = json.loads(synthetic_path.read_text(encoding='utf8'))
    if synthetic.get('status') != 'passed' or synthetic['implementation_sha256'] != code:
        raise ValueError('Current code lacks a matching local contract suite report')
    confirmation = json.loads((root/'confirmation.json').read_text(encoding='utf8'))
    if confirmation['status'] != 'confirmed_implementation_authorized':
        raise ValueError('This plan lacks explicit implementation authorization')
    if execution_revision is not None:
        if execution_revision != 'preprocess_rng_fix_20261024' or not isinstance(origin_binding, dict):
            raise ValueError('Only the explicitly audited zero-progress repair revision is supported')
        if set(origin_binding) != {'path', 'sha256'} or not re.fullmatch('[0-9a-f]{64}', origin_binding['sha256']):
            raise ValueError('The original zero-progress failure spec must be bound')
    entries = []
    for definition in build_engineering_matrix():
        baseline = baseline_doc['tasks'][f"{definition['domain']}_2021"]
        old_impl = PurePosixPath(baseline['output_dir']).parents[1]/'implementation'
        # Take hashes from the accepted original task; no scientific rerun.
        frozen = {name: dict(path=str(old_impl/(name+'.py')),
                            sha256=baseline['code_sha256'][name+'.py']) for name in FROZEN_NAMES}
        run_id = 'g50p2_eng_'+definition['domain']+'_'+definition['trajectory'].lower()
        output = SERVER_PLAN/'runs'
        if execution_revision is not None: output = output/execution_revision
        task = dict(schema='odr003-round2-task-v1', run_id=run_id, stage='engineering',
                    output_dir=str(output/run_id), definition=definition,
                    protocol=protocol, implementation_sha256=code, baseline_task=baseline,
                    frozen_modules=frozen, confirmation_sha256=sha256(root/'confirmation.json'))
        if execution_revision is not None:
            task['execution_revision'] = execution_revision
            if definition['domain'] == 'lightbox' and definition['trajectory'] == 'G50_reference':
                task['initial_checkpoint_import_required'] = True
                task['initial_checkpoint_origin'] = origin_binding
        filename = run_id+'.json'
        write_identical_or_new(root/'tasks'/filename, task)
        entries.append(dict(run_id=run_id, task_path=str(SERVER_PLAN/'tasks'/filename),
                            task_sha256=sha256(root/'tasks'/filename), definition=definition))
    index = dict(schema='odr003-round2-engineering-index-v1', stage='prepared_not_started',
                 protocol_sha256=protocol['protocol_sha256'], implementation_sha256=code,
                 tasks=entries, logical_trajectories=8, maximum_logical_observations=1024,
                 synthetic_contracts=dict(path=str(SERVER_PLAN/'local_validation/contract_report.json'),
                                          sha256=sha256(synthetic_path)),
                 actual_recomputed_work_separate=True, independent_verification_required=True,
                 shared_host_idle_gate_required=True,
                 next_stage='only_if_independent_engineering_proof_allows_complete_matrix')
    if execution_revision is not None:
        index['execution_revision'] = execution_revision
        index['pre_forward_repair_required'] = True
    write_identical_or_new(root/'engineering_index.json', index)
    write_identical_or_new(root/'complete_matrix_templates.json',
                         dict(status='inactive_templates_only', protocol=protocol,
                              tasks=build_complete_matrix(), max_new_complete_odr=18))
    return dict(engineering_index_sha256=sha256(root/'engineering_index.json'),
                implementation_count=len(code), task_count=len(entries),
                status='prepared_not_started', no_process_launched=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--baseline-bindings', required=True)
    p.add_argument('--execution-revision')
    p.add_argument('--origin-binding-json')
    args = p.parse_args()
    binding = json.loads(Path(args.origin_binding_json).read_text(encoding='utf8')) if args.origin_binding_json else None
    print(json.dumps(prepare(args.root, args.baseline_bindings, execution_revision=args.execution_revision, origin_binding=binding)))
