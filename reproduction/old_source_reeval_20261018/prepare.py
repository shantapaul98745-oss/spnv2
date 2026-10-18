"""CPU-only preparation for three full-domain evaluations of the old Lu source.

This builder never launches inference, pauses training, or accesses peer assets.
Run in an independent CPU screen.  Existing weights and frozen production code
are read-only.  Import acceptance certifies inference compatibility, not training.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

BASE = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai')
P = BASE / 'spnv2_runs/plans/rebaseline_20261017'
Q = BASE / 'spnv2_runs/plans/old_source_reeval_20261018'
PYTHON = '/home/ps/anaconda3/envs/SPNv2/bin/python'
FINAL = BASE / 'spnv2_runs/frozen/phi3_BN_single_gpu_localstyle_seed2021_epoch20/model_final.pth.tar'
FINAL_SHA = 'dbb9deb17c37ce48b01edc1d658ad1a64c09094cf88d28a3c79adb49fd4cad50'
MODEL_ID = 'lux_old_b4'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new(path, content):
    """Publish a new file atomically without replacing any previous evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Preserve existing preparation evidence: ' + str(path))
    temporary = path.with_name('.' + path.name + '.prepare.tmp')
    data = content if isinstance(content, bytes) else content.encode('utf-8')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def write_json(path, value):
    write_new(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + '\n')


def tensor_digest(state):
    import torch
    result = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        require(value.layout == torch.strided, 'Non-dense state tensor: ' + name)
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    require(os.environ.get('STY'), 'Preparation must run in its own CPU screen')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('TORCH_HOME', str(BASE / 'spnv2_runs/env/cache/torch'))
    sys.dont_write_bytecode = True
    import torch
    import yaml
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA')
    torch.set_num_threads(2)
    require(not (Q / 'manifest.json').exists() and not (Q / 'preflight.json').exists(),
            'Existing accepted bundle is preserved; no automatic rebuild')
    require((Q / 'verify_comparison.py').is_file(), 'Install the independent comparison verifier first')
    require(sha(FINAL) == FINAL_SHA, 'Previously selected old final changed')

    source = P / 'source'
    code = {str(f.relative_to(source)): sha(f) for folder in ('core', 'tools')
            for f in sorted((source / folder).rglob('*.py'))}
    source_manifest = json.loads((P / 'source_manifest.json').read_text())
    require(code == source_manifest['core_tools_sha256'], 'Frozen source mismatch')
    sys.path.insert(0, str(P / 'evaluation_entrypoints'))
    import entry_contract
    wrapper_protocol = entry_contract.verify_protocol()
    sys.path.insert(0, str(source / 'core'))
    from config import cfg as defaults, update_config
    from nets import build_spnv2
    from utils.training_state import assert_finite_weights

    templates = {d: yaml.safe_load((P / ('configs/phi3_bn_ehs_eval_' + d + '.yaml')).read_text())
                 for d in ('synthetic', 'lightbox', 'sunlamp')}
    template = templates['synthetic']
    require(template['TEST']['IMAGES_PER_GPU'] == 1 and not template['TRAIN']['REPRODUCIBLE'],
            'Unexpected evaluation batch/sampler protocol')
    require(template['TEST']['HEAD'] == ['heatmap', 'efficientpose'], 'Unexpected test heads')
    require(template['MODEL']['HEAD']['NAMES'] == ['heatmap', 'efficientpose', 'segmentation'],
            'Expected full EHS model')
    require(template['MODEL']['EFFICIENTDET_PHI'] == 3
            and not template['MODEL']['USE_GROUPNORM_BACKBONE']
            and template['MODEL']['USE_GROUPNORM_HEADS'], 'Expected phi3 BN model')
    require(template['DATASET']['INPUT_SIZE'] == [768, 512], 'Unexpected resolution')
    require(template['TEST']['SPEED_THRESHOLD_Q'] == 0.169
            and template['TEST']['SPEED_THRESHOLD_T'] == 0.002173, 'Unexpected HIL thresholds')
    normalized = []
    for value in templates.values():
        v = copy.deepcopy(value)
        v.pop('EXP_NAME')
        v['TEST'].pop('TEST_CSV')
        normalized.append(v)
    require(all(v == normalized[0] for v in normalized), 'Domain inference settings differ')

    cfg = defaults.clone()
    update_config(cfg, SimpleNamespace(cfg=str(P / 'configs/phi3_bn_ehs_eval_synthetic.yaml'), opts=[]))
    assets = {str(source / key): value for key, value in code.items()}
    geometry = {name: str(Path(path).resolve(strict=True)) for name, path in {
        'camera': cfg.DATASET.CAMERA, 'keypoints': cfg.DATASET.KEYPOINTS}.items()}
    for path in geometry.values():
        assets[path] = sha(path)
    domains = []
    for domain, expected in [('synthetic', 11994), ('lightbox', 6740), ('sunlamp', 2791)]:
        rel = templates[domain]['TEST']['TEST_CSV']
        path = (Path(cfg.DATASET.ROOT) / cfg.DATASET.DATANAME / rel).resolve(strict=True)
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        ids = [r[0] for r in rows]
        require(len(ids) == expected and len(set(ids)) == expected
                and all(len(r) >= 12 and r[0] for r in rows), 'Unexpected CSV samples: ' + domain)
        require(all(math.isfinite(float(x)) for r in rows for x in r[5:12]),
                'Invalid CSV ground truth')
        domains.append(dict(id=domain, csv=str(path), csv_sha256=sha(path),
                            expected_samples=expected, sample_ids_sha256=digest(ids),
                            relative_csv=rel))
        assets[str(path)] = sha(path)

    print('CPU_IMPORT ' + MODEL_ID, flush=True)
    state = torch.load(FINAL, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state
            and all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()),
            'Expected plain safe model state dictionary; no unsafe pickle fallback')
    require(all(bool(torch.isfinite(v).all()) for v in state.values()), 'Nonfinite saved tensor')
    model = build_spnv2(cfg).cpu()
    model.load_state_dict(state, strict=True)
    assert_finite_weights(model)
    state_digest = tensor_digest(state)
    require(tensor_digest(model.state_dict()) == state_digest, 'Loaded tensors changed')
    tensor_count = len(state)
    require(sha(FINAL) == FINAL_SHA and not torch.cuda.is_initialized(), 'Input/CUDA state changed')
    del model, state

    import_path = Q / 'imports' / (MODEL_ID + '.json')
    import_proof = dict(schema='spnv2-existing-source-inference-import-v1', status='passed',
                        scope='Fixed existing old final, safe CPU load, strict common-architecture reload and finite tensors; NOT historical training acceptance.',
                        is_training_acceptance=False, final=str(FINAL), final_sha256=FINAL_SHA,
                        model_id=MODEL_ID, strict_reload=True, finite_tensors=True,
                        cuda_initialized=False, tensor_sha256=state_digest, tensor_count=tensor_count,
                        source_manifest=str(P / 'source_manifest.json'),
                        source_manifest_sha256=sha(P / 'source_manifest.json'))
    # From this point all outputs are new files strictly inside Q.
    write_json(import_path, import_proof)
    write_new(Q / 'pipeline/progress.py', (P / 'pipeline/progress.py').read_bytes())
    paper_path = Q / 'reference/paper_reference.json'
    write_new(paper_path, (P / 'analysis_plots/reference/paper_reference.json').read_bytes())
    models = [dict(id=MODEL_ID, label='旧模型：batch4、不累积', physical_batch=4,
                   accum_steps=1, final=str(FINAL), final_sha256=FINAL_SHA,
                   import_report=str(import_path), import_report_sha256=sha(import_path),
                   tensor_sha256=state_digest, tensor_count=tensor_count)]
    for folder, mapping in [('evaluation_entrypoints', wrapper_protocol['wrapper_code_sha256']),
                            ('extended_evaluation', wrapper_protocol['observer_code_sha256'])]:
        for name, expected in mapping.items():
            assets[str(P / folder / name)] = expected
    for name, expected in wrapper_protocol['base_entries_sha256'].items():
        assets[str(P / name)] = expected
    for path in [P / 'source_manifest.json', P / 'evaluation_entrypoints/protocol_manifest.json',
                 P / 'extended_evaluation/preparation_manifest.json', Q / 'verify_comparison.py',
                 Q / 'pipeline/progress.py', paper_path, import_path, FINAL, Path(__file__).resolve()]:
        assets[str(path)] = sha(path)
    wrapper = P / 'evaluation_entrypoints/capture_source.py'
    base_entry = P / 'tasks/source_artifacts.py'
    old_queue = json.loads((P / 'queue_observed_stage_20261018.json').read_text())
    eval_template = next(t for t in old_queue['tasks'] if t['id'] == 'eval_phi3_synthetic')
    tasks = []
    for domain in domains:
        ident = 'reeval_' + MODEL_ID + '_' + domain['id']
        cfg_path = Q / 'configs' / (ident + '.yaml')
        raw_report = Q / 'reports' / (ident + '.json')
        base_verified = Q / 'reports' / (ident + '_base_verified.json')
        verified = Q / 'reports' / (ident + '_verified.json')
        output = Q / 'evaluations' / ident
        value = copy.deepcopy(templates[domain['id']])
        value['EXP_NAME'] = ident
        value['OUTPUT_DIR'] = str(Q / 'model_outputs_unused')
        value['LOG_DIR'] = str(Q / 'model_logs_unused')
        value['TEST']['MODEL_FILE'] = str(FINAL)
        write_new(cfg_path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        assets[str(cfg_path)] = sha(cfg_path)
        command = [PYTHON, '-B', '-s', '-u', str(wrapper), '--base-entry', str(base_entry),
                   'evaluate', '--cfg', str(cfg_path), '--source-report', str(import_path),
                   '--output', str(output), '--report', str(raw_report),
                   '--expected-samples', str(domain['expected_samples'])]
        base_verify = [PYTHON, '-B', '-s', '-u', str(wrapper), '--base-entry', str(base_entry),
                       'verify_eval', '--cfg', str(cfg_path), '--source-report', str(raw_report),
                       '--report', str(base_verified), '--expected-samples', str(domain['expected_samples'])]
        tasks.append(dict(id=ident, model_id=MODEL_ID, domain_id=domain['id'], cfg=str(cfg_path),
                          raw_report=str(raw_report), base_verified_report=str(base_verified),
                          verified_report=str(verified), base_verify_command=base_verify,
                          command=command, main_command=command,
                          verify_command=[PYTHON, '-B', '-s', '-u', str(Q / 'verify_comparison.py'),
                                          '--task-id', ident],
                          cwd=str(source), output=str(output), expected_samples=domain['expected_samples'],
                          env=copy.deepcopy(eval_template['env']),
                          session='spnv2_old_reeval_' + domain['id'] + '_20261018',
                          log=str(Q / 'logs' / (ident + '.log'))))

    for path, expected in assets.items():
        require(sha(path) == expected, 'Bound input changed during preparation: ' + path)
    preflight = dict(schema='spnv2-old-source-reevaluation-preflight-v1', status='passed',
                     checked_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds'),
                     model_count=1, domain_count=3, no_training=True, gpu_work_launched=False,
                     cuda_initialized=False, models=models, domains=domains, geometry=geometry,
                     strict_reload=True, finite_tensors=True, inference_protocol='same frozen source and same-forward H observer',
                     original_training_acceptance=False, production_state_modified=False)
    write_json(Q / 'preflight.json', preflight)
    assets[str(Q / 'preflight.json')] = sha(Q / 'preflight.json')
    manifest = dict(schema='spnv2-unified-source-comparison-v1', status='prepared_cpu_import_passed',
                    root=str(Q), production_plan=str(P), models=models, domains=domains,
                    shared_assets=assets, geometry=geometry, tasks=tasks,
                    paper_reference=str(paper_path), paper_reference_sha256=sha(paper_path),
                    preflight=str(Q / 'preflight.json'), expected_task_count=3,
                    protocol=dict(architecture='phi3-BN/full EHS', evaluation_batch=1,
                                  inference_heads=['heatmap', 'efficientpose'], input_size=[768, 512],
                                  deterministic=True, seed=2021, source_only=True,
                                  no_training=True, no_odr=True,
                                  required_coverage='N/N for E translation, H rotation and hybrid score; no missing-pose substitution',
                                  target_threshold_q_degrees=0.169, target_threshold_t_normalized=0.002173,
                                  synthetic_thresholding=False),
                    limitations=['Only the historical LuXinghai final is reevaluated here.',
                                 'The already accepted new-model results may be compared separately after identity/protocol verification.',
                                 'Import acceptance is inference compatibility evidence, not a reconstruction of historical training.'])
    write_json(Q / 'manifest.json', manifest)
    print(json.dumps(dict(status='prepared', manifest=str(Q / 'manifest.json'),
                          manifest_sha256=sha(Q / 'manifest.json'), evaluations=3,
                          gpu_work_launched=False, training_process_touched=False), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
