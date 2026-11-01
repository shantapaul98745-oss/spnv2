"""Bind only storage/metadata changes to the prepared original science queue."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    rev = Path(__file__).resolve().parent
    plan = rev.parent.parent
    dep = rev / 'deployment'
    dep.mkdir(exist_ok=False)
    manifest = json.loads((rev / 'candidate/manifest.json').read_text())
    state = json.loads((rev / 'candidate/candidate_state.json').read_text())
    source = plan / 'source/tools/train.py'
    # Snapshot the metadata-only entry. Keep all original math/exit semantics.
    entry = (plan / 'pipeline/task_entry.py').read_text()
    entry = entry.replace('import argparse\n', 'import argparse\nimport hashlib\nimport uuid\n', 1)
    entry = entry.replace('    started = now()\n', '    started = now()\n    launch_nonce = uuid.uuid4().hex\n', 1)
    entry = entry.replace('    environment.update(config.get("env") or {})\n',
        '    environment.update(config.get("env") or {})\n'
        '    environment["SPNV2_SOURCE150_TASK_NONCE"] = launch_nonce\n', 1)
    entry = entry.replace('        atomic_json(record_path, record)\n',
        '        record.update(boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),\n'
        '                      launch_nonce=launch_nonce,\n'
        '                      actual_cmdline_sha256=hashlib.sha256(Path("/proc/" + str(child.pid) + "/cmdline").read_bytes()).hexdigest())\n'
        '        atomic_json(record_path, record)\n'
        '        if read_json(record_path) != record:\n'
        '            raise RuntimeError("Task identity registration readback failed")\n', 1)
    # Dedicated system caches; the actual ImageNet cache remains the original
    # frozen source initialization asset, never downloaded or replaced here.
    entry = entry.replace('    cache_root = Path(config["config"]).parent / "cache" / Path(config["config"]).name.removesuffix(".config.json")',
        '    cache_root = Path("/tmp/spnv2_source150_20261101") / Path(config["config"]).name.removesuffix(".config.json")')
    compile(entry, str(rev / 'task_entry.py'), 'exec')
    (rev / 'task_entry.py').write_text(entry)
    (rev / 'progress.py').write_bytes((plan / 'pipeline/progress.py').read_bytes())
    from checkpoint_retention import SCHEMA, FIXED_LIMITS, NativeCheckpointRetention
    common = {str(rev / name): sha(rev / name) for name in
              ('storage_entry.py','checkpoint_retention.py','task_entry.py','manager_entry.py','progress.py')}
    estimates = []
    for task in manifest['tasks'][52:]:
        task['metadata']['asset_sha256'].update(common)
        if not task['id'].startswith('train_'):
            continue
        cfg = Path(task['metadata']['source']).read_text()
        out = Path(re.search(r'(?m)^OUTPUT_DIR: (.+)$', cfg).group(1))
        exp = re.search(r'(?m)^EXP_NAME: (.+)$', cfg).group(1)
        arch = re.search(r'(?m)^    NAME: (.+)$', cfg).group(1)
        policy = dict(schema=SCHEMA, retain_latest=2, run_output=str(out / arch / exp),
                      total_output_root=str(out), base_entry_sha256=sha(source), **FIXED_LIMITS)
        NativeCheckpointRetention(policy).preflight()
        policy_path = dep / (task['id'] + '.storage_policy.json')
        write(policy_path, policy)
        old = task['command']; i = old.index(str(source))
        task['command'] = old[:i] + [str(rev / 'storage_entry.py'), '--base-entry',
                           str(source), '--policy', str(policy_path), '--'] + old[i + 1:]
        task['metadata']['asset_sha256'][str(policy_path)] = sha(policy_path)
        task['metadata']['storage_policy'] = str(policy_path)
        task['metadata']['storage_only_entry_original_command'] = old
        estimates.append({'task': task['id'], 'run_output': policy['run_output'],
                          'max_run_bytes': policy['max_run_bytes'], 'ordinary_checkpoints': 2,
                          'protected': ['initial_state.pth.tar', 'model_final.pth.tar',
                                        'run_manifest.json', 'reproducible_run.json']})
    sys.path.insert(0, str(plan / 'pipeline'))
    from runner import validate_manifest, manifest_hash, initialize_state
    validate_manifest(manifest)
    state['tasks'] = state['tasks'][:52] + [dict(copy.deepcopy(t), status='queued') for t in manifest['tasks'][52:]]
    state['manifest_sha256'] = manifest_hash(manifest)
    state['source150_storage_revision'] = str(dep / 'binding.json')
    initialized = initialize_state(state, manifest)
    assert initialized['tasks'][:52] == state['tasks'][:52]
    assert len(initialized['completed']) == 52
    write(dep / 'manifest.json', manifest)
    write(dep / 'state.json', state)
    write(dep / 'storage_estimate.json', dict(schema='source150-storage-estimate-v1',
        source= 'Previous phi3 native checkpoint180635951B; phi6 max715357090B; no newphi6.',
        **FIXED_LIMITS, runs=estimates,
        preserved_initial_final_and_scientific_evidence=True,
        failed_native_publication_keeps_pending_and_requires_reconciliation=True))
    files = [p for p in dep.iterdir() if p.is_file()]
    files += [rev / name for name in ('activate_once.py','manager_entry.py','task_entry.py','progress.py',
                                     'storage_entry.py','checkpoint_retention.py','prepare_deployment.py')]
    write(dep / 'binding.json', dict(plan=str(plan),
        python='/home/ps/anaconda3/envs/SPNv2/bin/python',
        old_status_sha256=sha(rev / 'candidate/old55_status.json'),
        runner={'path': str(plan / 'pipeline/runner.py'), 'sha256': sha(plan / 'pipeline/runner.py')},
        files_sha256={str(p): sha(p) for p in files},
        manifest_canonical_sha256=state['manifest_sha256'],
        inherited_accepted_tasks=52, remaining_formal_epochs=150,
        new_training_count=8, new_domain_evaluation_count=24,
        scope='original190source_epoch_plan_only_no_G50P_reopening_or_extra_scans'))
    print(json.dumps({'status':'storage_bound_prepared', 'manifest': str(dep/'manifest.json'),
                      'sha256':state['manifest_sha256'], 'new_trainings':8,'epochs':150}))


if __name__ == '__main__':
    main()
