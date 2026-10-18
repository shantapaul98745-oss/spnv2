"""Prepare an inactive, bounded next-stage proposal; never control live processes."""
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

P = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
D = P / 'queue_proposals/observed_stage_20261018'
sys.path.insert(0, str(P / 'pipeline'))
from runner import validate_manifest
from pending_observers import revise_queued_observers
from revision import canonical_sha256

def read(path):
    return json.loads(Path(path).read_text())

cache = {}
def sha(path):
    path = Path(path).resolve()
    assert path.is_relative_to(P.parents[1]) and path.is_file(), path
    if str(path) not in cache:
        cache[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cache[str(path)]

def save(name, value):
    with (D/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

old = read(P/'queue_v2.json')
assert canonical_sha256(old) == '6562b36060f12e72dcdf9340b5262cf766cafea93c9a2fccc7ee1513c240d61b'
historical = P/'queue_proposals/v3b/full_main_v3_proposal.json'
assert sha(historical) == 'a1916236b15749016b5874eb3cb441197a7ea2ba5ca9f2febe8eeacdf60ad236'
blueprint = read(historical)
assert blueprint['tasks'][:9] == old['tasks']
fixed = [copy.deepcopy(t) for t in blueprint['tasks'] if t['kind'] == 'odr_fixed']
assert len(fixed) == 40
odr = read(P/'odr/manifest.json')
assert sha(P/'odr/manifest.json') == 'ac7c7b136d6f05ccdc0852dd3c182f11bdf1cb67b1fabefaecddd1da46ca6a77'
assert [t['id'] for t in fixed] == [t['id'] for t in odr['tasks'] if t['group'] == 'fixed']
specs = {t['id']: t for t in odr['tasks']}
e = P/'evaluation_entrypoints'
x = P/'extended_evaluation'
assert sha(e/'implementation_manifest.json') == '214f6f4d11653a85da01c281a4265e01cf6cffbda4e1cfd0e0b3b38d860a4810'
assert sha(x/'preparation_manifest.json') == '82285fc8f8cf6c1777ec4a40a1c315aab13c41d66891f9132d0660843a500cc5'
protocol = read(e/'protocol_manifest.json')
ready = read(e/'implementation_manifest.json')
assert ready['status'] == 'CPU_verified_ready_for_root_review'
for key in ['CPU_acceptance', 'independent_CPU_acceptance', 'contract_CPU']:
    proof = ready[key]
    assert proof['status'] == 'passed' and sha(proof['path']) == proof['sha256']
observer_assets = {}
for mapping, root in [(protocol['wrapper_code_sha256'],e), (protocol['observer_code_sha256'],x), (protocol['base_entries_sha256'],P)]:
    for name, digest in mapping.items():
        path = root/name
        assert sha(path) == digest, str(path)
        observer_assets[str(path)] = digest
for path in [e/'implementation_manifest.json', e/'protocol_manifest.json', x/'preparation_manifest.json', P/'source_manifest.json']:
    observer_assets[str(path)] = sha(path)
assert protocol['source_manifest_sha256'] == sha(P/'source_manifest.json')
assert protocol['observer_preparation_manifest_sha256'] == sha(x/'preparation_manifest.json')
source_pins = {str(P/'source'/k):v for k,v in read(P/'source_manifest.json')['core_tools_sha256'].items()}
frozen_config_pins = read(P/'queue_frozen_assets.json')['configs']

new = copy.deepcopy(old)
pending_ids = []
for task in new['tasks']:
    if task['kind'] != 'full_source_evaluation':
        continue
    pending_ids.append(task['id'])
    for field, mode in [('command','evaluate'), ('verify_command','verify_eval')]:
        cmd = task[field]
        i = cmd.index(str(P/'tasks/source_artifacts.py'))
        assert cmd[i+1] == mode
        task[field] = cmd[:i] + [str(e/'capture_source.py'),'--base-entry'] + cmd[i:]
    assets = task['metadata'].setdefault('asset_sha256', {})
    cfg = task['metadata']['source']
    assert cfg in assets or cfg in frozen_config_pins
    cfg_pin = assets[cfg] if cfg in assets else frozen_config_pins[cfg]
    for path, digest in {**source_pins, **observer_assets, cfg:cfg_pin}.items():
        assert path not in assets or assets[path] == digest
        assets[path] = digest
    task['metadata']['pending_observer_entry'] = str(e/'capture_source.py')
assert len(pending_ids) == 6

drifts = {}
pipeline_assets = {str(p):sha(p) for p in (P/'pipeline').glob('*.py')}
for task in fixed:
    spec = specs[task['id']]
    assert task['command'] == spec['queue_command'] and task['verify_command'] == spec['verify_command']
    for path, digest in task['metadata']['asset_sha256'].items():
        actual = sha(path)
        if actual != digest:
            assert path == str(P/'pipeline/runner.py')
            assert digest == '72385d6ca9f9fc4a2e7005a57761e9393a93922921226bbb254402639b9a7c47'
            assert actual == '35c3de5c4b0d3fc3dd320587da984958e5403dc0ccc5bd540f58d46941407e39'
            drifts[path] = {'historical':digest, 'current_verified':actual}
    md = task['metadata']
    md['base_parent_command'] = task['command']
    md['base_parent_verify_command'] = task['verify_command']
    task['command'] = spec['queue_command'][:4] + [str(e/'run_odr.py'),'--task',task['id'],'--execute']
    task['verify_command'] = spec['verify_command'][:4] + [str(e/'verify_odr.py'),'--task',task['id']]
    md['same_forward_extension_entrypoint'] = str(e/'run_odr.py')
    md['asset_sha256'].update(pipeline_assets)
    md['asset_sha256'].update(observer_assets)
    task['expected_outputs'].append(str(e/'reports'/(task['id']+'.json')))
    task['session'] = 'spnv2_odr_'+task['id']+'_obs_20261018'
    task['log'] = str(P/'logs'/('odr_'+task['id']+'_obs_20261018.log'))
    assert not Path(task['log']).exists()
    assert not Path(spec['output']).exists()
new['tasks'].extend(fixed)
new['scope'] = 'two_main_sources_observed_evaluations_and_fixed40_only'
new['backlog'] = old['backlog'][1:] + ['This 49-task stage is not full 190-epoch completion; remaining 8 sources, pilots, resources, scans, cases and final audit require later integration.']
assert len(new['tasks']) == 49
validate_manifest(new)
for task in new['tasks']:
    for path,digest in task['metadata'].get('asset_sha256', {}).items():
        assert sha(path) == digest, (task['id'],path)

# Pure validation against a single live snapshot; do not bind or save live state.
state = read(P/'status.json')
before = copy.deepcopy(state)
amended, audit = revise_queued_observers(state,old,new,task_ids=pending_ids,caller_context={'operation':'inactive_proposal_cpu_preflight_only'})
assert state == before and amended['tasks'][0] == state['tasks'][0]
assert all(new['tasks'][i] == old['tasks'][i] for i in [0,4,5])
assert state['active'] == 'train_phi3_EHS'
assert [t['id'] for t in state['tasks'] if t['status'] not in {'queued','pending'}] == ['train_phi3_EHS']
save('proposal.json',new)
raw = (D/'proposal.json').read_bytes()
with (D/'proposal.json.gz').open('xb') as f:
    f.write(gzip.compress(raw,mtime=0))
assert gzip.decompress((D/'proposal.json.gz').read_bytes()) == raw
receipt = {
    'status':'passed_inactive_proposal', 'activated':False,
    'operation':'wrap_six_never_started_source_evaluations_and_append_authorized_fixed40',
    'proposal_path':str(D/'proposal.json'), 'proposal_sha256':sha(D/'proposal.json'),
    'proposal_canonical_sha256':canonical_sha256(new),'proposal_gzip_sha256':sha(D/'proposal.json.gz'),
    'old_canonical_sha256':canonical_sha256(old),'pending_task_ids':pending_ids,
    'task_count':49, 'appended_fixed_odr':40,'reused_original_training_tasks':[old['tasks'][i]['id'] for i in [0,4,5]],
    'preserved_original_assets':True, 'old_original_args_env_gpu_dependencies_outputs_preserved':True,
    'current_worker':{k:state['current'].get(k) for k in ['pid','starttime','epoch','batch']},
    'current_manager':state['manager'],'changed_historical_bindings':drifts,
    'unique_verified_assets':len(cache), 'requires_user_exception':'one manager-only graceful restart; healthy worker untouched',
    'no_process_control':True, 'live_state_written':False,'scientific_results':False,
    'future_activation':{'manifest_flag':'--revise-pending-from','old_manifest':str(P/'queue_v2.json'),'exact_pending_tasks':pending_ids},
    'full_scope':'10 source configurations /190 epochs plus all planned evaluations, ODR, scans and final audit remains unchanged',
}
save('preflight.json',receipt)
save('pure_revision_preview.json',audit)
print(json.dumps(receipt,ensure_ascii=False))
