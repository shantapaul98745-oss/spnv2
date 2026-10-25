"""Zero-model-forward correction validation using the saved real decoder input."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    from budget import atomic_json
    task = json.loads(Path(task_path).read_text())
    def checked(spec):
        path = Path(spec['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Validation binding changed: '+str(path))
        return path
    if not admission['allow_start'] or not admission['boundary_verified'] or conflict_check():
        raise ValueError('Fresh shared-host admission required')
    pair = load_module('_candidate_pair_runner', checked(task['candidate_code']['pair_runner.py']))
    manager = load_module('_candidate_diagnostic_manager', checked(task['candidate_code']['manager.py']))
    idx = json.loads(checked(task['candidate_index']).read_text())
    manager.validate_index(idx)
    # Exercise the preservation guard that permits the old accepted first pair.
    broken = copy.deepcopy(idx)
    next(iter(broken['preserved_pairs'].values()))['accepted_state']['proof']['sha256'] = '0'*64
    try:
        manager.validate_index(broken)
    except ValueError:
        tamper_rejected = True
    else:
        raise ValueError('Preserved-pair proof tamper was not rejected')
    evidence = json.loads(checked(task['operator_evidence']).read_text())
    difference = json.loads(checked(task['first_difference']).read_text())
    source = checked(dict(path=evidence['source_path'], sha256=evidence['source_sha256']))
    core = source.parent.parent
    sys.path.insert(0, str(core))
    # Restore precisely the original deterministic backend before CUDA tensors.
    runtime_path = Path(task['producer_runtime']['path'])
    sys.path.insert(0, str(runtime_path.parent))
    runtime = load_module('_bound_deterministic_runtime', checked(task['producer_runtime']))
    runtime.prepare_environment()
    import torch
    deterministic = runtime.apply(torch)
    from utils import postprocess
    if Path(postprocess.__file__).resolve() != source.resolve():
        raise ValueError('Imported decoder source differs')
    raw = torch.tensor(evidence['raw_input'], dtype=torch.float32, device='cuda:0')
    if list(raw.shape) != [1, 6] or list(raw.stride()) != evidence['raw_stride']:
        raise ValueError('Saved actual decoder input shape/stride differs')
    cold = load_module('_rotation_decoder_cold_validation', source)
    cold_result = cold.rot_6d_to_matrix(raw).squeeze().cpu().tolist()
    if cold_result != difference['actual'][0]['arrival']['effi_R']:
        raise ValueError('Cold operator does not exactly match preserved failure')
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    cold_zero = pair.restore_rotation_decoder_runtime([], None, postprocess.rot_6d_to_matrix,
                                                      torch, torch.device('cuda:0'), budget)
    if cold_zero['synthetic_operator_calls'] != 0:
        raise ValueError('Zero-observation parents must remain cold')
    receipt = pair.restore_rotation_decoder_runtime([dict(kind='observation')]*64, None,
        postprocess.rot_6d_to_matrix, torch, torch.device('cuda:0'), budget)
    hot_result = postprocess.rot_6d_to_matrix(raw).squeeze().cpu().tolist()
    if hot_result != difference['expected'][0]['arrival']['effi_R']:
        raise ValueError('Correct-shape priming does not exactly recover original rotation')
    if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(before_cuda, torch.cuda.get_rng_state()):
        raise ValueError('Pure operator priming changed RNG')
    # On this observed failure the decoded rotation is the only differing field.
    reconstructed = copy.deepcopy(difference['actual'])
    reconstructed[0]['arrival']['effi_R'] = hot_result
    if reconstructed != difference['expected']:
        raise ValueError('Saved observation has additional unexplained differences')
    if pair.differences(difference['actual'], difference['expected']) != difference['differences']:
        raise ValueError('New first-difference capture does not retain the actual failure fields')
    path = Path(task['output_path'])
    if path.exists():
        raise ValueError('Validation evidence already exists')
    result = dict(schema='g50p-rotation-runtime-targeted-validation-v1', status='passed',
                  model_forwards=0, model_constructed=False, cold_exactly_matches_failure=True,
                  corrected_hot_exactly_matches_original=True, raw_input_shape=[1,6],
                  all_saved_observation_differences_explained=True, rng_unchanged=True,
                  zero_parent_kept_cold=True, runtime_recovery=receipt,
                  candidate_index=task['candidate_index'], candidate_code=task['candidate_code'],
                  first_difference=task['first_difference'], operator_evidence=task['operator_evidence'],
                  preserved_pair_lineage_valid=True, altered_proof_rejected=tamper_rejected,
                  actual_production_pair_still_pending=True, deterministic_runtime=deterministic,
                  scientific_protocol_changed=False, budget=budget.snapshot())
    atomic_json(path, result)
    return dict(status='rotation_runtime_fix_validation_passed', exit_code=0, model_forwards=0,
                proof=str(path), proof_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
