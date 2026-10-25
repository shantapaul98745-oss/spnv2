"""One unpersisted observation, exact diff capture, no downstream dispatch."""
from pathlib import Path
import json
import time


def differences(a, b, path='$', limit=40):
    result = []
    def visit(x, y, p):
        if len(result) >= limit:
            return
        if type(x) is not type(y):
            result.append(dict(path=p, actual_type=type(x).__name__, expected_type=type(y).__name__, actual=x, expected=y))
        elif isinstance(x, dict):
            for key in sorted(set(x) | set(y)):
                if key not in x or key not in y:
                    result.append(dict(path=p+'.'+key, actual_present=key in x, expected_present=key in y))
                else:
                    visit(x[key], y[key], p+'.'+key)
        elif isinstance(x, list):
            if len(x) != len(y):
                result.append(dict(path=p, actual_length=len(x), expected_length=len(y)))
            for i, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, p+'['+str(i)+']')
        elif x != y:
            result.append(dict(path=p, actual=x, expected=y))
    visit(a, b, path)
    return result


class OneForward:
    def __init__(self, budget):
        self.budget, self.count = budget, 0
    def __getattr__(self, name):
        return getattr(self.budget, name)
    def reserve_forward(self, label, count=1):
        if self.count + count > 1:
            raise RuntimeError('First-difference probe permits exactly one physical forward at most')
        self.count += count
        return self.budget.reserve_forward(label, count=count)


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    import pair_runner as pair
    task = pair.load_json(task_path)
    original_task_path = pair.binding(task['failed_task'])
    original = pair.load_json(original_task_path)
    frozen = pair.load_json(pair.binding(original['index_binding']))
    pair.require(original['selection'] in frozen['windows'], 'Probe selection is not frozen')
    pair.require(original['selection']['window_id'] == 'sunlamp:2021:9', 'Only the failed second pilot is permitted')
    pair.require(admission['allow_start'] and admission['boundary_verified'], 'Admission required')
    pair.require(not conflict_check(), 'Resource ownership changed before probe')
    output = Path(task['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    parent, metadata = pair.read_generation(original['selection']['parent_checkpoint'])
    pair.require(parent['next_index'] == 64, 'Failure parent cursor changed')
    trace = pair.reference_trace(original, parent['identity'], budget=budget)
    report = pair.load_json(pair.binding(frozen['runs']['sunlamp:2021']['original_report']))
    bounded = OneForward(budget)
    runtime = None
    try:
        runtime = pair.build_runtime(original, output, parent, budget=bounded, original_report=report)
        engine, equal, hook = runtime['engine'], runtime['equal'], runtime['hook']
        pair.require(equal(pair.science_state(engine.snapshot()), pair.science_state(parent)),
                     'Restored parent scientific state differs before the probe')
        pair.trace_prefix_matches(engine.scientific_trace, trace, equal)
        # All nonzero frozen parents have >=64 prior full scoring calls. Prime
        # only this pure scripted decoder to reproduce that already-hot runtime.
        import torch
        import importlib.util
        import sys
        from utils import postprocess
        before_warm = engine.learner.snapshot()
        decoder = postprocess.rot_6d_to_matrix
        dummy = torch.tensor([[[1., 0., 0., 0., 1., 0.]]], dtype=torch.float32, device='cuda:0')
        for _ in range(6):
            decoder(dummy)
        torch.cuda.synchronize()
        pair.require(equal(before_warm, engine.learner.snapshot()), 'Pure rotation warmup changed scientific learning state')
        captured = []
        def capture_rotation(raw):
            captured.append(raw.detach().clone())
            return decoder(raw)
        # The unchanged prediction module has an imported reference to this
        # operator. Capture its actual input without invoking the model again.
        prediction_module = sys.modules['prediction']
        pair.require(prediction_module.rot_6d_to_matrix is decoder, 'Bound prediction decoder differs')
        prediction_module.rot_6d_to_matrix = capture_rotation

        before = len(engine.scientific_trace)
        hook.phase = 'first_difference_observation_0064'
        engine.step(stop_after_prepare=True)
        prediction_module.rot_6d_to_matrix = decoder
        runtime['finite']()
        actual = engine.scientific_trace[before:]
        expected = trace[before:before+len(actual)]
        pair.require(bounded.count == 1 and engine.next_index == 65 and len(actual) == 1,
                     'Probe escaped the single failed observation')
        delta = differences(actual, expected)
        pair.require(len(captured) == 1, 'Exactly one rotation input must be captured')
        raw = captured[0]
        alias = '_g50p_cold_rotation_control'
        spec = importlib.util.spec_from_file_location(alias, postprocess.__file__)
        cold_module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = cold_module
        spec.loader.exec_module(cold_module)
        cold_first = cold_module.rot_6d_to_matrix(raw).squeeze().cpu().tolist()
        cold_series = [cold_first]
        for _ in range(6):
            cold_series.append(cold_module.rot_6d_to_matrix(raw).squeeze().cpu().tolist())
        prior = pair.load_json(pair.binding(task['prior_difference']))
        cold_matches_failed = equal(cold_first, prior['actual'][0]['arrival']['effi_R'])
        hot_matches_original = equal(cold_series[-1], expected[0]['arrival']['effi_R'])
        operator_evidence = dict(raw_input=raw.cpu().tolist(), raw_shape=list(raw.shape), raw_dtype=str(raw.dtype),
            raw_stride=list(raw.stride()), cold_first=cold_first, subsequent=cold_series[1:],
            cold_matches_preserved_failed=cold_matches_failed, hot_matches_frozen_original=hot_matches_original,
            pure_warmup_learning_state_equal=True, exact_full_observation_equal=equal(actual, expected),
            source_path=postprocess.__file__, source_sha256=pair.sha256(postprocess.__file__),
            unchanged_original_operator=True, model_forward_repeats_for_operator_control=0)
        pair.atomic_json(output/'operator_evidence.json', operator_evidence)
        pair.require(not delta and cold_matches_failed and hot_matches_original,
                     'Rotation cold/hot causal control did not exactly explain the saved divergence')
        # Save the produced state too: no repeat of this successfully persisted
        # diagnostic forward is needed merely to inspect a missing field.
        state = pair.save_pickle(output/'observed_state.bin', pair.science_state(engine.snapshot()))
        result = dict(schema='g50p-first-trace-difference-v1', status='difference_captured' if delta else 'single_observation_matched',
                      failed_task=task['failed_task'], parent_checkpoint=original['selection']['parent_checkpoint'],
                      observation_index=64, actual=actual, expected=expected, differences=delta,
                      exact_equal=equal(actual, expected), physical_forwards=bounded.count,
                      full_restored_parent_equal=True, observed_state=state,
                      downstream_dispatched=False, scientific_acceptance=False,
                      deterministic_runtime=runtime['deterministic_runtime'],
                      forward_categories=hook.counts, budget=budget.snapshot())
        pair.atomic_json(output/'difference.json', result)
        return dict(status='rotation_decoder_recovery_verified', exit_code=0, scientific_acceptance=False,
                    report=str(output/'difference.json'), report_sha256=pair.sha256(output/'difference.json'),
                    physical_forwards=bounded.count, first_difference=delta[0] if delta else None)
    finally:
        if runtime is not None:
            runtime['hook'].close()
            runtime['close']()
