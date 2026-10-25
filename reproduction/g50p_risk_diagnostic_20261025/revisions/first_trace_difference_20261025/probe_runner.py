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
        before = len(engine.scientific_trace)
        hook.phase = 'first_difference_observation_0064'
        engine.step(stop_after_prepare=True)
        runtime['finite']()
        actual = engine.scientific_trace[before:]
        expected = trace[before:before+len(actual)]
        pair.require(bounded.count == 1 and engine.next_index == 65 and len(actual) == 1,
                     'Probe escaped the single failed observation')
        delta = differences(actual, expected)
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
        return dict(status='technical_difference_probe_complete', exit_code=0, scientific_acceptance=False,
                    report=str(output/'difference.json'), report_sha256=pair.sha256(output/'difference.json'),
                    physical_forwards=bounded.count, first_difference=delta[0] if delta else None)
    finally:
        if runtime is not None:
            runtime['hook'].close()
            runtime['close']()
