"""Finite Flare acceptance/evaluation, delegating unchanged scientific entries."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import uuid

P = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017')
HERE = Path(__file__).resolve().parent
SELECTED = ('train_table3_flare', 'eval_table3_flare_synthetic',
            'eval_table3_flare_lightbox', 'eval_table3_flare_sunlamp')
ALLOWED = {(SELECTED[0], 'verify')} | {(x, phase) for x in SELECTED[1:] for phase in ('main', 'verify')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(path), sha256=sha(path))


def permitted(ident, phase):
    if (ident, phase) not in ALLOWED:
        raise RuntimeError('Outside the explicit Flare evaluation authorization: ' + ident + '/' + phase)


def main():
    started = time.monotonic()
    if not os.environ.get('STY'):
        raise RuntimeError('Use the original independent screen wrapper')
    request = json.loads((HERE / 'request.json').read_text())
    for name, digest in request['code_sha256'].items():
        if sha(name) != digest:
            raise RuntimeError('Bound implementation changed: ' + name)
    sys.path.insert(0, str(P / 'pipeline'))
    import runner
    from progress import Reporter, atomic_json, now
    from task_entry import process_identity
    # The unchanged Queue finds its metadata execution entry beside __file__.
    runner.__file__ = __file__

    class FiniteQueue(runner.Queue):
        def prepare_launch(self, task, phase):
            permitted(task['id'], phase)
            return super().prepare_launch(task, phase)

    # Targeted scope checks: a completed producer and all later training are denied.
    assert len(ALLOWED) == 7
    for ident, phase in ALLOWED:
        permitted(ident, phase)
    for ident, phase in ((SELECTED[0], 'main'), ('train_table4_phi0_bn', 'main'),
                         ('train_table4_phi3_gn', 'main')):
        try:
            permitted(ident, phase)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Training dispatch must be denied')

    with (P / 'pipeline.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (HERE / 'activation.json').exists():
            raise RuntimeError('This activation is one-time; inspect saved attempts rather than relaunching')
        if sha(P / 'status.json') != request['before_status_sha256']:
            raise RuntimeError('The source queue state changed during preparation')
        before = json.loads((P / 'status.json').read_text())
        manifest_path = Path(request['manifest'])
        manifest = json.loads(manifest_path.read_text())
        assert runner.manifest_hash(manifest) == before['manifest_sha256'] == request['manifest_sha256']
        producer = next(x for x in before['tasks'] if x['id'] == SELECTED[0])
        old_main = copy.deepcopy(producer['runtime'])
        assert old_main['phase'] == 'main' and old_main['exit_code'] == old_main['wrapper_exit_code'] == 0
        result = json.loads(Path(old_main['result']).read_text())
        assert result['exit_code'] == int(Path(old_main['log'] + '.exitcode').read_text()) == 0
        assert result['pid'] == request['native_worker']['pid']
        assert str(result['starttime']) == str(request['native_worker']['starttime'])
        assert not runner.process_alive(request['native_worker'])
        assert not runner.process_alive(before.get('manager'))
        for t in before['tasks']:
            if t['id'] in SELECTED[1:] or t['id'].startswith(('train_table4_', 'eval_table4_')):
                assert t['status'] == 'queued' and not t.get('runtime')
        prior_accepted = [copy.deepcopy(t) for t in before['tasks'] if t['status'] == 'accepted']
        forbidden = [copy.deepcopy(t) for t in before['tasks'] if t['id'].startswith(('train_table4_', 'eval_table4_'))]
        assert len(prior_accepted) == 72
        for name, value in (('before_status.json', before), ('before_cursor.json', json.loads((P / 'monitoring/guard_cursor.json').read_text()))):
            atomic_json(HERE / name, value)
        state = runner.initialize_state(before, manifest)
        state.update(downstream_blocked=False, automatic_dispatch_enabled=False,
                     authorized_phase_scope=[list(x) for x in sorted(ALLOWED)],
                     status='flare_evaluation_only', error=None)
        state['flare_evaluation_authorization'] = bind(HERE / 'request.json')
        identity = process_identity(os.getpid())
        identity.update(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        nonce=uuid.uuid4().hex, command=sys.argv,
                        cmdline_sha256=sha('/proc/self/cmdline'))
        atomic_json(HERE / 'manager_identity.json', identity)
        assert json.loads((HERE / 'manager_identity.json').read_text()) == identity
        state['manager'] = dict(identity, started_at=now(), mode='finite_flare_evaluation_only')
        reporter = Reporter(P / 'status.json', Path(manifest['report_path']), state,
                            interval=60, warning_path=HERE / 'reporter.warnings.log')
        queue = FiniteQueue(manifest_path, manifest, state, reporter)
        atomic_json(HERE / 'activation.json', dict(at=now(), request=bind(HERE / 'request.json'),
                    native_main_reused=True, allowed_phases=[list(x) for x in sorted(ALLOWED)],
                    manager=identity, later_training_disabled=True))
        def stop(signum, frame):
            queue.stop_requested = True
            state['manager']['signal'] = signum
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        reporter.publish(force=True)
        try:
            # Explicit finite selection; never call Queue.step on the original backlog.
            for ident in SELECTED:
                task = next(x for x in state['tasks'] if x['id'] == ident)
                next_resource_check = 0.0
                while task['status'] != 'accepted':
                    if queue.stop_requested:
                        raise RuntimeError('Finite controller stopped; healthy child is preserved')
                    if task['status'] == 'failed':
                        raise RuntimeError(task.get('error', ident + ' failed'))
                    if task.get('runtime'):
                        queue.observe(task)
                    elif time.monotonic() >= next_resource_check:
                        permitted(ident, 'main')
                        ready, reason = runner.gpu_available(task.get('gpu', False))
                        if ready:
                            ready, reason = runner.gpu_available(task.get('gpu', False))
                        if ready:
                            queue.prepare_launch(task, 'main')
                        else:
                            state.update(status='waiting_resources', active=ident, resource_wait=reason)
                            next_resource_check = time.monotonic() + 300
                    reporter.publish()
                    time.sleep(1)
                for output in task['expected_outputs']:
                    path = Path(output if isinstance(output, str) else output['path'])
                    assert json.loads(path.read_text())['status'] == 'passed'
                if ident == SELECTED[0]:
                    task.update(native_training_complete=True, scientific_accepted=True,
                                verification_deferred_by_user_stop=False)
                atomic_json(HERE / (ident + '.accepted.json'), dict(at=now(), task=task))
                print('FLARE_ACCEPTED ' + ident, flush=True)
            assert [t for t in state['tasks'] if t['id'].startswith(('train_table4_', 'eval_table4_'))] == forbidden
            assert [t for t in state['tasks'] if t['id'] in {x['id'] for x in prior_accepted}] == prior_accepted
            final_producer = next(t for t in state['tasks'] if t['id'] == SELECTED[0])
            assert any(h == old_main for h in final_producer['history'])
            state.update(status='flare_evaluation_complete_stopped', active=None,
                         downstream_blocked=True, automatic_dispatch_enabled=False, resource_wait=None)
            state['manager'].update(stopped_at=now(), terminal='finite_scope_complete')
            summary = dict(at=now(), status='passed_pending_controller_wrapper_exit',
                           selected_tasks=list(SELECTED), model_training_replayed=False,
                           independent_model_acceptance=bind(P / 'reports/source_table3_flare_verified.json'),
                           domains={d:bind(P / ('reports/eval_table3_flare_' + d + '_verified.json'))
                                    for d in ('synthetic','lightbox','sunlamp')},
                           original72acceptances_preserved=True, table4_undispatched=True,
                           wall_seconds=time.monotonic()-started,
                           cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                           cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime)
            atomic_json(HERE / 'completion.json', summary)
            state['flare_evaluation_completion'] = bind(HERE / 'completion.json')
            reporter.publish(force=True)
            print('FLARE_EVALUATION_COMPLETE', flush=True)
        except BaseException as error:
            state.update(status='flare_evaluation_failed', downstream_blocked=True,
                         automatic_dispatch_enabled=False, error=repr(error))
            atomic_json(HERE / 'failure.json', dict(at=now(), error=repr(error),
                        wall_seconds=time.monotonic()-started, no_automatic_retry=True))
            reporter.publish(force=True)
            raise


if __name__ == '__main__':
    main()
