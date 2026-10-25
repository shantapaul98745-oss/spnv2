"""Explicit conservative debit for the pre-admission failure and repair metadata.

Uses standard attempt journal events without changing the budget implementation,
limits, initial receipt or any prior event. The extra allowance is an estimate,
not claimed process runtime. Heavy validation has its own normal actual attempt.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import fcntl

D = Path('/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/g50p_risk_diagnostic_20261025')
R = D/'revisions/admission_template_fix_20261025'
sys.path.insert(0, str(D/'implementation'))
from budget import GlobalBudget, atomic_json


def main():
    receipt = R/'control_cost_settlement.json'
    if receipt.exists():
        raise ValueError('Control cost was already settled; never debit twice')
    with (D/'pipeline.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idx = json.loads((D/'index.json').read_text())
        original = json.loads((R/'before/status.json').read_text())['active']
        log = Path(original['log_path'])
        if log.with_name(log.name+'.exitcode').read_text().strip() != '1':
            raise ValueError('Original pre-admission failure exit changed')
        if 'Task implementation differs from frozen diagnostic index' not in log.read_text():
            raise ValueError('Different pre-admission failure')
        if Path(original['launch_directory']).exists():
            raise ValueError('Owned execution record exists; do not assume uncharged pre-admission failure')
        global_budget = GlobalBudget(D/'budget', dict(project_id=idx['diagnostic_id'],
            frozen_inputs_sha256=idx['frozen_inputs']['sha256']), initial_cost_receipt=idx['initial_cost_receipt'])
        before = global_budget.snapshot()
        if before['unresolved_attempt_ids'] or before['remaining_active_seconds_lower'] < 125:
            raise ValueError('Cannot safely settle bounded control cost')
        p = Path('/proc')/str(os.getpid())
        stat = (p/'stat').read_text()
        launch = dict(pid=os.getpid(), starttime=int(stat[stat.rfind(')')+2:].split()[19]),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            cmdline_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest(),
            run_nonce=os.environ['SPNV2_RUN_NONCE'], run_id='g50p_diag_control_cost_admission_bindingfix1')
        attempt = global_budget.begin_attempt(launch, allow_model_forwards=False)
        # The explicit extra upper charge covers the failed dispatch (<12s by
        # queue-to-block timestamps) plus bounded metadata preparation/deployment.
        # It is deliberately conservative and never refunded or called measured.
        elapsed = time.monotonic()-attempt.record['started_monotonic']
        extra, tail = 120.0, 2.0
        close = dict(attempt_id=attempt.record['attempt_id'], status='control_cost_upper_settled',
            charged_seconds_upper=elapsed+extra+tail, measured_seconds_before_finalization=elapsed,
            finalization_reserved_upper_seconds=tail, retrospective_control_overhead_upper_seconds=extra,
            exact_total_time_claim=False, unresolved_forward_tokens=[],
            cost_scope='pre-admission failed dispatch plus bounded metadata repair/preparation/deployment; separate model-free validation remains additionally charged',
            original_failed_task_sha256=original['task_sha256'],
            original_failure_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        global_budget._append('attempt_closed', close)
        attempt.closed = True
        after = global_budget.snapshot()
        atomic_json(receipt, dict(schema='g50p-control-cost-conservative-settlement-v1', status='settled',
            physical_forwards=0, original_failed_attempt=original, settlement_launch=launch,
            cost=close, budget_before=before, budget_after=after))
        print(json.dumps(dict(receipt=str(receipt), charged_seconds_upper=close['charged_seconds_upper'],
                              total_active_seconds_upper=after['active_seconds_charged_upper'])))


if __name__ == '__main__':
    main()
