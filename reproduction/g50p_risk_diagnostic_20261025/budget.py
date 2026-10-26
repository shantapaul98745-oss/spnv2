"""Conservative global physical-work budget for the fixed diagnostic.

Forward reservations are durable before dispatch and are never refunded. A
crashed unresolved attempt retains its assigned bound (or an unknown upper when time is unlimited);
it blocks further execution instead of guessing a smaller cost from a log line.
No torch import, model construction, subprocess or host query occurs here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

MAX_FORWARDS = 40000
MAX_ACTIVE_SECONDS = 14400.
FINALIZATION_UPPER_SECONDS = 2.
WORK_STOP_MARGIN_SECONDS = 60.


class BudgetExhausted(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf8') as handle:
        handle.write(canonical(value)+'\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


class GlobalBudget:
    def __init__(self, root, identity, *, clock=None, initial_cost_receipt=None, budget_policy=None, activate_policy=False):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.clock = clock or time
        identity_path = self.root/'identity.json'
        value = dict(schema='g50p-diagnostic-global-budget-v1', identity=self.identity,
                     max_physical_forwards=MAX_FORWARDS, max_active_seconds=MAX_ACTIVE_SECONDS)
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf8')) != value:
                raise ValueError('Global diagnostic budget identity changed')
        else: atomic_json(identity_path, value)
        self.path = self.root/'journal.jsonl'
        self.rows = []
        self._reload()
        self._attempt = None
        if initial_cost_receipt is not None: self._initial_cost(initial_cost_receipt)
        self._configure_policy(budget_policy, activate_policy=activate_policy)

    def _configure_policy(self, binding, *, activate_policy):
        self.budget_policy = None
        self.time_hard_stop_enabled = True
        self.max_active_seconds = MAX_ACTIVE_SECONDS
        applied = [r['payload'] for r in self.rows if r['kind'] == 'budget_policy_applied']
        if len(applied) > 1:
            raise ValueError('Ambiguous diagnostic budget policy history')
        if binding is None and applied:
            binding = applied[0]['budget_policy']
        if binding is None:
            return
        def checked(spec):
            path = Path(spec['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError('Budget amendment binding changed: '+str(path))
            return json.loads(path.read_text(encoding='utf8'))
        policy = checked(binding)
        if (policy.get('schema') != 'g50p-diagnostic-budget-policy-v2'
                or policy.get('project_id') != self.identity.get('project_id')
                or policy.get('frozen_inputs_sha256') != self.identity.get('frozen_inputs_sha256')
                or policy.get('max_physical_forwards') != MAX_FORWARDS
                or policy.get('max_active_seconds') is not None
                or policy.get('time_hard_stop_enabled') is not False
                or policy.get('preserve_all_past_charges') is not True):
            raise ValueError('Only the authorized time-cap removal is allowed')
        original_identity = checked(policy['original_budget_identity'])
        if original_identity != json.loads((self.root/'identity.json').read_text(encoding='utf8')):
            raise ValueError('Original budget identity must remain byte-bound and unchanged')
        original_protocol = checked(policy['original_protocol'])
        if original_protocol.get('max_physical_forwards') != MAX_FORWARDS:
            raise ValueError('Original physical-forward cap differs')
        authorization = checked(policy['authorization'])
        if authorization.get('authorization') != 'user_2026-10-26_remove_cumulative_time_limit':
            raise ValueError('Missing explicit time-limit-removal authorization')
        prefix = policy['historical_journal_prefix']
        count = prefix['events']
        if (type(count) is not int or count < 1 or count > len(self.rows)
                or self.rows[count-1]['sha256'] != prefix['tail_sha256']):
            raise ValueError('Budget amendment does not preserve the historical journal prefix')
        if applied:
            if applied[0]['budget_policy'] != binding:
                raise ValueError('Applied budget policy cannot be silently replaced')
        else:
            if not activate_policy:
                raise ValueError('Budget policy requires explicit append-only activation')
            attempts = {r['payload']['attempt_id'] for r in self.rows if r['kind']=='attempt_started'}
            closed = {r['payload']['attempt_id'] for r in self.rows if r['kind']=='attempt_closed'}
            if attempts-closed:
                raise ValueError('Cannot amend time policy with an unresolved execution')
            self._append('budget_policy_applied', dict(budget_policy=dict(binding),
                previous_active_time_limit_seconds=MAX_ACTIVE_SECONDS,
                active_time_limit_seconds=None, time_hard_stop_enabled=False,
                max_physical_forwards=MAX_FORWARDS, historical_journal_prefix=dict(prefix),
                authorization=policy['authorization'], all_historical_charges_retained=True))
        self.budget_policy = dict(binding)
        self.time_hard_stop_enabled = False
        self.max_active_seconds = None

    def _initial_cost(self,binding):
        path=Path(binding['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
            raise ValueError('Initial diagnostic cost receipt hash changed')
        receipt=json.loads(path.read_text(encoding='utf8'))
        forwards,seconds=receipt['physical_forwards_upper'],receipt['active_seconds_upper']
        if (receipt.get('schema')!='g50p-diagnostic-initial-cost-v1' or receipt.get('status')!='passed'
                or type(forwards) is not int or not 0<=forwards<=MAX_FORWARDS
                or type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=MAX_ACTIVE_SECONDS):
            raise ValueError('Invalid initial physical-work cost receipt')
        previous=[row['payload'] for row in self.rows if row['kind']=='initial_cost_imported']
        payload=dict(binding=binding,physical_forwards_upper=forwards,active_seconds_upper=seconds)
        if previous:
            if previous!=[payload]: raise ValueError('Initial diagnostic cost binding changed or duplicated')
        else:
            if self.rows: raise ValueError('Initial diagnostic cost must be imported before real work')
            self._append('initial_cost_imported',payload)

    def _reload(self):
        previous = '0'*64
        rows = []
        if self.path.exists():
            with self.path.open('rb') as handle:
                for line in handle:
                    if not line.endswith(b'\n'):
                        raise ValueError('Torn global budget journal; preserve and block')
                    row = json.loads(line)
                    digest = row.pop('sha256')
                    if (row['sequence'] != len(rows)+1 or row['previous_sha256'] != previous
                            or hashlib.sha256(canonical(row).encode()).hexdigest() != digest):
                        raise ValueError('Global budget journal checksum/sequence differs')
                    row['sha256'] = digest; rows.append(row); previous = digest
        self.rows = rows
        self._reserved_forward_count = sum(row['payload']['count'] for row in rows if row['kind']=='forward_reserved')+sum(
            row['payload']['physical_forwards_upper'] for row in rows if row['kind']=='initial_cost_imported')

    def _append(self, kind, payload):
        row = dict(sequence=len(self.rows)+1, previous_sha256=self.rows[-1]['sha256'] if self.rows else '0'*64,
                   kind=kind, payload=payload, wall_time=self.clock.time())
        row['sha256'] = hashlib.sha256(canonical(row).encode()).hexdigest()
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w', encoding='utf8') as handle:
            handle.write(canonical(row)+'\n'); handle.flush(); os.fsync(handle.fileno())
        self.rows.append(row)
        if kind == 'forward_reserved': self._reserved_forward_count += payload['count']
        if kind == 'initial_cost_imported': self._reserved_forward_count += payload['physical_forwards_upper']
        return row

    def snapshot(self):
        attempts, forwards, finalizations = {}, {}, {}
        waits = 0.
        for row in self.rows:
            p = row['payload']
            if row['kind'] == 'attempt_started': attempts[p['attempt_id']] = dict(p)
            elif row['kind'] == 'attempt_closed': attempts[p['attempt_id']]['close'] = p
            elif row['kind'] == 'forward_reserved': forwards[p['token']] = dict(p)
            elif row['kind'] == 'forward_confirmed': forwards[p['token']]['confirmed'] = True
            elif row['kind'] == 'queue_wait': waits += p['seconds']
            elif row['kind'] == 'time_finalization_observed': finalizations[p['attempt_id']] = p
        initial=[row['payload'] for row in self.rows if row['kind']=='initial_cost_imported']
        violated={row['payload']['attempt_id'] for row in self.rows if row['kind']=='finalization_bound_violated'}
        initial_time = sum(row['active_seconds_upper'] for row in initial)
        known_charged, closed_measured, unsettled_upper = initial_time, 0., []
        for key, p in attempts.items():
            close, observed = p.get('close'), finalizations.get(key)
            if close is not None:
                measured = close.get('measured_seconds_before_finalization', 0.)
                if observed: measured = max(measured, observed['measured_total_seconds'])
                closed_measured += measured
                charge = close['charged_seconds_upper']
                if observed: charge = max(charge, observed['charged_seconds_upper'])
                if key in violated: charge = p['allocated_seconds_upper']
            else:
                charge = p['allocated_seconds_upper']
            if charge is None: unsettled_upper.append(key)
            else: known_charged += charge
        charged = None if unsettled_upper else known_charged
        reserved = sum(p['count'] for p in forwards.values())+sum(row['physical_forwards_upper'] for row in initial)
        confirmed = sum(p['count'] for p in forwards.values() if p.get('confirmed'))
        closed_charge = initial_time + sum(max(p['close']['charged_seconds_upper'],
            finalizations.get(key,{}).get('charged_seconds_upper',0.)) for key,p in attempts.items() if 'close' in p)
        return dict(schema='g50p-diagnostic-budget-snapshot-v1', physical_forwards_reserved_upper=reserved,
                    physical_forwards_confirmed=confirmed,
                    physical_forwards_unknown_upper=sum(p['count'] for p in forwards.values() if not p.get('confirmed')),
                    remaining_forward_reservations=MAX_FORWARDS-reserved,
                    active_seconds_charged_upper=charged,
                    settled_active_seconds_charged_upper=closed_charge,
                    remaining_active_seconds_lower=(MAX_ACTIVE_SECONDS-charged) if self.time_hard_stop_enabled and charged is not None else None,
                    time_hard_stop_enabled=self.time_hard_stop_enabled, max_active_seconds=self.max_active_seconds,
                    budget_policy=self.budget_policy,
                    time_accounting=dict(closed_measured_seconds=closed_measured,
                        initial_conservative_seconds=initial_time,
                        closed_conservative_addition_seconds=closed_charge-closed_measured,
                        unsettled_upper_attempt_ids=unsettled_upper,
                        exact_total_time_claim=False),
                    unresolved_attempt_ids=[key for key,p in attempts.items() if 'close' not in p],
                    unknown_forward_tokens=[key for key,p in forwards.items() if not p.get('confirmed')],
                    queue_wait_seconds=waits, initial_cost_receipts=initial,
                    active_time_bound_violation_attempt_ids=sorted(violated),
                    attempts=list(attempts.values()), journal_events=len(self.rows))

    def begin_attempt(self, launch_record, *, allow_model_forwards=True):
        if self._attempt is not None: raise ValueError('An attempt is already live')
        # The caller holds the diagnostic pipeline lock for this entire lifetime.
        # An unresolved attempt is never assumed dead or reused automatically.
        self._reload()
        if self.time_hard_stop_enabled and any(row['kind']=='finalization_bound_violated' for row in self.rows):
            raise BudgetExhausted('An earlier active-time bound was violated; preserve and block')
        snapshot = self.snapshot()
        if snapshot['unresolved_attempt_ids']:
            raise BudgetExhausted('Unresolved prior active-time allocation; preserve and block')
        remaining = snapshot['remaining_active_seconds_lower']
        if (self.time_hard_stop_enabled and remaining <= WORK_STOP_MARGIN_SECONDS) or (allow_model_forwards and snapshot['remaining_forward_reservations'] <= 0):
            raise BudgetExhausted('Global diagnostic physical-work budget exhausted')
        birth = {key:launch_record[key] for key in ('pid','starttime','boot_id','cmdline_sha256','run_nonce','run_id')}
        p = dict(attempt_id=uuid.uuid4().hex, launch=birth, started_monotonic=self.clock.monotonic(),
                 allocated_seconds_upper=remaining, started_wall_time=self.clock.time(),
                 model_forwards_allowed=bool(allow_model_forwards))
        p['deadline_monotonic'] = p['started_monotonic']+remaining if self.time_hard_stop_enabled else None
        p['budget_policy'] = self.budget_policy
        p['time_hard_stop_enabled'] = self.time_hard_stop_enabled
        self._append('attempt_started', p)
        self._attempt = AttemptBudget(self, p)
        return self._attempt

    def record_queue_wait(self, seconds, *, reason):
        if self._attempt is not None: raise ValueError('Active execution cannot be relabelled queue waiting')
        if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Queue wait must be finite nonnegative seconds')
        self._append('queue_wait', dict(seconds=float(seconds), reason=str(reason)))


class AttemptBudget:
    def __init__(self, parent, record):
        self.parent, self.record = parent, record
        self.attempt_id = record['attempt_id']
        self.deadline_monotonic = record['deadline_monotonic']
        self.pending = {}
        self.closed = False

    def check_time(self, reason='safe_point'):
        if self.closed: raise ValueError('Closed diagnostic attempt cannot execute more work')
        if self.deadline_monotonic is not None and self.parent.clock.monotonic() >= self.deadline_monotonic-WORK_STOP_MARGIN_SECONDS:
            raise BudgetExhausted('Diagnostic active-time deadline reached at '+str(reason))

    def reserve_forward(self, label, count=1):
        self.check_time('before_model_forward')
        if not self.record['model_forwards_allowed']:
            raise ValueError('This admitted independent verification is zero-forward-only')
        if type(count) is not int or count < 1: raise ValueError('Forward count must be a positive integer')
        used = self.parent._reserved_forward_count
        if used+count > MAX_FORWARDS: raise BudgetExhausted('Global physical-forward cap would be exceeded')
        token = uuid.uuid4().hex
        payload = dict(attempt_id=self.attempt_id, token=token, count=count, label=str(label))
        self.parent._append('forward_reserved', payload)
        self.pending[token] = count
        self.check_time('after_forward_reservation_fsync')
        return token

    def finish_forward(self, token, count=None):
        if self.closed or token not in self.pending: raise ValueError('Forward token missing, closed, or already confirmed')
        if count is not None and count != self.pending[token]: raise ValueError('Forward completion count differs from reservation')
        self.parent._append('forward_confirmed', dict(attempt_id=self.attempt_id, token=token, count=self.pending[token]))
        del self.pending[token]
        self.check_time('after_model_forward_and_confirmation_fsync')

    def snapshot(self):
        result = self.parent.snapshot()
        elapsed = self.parent.clock.monotonic()-self.record['started_monotonic']
        result.update(current_attempt_id=self.attempt_id, current_elapsed_seconds=elapsed,
                      deadline_monotonic=self.deadline_monotonic,
                      remaining_current_seconds=max(0.,self.deadline_monotonic-self.parent.clock.monotonic()) if self.deadline_monotonic is not None else None)
        return result

    def close(self, status='completed'):
        if self.closed: raise ValueError('Attempt already closed')
        if self.deadline_monotonic is not None and self.parent.clock.monotonic() >= self.deadline_monotonic-FINALIZATION_UPPER_SECONDS:
            raise BudgetExhausted('Only unresolved upper-bound accounting remains; no time for final publication')
        before = self.parent.clock.monotonic()
        elapsed = before-self.record['started_monotonic']
        charged = elapsed+FINALIZATION_UPPER_SECONDS
        if self.record['allocated_seconds_upper'] is not None and charged > self.record['allocated_seconds_upper']:
            raise BudgetExhausted('No remaining budget for finalization upper bound')
        payload = dict(attempt_id=self.attempt_id, status=str(status), measured_seconds_before_finalization=elapsed,
                       finalization_reserved_upper_seconds=FINALIZATION_UPPER_SECONDS,
                       charged_seconds_upper=charged, unresolved_forward_tokens=list(self.pending),
                       exact_total_time_claim=False)
        self.parent._append('attempt_closed', payload)
        if self.deadline_monotonic is not None and self.parent.clock.monotonic()-before > FINALIZATION_UPPER_SECONDS:
            # Do not silently treat an overlong fsync as a valid bounded close.
            self.parent._append('finalization_bound_violated', dict(attempt_id=self.attempt_id))
            self.closed=True; self.parent._attempt=None
            raise BudgetExhausted('Finalization exceeded reserved time upper bound; block diagnostic')
        self.closed = True; self.parent._attempt = None
        return self.parent.snapshot()

    def check_final_receipt(self):
        """After the entry's last small receipt, verify the reserved tail held."""
        if not self.closed: raise ValueError('Attempt is not finalized')
        closes=[row['payload'] for row in self.parent.rows if row['kind']=='attempt_closed'
                and row['payload']['attempt_id']==self.attempt_id]
        if len(closes)!=1: raise ValueError('Attempt finalization is ambiguous')
        if self.deadline_monotonic is None:
            observed = self.parent.clock.monotonic()-self.record['started_monotonic']
            self.parent._append('time_finalization_observed',dict(attempt_id=self.attempt_id,
                measured_total_seconds=observed, charged_seconds_upper=max(closes[0]['charged_seconds_upper'],observed+FINALIZATION_UPPER_SECONDS),
                time_hard_stop_enabled=False, exact_total_time_claim=False))
            return
        if self.parent.clock.monotonic()>self.record['started_monotonic']+closes[0]['charged_seconds_upper']:
            self.parent._append('finalization_bound_violated',dict(attempt_id=self.attempt_id))
            raise BudgetExhausted('Final entry receipt exceeded active-time upper bound; diagnostic blocked')


class AReadOnlyBudget:
    """Separate saved-state extraction accounting; zero model forwards allowed."""
    def __init__(self, root, launch_record, *, max_seconds):
        if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or max_seconds <= 60:
            raise ValueError('A extraction needs an explicit finite technical timeout over 60 seconds')
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.attempt_id=uuid.uuid4().hex; self.started=time.monotonic()
        self.deadline_monotonic=self.started+max_seconds; self.closed=False
        self.path=self.root/(self.attempt_id+'.json')
        self.record=dict(schema='g50p-diagnostic-A-readonly-cost-v1',attempt_id=self.attempt_id,
                         launch_record=launch_record,started_at=time.time(),allocated_seconds_upper=max_seconds,
                         model_forwards=0,deducted_from_B_time=False,status='running')
        atomic_json(self.path,self.record)

    def reserve_forward(self,*args,**kwargs):
        raise ValueError('A extraction is saved-state-only: model forwarding is forbidden')

    def finish_forward(self,*args,**kwargs):
        raise ValueError('A extraction cannot confirm a model forward')

    def check_time(self,reason='A_saved_state_IO'):
        if self.closed or time.monotonic() >= self.deadline_monotonic-WORK_STOP_MARGIN_SECONDS:
            raise BudgetExhausted('A saved-state extraction technical timeout at '+str(reason))

    def snapshot(self):
        return {**self.record,'measured_elapsed_seconds':time.monotonic()-self.started,
                'deadline_monotonic':self.deadline_monotonic}

    def close(self,status='completed'):
        if self.closed: raise ValueError('A extraction budget already closed')
        self.record.update(status=status,measured_seconds_before_finalization=time.monotonic()-self.started,
                           finalization_reserved_upper_seconds=FINALIZATION_UPPER_SECONDS,finished_at=time.time())
        atomic_json(self.path,self.record); self.closed=True
        return dict(self.record)
