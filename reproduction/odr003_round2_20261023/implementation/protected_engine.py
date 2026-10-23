"""Injected G50-P/RW/R-P execution with complete-checkpoint authority.

This module does not load real images, labels, models, or scheduling resources.
The caller supplies deterministic, RNG-isolated data access and pure scorers.
Journal entries charge physical work; only published complete states determine
scientific observations, decisions, and successful updates.
"""
from __future__ import annotations

import copy
import math
import random
import time
from typing import Any, Mapping

import numpy as np

from durable_state import DurableRunStore
from guard_pools import GuardPools, assess_protection_window
from state_adapter import equal_tree, rng_state


SCHEMA = "odr003-round2-protected-engine-v1"
FAULT_POINTS = ("candidate_2_tentative", "observation_31")


class ControlledInterruption(RuntimeError):
    """The caller exits this owned run and resumes its complete checkpoint."""


def _science(value):
    """Remove physical timing from deterministic scientific records."""
    if isinstance(value, np.ndarray):
        return _science(value.tolist())
    if isinstance(value, np.generic):
        return _science(value.item())
    if isinstance(value, Mapping):
        return {key: _science(item) for key, item in value.items()
                if key not in {"timing_seconds", "elapsed_seconds", "compute_seconds",
                               "wall_time_seconds", "timestamp", "created_at"}}
    if isinstance(value, (tuple, list)):
        return [_science(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("Nonfinite scientific signal")
    return copy.deepcopy(value)


def _cursors(state):
    counts = state["counters"]
    return dict(observed=counts["observed"], candidates=counts["candidates"],
                commits=counts["commits"], rollbacks=counts["rollbacks"],
                next_index=state["next_index"])


def validate_engine_state(state):
    """Complete engine envelope validator for DurableRunStore."""
    required = {"schema", "identity", "mode", "stream_records", "learner", "pools",
                "provider_state", "counters", "next_index", "window_ids", "transaction",
                "rw_rng_state", "rw_probability", "rw_seed", "rp_mask", "scientific_trace",
                "finish_counts", "finished", "checkpoint_every", "ledger_boundary"}
    if not isinstance(state, Mapping) or not required <= state.keys() or state["schema"] != SCHEMA:
        raise ValueError("Incomplete engine checkpoint envelope")
    if state["mode"] not in ("P", "disabled", "RW", "RP"):
        raise ValueError("Unknown method in engine checkpoint")
    for name in ("observed", "accepted", "candidates", "commits", "rollbacks", "tail_discarded",
                 "consecutive_rollbacks", "max_consecutive_rollbacks"):
        if type(state["counters"].get(name)) is not int or state["counters"][name] < 0:
            raise ValueError(f"Invalid scientific counter {name}")
    counts = state["counters"]
    if (counts["observed"] != state["next_index"] or counts["accepted"] > counts["observed"]
            or not 0 <= state["next_index"] <= len(state["stream_records"])):
        raise ValueError("Input/scientific cursors disagree")
    pending = state["transaction"] is not None
    if counts["candidates"] != counts["commits"] + counts["rollbacks"] + int(pending):
        raise ValueError("Candidate decisions disagree with transaction state")
    window = state["window_ids"]
    learning = state["learner"]
    if not isinstance(window, list) or len(window) > 4 or len(set(window)) != len(window):
        raise ValueError("Invalid learning window")
    if pending:
        transaction = state["transaction"]
        if (transaction["phase"] != "prepared" or len(window) != 4
                or learning["phase"] != "collecting" or learning["window_size"] != 3
                or len(transaction["pre_signals"]) != len(transaction["protection_records"])):
            raise ValueError("Prepared checkpoint must contain fourth ID and only three gradients")
    elif not state["finished"] and learning["window_size"] != len(window):
        raise ValueError("Partial gradient/window identities disagree")
    adapter_counts = learning["learning"]["counters"]
    if (adapter_counts["observed"] != counts["observed"]
            or adapter_counts["accepted"] != counts["accepted"] - int(pending)
            or adapter_counts["updates"] != counts["commits"]):
        raise ValueError("Learning/scientific counters disagree")
    trace_ids = [event["id"] for event in state["scientific_trace"]]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("Duplicate scientific event IDs")
    if sum(event["kind"] == "observation" for event in state["scientific_trace"]) != counts["observed"]:
        raise ValueError("Observation trace length differs from complete cursor")
    if sum(event["kind"] == "window" for event in state["scientific_trace"]) != counts["commits"] + counts["rollbacks"]:
        raise ValueError("Decision trace length differs from complete cursor")
    observations = [event for event in state["scientific_trace"] if event["kind"] == "observation"]
    for index, event in enumerate(observations):
        if (event["id"] != f"observation:{index:04d}" or event["observation_index"] != index
                or event["sample_id"] != state["stream_records"][index]["sample_id"]
                or type(event["accepted"]) is not bool):
            raise ValueError("Observation trace diverges from the frozen stream")
    if sum(event["accepted"] for event in observations) != counts["accepted"]:
        raise ValueError("Accepted observation trace/counter mismatch")
    windows = [event for event in state["scientific_trace"] if event["kind"] == "window"]
    for number, event in enumerate(windows, start=1):
        if event["id"] != f"candidate:{number:04d}" or event["number"] != number:
            raise ValueError("Window scientific IDs do not form a stable sequence")
    if (sum(event["decision"] == "commit" for event in windows) != counts["commits"]
            or sum(event["decision"] == "rollback" for event in windows) != counts["rollbacks"]):
        raise ValueError("Window decisions differ from scientific counters")
    accepted_ids = [event["sample_id"] for event in observations if event["accepted"]]
    resolved_images = 4 * (counts["commits"] + counts["rollbacks"])
    unresolved = accepted_ids[resolved_images:]
    if state["finished"]:
        if len(unresolved) != counts["tail_discarded"] or window:
            raise ValueError("Finished tail does not explain unresolved accepted images")
    elif window != unresolved:
        raise ValueError("Pending window omits or reorders accepted images")
    if pending and state["transaction"]["number"] != counts["candidates"]:
        raise ValueError("Prepared candidate identity mismatch")
    if state["finished"] and (pending or learning["phase"] != "finished" or state["finish_counts"] is None):
        raise ValueError("Final checkpoint is incomplete")


class ProtectedEngine:
    def __init__(self, *, mode, stream_records, learner, pools: GuardPools,
                 store: DurableRunStore, image_provider, arrival_scorer,
                 protection_scorer, gate, sync=lambda: None,
                 rw_probability=None, rw_seed=None, rp_mask=None,
                 checkpoint_every=64, fault_points=(), verify_purity=True):
        if mode not in ("P", "disabled", "RW", "RP"):
            raise ValueError("Mode must be P, disabled, RW, or RP")
        if store.state_validator is not validate_engine_state:
            raise ValueError("Construct DurableRunStore with state_validator=validate_engine_state")
        records = _science(stream_records)
        ids = [record["sample_id"] for record in records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Frozen stream must contain unique sample IDs")
        for record in records:
            if not {"sample_id", "image_sha256", "preprocess_sha256"} <= record.keys():
                raise ValueError("Bound asset/preprocessing identities required")
        if checkpoint_every != 64:
            raise ValueError("The fixed regular checkpoint interval is 64 observations")
        if set(fault_points) - set(FAULT_POINTS):
            raise ValueError("Unapproved controlled interruption point")
        if mode == "RW":
            if (isinstance(rw_probability, bool) or rw_probability is None
                    or not math.isfinite(rw_probability) or not 0 <= rw_probability <= 1
                    or type(rw_seed) is not int or rw_seed < 0):
                raise ValueError("RW requires fixed finite p and an independent seed")
        elif rw_probability is not None or rw_seed is not None:
            raise ValueError("RW parameters belong only to RW")
        if mode == "RP":
            if (rp_mask is None or len(rp_mask) != len(ids) or any(type(x) is not bool for x in rp_mask)
                    or sum(rp_mask) % 4):
                raise ValueError("RP requires its frozen exact multiple-of-four position mask")
        elif rp_mask is not None:
            raise ValueError("Position mask belongs only to RP")
        self.mode, self.stream_records = mode, records
        self.learner, self.pools, self.store = learner, pools, store
        self.image_provider, self.arrival_scorer = image_provider, arrival_scorer
        self.protection_scorer, self.gate, self.sync = protection_scorer, gate, sync
        self.rw_probability, self.rw_seed = rw_probability, rw_seed
        self.rw_rng = random.Random(rw_seed) if mode == "RW" else None
        self.rp_mask = list(rp_mask) if rp_mask is not None else None
        self.checkpoint_every = checkpoint_every
        self.fault_points, self.verify_purity = tuple(fault_points), bool(verify_purity)
        self.next_index = 0
        self.window_ids = []
        self.transaction = None
        self.scientific_trace = []
        self.counters = dict(observed=0, accepted=0, candidates=0, commits=0, rollbacks=0,
                             tail_discarded=0, consecutive_rollbacks=0, max_consecutive_rollbacks=0)
        self.finish_counts, self.finished = None, False
        self._ready = False
        self._previous_work = {}
        self._fault_markers = set()

    def _index_journal(self):
        records, _ = self.store.read_journal()
        for record in records:
            if record["kind"] == "work_started":
                item = record["payload"]
                self._previous_work[item["logical_key"]] = item["attempt_id"]
            if record["kind"] == "controlled_interruption":
                self._fault_markers.add(record["payload"]["point"])

    def _work(self, kind, logical_key, category, function):
        reserved = {} if category is None else {category: 1}
        previous = self._previous_work.get(logical_key)
        attempt = self.store.begin_work(kind, reserved, logical_key=logical_key, replay_of=previous)
        self._previous_work[logical_key] = attempt
        self.sync()
        started = time.perf_counter()
        # If dispatch fails, leave its reservation unresolved rather than
        # inventing an exact number of completed hardware forwards.
        result = function()
        self.sync()
        self.store.finish_work(attempt, completed_forwards=reserved,
                               compute_seconds=time.perf_counter() - started)
        return result

    def _image(self, sample_id, logical_key):
        before = copy.deepcopy(rng_state())
        image = self._work("image_read", logical_key, None, lambda: self.image_provider(sample_id))
        if not equal_tree(before, rng_state()):
            raise RuntimeError("image_provider changed global model-compute RNG")
        return image

    def _score(self, scorer, image, logical_key, category):
        before = self.learner.snapshot() if self.verify_purity else None
        result = self._work(category, logical_key, category, lambda: scorer(image))
        if before is not None and not equal_tree(before, self.learner.snapshot()):
            raise RuntimeError("Scoring mutated learning state, gradient, cache, mode, or RNG")
        return _science(result)

    def _provider_state(self):
        if hasattr(self.image_provider, "state_dict"):
            if not hasattr(self.image_provider, "load_state_dict"):
                raise ValueError("Stateful provider needs load_state_dict")
            state = self.image_provider.state_dict()
            if state is None:
                raise ValueError("A stateful provider must expose an explicit non-None state")
            return copy.deepcopy(state)
        return None

    def snapshot(self):
        state = dict(schema=SCHEMA, identity=dict(self.store.identity), mode=self.mode,
                     stream_records=copy.deepcopy(self.stream_records), learner=self.learner.snapshot(),
                     pools=self.pools.state_dict(), provider_state=self._provider_state(),
                     counters=dict(self.counters), next_index=self.next_index,
                     window_ids=list(self.window_ids), transaction=copy.deepcopy(self.transaction),
                     rw_rng_state=self.rw_rng.getstate() if self.rw_rng else None,
                     rw_probability=self.rw_probability, rw_seed=self.rw_seed,
                     rp_mask=copy.deepcopy(self.rp_mask), scientific_trace=copy.deepcopy(self.scientific_trace),
                     finish_counts=copy.deepcopy(self.finish_counts), finished=self.finished,
                     checkpoint_every=self.checkpoint_every, ledger_boundary=self.store.journal_boundary())
        validate_engine_state(state)
        return state

    def checkpoint(self, reason="explicit"):
        self.sync()
        started = time.perf_counter()
        state = self.snapshot()
        info = self.store.save(state, _cursors(state))
        checkpoint_seconds = time.perf_counter() - started
        self.store.append_event("checkpoint_published", {"generation": info.generation,
                                "reason": reason, "scientific_cursors": info.scientific_cursors,
                                "checkpoint_seconds": checkpoint_seconds,
                                "timed_scope": "state_capture_and_complete_generation_save_with_fsync"})
        return info

    def resume(self):
        if self._ready:
            raise RuntimeError("Engine already initialized; instantiate a fresh engine to recover")
        recovered = self.store.recover()
        self._index_journal()
        if recovered is None:
            self._ready = True
            self.checkpoint("initial")
            return None
        state = recovered.state
        validate_engine_state(state)
        for name in ("mode", "stream_records", "rw_probability", "rw_seed", "rp_mask", "checkpoint_every"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Recovery protocol mismatch: {name}")
        if recovered.scientific_cursors != _cursors(state):
            raise ValueError("Published metadata and complete state cursors differ")
        if (state["provider_state"] is not None) != hasattr(self.image_provider, "state_dict"):
            raise ValueError("Provider state contract changed across checkpoint recovery")
        self.pools.load_state_dict(state["pools"])
        if state["provider_state"] is not None:
            if not hasattr(self.image_provider, "load_state_dict"):
                raise ValueError("Checkpoint requires provider state restoration")
            self.image_provider.load_state_dict(copy.deepcopy(state["provider_state"]))
        self.learner.load(state["learner"])
        if self.rw_rng:
            self.rw_rng.setstate(state["rw_rng_state"])
        for name in ("counters", "next_index", "window_ids", "transaction", "scientific_trace",
                     "finish_counts", "finished"):
            setattr(self, name, copy.deepcopy(state[name]))
        self._ready = True
        return recovered.info

    def _fault_once(self, point):
        if point not in self.fault_points or point in self._fault_markers:
            return
        self.store.append_event("controlled_interruption", {"point": point},
                                event_id="controlled-interruption:" + point)
        self._fault_markers.add(point)
        raise ControlledInterruption(point)

    def _prepare(self):
        number = self.counters["candidates"] + 1
        transaction_id = f"candidate:{number:04d}"
        protected = self.pools.protection_records(self.window_ids) if self.mode == "P" else []
        pre = []
        for record in protected:
            sample_id = record["sample_id"]
            image = self._image(sample_id, f"{transaction_id}:pre:{sample_id}:read")
            pre.append(self._score(self.protection_scorer, image,
                                  f"{transaction_id}:pre:{sample_id}", "protection_pre"))
        self.counters["candidates"] = number
        self.transaction = dict(id=transaction_id, number=number, phase="prepared",
                                protection_records=protected, pre_signals=pre)
        # Exactly four observed accepted IDs, but only the first three have
        # contributed gradients. Normally this is an in-memory transaction:
        # full checkpoints are every 64 observations and normal exit. The
        # predeclared candidate-2 engineering fault alone adds a prepared save.
        if (number == 2 and "candidate_2_tentative" in self.fault_points
                and "candidate_2_tentative" not in self._fault_markers):
            self.checkpoint("controlled_candidate_2_prepared")

    def _resolve_prepared(self):
        transaction = self.transaction
        if transaction is None or transaction["phase"] != "prepared":
            raise RuntimeError("No prepared candidate to resolve")
        key = transaction["id"]
        fourth_id = self.window_ids[3]
        image = self._image(fourth_id, key + ":fourth:read")
        loss = self._work("temporary_update", key + ":temporary_update", "adaptation",
                          lambda: self.learner.candidate(image))
        if transaction["number"] == 2:
            self._fault_once("candidate_2_tentative")
        protection = None
        protection_pairs = None
        coin = None
        if self.mode == "P":
            pairs = []
            for record, before in zip(transaction["protection_records"], transaction["pre_signals"]):
                sample_id = record["sample_id"]
                protected_image = self._image(sample_id, f"{key}:post:{sample_id}:read")
                after = self._score(self.protection_scorer, protected_image,
                                    f"{key}:post:{sample_id}", "protection_post")
                pairs.append(dict(sample_id=sample_id, pre=before, post=after))
            protection = assess_protection_window(pairs)
            protection_pairs = pairs
            self.pools.record_check(key, transaction["protection_records"])
            commit = not protection["rollback"]
        elif self.mode == "RW":
            coin = self.rw_rng.random()
            commit = coin < self.rw_probability
        else:
            commit = True
        decision = "commit" if commit else "rollback"
        self._work("window_" + decision, key + ":" + decision, None,
                   getattr(self.learner, decision))
        self.counters["commits" if commit else "rollbacks"] += 1
        self.counters["consecutive_rollbacks"] = 0 if commit else self.counters["consecutive_rollbacks"] + 1
        self.counters["max_consecutive_rollbacks"] = max(self.counters["max_consecutive_rollbacks"],
                                                         self.counters["consecutive_rollbacks"])
        pool_result = self.pools.finish_window() if self.mode != "RP" else None
        event = dict(id=key, kind="window", number=transaction["number"],
                     sample_ids=list(self.window_ids), decision=decision, fourth_loss=float(loss),
                     protection_records=transaction["protection_records"], protection=protection,
                     protection_pairs=protection_pairs,
                     rw_coin=coin, rw_probability=self.rw_probability,
                     pool_transition=pool_result, counters=dict(self.counters))
        self.scientific_trace.append(_science(event))
        # This journal record is diagnostic only. The decision is durable for
        # science only once a complete checkpoint containing it is published.
        self.store.append_event("window_decision_audit", {"id": key, "decision": decision,
                                "counters": dict(self.counters)})
        self.window_ids = []
        self.transaction = None

    def _after_observation(self):
        if self.next_index == 31 and "observation_31" in self.fault_points and "observation_31" not in self._fault_markers:
            self.checkpoint("controlled_observation_31_safe_point")
            self._fault_once("observation_31")
        if self.next_index % self.checkpoint_every == 0:
            self.checkpoint("regular_observation_boundary")

    def step(self, *, stop_after_prepare=False):
        """Process one new observation, or resolve one saved prepared candidate."""
        if type(stop_after_prepare) is not bool:
            raise ValueError("stop_after_prepare is an explicit engineering control")
        if not self._ready:
            self.resume()
        if self.finished:
            return False
        if self.transaction is not None:
            self._resolve_prepared()
            self._after_observation()
            return True
        if self.next_index == len(self.stream_records):
            self._work("finish_tail", "finish_tail", None,
                       lambda: self._finish_learning())
            self.finished = True
            self.checkpoint("normal_exit")
            return False
        index = self.next_index
        record = self.stream_records[index]
        sample_id = record["sample_id"]
        key = f"observation:{index:04d}"
        image = self._image(sample_id, key + ":read")
        if self.mode == "RP":
            arrival = {"heat_peaks": None, "scoring_skipped": "frozen_random_position_mask"}
            accepted, gate_reason, gate_score = self.rp_mask[index], "random_budget_mask", None
        else:
            arrival = self._score(self.arrival_scorer, image, key + ":arrival", "arrival_scoring")
            gate_result = self.gate(arrival)
            if type(gate_result) is bool:
                accepted, gate_reason, gate_score = gate_result, "injected_gate", None
            else:
                accepted, gate_reason, gate_score = gate_result
                if type(accepted) is not bool:
                    raise ValueError("Gate decision must be an explicit bool")
        self.next_index += 1
        self.counters["observed"] += 1
        self.learner.adapter.observed += 1
        self.counters["accepted"] += int(accepted)
        arrival_record = dict(sample_id=sample_id, observation_index=index,
                              image_sha256=record["image_sha256"], preprocess_sha256=record["preprocess_sha256"],
                              g50_accepted=accepted, arrival=arrival)
        if self.mode != "RP":
            self.pools.observe(arrival_record)
        self.scientific_trace.append(_science(dict(id=key, kind="observation", sample_id=sample_id,
                                      observation_index=index, accepted=accepted, gate_reason=gate_reason,
                                      gate_score=gate_score, arrival=arrival)))
        if accepted:
            if not self.window_ids:
                self.learner.begin_window()
            self.window_ids.append(sample_id)
            if len(self.window_ids) < 4:
                self._work("gradient_accumulation", key + ":accumulate", "adaptation",
                           lambda: self.learner.accumulate(image))
            else:
                self._prepare()
                if stop_after_prepare:
                    return True
                self._resolve_prepared()
        self._after_observation()
        return True

    def _finish_learning(self):
        self.finish_counts = self.learner.finish_tail(legacy_reference=self.mode == "disabled")
        self.counters["tail_discarded"] = self.finish_counts["tail"]
        self.window_ids = []
        return self.finish_counts

    def run(self):
        while self.step():
            pass
        return dict(counters=dict(self.counters), finish_counts=copy.deepcopy(self.finish_counts),
                    scientific_trace=copy.deepcopy(self.scientific_trace), pools=self.pools.diagnostics(),
                    physical_work=self.store.work_summary())
