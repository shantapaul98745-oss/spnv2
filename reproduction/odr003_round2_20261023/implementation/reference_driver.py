"""Independent original-G50 engineering reference loop.

Learning uses OriginalAAdapter.accept/finish directly. WindowLearner is only a
complete-state serializer/restorer; none of its transactional update methods is
called. Production scope is the frozen 128-observation engineering prefix or
the explicitly selected 1024-observation deterministic A/G50 baseline run.
"""
from __future__ import annotations

import copy
import time

from durable_state import DurableRunStore
from protected_engine import _science
from protected_learning import WindowLearner
from state_adapter import OriginalAAdapter, equal_tree, rng_state


SCHEMA = "odr003-round2-original-g50-reference-v1"


def _cursors(state):
    learning = state["learning"]["counters"]
    return dict(observed=learning["observed"], candidates=learning["updates"],
                commits=learning["updates"], rollbacks=0, next_index=state["next_index"])


def validate_reference_state(state):
    required = {"schema", "identity", "stream_records", "learning", "provider_state",
                "next_index", "observation_trace", "finished", "finish_counts", "ledger_boundary"}
    if not isinstance(state, dict) or not required <= state.keys() or state["schema"] != SCHEMA:
        raise ValueError("Incomplete original-G50 reference state")
    next_index = state["next_index"]
    size = len(state["stream_records"])
    if (type(next_index) is not int or not 0 <= next_index <= size
            or not (1 <= size <= 128 or size == 1024)):
        raise ValueError("Reference prefix/cursor exceeds fixed engineering/full-run scope")
    learning = state["learning"]
    needed = {"model", "buffers", "optimizer", "optimizer_type", "optimizer_parameter_names",
              "modes", "requires_grad", "gradients", "bn_features", "bn_mean", "bn_mean_sq",
              "boundary_buffers", "rng", "scaler", "counters"}
    if not isinstance(learning, dict) or not needed <= learning.keys():
        raise ValueError("Incomplete reference learning checkpoint")
    counts = learning["counters"]
    if any(type(counts.get(key)) is not int or counts[key] < 0 for key in ("observed", "accepted", "updates", "tail")):
        raise ValueError("Invalid reference counter")
    if (counts["observed"] != next_index or counts["accepted"] > next_index
            or counts["updates"] != counts["accepted"] // 4):
        raise ValueError("Original-A update counters disagree with accepted images")
    trace = state["observation_trace"]
    if len(trace) != next_index:
        raise ValueError("Observation trace length differs from complete cursor")
    for index, event in enumerate(trace):
        if (event["id"] != f"observation:{index:04d}" or event["kind"] != "observation"
                or event["observation_index"] != index or type(event["accepted"]) is not bool
                or event["sample_id"] != state["stream_records"][index]["sample_id"]):
            raise ValueError("Reference trace diverges from its frozen prefix")
    if sum(event["accepted"] for event in trace) != counts["accepted"]:
        raise ValueError("Reference acceptance trace/counter mismatch")
    if type(state["finished"]) is not bool:
        raise ValueError("Reference finalization state must be explicit")
    if state["finished"]:
        if next_index != len(state["stream_records"]) or state["finish_counts"] is None:
            raise ValueError("Final reference state has not completed its prefix")
        if counts["tail"] != counts["accepted"] % 4:
            raise ValueError("Final reference tail mismatch")
    elif state["finish_counts"] is not None:
        raise ValueError("Unfinished reference has final counts")


class ReferenceDriver:
    def __init__(self, *, stream_records, adapter: OriginalAAdapter,
                 store: DurableRunStore, image_provider, scorer, gate,
                 sync=lambda: None, verify_purity=True, allow_stub_prefix=False,
                 complete_run=False):
        if not isinstance(adapter, OriginalAAdapter) or adapter.batch != 4:
            raise ValueError("Reference requires the unchanged batch-four OriginalAAdapter")
        if store.state_validator is not validate_reference_state:
            raise ValueError("Store must use validate_reference_state")
        records = _science(stream_records)
        if type(complete_run) is not bool or type(allow_stub_prefix) is not bool:
            raise ValueError("Reference scope flags must be explicit booleans")
        expected = 1024 if complete_run else 128
        if (complete_run and allow_stub_prefix) or (not allow_stub_prefix and len(records) != expected):
            raise ValueError("Production reference must use its exact frozen 128/1024 scope")
        if allow_stub_prefix and not 1 <= len(records) <= 128:
            raise ValueError("Synthetic engineering prefixes cannot expand the production budget")
        ids = [record["sample_id"] for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError("Reference images must be distinct")
        if any(not {"sample_id", "image_sha256", "preprocess_sha256"} <= record.keys() for record in records):
            raise ValueError("Bound image/preprocessing identities required")
        self.stream_records, self.adapter, self.store = records, adapter, store
        self.image_provider, self.scorer, self.gate = image_provider, scorer, gate
        self.sync, self.verify_purity = sync, bool(verify_purity)
        self._capture_tool = WindowLearner(adapter)
        self.next_index, self.observation_trace = 0, []
        self.finished, self.finish_counts = False, None
        self._ready, self._previous_work = False, {}

    def _work(self, kind, key, category, function):
        forwards = {} if category is None else {category: 1}
        attempt = self.store.begin_work(kind, forwards, logical_key=key,
                                       replay_of=self._previous_work.get(key))
        self._previous_work[key] = attempt
        self.sync()
        started = time.perf_counter()
        result = function()
        self.sync()
        self.store.finish_work(attempt, completed_forwards=forwards,
                               compute_seconds=time.perf_counter() - started)
        return result

    def _provider_state(self):
        if not hasattr(self.image_provider, "state_dict"):
            return None
        if not hasattr(self.image_provider, "load_state_dict"):
            raise ValueError("Stateful image provider lacks recovery interface")
        value = self.image_provider.state_dict()
        if value is None:
            raise ValueError("Stateful provider must expose an explicit state")
        return copy.deepcopy(value)

    def snapshot(self):
        state = dict(schema=SCHEMA, identity=dict(self.store.identity),
                     stream_records=copy.deepcopy(self.stream_records),
                     learning=self._capture_tool._capture_learning(),
                     provider_state=self._provider_state(), next_index=self.next_index,
                     observation_trace=copy.deepcopy(self.observation_trace),
                     finished=self.finished, finish_counts=copy.deepcopy(self.finish_counts),
                     ledger_boundary=self.store.journal_boundary())
        validate_reference_state(state)
        return state

    def checkpoint(self, reason="explicit"):
        self.sync()
        started = time.perf_counter()
        state = self.snapshot()
        info = self.store.save(state, _cursors(state))
        elapsed = time.perf_counter() - started
        self.store.append_event("checkpoint_published", dict(generation=info.generation, reason=reason,
                                checkpoint_seconds=elapsed, scientific_cursors=info.scientific_cursors,
                                timed_scope="state_capture_and_complete_generation_save_with_fsync"))
        return info

    def resume(self):
        if self._ready:
            raise RuntimeError("Instantiate a fresh reference driver for recovery")
        recovered = self.store.recover()
        journal, _ = self.store.read_journal()
        for event in journal:
            if event["kind"] == "work_started":
                payload = event["payload"]
                self._previous_work[payload["logical_key"]] = payload["attempt_id"]
        if recovered is None:
            self._ready = True
            self.checkpoint("initial")
            return None
        state = recovered.state
        if state["stream_records"] != self.stream_records or recovered.scientific_cursors != _cursors(state):
            raise ValueError("Reference checkpoint prefix/scientific cursor mismatch")
        if (state["provider_state"] is not None) != hasattr(self.image_provider, "state_dict"):
            raise ValueError("Provider state contract changed across recovery")
        if state["provider_state"] is not None:
            self.image_provider.load_state_dict(copy.deepcopy(state["provider_state"]))
        self._capture_tool._restore_learning(state["learning"])
        self.next_index = state["next_index"]
        self.observation_trace = copy.deepcopy(state["observation_trace"])
        self.finished, self.finish_counts = state["finished"], copy.deepcopy(state["finish_counts"])
        self._ready = True
        return recovered.info

    def step(self):
        if not self._ready:
            self.resume()
        if self.finished:
            return False
        if self.next_index == len(self.stream_records):
            # This is the original adapter's tail behavior, including its cache
            # clearing and its deliberately unmodified compute-RNG semantics.
            self.finish_counts = self._work("original_finish", "original_finish", None,
                                            self.adapter.finish)
            self.finished = True
            self.checkpoint("normal_exit")
            return False
        index, record = self.next_index, self.stream_records[self.next_index]
        sample_id, key = record["sample_id"], f"observation:{index:04d}"
        before_data = copy.deepcopy(rng_state())
        image = self._work("image_read", key + ":read", None, lambda: self.image_provider(sample_id))
        if not equal_tree(before_data, rng_state()):
            raise RuntimeError("Reference provider changed global compute RNG")
        before_score = self._capture_tool._capture_learning() if self.verify_purity else None
        arrival = self._work("arrival_scoring", key + ":arrival", "arrival_scoring", lambda: self.scorer(image))
        if before_score is not None and not equal_tree(before_score, self._capture_tool._capture_learning()):
            raise RuntimeError("Reference scorer changed learning state or RNG")
        arrival = _science(arrival)
        decision = self.gate(arrival)
        if type(decision) is bool:
            accepted, reason, score = decision, "injected_gate", None
        else:
            accepted, reason, score = decision
            if type(accepted) is not bool:
                raise ValueError("Gate decision must be a bool")
        self.adapter.observed += 1
        if accepted:
            # Independent reference update path: no transactional learner call.
            self._work("original_accept", key + ":accept", "adaptation", lambda: self.adapter.accept(image))
        self.observation_trace.append(_science(dict(id=key, kind="observation", sample_id=sample_id,
                                      observation_index=index, accepted=accepted, gate_reason=reason,
                                      gate_score=score, arrival=arrival)))
        self.next_index += 1
        if self.next_index % 64 == 0:
            self.checkpoint("regular_observation_boundary")
        return True

    def run(self):
        while self.step():
            pass
        return dict(learning=self._capture_tool._capture_learning(),
                    observation_trace=copy.deepcopy(self.observation_trace),
                    finish_counts=copy.deepcopy(self.finish_counts), physical_work=self.store.work_summary())
