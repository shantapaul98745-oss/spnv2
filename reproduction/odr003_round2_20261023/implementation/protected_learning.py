"""Whole-window transactions around the unchanged original-A adapter.

No data cursor, protection pool, random-window decision RNG, or durable-log
ownership lives here. The outer runner checkpoints those together with this
module. A normal rollback restores learning state but preserves every already
observed/accepted input count. A full ``load`` restores both learning and counts.

Snapshots contain independent CPU tensors and support partially accumulated
windows. They are not durable until the outer checkpoint transaction commits.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np
import torch

from state_adapter import OriginalAAdapter, rng_state, restore_rng


SCHEMA = "odr003-round2-window-learner-v1"


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, OrderedDict):
        result = OrderedDict((key, _cpu_tree(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _on_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: _on_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_on_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_on_device(item, device) for item in value)
    return copy.deepcopy(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative counter {name}")
    return value


class WindowLearner:
    """Keep original loss/AdamW/BN naming and math; add explicit transactions.

    API:
      begin_window(); accumulate(image) x3; candidate(image4);
      commit() or rollback(); finish_tail() at end of the fixed stream.

    ``adapter.observed`` remains the outer runner's responsibility. Full state
    snapshots can be taken at any phase, but the root must not label a pending
    candidate as scientifically committed. Normal rollback is never a retry.
    """
    def __init__(self, adapter: OriginalAAdapter):
        if not isinstance(adapter, OriginalAAdapter) or adapter.batch != 4:
            raise ValueError("A frozen OriginalAAdapter with batch four is required")
        self.adapter = adapter
        self.phase = "idle"
        self.window_size = 0
        self._base: dict | None = None
        self.finish_mode: str | None = None

    def _optimizer_names(self) -> list[list[str]]:
        names = {id(parameter): name for name, parameter in self.adapter.model.named_parameters()}
        try:
            return [[names[id(parameter)] for parameter in group["params"]]
                    for group in self.adapter.optimizer.param_groups]
        except KeyError as error:
            raise ValueError("Optimizer owns a parameter outside the bound model") from error

    def _capture_learning(self) -> dict:
        a = self.adapter
        return dict(model=_cpu_tree(a.model.state_dict()),
                    # state_dict omits non-persistent buffers, so include every
                    # named buffer explicitly as part of the rollback contract.
                    buffers={name: _cpu_tree(buffer) for name, buffer in a.model.named_buffers()},
                    optimizer=_cpu_tree(a.optimizer.state_dict()),
                    optimizer_type=f"{type(a.optimizer).__module__}.{type(a.optimizer).__qualname__}",
                    optimizer_parameter_names=self._optimizer_names(),
                    modes={name: module.training for name, module in a.model.named_modules()},
                    requires_grad={name: parameter.requires_grad for name, parameter in a.model.named_parameters()},
                    gradients={name: None if parameter.grad is None else _cpu_tree(parameter.grad)
                               for name, parameter in a.model.named_parameters()},
                    bn_features=_cpu_tree(a.bn_features), bn_mean=_cpu_tree(a.bn_mean),
                    bn_mean_sq=_cpu_tree(a.bn_mean_sq), boundary_buffers=_cpu_tree(a._boundary_buffers),
                    rng=_cpu_tree(rng_state()), scaler=None,
                    counters={name: int(getattr(a, name)) for name in ("accepted", "observed", "updates", "tail")})

    def _validate_learning(self, state: Mapping) -> None:
        a = self.adapter
        if state.get("scaler") is not None:
            raise ValueError("The frozen FP32 protocol has no scaler")
        names = dict(a.model.named_parameters())
        buffers = dict(a.model.named_buffers())
        if set(state["gradients"]) != set(names) or set(state["requires_grad"]) != set(names):
            raise ValueError("Checkpoint parameter identities do not match")
        if set(state["buffers"]) != set(buffers) or set(state["boundary_buffers"]) != set(buffers):
            raise ValueError("Checkpoint buffer identities do not match")
        if set(state["modes"]) != {name for name, _ in a.model.named_modules()}:
            raise ValueError("Checkpoint module identities do not match")
        expected_type = f"{type(a.optimizer).__module__}.{type(a.optimizer).__qualname__}"
        if state["optimizer_type"] != expected_type or state["optimizer_parameter_names"] != self._optimizer_names():
            raise ValueError("Checkpoint optimizer type/group order does not match")
        current_model = a.model.state_dict()
        if set(state["model"]) != set(current_model):
            raise ValueError("Checkpoint state_dict identities do not match")
        for label, stored, current in (("model", state["model"], current_model),
                                      ("buffers", state["buffers"], buffers),
                                      ("boundary_buffers", state["boundary_buffers"], buffers)):
            for name, actual in current.items():
                saved = stored[name]
                if not isinstance(saved, torch.Tensor) or saved.shape != actual.shape or saved.dtype != actual.dtype:
                    raise ValueError(f"Checkpoint {label} shape/dtype mismatch: {name}")
        for name, parameter in names.items():
            saved = state["gradients"][name]
            if saved is not None and (not isinstance(saved, torch.Tensor)
                                      or saved.shape != parameter.shape or saved.dtype != parameter.dtype):
                raise ValueError(f"Checkpoint gradient mismatch: {name}")
            if type(state["requires_grad"][name]) is not bool:
                raise ValueError("requires_grad must remain a bool")
        if any(type(mode) is not bool for mode in state["modes"].values()):
            raise ValueError("Module training mode must remain a bool")
        for name in ("accepted", "observed", "updates", "tail"):
            _nonnegative_int(state["counters"][name], name)
        if state["counters"]["tail"] >= a.batch:
            raise ValueError("Invalid incomplete-tail counter")

    def _restore_learning(self, state: Mapping) -> None:
        self._validate_learning(state)
        a = self.adapter
        a.model.load_state_dict(state["model"], strict=True)
        with torch.no_grad():
            for name, buffer in a.model.named_buffers():
                buffer.copy_(state["buffers"][name].to(device=buffer.device))
        # PyTorch's official loader follows each group's capturable/fused policy:
        # normal AdamW 'step' remains CPU, parameter moments go to the parameter
        # device. Do not blanket-move the optimizer state to CUDA.
        a.optimizer.load_state_dict(_cpu_tree(state["optimizer"]))
        for name, parameter in a.model.named_parameters():
            parameter.requires_grad_(state["requires_grad"][name])
            gradient = state["gradients"][name]
            parameter.grad = None if gradient is None else gradient.detach().to(device=parameter.device, copy=True)
        for name, module in a.model.named_modules():
            # Calling module.train() recursively would overwrite child modes.
            module.training = state["modes"][name]
        device = torch.device(a.device)
        for attr in ("bn_features", "bn_mean", "bn_mean_sq"):
            cache = getattr(a, attr)
            cache.clear()
            cache.update(_on_device(state[attr], device))
        buffer_devices = {name: buffer.device for name, buffer in a.model.named_buffers()}
        a._boundary_buffers = {name: value.detach().to(device=buffer_devices[name], copy=True)
                               for name, value in state["boundary_buffers"].items()}
        for name, count in state["counters"].items():
            setattr(a, name, count)
        restore_rng(_cpu_tree(state["rng"]))

    def begin_window(self) -> None:
        if self.phase != "idle" or self._base is not None or self.window_size:
            raise RuntimeError("A learning window is already active or the run is finished")
        if self.adapter.accepted % 4:
            raise RuntimeError("New windows must begin at a complete accepted-input boundary")
        if any(parameter.grad is not None for parameter in self.adapter.model.parameters()):
            raise RuntimeError("Unexpected pending gradient at a complete-window boundary")
        self._base = self._capture_learning()
        self.phase = "collecting"

    def _check_active_counters(self) -> None:
        if self._base is None:
            raise RuntimeError("Missing learning-window base")
        if self.adapter.accepted != self._base["counters"]["accepted"] + self.window_size:
            raise RuntimeError("Accepted input counter changed outside WindowLearner")
        expected_updates = self._base["counters"]["updates"] + int(self.phase == "candidate")
        if self.adapter.updates != expected_updates:
            raise RuntimeError("Optimizer update counter changed outside WindowLearner")
        if self.adapter.observed < self._base["counters"]["observed"]:
            raise RuntimeError("Observation cursor cannot move backwards within a window")

    def accumulate(self, image: torch.Tensor) -> float:
        if self.phase != "collecting" or not 0 <= self.window_size < 3:
            raise RuntimeError("accumulate is only the first three accepted images")
        self._check_active_counters()
        loss = self.adapter.accept(image)
        self.window_size += 1
        self._check_active_counters()
        return loss

    def candidate(self, image: torch.Tensor) -> float:
        if self.phase != "collecting" or self.window_size != 3:
            raise RuntimeError("candidate requires three accumulated images")
        self._check_active_counters()
        loss = self.adapter.accept(image)
        self.window_size = 4
        self.phase = "candidate"
        self._check_active_counters()
        return loss

    def commit(self) -> dict:
        if self.phase != "candidate" or self.window_size != 4:
            raise RuntimeError("Only a complete temporary candidate can commit")
        self._check_active_counters()
        result = dict(decision="commit", accepted=self.adapter.accepted,
                      observed=self.adapter.observed, updates=self.adapter.updates)
        self._base = None
        self.window_size = 0
        self.phase = "idle"
        return result

    def rollback(self) -> dict:
        if self.phase != "candidate" or self.window_size != 4:
            raise RuntimeError("Only a complete temporary candidate can roll back")
        self._check_active_counters()
        accepted, observed = self.adapter.accepted, self.adapter.observed
        self._restore_learning(self._base)
        # These images were consumed even though their gradient step was vetoed.
        # Preserving accepted modulo four makes the next window's original-A
        # accumulation/start logic correct without replaying any input.
        self.adapter.accepted, self.adapter.observed = accepted, observed
        result = dict(decision="rollback", accepted=accepted, observed=observed,
                      updates=self.adapter.updates)
        self._base = None
        self.window_size = 0
        self.phase = "idle"
        return result

    def finish_tail(self, *, legacy_reference: bool = False) -> dict:
        """Finalize tail; legacy mode is ONLY the original/reference comparison.

        P/RW restore the complete partial-window base, including learning RNG
        and caches. The protection-disabled engineering control deliberately
        calls the untouched adapter.finish(), which clears tail caches and does
        not rewind RNG. Both paths preserve the model/optimizer update math.
        """
        if type(legacy_reference) is not bool:
            raise ValueError("legacy_reference must be an explicit bool")
        if self.phase in ("candidate", "finished"):
            raise RuntimeError("A pending candidate must be decided before finalizing")
        accepted, observed = self.adapter.accepted, self.adapter.observed
        tail = self.window_size
        if accepted % 4 != tail:
            raise RuntimeError("Partial accepted count and tail do not agree")
        if self.phase == "collecting":
            self._check_active_counters()
            if not legacy_reference:
                self._restore_learning(self._base)
                self.adapter.accepted, self.adapter.observed = accepted, observed
        elif self.phase != "idle":
            raise RuntimeError("Unknown window phase")
        if legacy_reference:
            result = self.adapter.finish()
            self.finish_mode = "original_adapter_finish"
        else:
            self.adapter.tail = tail
            result = dict(observed=observed, accepted=accepted, rejected=observed - accepted,
                          updates=self.adapter.updates, used=self.adapter.updates * 4, tail=tail)
            self.finish_mode = "rollback_partial_window"
        self._base = None
        self.window_size = 0
        self.phase = "finished"
        return result

    def snapshot(self) -> dict:
        if self.phase in ("collecting", "candidate"):
            self._check_active_counters()
        return dict(schema=SCHEMA, batch=4, phase=self.phase, window_size=self.window_size,
                    finish_mode=self.finish_mode,
                    learning=self._capture_learning(), window_base=_cpu_tree(self._base))

    def load(self, snapshot: Mapping) -> None:
        value = _cpu_tree(snapshot)
        if value.get("schema") != SCHEMA or value.get("batch") != 4:
            raise ValueError("Window-learner checkpoint schema/protocol mismatch")
        phase, size = value["phase"], value["window_size"]
        finish_mode = value["finish_mode"]
        _nonnegative_int(size, "window_size")
        base, learning = value["window_base"], value["learning"]
        if phase not in ("idle", "collecting", "candidate", "finished"):
            raise ValueError("Unknown saved transaction phase")
        if ((phase == "finished" and finish_mode not in ("original_adapter_finish", "rollback_partial_window"))
                or (phase != "finished" and finish_mode is not None)):
            raise ValueError("Checkpoint finalization mode/phase disagree")
        self._validate_learning(learning)
        if phase in ("idle", "finished"):
            if size or base is not None:
                raise ValueError("Inactive transaction has a partial window/base")
            if phase == "idle" and learning["counters"]["accepted"] % 4:
                raise ValueError("Idle transaction is not at an accepted-input boundary")
            if phase == "finished" and learning["counters"]["accepted"] % 4 != learning["counters"]["tail"]:
                raise ValueError("Finished transaction has inconsistent tail")
        else:
            if not isinstance(base, Mapping):
                raise ValueError("Active transaction lacks a full rollback base")
            self._validate_learning(base)
            if (phase == "collecting" and size not in (0, 1, 2, 3)) or (phase == "candidate" and size != 4):
                raise ValueError("Transaction phase/window size disagree")
            if base["counters"]["accepted"] % 4:
                raise ValueError("Rollback base is not a full accepted-input boundary")
            if learning["counters"]["accepted"] != base["counters"]["accepted"] + size:
                raise ValueError("Checkpoint partial gradients/accepted counters disagree")
            if learning["counters"]["updates"] != base["counters"]["updates"] + int(phase == "candidate"):
                raise ValueError("Checkpoint candidate/update counters disagree")
            if learning["counters"]["observed"] < base["counters"]["observed"]:
                raise ValueError("Checkpoint observation cursor moved backwards")
        self._restore_learning(learning)
        self._base = base
        self.phase, self.window_size = phase, size
        self.finish_mode = finish_mode

    state_dict = snapshot
    load_state_dict = load
