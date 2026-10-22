"""Gated adapter that preserves the reviewed original A's update math verbatim.

In particular the original hook names are relative to model.backbone while the
running-stat loop uses model.named_modules(). We deliberately do not repair that
historical name mismatch here: doing so would mix the B2 change into geometry G.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [clone_tree(x) for x in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(x) for x in value)
    return copy.deepcopy(value)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a == b


@contextmanager
def isolated_scoring(model, bn_features):
    """Transactional scoring: no optimizer access, gradients disabled, RNG restored.

    Existing feature tensors are detached read-only references. Hooks replace the
    dictionary entries, so restoring the dictionary cannot mutate earlier tensors.
    Model buffers are restored even if an unexpected forward writes one.
    """
    rng = rng_state()
    modes = {n:m.training for n,m in model.named_modules()}
    buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    features = dict(bn_features)
    try:
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for n,b in model.named_buffers():
                b.copy_(buffers[n])
        for n,m in model.named_modules():
            m.training = modes[n]
        bn_features.clear(); bn_features.update(features)
        restore_rng(rng)


class OriginalAAdapter:
    def __init__(self, model, optimizer, device, batch=4):
        self.model, self.optimizer, self.device, self.batch = model, optimizer, device, batch
        self.bn_features, self.bn_mean, self.bn_mean_sq = {}, {}, {}
        self.handles = []
        self.accepted = self.updates = self.observed = self.tail = 0
        model.eval(); model.requires_grad_(False)
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(True); module.eval()
                def hook(m, inputs, output, key=name):
                    self.bn_features[key] = inputs[0].detach()
                self.handles.append(module.register_forward_hook(hook))
        self._boundary_buffers = {n:b.detach().clone() for n,b in model.named_buffers()}

    @staticmethod
    def configure(model, device, batch=4):
        model.eval(); model.requires_grad_(False)
        for m in model.backbone.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True); m.eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=1e-4, betas=(.9,.999), weight_decay=0., eps=1e-8)
        return OriginalAAdapter(model, optimizer, device, batch)

    def accept(self, image):
        loss, items = self.model(image, is_train=True, gpu=self.device)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError('Nonfinite adaptation loss')
        (loss / self.batch).backward()
        self.accepted += 1
        for name, feature in self.bn_features.items():
            mean = torch.mean(feature, (0,2,3)) / self.batch
            mean_sq = torch.mean(feature.square(), (0,2,3)) / self.batch
            if self.batch == 1 or self.accepted % self.batch == 1:
                self.bn_mean[name], self.bn_mean_sq[name] = mean, mean_sq
            else:
                self.bn_mean[name] = self.bn_mean[name] + mean
                self.bn_mean_sq[name] = self.bn_mean_sq[name] + mean_sq
        if self.accepted % self.batch == 0:
            for p in self.model.parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise FloatingPointError('Nonfinite adaptation gradient')
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Exact original A names and equations, deliberately not B2.
            for name,module in self.model.named_modules():
                if name in self.bn_mean:
                    mean = self.bn_mean[name]
                    var = self.bn_mean_sq[name] - mean.square()
                    mean = (1-module.momentum)*module.running_mean + module.momentum*mean
                    var = (1-module.momentum)*module.running_var + module.momentum*var
                    with torch.no_grad():
                        module.running_mean.copy_(mean); module.running_var.copy_(var)
            self.bn_mean, self.bn_mean_sq = {}, {}
            self.updates += 1
            self._boundary_buffers = {n:b.detach().clone() for n,b in self.model.named_buffers()}
        return float(items['ent'])

    def finish(self):
        self.tail = self.accepted % self.batch
        if self.tail:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for n,b in self.model.named_buffers():
                    b.copy_(self._boundary_buffers[n])
            self.bn_mean, self.bn_mean_sq = {}, {}
            self.bn_features.clear()
        return dict(observed=self.observed, accepted=self.accepted, rejected=self.observed-self.accepted,
                    updates=self.updates, used=self.updates*self.batch, tail=self.tail)

    def checkpoint(self):
        if self.accepted % self.batch:
            raise ValueError('Only checkpoint complete update windows')
        return dict(model=clone_tree(self.model.state_dict()), optimizer=clone_tree(self.optimizer.state_dict()),
                    rng=clone_tree(rng_state()), accepted=self.accepted, updates=self.updates, observed=self.observed,
                    modes={n:m.training for n,m in self.model.named_modules()}, batch=self.batch)

    def restore(self, checkpoint):
        if checkpoint['batch'] != self.batch or checkpoint['accepted'] % self.batch:
            raise ValueError('Incompatible checkpoint boundary')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.zero_grad(set_to_none=True)
        self.accepted, self.updates, self.observed = [checkpoint[k] for k in ('accepted','updates','observed')]
        for n,m in self.model.named_modules():
            m.training = checkpoint['modes'][n]
        self.bn_features.clear(); self.bn_mean.clear(); self.bn_mean_sq.clear()
        self._boundary_buffers = {n:b.detach().clone() for n,b in self.model.named_buffers()}
        restore_rng(checkpoint['rng'])

    def close(self):
        for handle in self.handles:
            handle.remove()
