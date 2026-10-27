"""Isolated, budgeted paired-window diagnosis of the frozen G50-P producer.

This module does not edit, signal, or resume a producer run. It copies one
already complete checkpoint into its own durable store, replays the recorded
trajectory, measures both outcomes of one update, and proves the actual branch
rejoins an original complete checkpoint. All heavy imports follow admission.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from collections import Counter
import numeric_compare as numeric


SCHEMA = "g50p-risk-diagnostic-task-v1"
OUTPUT_SCHEMA = "g50p-risk-diagnostic-pair-report-v1"
PRODUCER_NAME = "odr003_round2_20261023"


class ResourceYield(RuntimeError):
    pass


def require(condition, message):
    if not condition: raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_json(path):
    def reject(value): raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf8"), parse_constant=reject)


def binding(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "invalid immutable binding")
    path = Path(value["path"]).resolve(strict=True)
    require(path.is_file() and sha256(path) == value["sha256"], "bound file changed: " + str(path))
    return path


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "refusing to replace diagnostic state evidence")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush(); os.fsync(handle.fileno())
    return dict(path=str(path), sha256=sha256(path))


def tree_digest(value):
    """Stable typed state digest, including tensor metadata and exact RNG bytes."""
    h = hashlib.sha256()
    def visit(item):
        if type(item).__module__.startswith("torch") and hasattr(item, "detach"):
            tensor = item.detach().cpu().contiguous()
            h.update(b"tensor\0" + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode() + b"\0")
            h.update(tensor.numpy().tobytes())
        elif type(item).__module__.startswith("numpy") and hasattr(item, "dtype"):
            import numpy as np
            array = np.asarray(item)
            h.update(b"numpy\0" + str(array.dtype).encode() + repr(array.shape).encode() + b"\0" + array.tobytes())
        elif isinstance(item, dict):
            h.update(b"dict\0")
            if hasattr(item, "_metadata"): visit(item._metadata)
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))): visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update((type(item).__name__ + ":" + str(len(item))).encode())
            for part in item: visit(part)
        else:
            require(not isinstance(item, float) or math.isfinite(item), "nonfinite scientific state")
            h.update((type(item).__name__ + ":" + repr(item) + "\0").encode())
    visit(value)
    return h.hexdigest()


def science_state(state):
    """Only the producer's physical journal boundary is excluded from matching."""
    return {key: value for key, value in state.items() if key != "ledger_boundary"}


def read_generation(spec):
    """Read one explicitly bound complete generation without touching CURRENT."""
    directory = Path(spec["directory"]).resolve(strict=True)
    metadata_path, state_path = directory/"metadata.json", directory/"state.bin"
    require(sha256(metadata_path) == spec["metadata_sha256"]
            and sha256(state_path) == spec["state_sha256"], "original generation binding changed")
    require((directory/"metadata.sha256").read_text(encoding="ascii").strip() == spec["metadata_sha256"], "original metadata checksum differs")
    metadata = load_json(metadata_path)
    require(metadata["format_version"] == 1 and metadata["state_sha256"] == spec["state_sha256"]
            and metadata["state_bytes"] == state_path.stat().st_size, "original generation is incomplete")
    with state_path.open("rb") as handle: state = pickle.load(handle)
    require(state["identity"] == metadata["identity"], "original complete checkpoint identity differs")
    return state, metadata


def reference_trace(task, parent_identity, *, budget):
    selection = task["selection"]
    budget.check_time("read zero-forward scientific-only trace extraction")
    # The A proof is produced before B. Its path is frozen in the dispatch, but
    # its output checksum cannot be known before that actual extraction runs.
    # Never fall back to loading old final states, which contain developer
    # evaluation fields unrelated to replaying the scientific trajectory.
    proof_path = Path(task["reference_trace_proof_path"]).resolve(strict=True)
    expected = Path(task["output_dir"]).resolve().parent.parent/"inputs/extraction_proof.json"
    require(proof_path == expected, "trace extraction proof escapes this diagnostic's fixed inputs")
    proof = load_json(proof_path)
    key = f"P:{selection['domain']}:{selection['stream_seed']}"
    require(proof.get("schema") == "g50p-risk-diagnostic-A-extraction-proof-v1"
            and proof.get("status") == "passed" and proof.get("model_forwards") == 0
            and proof.get("frozen_inputs") == task["index_binding"]
            and proof.get("parent_final_state_sha256", {}).get(key) == selection["original_final_checkpoint"]["state_sha256"],
            "recorded trace lacks the actual matching zero-forward extraction proof")
    target = proof_path.parent/(parent_identity["run_id"]+".json")
    outputs = [value for value in proof["outputs"] if Path(value["path"]).resolve() == target]
    require(len(outputs) == 1, "trace extraction must contain one uniquely bound original run output")
    document = load_json(binding(outputs[0]))
    allowed = {"identity", "run_key", "run_id", "source_sha256", "protocol_sha256", "state_sha256",
               "scientific_trace", "pools", "stream_records", "counters"}
    require(set(document) == allowed, "scientific trace extraction contains missing or out-of-scope/GT-derived fields")
    require(document["identity"] == parent_identity and document["run_id"] == parent_identity["run_id"]
            and document["run_key"] == key
            and document["state_sha256"] == selection["original_final_checkpoint"]["state_sha256"],
            "extracted reference trace belongs to another original complete state")
    return document["scientific_trace"]


def derive_branch_snapshots(candidate):
    """Expected commit vs whole-window rollback, never the three-gradient point."""
    require(candidate["phase"] == "candidate" and candidate["window_size"] == 4
            and candidate["window_base"] is not None, "branching requires the temporary full update")
    accepted = copy.deepcopy(candidate)
    accepted.update(phase="idle", window_size=0, window_base=None, finish_mode=None)
    rejected = copy.deepcopy(accepted)
    rejected["learning"] = copy.deepcopy(candidate["window_base"])
    for counter in ("accepted", "observed"):
        rejected["learning"]["counters"][counter] = candidate["learning"]["counters"][counter]
    require(accepted["learning"]["counters"]["updates"] == rejected["learning"]["counters"]["updates"] + 1,
            "candidate is not exactly one update after its whole-window base")
    return {"accept": accepted, "rollback": rejected}


def capture_candidate(engine):
    """Execute the original resolver once, recording its exact pre-decision state."""
    require(engine.transaction is not None and engine.transaction["phase"] == "prepared", "target window not prepared")
    captured = []
    commit, rollback = engine.learner.commit, engine.learner.rollback
    def wrapper(function):
        def call():
            captured.append(engine.learner.snapshot())
            return function()
        return call
    engine.learner.commit, engine.learner.rollback = wrapper(commit), wrapper(rollback)
    try:
        engine._resolve_prepared()
        engine._after_observation()
    finally:
        engine.learner.commit, engine.learner.rollback = commit, rollback
    require(len(captured) == 1, "original candidate resolver did not make one complete decision")
    return captured[0]


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


class TraceMismatch(ValueError):
    def __init__(self, actual, expected):
        self.trace_difference = dict(actual=actual, expected=expected,
                                     first_differences=differences(actual, expected), numeric_policy_failure=numeric.LAST_FAILURE)
        super().__init__('replay prediction/gate/protection/decision trace diverged from frozen producer')


def trace_prefix_matches(actual, original, equal):
    equal = numeric.equal
    expected = original[:len(actual)]
    if len(actual) > len(original) or not equal(actual, expected):
        for index, (left, right) in enumerate(zip(actual, expected)):
            if not equal(left, right):
                raise TraceMismatch(dict(index=index,event=left), dict(index=index,event=right))
        raise TraceMismatch(dict(length=len(actual)), dict(length=len(expected)))


class ForwardBudgetHook:
    """One backbone invocation per input image covers extract's manual HE/EHS."""
    def __init__(self, backbone, budget, *, synchronize):
        require(not getattr(backbone, "_risk_diagnostic_budget_hook", False), "duplicate forward-budget hook")
        backbone._risk_diagnostic_budget_hook = True
        self.backbone, self.budget, self.synchronize = backbone, budget, synchronize
        self.phase = "initialization"
        self.pending = []
        self.counts = {}
        self.handles = (backbone.register_forward_pre_hook(self.before), backbone.register_forward_hook(self.after))

    def before(self, module, inputs):
        self.budget.check_time("before forward: " + self.phase)
        require(inputs and hasattr(inputs[0], "shape") and len(inputs[0].shape) >= 1, "cannot count backbone input images")
        count = int(inputs[0].shape[0])
        require(count > 0, "empty backbone batch")
        token = self.budget.reserve_forward(self.phase, count=count)
        self.pending.append((token, count))
        self.counts[self.phase] = self.counts.get(self.phase, 0) + count

    def after(self, module, inputs, output):
        require(bool(self.pending), "forward completion without reservation")
        token, count = self.pending.pop()
        self.synchronize()
        self.budget.finish_forward(token, count=count)
        self.budget.check_time("after forward: " + self.phase)

    def close(self):
        for handle in self.handles: handle.remove()
        self.backbone._risk_diagnostic_budget_hook = False


def _metrics_summary(rows, offline):
    require(len(rows) == len(offline) == 128
            and [r["sample_id"] for r in rows] == [r["sample_id"] for r in offline], "fixed128 panel order changed")
    complete = all(row["metrics"]["valid"] and o["valid"] for row, o in zip(rows, offline))
    errors = [row["metrics"]["error"] for row in rows]
    def mean(values): return math.fsum(value/len(values) for value in values)
    d = [max(error-o["error"], 0.0) for error, o in zip(errors, offline)] if complete else None
    return dict(panel_count=128, reference_count=128, tail_count=7,
                valid_count=sum(row["metrics"]["valid"] for row in rows), complete=complete,
                mean_error=mean(errors) if complete else None,
                mean_D=mean(d) if complete else None,
                tail_D=mean(sorted(d, reverse=True)[:7]) if complete else None,
                missing_ids=[row["sample_id"] for row in rows if not row["metrics"]["valid"]],
                missing_policy="None for every fixed-panel metric; never shrink the denominator")


def evaluate_branches(engine, candidate, panel_ids, evaluate_one, offline_rows, *, equal, budget,
                      phase_hook=None, progress=None, saved_progress=None,
                      reconstruct_panel=None, resource_check=None):
    """Evaluate frozen branches, restoring the actual complete engine every time."""
    require(len(panel_ids) == len(set(panel_ids)) == 128, "exactly128 fixed panel IDs required")
    actual = engine.snapshot()
    branches = derive_branch_snapshots(candidate)
    result = {}
    for name in ("accept", "rollback"):
        budget.check_time("load branch " + name)
        engine.learner.load(candidate)
        getattr(engine.learner, "commit" if name == "accept" else "rollback")()
        require(equal(engine.learner.snapshot(), branches[name]), "actual whole-window branch differs from saved derivation")
        before = engine.learner.snapshot()
        provider_before = copy.deepcopy(engine.image_provider.state_dict())
        if phase_hook is not None: phase_hook.phase = "panel_" + name
        before_digest = tree_digest(before)
        rows = saved_progress(name, before_digest) if saved_progress is not None else []
        require([row["sample_id"] for row in rows] == panel_ids[:len(rows)] and len(rows) <= 128,
                "saved branch prefix IDs/order differ")
        try:
            if reconstruct_panel is not None:
                for row in rows:
                    budget.check_time("reconstruct prior panel input without forward")
                    reconstruct_panel(row["sample_id"])
            for sample_id in panel_ids[len(rows):]:
                if resource_check is not None: resource_check()
                budget.check_time("panel image " + name)
                row = evaluate_one(sample_id)
                require(row["sample_id"] == sample_id and set(row) == {"sample_id", "prediction", "metrics"}, "panel evaluation record schema/order differs")
                require(equal(engine.learner.snapshot(), before)
                        and equal(engine.image_provider.state_dict(), provider_before),
                        "branch evaluation changed learning/RNG/gradient/cache/provider state")
                rows.append(row)
                if progress is not None: progress(name, rows, before_digest)
            result[name] = dict(rows=rows, summary=_metrics_summary(rows, offline_rows),
                                before_learning_sha256=tree_digest(before), after_learning_sha256=tree_digest(engine.learner.snapshot()),
                                learning_unchanged=True, producer_provider_unchanged=True)
        finally:
            engine.learner.load(actual["learner"])
            require(equal(science_state(engine.snapshot()), science_state(actual)), "actual engine not restored after counterfactual evaluation")
    changes = []
    for accept, rollback, offline in zip(result["accept"]["rows"], result["rollback"]["rows"], offline_rows):
        am, rm = accept["metrics"], rollback["metrics"]
        changes.append(dict(sample_id=accept["sample_id"], accept=am, rollback=rm, offline=offline,
                    error_change=None if not (am["valid"] and rm["valid"]) else am["error"]-rm["error"],
                    rotation_change_deg=None if am["rotation_deg"] is None or rm["rotation_deg"] is None else am["rotation_deg"]-rm["rotation_deg"],
                    translation_change_m=None if am["translation_m"] is None or rm["translation_m"] is None else am["translation_m"]-rm["translation_m"],
                    accept_new_missing=rm["valid"] and not am["valid"], rollback_new_missing=am["valid"] and not rm["valid"]))
    result["per_image_changes"] = changes
    result["paired_summary"] = {metric: (None if result["accept"]["summary"][metric] is None or result["rollback"]["summary"][metric] is None
                                      else result["accept"]["summary"][metric]-result["rollback"]["summary"][metric])
                                for metric in ("mean_error", "mean_D", "tail_D")}
    return result


def load_producer(task):
    producer = task["producer"]
    directory = Path(producer["implementation_dir"]).resolve(strict=True)
    require(directory.parent.name == PRODUCER_NAME, "producer implementation is not the frozen round-two experiment")
    require(producer.get("commit", "").startswith("3ef"), "diagnostic must use the confirmed producer revision")
    require({path.name for path in directory.glob("*.py")} == set(producer["implementation_sha256"]),
            "producer Python file coverage changed")
    for name, digest in producer["implementation_sha256"].items():
        require(Path(name).name == name and sha256(directory/name) == digest, "producer code binding changed: " + name)
    sys.path.insert(0, str(directory))
    def get(name):
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == directory/(name+".py")
                and sha256(module.__file__) == producer["implementation_sha256"][name+".py"], "producer module shadowed: " + name)
        return module
    return get


def _cursors(state):
    c = state["counters"]
    return dict(observed=c["observed"], candidates=c["candidates"], commits=c["commits"], rollbacks=c["rollbacks"], next_index=state["next_index"])


def configure_frozen_modules(backend, original, *, output_dir, run_id, phase):
    """Follow the producer's exact core-path-before-prediction import order.

    prediction imports ``utils.postprocess`` and ``utils.metrics`` at module
    scope. The producer's reviewed configure() installs its verified source-A
    core path. Importing prediction before that bootstrap fails even though the
    immutable prediction bytes are correct. No model is constructed here.
    """
    frozen = {name: backend.import_bound_module(name, spec)
              for name, spec in original["frozen_modules"].items() if name != "prediction"}
    baseline = frozen["run_exploration"]
    boundary = baseline.GTBoundary(original["baseline_task"]["bindings"]["developer_labels"]["path"])
    boundary.phase = phase
    paths, receipt, core, prepared = baseline.verify_bindings(original["baseline_task"])
    cfg, _ = baseline.configure({**original["baseline_task"], "output_dir":str(output_dir),
                                "run_id":run_id}, paths, core)
    frozen["prediction"] = backend.import_bound_module("prediction", original["frozen_modules"]["prediction"])
    return frozen, boundary, paths, receipt, core, prepared, cfg


def restore_rotation_decoder_runtime(scientific_trace, transaction, decoder, torch, device, budget):
    """Restore scripted operator call phase, without a model forward or RNG draw.

    The original decoder specializes its cold/hot graph by rank. Actual E-head
    selected rotations have shape [1,6], not [1,1,6]. A zero-observation parent
    must stay cold; a resumed parent must retain its original call phase.
    """
    calls = sum(event['kind'] == 'observation' for event in scientific_trace)
    calls += sum(2 * len(event['protection_records']) for event in scientific_trace if event['kind'] == 'window')
    if transaction is not None:
        calls += len(transaction['pre_signals'])
    primed = min(calls, 6)
    if primed:
        budget.check_time('restore original scripted rotation decoder phase')
        dummy = torch.tensor([[1., 0., 0., 0., 1., 0.]], dtype=torch.float32, device=device)
        for _ in range(primed):
            decoder(dummy)
        torch.cuda.synchronize(device)
    return dict(schema='g50p-scripted-rotation-runtime-recovery-v1', prior_calls=calls,
                synthetic_operator_calls=primed, model_forwards=0,
                input_shape=[1, 6], dtype='torch.float32', original_operator_unchanged=True)


def build_runtime(task, output, parent, *, budget, original_report):
    """Actual frozen SPNv2 wiring; no alternative decoder or learning rule."""
    budget.check_time("load producer code")
    old = load_producer(task)
    deterministic_module = old("deterministic_runtime")
    deterministic_module.prepare_environment()
    old_backend = old("frozen_backend")
    original = load_json(binding(task["selection"]["original_task"]))
    require(original["stage"] == "complete" and original["definition"]["group"] == "P", "diagnose only existing accepted P windows")
    require(parent["identity"] == dict(run_id=original["run_id"], source_sha256=original["protocol"]["source_sha256"],
                    config_sha256=task["selection"]["original_task"]["sha256"],
                    manifest_sha256=original["baseline_task"]["bindings"]["stream_manifest"]["sha256"]),
            "isolated parent is not the exact original task/source/stream identity")
    for name, digest in original["implementation_sha256"].items():
        require(task["producer"]["implementation_sha256"].get(name) == digest, "source task and producer code differ")
    budget.check_time("verify frozen producer assets")
    frozen, boundary, paths, receipt, core, prepared, cfg = configure_frozen_modules(
        old_backend, original, output_dir=output, run_id=task["run_id"], phase="adaptation")
    baseline, prediction = frozen["run_exploration"], frozen["prediction"]
    budget.check_time("configured frozen model and prediction imports")
    from nets.build import SPNv2
    from dataset.transforms.build import build_transforms
    from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints, set_seeds_cudnn
    import torch
    import numpy as np
    require(str(np.__version__) == original_report["numpy_version"], "diagnostic NumPy version differs from the frozen producer")
    require(torch.cuda.is_available(), "real paired-window diagnosis requires the admitted GPU")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    set_seeds_cudnn(cfg, seed=original["definition"]["stream_seed"])
    deterministic = deterministic_module.apply(torch)
    require(deterministic == original_report["deterministic_runtime"], "diagnostic backend/software differs from the frozen producer")
    torch.cuda.reset_peak_memory_stats(device)
    budget.check_time("construct and transfer frozen model")
    model = SPNv2(cfg).float().to(device)
    adapter = frozen["state_adapter"].OriginalAAdapter.configure(model, device, batch=4)
    budget.check_time("model build and checkpoint setup")
    camera = load_camera_intrinsics(str(paths["camera"])); points = load_tango_3d_keypoints(str(paths["keypoints"]))
    calibration = baseline.load_calibration(original["baseline_task"], paths, prepared)
    transforms = build_transforms(cfg, is_train=False, load_labels=False)
    data = frozen["data_boundary"]
    dataset = data.ImageManifestDataset(paths["stream_manifest"], transforms=transforms)
    preprocess = old_backend.canonical_sha(dict(config=original["baseline_task"]["bindings"]["config"],
                     core=original["baseline_task"]["bindings"]["source_core_manifest"], data_code=original["frozen_modules"]["data_boundary"]))
    provider = old_backend.DeterministicImageProvider(dataset, preprocess_sha256=preprocess,
                     rng_state=frozen["state_adapter"].rng_state, restore_rng=frozen["state_adapter"].restore_rng,
                     equal_tree=frozen["state_adapter"].equal_tree,
                     manifest_file_sha256=original["baseline_task"]["bindings"]["stream_manifest"]["sha256"])
    records = old_backend.stream_records(dataset, preprocess, limit=1024)
    learner = old("protected_learning").WindowLearner(adapter)
    seeds = original["definition"]["rng_seeds"]
    pools = old("guard_pools").GuardPools(seeds["uniform_pool"], seeds["vulnerable_pool"], heatmap_threshold=float(cfg.TEST.HEATMAP_THRESHOLD))
    state_module = old("protected_engine")
    store = old("durable_state").DurableRunStore(output/"isolated_durable", parent["identity"], state_validator=state_module.validate_engine_state)
    recovered = store.recover()
    if recovered is None:
        store.save(parent, _cursors(parent))
    scorers = old_backend.FrozenScorers(model=model, adapter=adapter, cfg=cfg, camera=camera, points=points,
                                      calibration=calibration, prediction_module=prediction, gate_module=frozen["gate_core"])
    sync = lambda: torch.cuda.synchronize(device)
    engine = state_module.ProtectedEngine(mode="P", stream_records=records, learner=learner, pools=pools, store=store,
                       image_provider=provider, arrival_scorer=scorers.arrival, protection_scorer=scorers.protection,
                       gate=scorers.gate, sync=sync, verify_purity=True)
    budget.check_time("restore complete isolated producer checkpoint")
    engine.resume()
    operator_warmup = prediction.warmup_bbox_decoder(model)
    # The pure decoder warmup must not have touched model/RNG before replay.
    require(frozen["state_adapter"].equal_tree(engine.snapshot()["learner"], store.recover().state["learner"]), "operator warmup changed checkpoint learning state")
    before_rotation_recovery = engine.learner.snapshot()
    rotation_runtime = restore_rotation_decoder_runtime(engine.scientific_trace, engine.transaction,
        prediction.rot_6d_to_matrix, torch, device, budget)
    require(frozen['state_adapter'].equal_tree(before_rotation_recovery, engine.learner.snapshot()),
            'scripted rotation runtime recovery changed scientific learning state')
    hook = ForwardBudgetHook(model.backbone, budget, synchronize=sync)
    panel_path = binding(task["panel"]["manifest"])
    panel_doc = load_json(panel_path)
    panel_ids = [row["sample_id"] for row in panel_doc["items"]]
    require(len(panel_ids) == len(set(panel_ids)) == 128 and not set(panel_ids) & {row["sample_id"] for row in records}, "panel overlaps adaptation or differs from frozen128")
    require(task["panel"]["labels"] == original["baseline_task"]["bindings"]["developer_labels"], "panel GT is not the existing developer partition")
    # The diagnostic's manifest is an immutable selection document, not a new
    # data protocol. Reuse the original image-only dataset and exact records.
    developer_dataset = data.ImageManifestDataset(paths["developer_manifest"], transforms=transforms)
    developer_indices = {row["sample_id"]: index for index,row in enumerate(developer_dataset.manifest["items"])}
    require(set(panel_ids) <= developer_indices.keys(), "panel escapes the original developer image partition")
    require(panel_doc["items"] == [developer_dataset.manifest["items"][developer_indices[sid]] for sid in panel_ids], "panel image records differ from frozen developer assets")
    class PanelDataset:
        def __init__(self):
            self.manifest = {**developer_dataset.manifest, "items": panel_doc["items"]}
        def __getitem__(self, index): return developer_dataset[developer_indices[panel_ids[index]]]
        def __len__(self): return len(panel_ids)
    panel_dataset = PanelDataset()
    labels = None
    panel_provider = old_backend.DeterministicImageProvider(panel_dataset, preprocess_sha256=preprocess,
                     rng_state=frozen["state_adapter"].rng_state, restore_rng=frozen["state_adapter"].restore_rng,
                     equal_tree=frozen["state_adapter"].equal_tree, manifest_file_sha256=task["panel"]["manifest"]["sha256"])
    # A cooperative resumption may start from the later isolated checkpoint.
    # Keep panel reconstruction tied to this pair's immutable original parent.
    panel_provider.initial_data_rng = copy.deepcopy(parent["learner"]["learning"]["rng"])
    panel_provider.data_rng = copy.deepcopy(panel_provider.initial_data_rng)
    def evaluate_one(sample_id):
        nonlocal labels
        before_phase = boundary.phase
        boundary.phase = "evaluation"
        try:
            if labels is None:
                budget.check_time("open only frozen developer GT for branch evaluation")
                label_path = binding(task["panel"]["labels"])
                labels = {row["sample_id"]: row for row in load_json(label_path)["items"]}
                require(set(panel_ids) <= labels.keys(), "panel includes non-developer/withheld GT")
            # Private panel reconstruction RNG is separate from the producer stream.
            raw = prediction.extract(model, adapter, panel_provider(sample_id), cfg, camera, points, profile="full")
            metric = prediction.developer_metrics(raw, dict(quaternion=labels[sample_id]["quaternion_wxyz"], translation=labels[sample_id]["translation_m"]), cfg)
            raw.pop("timing_seconds", None)
            return dict(sample_id=sample_id, prediction=raw, metrics=metric)
        finally:
            boundary.phase = before_phase
    def finite():
        baseline.finite_state(model, adapter.optimizer)
        deterministic_module.validate(deterministic, torch_module=torch)
    return dict(engine=engine, learner=learner, equal=frozen["state_adapter"].equal_tree,
                evaluate_one=evaluate_one, panel_materialize=panel_provider,
                panel_ids=panel_ids, hook=hook, close=adapter.close,
                finite=finite, gt_boundary=boundary,
                deterministic_runtime=deterministic, operator_warmup=operator_warmup, rotation_runtime=rotation_runtime,
                gpu_memory=lambda: dict(peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                                       peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved(device))))


def _offline_rows(task, panel_ids):
    path = binding(task["panel"]["offline_predictions"])
    doc = load_json(path) if path.suffix == ".json" else [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    rows = doc.get("rows", doc.get("items")) if isinstance(doc, dict) else doc
    by_id = {}
    for row in rows:
        require(row["sample_id"] not in by_id, "duplicate source Offline prediction")
        by_id[row["sample_id"]] = row.get("metrics", row)
    require(set(panel_ids) <= by_id.keys(), "Offline reference omits frozen panel IDs")
    result = [{**by_id[sid], "sample_id": sid} for sid in panel_ids]
    require(all(row["valid"] is True and type(row["error"]) in (int,float) and math.isfinite(row["error"]) for row in result), "panel reference is not entirely source-finite")
    return result


def record_attempt_cost(output, launch, before, after, counts, started, *, status):
    nonce = launch["run_nonce"]
    require(len(nonce) >= 16 and Path(nonce).name == nonce, "invalid actual launch nonce")
    path = Path(output)/"attempt_costs"/(nonce+".json")
    doc = dict(schema="g50p-risk-diagnostic-attempt-cost-v1", run_nonce=nonce, status=status,
               launch_record_sha256=canonical_sha(launch), budget_before=before, budget=after,
               forward_categories=dict(counts), elapsed_seconds=time.monotonic()-started)
    require(not path.exists(), "an actual diagnostic process cost was recorded twice")
    atomic_json(path, doc)
    paths = sorted(path.parent.glob("*.json"))
    rows = [load_json(p) for p in paths]
    summed = Counter()
    for row in rows:
        require(row["status"] in ("waiting_resources", "pair_main_complete"), "unknown diagnostic attempt outcome")
        summed.update(row["forward_categories"])
    return rows, dict(summed), [dict(path=str(p), sha256=sha256(p)) for p in paths]


def execute(task_path, *, admission, launch_record, conflict_check, budget):
    started = time.monotonic()
    if not admission.get("allow_start") or not admission.get("boundary_verified"):
        raise RuntimeError("live shared-host admission is required before paired diagnosis")
    task_path = Path(task_path).resolve(strict=True); task = load_json(task_path)
    require(task.get("schema") == SCHEMA and task.get("action") == "run_pair", "wrong paired-window task")
    frozen_inputs = load_json(binding(task["index_binding"]))
    require(frozen_inputs.get("schema") == "g50p-risk-diagnostic-inputs-v1"
            and task["producer"] == frozen_inputs["producer"]
            and task["selection"] in frozen_inputs["windows"], "task changed the frozen48 window selection or producer")
    selection = task["selection"]
    pair_key = f"{selection['domain']}:{selection['stream_seed']}"
    require(task["panel"] == frozen_inputs["panels"][pair_key], "task changed its pre-frozen128 panel")
    require(launch_record["run_id"] == task["run_id"], "diagnostic launch identity differs")
    output = Path(task["output_dir"]).resolve()
    require(output.name == task["run_id"] and output.parent.name == "pairs"
            and output.parent.parent.name == "g50p_risk_diagnostic_20261025", "diagnostic output escapes new independent plan")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/"report.json").exists(), "completed pair must be verified/reused, never rerun")
    require(not (output/"failure.json").exists(), "preserved pair failure blocks blind retry")
    if (output/"task.json").exists(): require(load_json(output/"task.json") == task, "existing isolated pair belongs to another immutable task")
    else: atomic_json(output/"task.json", task)
    budget_before = budget.snapshot()
    budget.check_time("before loading parent complete checkpoint")
    parent, parent_meta = read_generation(selection["parent_checkpoint"])
    budget.check_time("before loading matching original checkpoint")
    expected, expected_meta = read_generation(selection["match_checkpoint"])
    budget.check_time("before loading original recorded trajectory")
    original_trace = reference_trace(task, parent["identity"], budget=budget)
    number = selection["candidate_number"]
    require(type(number) is int and number >= 1 and parent["identity"] == expected["identity"], "parent/match/original identity differs")
    require(parent["counters"]["candidates"] < number <= expected["counters"]["candidates"], "parent/match checkpoints do not bracket target")
    original_target = [event for event in original_trace if event["kind"] == "window" and event["number"] == number]
    require(len(original_target) == 1 and original_target[0]["decision"] == selection["original_decision"], "selected original candidate differs")
    runtime = None
    try:
        budget.check_time("read frozen producer and checkpoints")
        if conflict_check():
            record_attempt_cost(output, launch_record, budget_before, budget.snapshot(), {}, started, status="waiting_resources")
            return dict(status="waiting_resources", exit_code=75, reason="foreign compute before model build")
        original_record = frozen_inputs["runs"][pair_key]
        original_report = load_json(binding(original_record["original_report"]))
        original_proof = load_json(binding(original_record["original_proof"]))
        require(original_proof["status"] == "passed"
                and original_proof["report_sha256"] == original_record["original_report"]["sha256"], "original selected run is not independently accepted")
        runtime = build_runtime(task, output, parent, budget=budget, original_report=original_report)
        engine, equal, hook = runtime["engine"], runtime["equal"], runtime["hook"]
        last_resource_check = time.monotonic()
        def check_resources():
            nonlocal last_resource_check
            if time.monotonic()-last_resource_check < 60: return
            last_resource_check = time.monotonic()
            reason = conflict_check()
            if reason: raise ResourceYield(str(reason))
        hook.phase = "replay_to_target"
        trace_prefix_matches(engine.scientific_trace, original_trace, equal)
        candidate_receipt = output/"candidate_binding.json"
        if candidate_receipt.exists():
            candidate_binding = load_json(candidate_receipt)
            with binding(candidate_binding).open("rb") as handle: candidate = pickle.load(handle)
            require(engine.counters["candidates"] >= number and engine.transaction is None, "candidate evidence precedes durable actual decision")
        else:
            while engine.transaction is None or engine.transaction["number"] != number:
                check_resources(); budget.check_time("replay to fixed candidate")
                require(engine.next_index < expected["next_index"], "passed selected target without preparing it")
                engine.step(stop_after_prepare=True)
                runtime["finite"]()
                trace_prefix_matches(engine.scientific_trace, original_trace, equal)
            prepared = engine.snapshot()
            require(prepared["transaction"]["number"] == number and prepared["learner"]["window_size"] == 3,
                    "target must be before fourth gradient, not a different update")
            hook.phase = "target_candidate_and_protection"
            candidate = capture_candidate(engine)
            runtime["finite"]()
            budget.check_time("persist actual resolved window before panel evaluation")
            engine.checkpoint("diagnostic_actual_target_resolved")
            budget.check_time("persist candidate and whole-window base")
            candidate_binding = save_pickle(output/"candidate_learner.bin", candidate)
            atomic_json(candidate_receipt, candidate_binding)
        target = [event for event in engine.scientific_trace if event["kind"] == "window" and event["number"] == number][0]
        require(numeric.equal(target, original_target[0]), "target update/protection/actual decision did not reproduce original")
        actual_resolved = engine.snapshot()
        panel_ids = runtime["panel_ids"]
        offline = _offline_rows(task, panel_ids)
        def progress(branch, rows, digest):
            atomic_json(output/("progress_"+branch+".json"), dict(branch=branch, rows=rows,
                        task_sha256=sha256(task_path), candidate_learner=candidate_binding,
                        branch_learning_sha256=digest))
        def saved_progress(branch, digest):
            path = output/("progress_"+branch+".json")
            if not path.exists(): return []
            doc = load_json(path)
            require(doc["branch"] == branch and doc["task_sha256"] == sha256(task_path)
                    and doc["candidate_learner"] == candidate_binding
                    and doc["branch_learning_sha256"] == digest, "saved branch evaluation identity changed")
            return doc["rows"]
        branches = evaluate_branches(engine, candidate, panel_ids, runtime["evaluate_one"], offline,
                           equal=equal, budget=budget, phase_hook=hook, progress=progress,
                           saved_progress=saved_progress, reconstruct_panel=runtime["panel_materialize"],
                           resource_check=check_resources)
        require(equal(science_state(engine.snapshot()), science_state(actual_resolved)), "actual decision was not fully restored before continuation")
        hook.phase = "replay_to_original_checkpoint"
        while engine.next_index < expected["next_index"] or (engine.transaction is not None and expected["transaction"] is None):
            check_resources(); budget.check_time("continue actual producer branch")
            engine.step()
            runtime["finite"]()
            trace_prefix_matches(engine.scientific_trace, original_trace, equal)
        if expected["finished"] and not engine.finished:
            require(engine.next_index == len(engine.stream_records), "cannot finalize a nonfinal producer boundary")
            engine.step()
        actual_match = engine.snapshot()
        require(numeric.equal(science_state(actual_match), science_state(expected)), "full model/optimizer/gradient/RNG/pool/provider scientific checkpoint differs")
        runtime["finite"]()
        require(runtime["gt_boundary"].denied == [] and runtime["gt_boundary"].phase == "adaptation", "GT boundary was violated or not restored before continuation")
        budget.check_time("persist matched scientific state evidence")
        matched_binding = save_pickle(output/"matched_scientific_state.bin", science_state(actual_match))
        costs, counts, cost_bindings = record_attempt_cost(output, launch_record, budget_before, budget.snapshot(),
                                      hook.counts, started, status="pair_main_complete")
        report = dict(schema=OUTPUT_SCHEMA, status="pair_main_complete_pending_independent_verification",
                      run_id=task["run_id"], task_sha256=sha256(task_path), selection=selection,
                      producer=task["producer"], panel=task["panel"], original_target=target,
                      parent_checkpoint_sha256=parent_meta["state_sha256"], match_checkpoint_sha256=expected_meta["state_sha256"],
                      candidate_learner=candidate_binding, matched_scientific_state=matched_binding,
                      candidate_learner_sha256=tree_digest(candidate), branch_results=branches,
                      expected_branch_learning_sha256={name: tree_digest(value) for name,value in derive_branch_snapshots(candidate).items()},
                      matched_scientific_sha256=tree_digest(science_state(actual_match)),
                      expected_scientific_sha256=tree_digest(science_state(expected)),
                      original_trace_reproduced_stepwise=False, full_scientific_checkpoint_equal=False,
                      cross_gpu_trace_reproduced_stepwise=True, cross_gpu_full_state_within_tolerance=False,
                      scientific_state_accepted_under_policy=True, training_tensors_within_tolerance=True, discrete_trajectory_exact=True,
                      numeric_acceptance=numeric.receipt(), internal_branch_and_rollback_exact=True,
                      excluded_match_fields=["ledger_boundary"], original_producer_files_modified=False,
                      deterministic_runtime=runtime["deterministic_runtime"], operator_warmup=runtime["operator_warmup"], rotation_runtime=runtime["rotation_runtime"],
                      gt_boundary=dict(developer_label_reads=runtime["gt_boundary"].developer_reads,
                                       forbidden_attempts=runtime["gt_boundary"].denied,
                                       restored_phase=runtime["gt_boundary"].phase),
                      budget_before=budget_before, budget=budget.snapshot(), forward_categories=counts,
                      attempt_costs=costs, attempt_cost_bindings=cost_bindings, **runtime["gpu_memory"](),
                      actual_attempt_elapsed_seconds=time.monotonic()-started,
                      launch_record_sha256=canonical_sha(launch_record),
                      scientific_interpretation="paired diagnostic association on fixed128 developer panel; not method validation or threshold selection")
        budget.check_time("publish complete paired result")
        atomic_json(output/"report.json", report)
        return dict(status="pair_main_complete", exit_code=0, run_id=task["run_id"], report=str(output/"report.json"), report_sha256=sha256(output/"report.json"))
    except ResourceYield as exc:
        budget.check_time("cooperative checkpoint after foreign compute")
        runtime["engine"].checkpoint("diagnostic_shared_host_cooperative_yield")
        record_attempt_cost(output, launch_record, budget_before, budget.snapshot(), runtime["hook"].counts,
                            started, status="waiting_resources")
        atomic_json(output/"yield.json", dict(status="waiting_resources", reason=str(exc),
                    task_sha256=sha256(task_path), budget=budget.snapshot()))
        return dict(status="waiting_resources", exit_code=75, reason=str(exc), run_id=task["run_id"])
    except Exception as exc:
        atomic_json(output/"failure.json", dict(status="failed", type=type(exc).__name__, message=str(exc),
                    action="preserve_and_block_no_automatic_replay", budget=budget.snapshot(),
                    trace_difference=getattr(exc, "trace_difference", None),
                    elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if runtime is not None:
            runtime["hook"].close(); runtime["close"]()

