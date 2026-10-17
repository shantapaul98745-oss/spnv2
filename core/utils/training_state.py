"""Boundary-only training state and deterministic data, opt-in from TRAIN.REPRODUCIBLE.

The sampler yields (epoch, index), so prefetched worker augmentation is independent
of worker count, scheduling and resume position. No model RNG is used by a loader.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def keyed_seed(seed, epoch, index=0, purpose="sample"):
    key = f"spnv2-v1:{int(seed)}:{int(epoch)}:{int(index)}:{purpose}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


class SeededDataset(Dataset):
    """Keep the original dataset intact and seed each complete __getitem__ call."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        seed = keyed_seed(self.seed, epoch, index)
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            # Do not seed or initialize CUDA in loader workers.
            torch.random.default_generator.manual_seed(seed)
            transforms = getattr(self.dataset, "transforms", None)
            if hasattr(transforms, "set_random_seed"):
                transforms.set_random_seed(seed % 2**32)
            item = self.dataset[index]
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                targets = dict(item[1])
                targets["_sample_index"] = index
                return item[0], targets
            return item
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)


class EpochBatchSampler(Sampler):
    """Deterministic epoch ordering; truncate before loading any partial window."""
    version = 1

    def __init__(self, size, batch_size, accum_steps=1, seed=2021, shuffle=True):
        self.size, self.batch_size = int(size), int(batch_size)
        self.accum_steps, self.seed, self.shuffle = int(accum_steps), int(seed), bool(shuffle)
        if min(self.batch_size, self.accum_steps) < 1:
            raise ValueError("batch_size and accum_steps must be positive")
        self.epoch, self.start_batch = 0, 0
        self.total_batches = self.size // (self.batch_size * self.accum_steps) * self.accum_steps

    def set_epoch(self, epoch, start_batch=0):
        if not 0 <= start_batch <= self.total_batches or start_batch % self.accum_steps:
            raise ValueError("resume position must be a complete update boundary")
        self.epoch, self.start_batch = int(epoch), int(start_batch)

    def indices(self):
        if self.shuffle:
            gen = torch.Generator().manual_seed(keyed_seed(self.seed, self.epoch, purpose="order"))
            order = torch.randperm(self.size, generator=gen).tolist()
        else:
            order = list(range(self.size))
        return order[:self.total_batches * self.batch_size]

    def __iter__(self):
        order = self.indices()
        for batch in range(self.start_batch, self.total_batches):
            yield [(self.epoch, i) for i in order[batch*self.batch_size:(batch+1)*self.batch_size]]

    def __len__(self):
        return self.total_batches - self.start_batch

    def state_dict(self):
        order = self.indices()
        return dict(version=self.version, size=self.size, batch_size=self.batch_size,
                    accum_steps=self.accum_steps, seed=self.seed, shuffle=self.shuffle,
                    epoch=self.epoch, next_batch=self.start_batch, total_batches=self.total_batches,
                    order_sha256=hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest())


class TrainingState:
    def __init__(self, **values):
        self.epoch = 0
        self.next_batch = 0
        self.update_attempts = 0
        self.optimizer_updates = 0
        self.amp_skips = 0
        self.consecutive_amp_skips = 0
        self.samples_seen = 0
        self.microbatches_seen = 0
        self.last_loss = None
        for key, value in values.items():
            if key not in self.__dict__:
                raise ValueError(f"Unknown training state field: {key}")
            setattr(self, key, value)

    def state_dict(self):
        return dict(self.__dict__)


def emit_event(event, state=None, **fields):
    record = dict(event=event, phase="train", timestamp=time.time())
    if state is not None:
        record.update(state.state_dict())
        record["epoch"] = state.epoch + 1
        record["batch"] = state.next_batch
        record["loss"] = state.last_loss
    record.update(fields)
    line = "SPNV2_EVENT " + json.dumps(record, allow_nan=False, sort_keys=True)
    print(line, flush=True)
    event_file = os.environ.get("SPNV2_EVENT_FILE")
    if event_file:
        try:
            with open(event_file, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
        except OSError as exc:
            # stdout remains primary. The queue must flag this warning and block
            # downstream work, while healthy training preserves its live state.
            warning = dict(event="progress_write_warning", phase=record.get("phase", "train"),
                           timestamp=time.time(), error=str(exc), path=event_file,
                           queue_blocked=True)
            print("SPNV2_EVENT " + json.dumps(warning, allow_nan=False, sort_keys=True), flush=True)
            print(f"SPNV2_EVENT_FILE write warning: {exc}", file=sys.stderr, flush=True)
    return record


def capture_rng_state():
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=np_state[0], keys=np_state[1].tolist(),
                pos=np_state[2], has_gauss=np_state[3], cached_gaussian=np_state[4]),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng_state(states):
    random.setstate(states["python"])
    n = states["numpy"]
    np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32), n["pos"],
                         n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(states["torch_cpu"].cpu())
    if states["torch_cuda"]:
        if len(states["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Exact resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all([v.cpu() for v in states["torch_cuda"]])


def tensors_are_finite(values):
    # Reduce on each device before the host check, avoiding a CUDA sync per tensor.
    by_device = {}
    for value in values:
        if value is not None and (value.is_floating_point() or value.is_complex()):
            by_device.setdefault(value.device, []).append(torch.isfinite(value).all())
    return all(bool(torch.stack(flags).all()) for flags in by_device.values())


def assert_finite_weights(model):
    values = model.state_dict()
    if not tensors_are_finite(values.values()):
        for name, value in values.items():
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite model weight/buffer: {name}")


def atomic_torch_save(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_reproducible_inputs(cfg):
    for field, value in (("MODEL.PRETRAIN_FILE", cfg.MODEL.PRETRAIN_FILE),
                         ("MODEL.BACKBONE.PRETRAINED", cfg.MODEL.BACKBONE.PRETRAINED)):
        if value and (not Path(value).is_absolute() or not Path(value).is_file()):
            raise ValueError(f"Reproducible training requires an absolute existing {field}: {value}")


def data_inventory(datasets):
    """Freeze path/size/mtime metadata, not raw-image content, for actual inputs.

    Uses the already constructed original Dataset so folder/domain/mask/style
    selection follows precisely the training and validation dataset definitions.
    """
    records = []
    coverage = {}
    for split, original in sorted(datasets.items()):
        dataset = original.dataset if isinstance(original, SeededDataset) else original
        folders = [dataset.imagefolder]
        if dataset.load_labels and dataset.load_masks:
            folders.append(dataset.maskfolder)
        if dataset.is_train and dataset.styleAug:
            folders.append(dataset.stylefolder)
        count = 0
        for row in range(len(dataset)):
            filename = str(dataset.csv.iloc[row, 0])
            for folder in folders:
                path = Path(dataset.root)/dataset.domain/folder/filename
                stat = path.stat()  # A missing referenced asset blocks training.
                records.append([split, row, str(path.absolute()), stat.st_size, stat.st_mtime_ns])
                count += 1
        coverage[split] = dict(samples=len(dataset), folders=folders, files=count)
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return dict(version=1, hash_kind="sha256-of-path-size-mtime_ns-records",
                raw_content_hashed=False, sha256=hashlib.sha256(encoded).hexdigest(),
                records=records, coverage=coverage)


def build_manifest(cfg, model=None, datasets=None):
    validate_reproducible_inputs(cfg)
    repo = Path(__file__).resolve().parents[2]
    normalized = cfg.clone()
    normalized.defrost()
    normalized.TRAIN.RESUME_FILE = ""
    normalized.AUTO_RESUME = False
    config = normalized.dump()
    code = {}
    for folder in ("core", "tools"):
        for path in sorted((repo/folder).rglob("*.py")):
            code[str(path.relative_to(repo))] = sha256_file(path)
    csvs = {}
    for name in (cfg.TRAIN.TRAIN_CSV, cfg.TRAIN.VAL_CSV):
        path = Path(cfg.DATASET.ROOT)/cfg.DATASET.DATANAME/name
        csvs[str(path)] = sha256_file(path)
    init = {}
    for name in (cfg.MODEL.PRETRAIN_FILE, cfg.MODEL.BACKBONE.PRETRAINED):
        if name:
            path = Path(name)
            init[str(path)] = sha256_file(path)
    if not cfg.MODEL.USE_GROUPNORM_BACKBONE:
        # EfficientDet always requests pretrained=True on its BN EfficientNet.
        from nets.backbones.efficientnet import model_urls
        arch = f"efficientnet_b{cfg.MODEL.EFFICIENTDET_PHI}"
        url = model_urls[arch]
        cached = Path(torch.hub.get_dir())/"checkpoints"/url.rsplit("/", 1)[-1]
        if not cached.is_file():
            raise RuntimeError(f"Cannot hash the ImageNet initialization actually requested: {cached}")
        init["imagenet_backbone"] = dict(architecture=arch, source_url=url,
            path=str(cached.resolve()), sha256=sha256_file(cached))
    geometry = {}
    for field in ("CAMERA", "KEYPOINTS", "CADMODEL"):
        path = Path(getattr(cfg.DATASET, field))
        geometry[field] = dict(path=str(path.absolute()),
                               sha256=sha256_file(path) if path.is_file() else None,
                               exists=path.is_file())
    inventory = data_inventory(datasets) if datasets is not None else None
    import platform
    libraries = {name: str(getattr(sys.modules[name], "__version__", "unknown"))
                 for name in ("torchvision", "albumentations", "cv2", "scipy", "pandas")
                 if name in sys.modules}
    manifest = dict(version=1, config=config, config_sha256=hashlib.sha256(config.encode()).hexdigest(),
                    code=code, csv=csvs, initialization=init, auxiliary_geometry=geometry,
                    data_inventory=inventory,
                    runtime=dict(torch=str(torch.__version__), numpy=str(np.__version__),
                                 python=platform.python_version(), cuda=str(torch.version.cuda),
                                 cudnn=torch.backends.cudnn.version(), libraries=libraries,
                                 threads=torch.get_num_threads(),
                                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                 cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                    if torch.cuda.is_initialized() else []),
                    sampler="epoch-keyed-v1", augmentation="sample-keyed-v1")
    if model is not None:
        manifest["initial_state_sha256"] = model_digest(model)
    return manifest


def verify_manifest(saved, current):
    keys = ["version", "config_sha256", "code", "csv", "initialization", "runtime", "sampler", "augmentation", "auxiliary_geometry", "data_inventory"]
    if "initial_state_sha256" in current:
        keys.append("initial_state_sha256")
    for key in keys:
        if saved.get(key) != current.get(key):
            raise RuntimeError(f"Exact resume manifest mismatch: {key}")


def save_training_checkpoint(path, model, optimizer, scaler, state, sampler, manifest):
    if state.next_batch % sampler.accum_steps:
        raise ValueError("Refusing checkpoint inside an accumulation window")
    assert_finite_weights(model)
    sampler.set_epoch(state.epoch, state.next_batch)
    payload = dict(format="spnv2-boundary-v1", state_dict=model.state_dict(),
                   optimizer=optimizer.state_dict(), scaler=scaler.state_dict() if scaler is not None else None,
                   training_state=state.state_dict(), rng=capture_rng_state(),
                   sampler=sampler.state_dict(), manifest=manifest)
    atomic_torch_save(payload, path)
    return payload


def load_training_checkpoint(path, model, optimizer, scaler, sampler, manifest, device="cpu"):
    # All entries use tensors and primitive types, including the NumPy RNG state.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "spnv2-boundary-v1":
        raise RuntimeError("Explicit resume requires a new boundary checkpoint")
    verify_manifest(payload["manifest"], manifest)
    state = TrainingState(**payload["training_state"])
    sampler.set_epoch(state.epoch, state.next_batch)
    if payload["sampler"] != sampler.state_dict():
        raise RuntimeError("Exact resume sampler mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    assert_finite_weights(model)
    optimizer.load_state_dict(payload["optimizer"])
    if (scaler is None) != (payload["scaler"] is None):
        raise RuntimeError("Exact resume AMP/scaler mismatch")
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"])
    return state, payload["manifest"]


def prepare_output_directory(cfg):
    """Reserve new outputs exclusively, or require explicit ownership on resume."""
    output = Path(cfg.OUTPUT_DIR)/cfg.MODEL.BACKBONE.NAME/cfg.EXP_NAME
    marker = output/"reproducible_run.json"
    resume = cfg.TRAIN.RESUME_FILE
    if resume:
        resume = Path(resume).resolve()
        if not marker.is_file() or resume.parent != output.resolve() or not resume.is_file():
            raise RuntimeError("Resume requires an explicit checkpoint in its owned reproducible output directory")
        if json.loads(marker.read_text()).get("format") != "spnv2-reproducible-v1":
            raise RuntimeError("Unrecognized reproducible output ownership marker")
    else:
        output.mkdir(parents=True, exist_ok=False)
        marker.write_text(json.dumps(dict(format="spnv2-reproducible-v1", created=time.time()), indent=2))
    return str(output)
