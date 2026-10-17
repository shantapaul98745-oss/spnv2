"""Pilot evidence helpers. No CUDA initialization or external data operations."""
import hashlib
import json
import math
import os
from pathlib import Path

HEADS = ("heatmap", "efficientpose", "segmentation")
ARTIFACT_NAMES = (
    "manifest.json", "model_initial.pth.tar", "initial_checkpoint.pth.tar",
    "boundary_checkpoint.pth.tar", "model_final.pth.tar",
    "continuous_final_checkpoint.pth.tar", "model_resumed_final.pth.tar",
    "resumed_final_checkpoint.pth.tar",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def exact_tree(left, right, name="root"):
    import torch
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu())):
            raise AssertionError("Exact tensor mismatch: " + name)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError("Exact dictionary keys mismatch: " + name)
        for key in left:
            exact_tree(left[key], right[key], name + "." + str(key))
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("Exact sequence mismatch: " + name)
        for index, (a, b) in enumerate(zip(left, right)):
            exact_tree(a, b, name + "[" + str(index) + "]")
    elif type(left) is not type(right) or left != right:
        raise AssertionError("Exact scalar mismatch: " + name)


def tree_sha256(value):
    """Hash every tensor byte and every typed scalar, key, and sequence position."""
    import torch
    digest = hashlib.sha256()
    def add(raw):
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            add(b"tensor")
            add(str(tensor.dtype).encode())
            add(json.dumps(list(tensor.shape)).encode())
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite tensor in checkpoint evidence")
            add(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            add(b"dict")
            # torch optimizer state contains integer keys; retain typed keys.
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
            add(b"end_dict")
        elif isinstance(item, (list, tuple)):
            add(type(item).__name__.encode())
            for child in item:
                visit(child)
            add(b"end_sequence")
        elif item is None or type(item) in (str, int, bool, float):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite scalar in checkpoint evidence")
            add(type(item).__name__.encode())
            add(repr(item).encode())
        else:
            raise ValueError("Unsupported checkpoint evidence type: " + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def model_sha256(state_dict):
    """Production model digest, independently calculated from saved tensors."""
    import torch
    digest = hashlib.sha256()
    for key, value in state_dict.items():
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def protocol(cfg, attempts, checkpoint):
    """Check effective config; head ordering and loss factors are intentional."""
    names = list(cfg["MODEL"]["HEAD"]["NAMES"])
    losses = list(cfg["MODEL"]["HEAD"]["LOSS_HEADS"])
    phi = cfg["MODEL"]["EFFICIENTDET_PHI"]
    train = cfg["TRAIN"]
    gn = bool(cfg["MODEL"]["USE_GROUPNORM_BACKBONE"])
    allowed = ((phi == 6 and gn and names == list(HEADS))
               or (phi == 3 and not gn and names in
                   [["heatmap"], ["efficientpose"], ["heatmap", "efficientpose"]]))
    checks = {
        "supported_architecture_and_heads": allowed,
        "head_names_equal_loss_heads": names == losses and len(names) == len(set(names)),
        "unit_loss_factors": list(cfg["MODEL"]["HEAD"]["LOSS_FACTORS"]) == [1.0] * len(names),
        "loss_counts": list(cfg["MODEL"]["HEAD"]["LOSS_NUMS"]) == [3 if x == "efficientpose" else 1 for x in names],
        "resolution": list(cfg["DATASET"]["INPUT_SIZE"]) == [768, 512],
        "seed": cfg["SEED"] == 2021, "cuda": cfg["CUDA"] is True, "amp": cfg["FP16"] is True,
        "adamw": train["OPTIMIZER"] == "AdamW", "learning_rate": train["LR"] == .001,
        "weight_decay": train["WD"] == .0001,
        "schedule": train["SCHEDULER"] == "step" and list(train["LR_STEP"]) == [15, 18] and train["LR_FACTOR"] == .1,
        "physical_batch": train["IMAGES_PER_GPU"] == (1 if phi == 6 else 4),
        "accumulation": train["ACCUM_STEPS"] == 4,
        "heads_groupnorm": cfg["MODEL"]["USE_GROUPNORM_HEADS"] is True,
        "deterministic_cudnn": cfg["CUDNN"]["DETERMINISTIC"] is True and cfg["CUDNN"]["BENCHMARK"] is False,
        "reproducible": train["REPRODUCIBLE"] is True,
        "single_process": cfg["DIST"]["MULTIPROCESSING_DISTRIBUTED"] is False,
        "fresh_initialization": not cfg["AUTO_RESUME"] and not train["RESUME_FILE"]
            and not cfg["MODEL"]["PRETRAIN_FILE"] and not cfg["MODEL"]["BACKBONE"]["PRETRAINED"],
        "engineering_windows": type(attempts) is int and type(checkpoint) is int and attempts == 32 and checkpoint == 16,
        "no_entropy_training": not cfg["ODR"]["MIN_ENTROPY"],
    }
    for name in ("ADJUST_BRIGHTNESS_CONTRAST", "APPLY_BLUR", "APPLY_SOLAR_FLARE",
                 "APPLY_NOISE", "APPLY_RANDOM_ERASE", "APPLY_TEXTURE_RANDOMIZATION"):
        checks["augmentation_" + name] = cfg["AUGMENT"][name] is True
    if not all(checks.values()):
        raise ValueError("Pilot protocol mismatch: " + ", ".join(k for k, v in checks.items() if not v))
    return checks


def parameter_changes(initial, final, parameter_names, heads):
    import torch
    components = {"backbone": {"prefix": "backbone.", "changed_tensors": 0,
                              "changed_elements": 0, "max_abs_change": 0.0}}
    for index, name in enumerate(heads):
        components[name] = {"prefix": "heads." + str(index) + ".", "changed_tensors": 0,
                            "changed_elements": 0, "max_abs_change": 0.0}
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("Duplicate trainable parameter evidence")
    for name in parameter_names:
        before, after = initial[name].cpu(), final[name].cpu()
        if before.dtype != after.dtype or before.shape != after.shape:
            raise ValueError("Parameter schema changed: " + name)
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError("Nonfinite trainable parameter: " + name)
        difference = (after - before).abs()
        for item in components.values():
            if name.startswith(item["prefix"]):
                changed = int(torch.count_nonzero(difference))
                item["changed_tensors"] += int(changed > 0)
                item["changed_elements"] += changed
                item["max_abs_change"] = max(item["max_abs_change"], float(difference.max()))
    if not all(x["changed_elements"] > 0 for x in components.values()):
        raise ValueError("At least one active head/backbone has no changed parameter")
    return components
