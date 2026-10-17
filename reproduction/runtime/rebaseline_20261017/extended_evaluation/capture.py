"""Nonintrusive numerical H-pose observer. Frozen inference/solver files stay unchanged."""
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import threading

_LOCK = threading.RLock()
_ACTIVE = set()


class CaptureError(RuntimeError):
    pass


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation():
    root = Path(__file__).resolve().parent
    return {str(path): sha(path) for path in sorted(root.glob("*.py"))}


def read_ids(csv_path):
    with Path(csv_path).open(newline="") as stream:
        ids = [str(row[0]) for row in csv.reader(stream)]
    if not ids or any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("CSV IDs must be nonempty and unique")
    return ids


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush()
        __import__("os").fsync(stream.fileno())
    temporary.replace(path)


def numeric(value, shape):
    import numpy as np
    if value is None:
        return None, "solver_returned_none", None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if tuple(array.shape) != tuple(shape):
        return None, "wrong_shape:" + str(list(array.shape)), None
    if not np.isfinite(array).all():
        return None, "nonfinite", None
    if shape == (4,) and np.linalg.norm(array) == 0:
        return None, "zero_quaternion", None
    if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        return None, "unsupported_dtype:" + str(array.dtype), None
    return array.astype(np.float64, copy=True).tolist(), None, str(array.dtype)


class HeatmapPoseCapture:
    def __init__(self, module, *, csv_path, output_dir, heads, context=None):
        self.module = module
        self.csv_path = Path(csv_path).resolve()
        self.output = Path(output_dir).resolve()
        self.ids = read_ids(self.csv_path)
        self.heads = list(heads)
        if len(self.heads) != len(set(self.heads)) or not set(self.heads) <= {"heatmap", "efficientpose", "segmentation"}:
            raise ValueError("Unique known inference heads required")
        self.applicable = "heatmap" in self.heads
        if context is not None and not isinstance(context, dict):
            raise ValueError("Caller context must be a finite JSON object")
        self.context = copy.deepcopy(context or {})
        digest(self.context)  # Strict JSON, finite caller evidence.
        self.calls, self.records, self.errors = 0, [], []
        self.original = None
        self.stream = None

    def __enter__(self):
        with _LOCK:
            if id(self.module) in _ACTIVE:
                raise CaptureError("Nested observer for this inference module is forbidden")
            self.output.mkdir(parents=True, exist_ok=False)
            self.original = self.module.solve_pose_from_heatmaps
            self.raw = self.output / "heatmap_pose_capture.jsonl"
            self.stream = self.raw.open("x", encoding="utf-8")
            _ACTIVE.add(id(self.module))
            self.module.solve_pose_from_heatmaps = self._wrapped
        return self

    def _wrapped(self, *args, **kwargs):
        # No extra solver call, heatmap operation, model access or RNG operation.
        result = self.original(*args, **kwargs)
        index = self.calls
        self.calls += 1
        try:
            if not self.applicable:
                raise CaptureError("Unexpected solver call while heatmap head is not applicable")
            if not isinstance(result, (tuple, list)) or len(result) != 4:
                raise CaptureError("Unexpected original solver return structure")
            keypoints, q, t, reject = result
            q_value, q_reason, q_dtype = numeric(q, (4,))
            t_value, t_reason, t_dtype = numeric(t, (3,))
            if keypoints is None:
                k_value, k_reason = None, "solver_returned_none"
            else:
                import numpy as np
                k = keypoints.detach().cpu().numpy() if hasattr(keypoints, "detach") else np.asarray(keypoints)
                if k.ndim != 2 or k.shape[1] != 2 or not np.isfinite(k).all():
                    k_value, k_reason = None, "invalid_keypoints"
                else:
                    k_value, k_reason = k.astype(np.float64, copy=True).tolist(), None
            if index >= len(self.ids):
                raise CaptureError("More solver calls than actual CSV samples")
            record = {
                "index": index, "sample_id": self.ids[index], "reject": bool(reject),
                "keypoints": k_value, "q": q_value, "t": t_value, "q_dtype": q_dtype, "t_dtype": t_dtype,
                "q_numeric": q_value is not None, "t_numeric": t_value is not None,
                "numeric_pose_available": q_value is not None and t_value is not None,
                "missing_reason": {"q": q_reason, "t": t_reason, "keypoints": k_reason},
            }
            self.stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            self.stream.flush()
            self.records.append(record)
        except Exception as error:
            self.errors.append(type(error).__name__ + ": " + str(error))
        return result  # Preserve the exact original result object, even on capture write error.

    def __exit__(self, exc_type, exc, tb):
        import os
        with _LOCK:
            current = self.module.solve_pose_from_heatmaps
            if getattr(current, "__self__", None) is not self:
                self.errors.append("Inference solver symbol changed during observer context")
            self.module.solve_pose_from_heatmaps = self.original
            _ACTIVE.discard(id(self.module))
        try:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
            expected = len(self.ids) if self.applicable else 0
            if self.calls != expected or len(self.records) != expected:
                self.errors.append("Solver calls/records must equal actual N when H exists, otherwise zero")
            if sha(self.csv_path) != getattr(self, "_csv_sha", sha(self.csv_path)):
                self.errors.append("CSV changed during observation")
            source_file = inspect.getsourcefile(self.original)
            manifest = {
                "schema": "spnv2-heatmap-pose-capture-v1",
                "status": "failed" if exc is not None or self.errors else ("complete" if self.applicable else "not_applicable"),
                "heads": self.heads, "heatmap_applicable": self.applicable, "sample_count": len(self.ids),
                "solver_calls": self.calls, "record_count": len(self.records),
                "csv": {"path": str(self.csv_path), "sha256": sha(self.csv_path), "sample_id_sha256": digest(self.ids)},
                "records": {"path": str(self.raw), "sha256": sha(self.raw)},
                "context": self.context, "context_sha256": digest(self.context),
                "observer_code_sha256": implementation(),
                "original_solver": {"file": str(source_file), "file_sha256": sha(source_file) if source_file else None,
                                    "qualname": getattr(self.original, "__qualname__", repr(self.original))},
                "inference_module_file": getattr(self.module, "__file__", None),
                "inference_module_file_sha256": sha(self.module.__file__) if getattr(self.module, "__file__", None) else None,
                "errors": self.errors, "body_exception": type(exc).__name__ + ": " + str(exc) if exc else None,
                "contract": "Same original return object, no additional forward/solver/RNG, numeric geometry rejects kept, None never substituted",
            }
            atomic(self.output / "capture_manifest.json", manifest)
        except Exception as error:
            self.errors.append("Final capture evidence write failed: " + repr(error))
        if exc is None and self.errors:
            raise CaptureError("; ".join(self.errors))
        return False


def observe_heatmap_poses(inference_module, *, csv_path, output_dir, heads, context=None):
    observer = HeatmapPoseCapture(inference_module, csv_path=csv_path, output_dir=output_dir,
                                  heads=heads, context=context)
    observer._csv_sha = sha(observer.csv_path)
    return observer
