"""Read-only baseline reconciliation and explicit-denominator extended pose report."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import yaml

from .capture import sha, digest, implementation, read_ids, atomic

UNITS = {"eR": "deg", "eT": "m", "eR_HIL": "rad", "eT_norm_HIL": "m/m", "pose": "score"}


def require(value, message):
    if not value:
        raise ValueError(message)


def same_numeric(left, right, label):
    require(np.allclose(np.asarray(left), np.asarray(right), rtol=2e-5, atol=2e-5, equal_nan=True),
            "Original accepted numerical evidence differs: " + label)


def matrix_from_q(q):
    q = np.asarray(q, dtype=np.float32)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [2*w**2-1+2*x**2, 2*x*y+2*w*z, 2*x*z-2*w*y],
        [2*x*y-2*w*z, 2*w**2-1+2*y**2, 2*y*z+2*w*x],
        [2*x*z+2*w*y, 2*y*z-2*w*x, 2*w**2-1+2*z**2],
    ], dtype=np.float32)


def pose_metrics(q, t, q_gt, t_gt, *, rotation_matrix=None, threshold, theta_q, theta_t):
    values = {key: None for key in UNITS}
    if q is not None or rotation_matrix is not None:
        if rotation_matrix is not None:
            a = np.asarray(rotation_matrix, dtype=np.float32)
            gt = matrix_from_q(q_gt)
            angle = np.arccos(np.clip((np.trace(a @ gt.T) - 1) / 2, -1, 1))
        else:
            angle = 2 * np.arccos(np.clip(abs(np.dot(np.asarray(q), q_gt)), 0, 1))
        degrees = float(np.rad2deg(angle))
        values["eR"] = degrees
        values["eR_HIL"] = 0.0 if threshold and degrees < theta_q else float(angle)
    if t is not None:
        # Preserve frozen source arithmetic before the discontinuous HIL test:
        # E subtract/square/sum/sqrt/divide remain float32; PnP H stays float64,
        # while the actual CSV GT norm is always calculated in float32.
        error = np.sqrt(np.sum(np.square(np.asarray(t_gt) - np.asarray(t))))
        normalized = error / np.sqrt(np.sum(np.square(t_gt)))
        values["eT"] = float(error)
        values["eT_norm_HIL"] = 0.0 if threshold and normalized < theta_t else float(normalized)
    if values["eR_HIL"] is not None and values["eT_norm_HIL"] is not None:
        values["pose"] = values["eR_HIL"] + values["eT_norm_HIL"]
    return values


def load_inputs(*, capture_dir, result_dir, csv_path, cfg_path):
    capture_dir, result_dir = Path(capture_dir).resolve(), Path(result_dir).resolve()
    csv_path, cfg_path = Path(csv_path).resolve(), Path(cfg_path).resolve()
    ids = read_ids(csv_path)
    with csv_path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    require(all(len(row) >= 12 for row in rows), "Pose GT columns missing")
    ground_truth = [(np.asarray(row[5:9], dtype=np.float32), np.asarray(row[9:12], dtype=np.float32)) for row in rows]
    require(all(np.isfinite(q).all() and np.isfinite(t).all() and np.linalg.norm(t) > 0 and abs(np.linalg.norm(q)-1) < 1e-3 for q, t in ground_truth),
            "Invalid actual CSV ground truth")
    cfg = yaml.safe_load(cfg_path.read_text())
    require((Path(cfg["DATASET"]["ROOT"]) / cfg["DATASET"]["DATANAME"] / cfg["TEST"]["TEST_CSV"]).resolve() == csv_path, "Actual CSV differs from config test domain")
    heads = list(cfg["TEST"]["HEAD"])
    require(len(heads) == len(set(heads)) and set(heads) <= {"heatmap", "efficientpose", "segmentation"}, "Invalid config head list")
    require(all(math.isfinite(float(cfg["TEST"][k])) and cfg["TEST"][k] >= 0 for k in ("SPEED_THRESHOLD_Q", "SPEED_THRESHOLD_T")), "Invalid HIL thresholds")
    require("heatmap" in heads or "efficientpose" in heads, "No pose head is applicable")
    paths = {"csv": csv_path, "cfg": cfg_path, "capture_manifest": capture_dir / "capture_manifest.json",
             "capture_records": capture_dir / "heatmap_pose_capture.jsonl",
             "mean": result_dir / "mean_performance.mat", "per_sample": result_dir / "per_sample_performance.mat",
             "predictions": result_dir / "predictions_pose.mat"}
    cap = json.loads(paths["capture_manifest"].read_text())
    require(cap["schema"] == "spnv2-heatmap-pose-capture-v1" and cap["heads"] == heads,
            "Capture head/schema mismatch")
    h, e, n = "heatmap" in heads, "efficientpose" in heads, len(ids)
    require(cap["status"] == ("complete" if h else "not_applicable") and not cap["errors"] and cap["body_exception"] is None,
            "Capture is not complete/applicable")
    require(cap["sample_count"] == n and cap["solver_calls"] == cap["record_count"] == (n if h else 0),
            "Original solver call count must be exactly actual N or zero when H absent")
    require(cap["csv"] == {"path": str(csv_path), "sha256": sha(csv_path), "sample_id_sha256": digest(ids)},
            "Capture CSV/actual ID binding mismatch")
    require(cap["records"]["path"] == str(paths["capture_records"])
            and cap["records"]["sha256"] == sha(paths["capture_records"]), "Capture record hash/path mismatch")
    require(cap["observer_code_sha256"] == implementation(), "Observer implementation changed")
    require(cap["context_sha256"] == digest(cap["context"]), "Caller context binding mismatch")
    if cap["original_solver"]["file_sha256"]:
        require(sha(cap["original_solver"]["file"]) == cap["original_solver"]["file_sha256"],
                "Frozen solver file changed")
    if cap.get("inference_module_file_sha256"):
        require(sha(cap["inference_module_file"]) == cap["inference_module_file_sha256"], "Frozen inference module file changed")
    records = [json.loads(line) for line in paths["capture_records"].read_text().splitlines() if line]
    require(len(records) == (n if h else 0), "Capture record count mismatch")
    for index, record in enumerate(records):
        require(record["index"] == index and record["sample_id"] == ids[index], "Capture order/ID mismatch")
        require(type(record["reject"]) is bool, "Captured reject flag is not Boolean")
        if record["keypoints"] is not None:
            require(np.asarray(record["keypoints"]).shape == (cfg["DATASET"]["NUM_KEYPOINTS"], 2)
                    and np.isfinite(record["keypoints"]).all(), "Captured keypoint shape/values differ from config")
        else:
            require(record["missing_reason"]["keypoints"] is not None, "Missing keypoint reason absent")
        for key, shape in (("q", (4,)), ("t", (3,))):
            value = record[key]
            require(record[key + "_numeric"] is (value is not None), "Numeric availability flag mismatch")
            if value is not None:
                require(record[key + "_dtype"] in ("float32", "float64"), "Captured pose dtype missing/unsupported")
                require(np.asarray(value).shape == shape and np.isfinite(value).all(), "Invalid captured numeric pose")
                if key == "q":
                    require(abs(np.linalg.norm(value)-1) < 1e-3, "Captured quaternion must have unit norm")
            else:
                require(record[key + "_dtype"] is None, "Missing pose cannot claim a numeric dtype")
                require(record["missing_reason"][key] is not None, "Missing pose reason absent")
        require(record["numeric_pose_available"] is (record["q"] is not None and record["t"] is not None),
                "Complete numeric pose flag mismatch")
    mean = loadmat(paths["mean"], simplify_cells=True)
    per = loadmat(paths["per_sample"], simplify_cells=True)
    pred = loadmat(paths["predictions"], simplify_cells=True)
    for obj in (per, pred):
        require([str(x) for x in np.atleast_1d(obj["sample_id"]).tolist()] == ids, "MAT/CSV actual sample ID order mismatch")
    require(float(np.asarray(mean["n_samples"]).item()) == float(np.asarray(per["n_samples"]).item()) == n,
            "Original result sample count mismatch")
    accepted = np.ones(n, dtype=bool)
    if h:
        rejects = np.asarray(pred["reject"]).reshape(-1)
        require(rejects.size == n and set(rejects.tolist()) <= {0, 1, False, True}, "Invalid original reject mask")
        rejects = rejects.astype(bool)
        per_rejects = np.asarray(per["h_reject"]).reshape(-1)
        require(set(per_rejects.tolist()) <= {0, 1, False, True}, "Invalid per-sample original reject mask")
        require(np.array_equal(rejects, per_rejects.astype(bool))
                and np.array_equal(rejects, [r["reject"] for r in records]), "Original/captured reject masks differ")
        accepted = ~rejects
        require(float(np.asarray(mean["heatmap_valid"]).item()) == float(accepted.sum()), "Original H valid denominator changed")
        require(float(np.asarray(mean["h_reject"]).item()) == float(rejects.sum()), "Original rejection denominator mismatch")
        for key, shape in (("heat_q", (n, 4)), ("heat_t", (n, 3))):
            values = np.asarray(pred[key]).reshape(shape)
            require(np.isnan(values[~accepted]).all(), "Original rejected pose must remain NaN; no zero filling")
            captured = np.asarray([r["q" if key == "heat_q" else "t"] if r["q" if key == "heat_q" else "t"] is not None
                                   else [np.nan] * shape[1] for r in records], dtype=np.float32)
            require(np.array_equal(values[accepted], captured[accepted]), "Accepted pose changed versus original MAT")
    valid = np.asarray(per["final_pose_valid"]).reshape(-1)
    require(set(valid.tolist()) <= {0, 1, False, True}, "Invalid original primary mask")
    require(np.array_equal(valid.astype(bool), accepted),
            "Original primary accepted subset changed")
    efficient = {}
    if e:
        efficient["R"] = np.asarray(pred["effi_R"]).reshape(n, 3, 3)
        efficient["t"] = np.asarray(pred["effi_t"]).reshape(n, 3)
        require(np.isfinite(efficient["R"]).all() and np.isfinite(efficient["t"]).all(), "Original E numeric pose missing/nonfinite")
        require(np.allclose(efficient["R"] @ efficient["R"].transpose(0, 2, 1), np.eye(3), atol=1e-5)
                and np.allclose(np.linalg.det(efficient["R"]), 1, atol=1e-5), "Original E rotation is not a proper DCM")
    domain = cfg["TEST"]["TEST_CSV"].split("/")[0]
    return {"ids": ids, "gt": ground_truth, "cfg": cfg, "heads": heads, "h": h, "e": e,
            "records": records, "cap": cap, "mean": mean, "per": per, "pred": pred,
            "accepted": accepted, "efficient": efficient, "domain": domain,
            "paths": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()}}


def statistics(values, total, *, mask_name, original=False):
    available = [float(x) for x in values if x is not None]
    require(all(math.isfinite(x) for x in available), "Nonfinite extended numerical metric")
    count = len(available)
    return {"mean": float(np.mean(available, dtype=np.float64)) if count else None,
            "count": count, "total": total, "missing": total - count, "coverage": count / total,
            "denominator_scope": "original_accepted_subset" if original else
                ("all_samples" if count == total else "numeric_pose_available_subset"), "mask": mask_name}


def calculate(data, metric_function=pose_metrics):
    n, h, e = len(data["ids"]), data["h"], data["e"]
    threshold = data["domain"] in ("lightbox", "sunlamp")
    options = {"threshold": threshold, "theta_q": data["cfg"]["TEST"]["SPEED_THRESHOLD_Q"],
               "theta_t": data["cfg"]["TEST"]["SPEED_THRESHOLD_T"]}
    rows = []
    for index, (ident, (q_gt, t_gt)) in enumerate(zip(data["ids"], data["gt"])):
        record = data["records"][index] if h else None
        heat = metric_function(
            np.asarray(record["q"], dtype=record["q_dtype"]) if record["q"] is not None else None,
            np.asarray(record["t"], dtype=record["t_dtype"]) if record["t"] is not None else None, q_gt, t_gt, **options) if h else None
        eff = metric_function(None, data["efficient"]["t"][index], q_gt, t_gt,
                              rotation_matrix=data["efficient"]["R"][index], **options) if e else None
        hybrid = None
        if h and e:
            hybrid = {"eR": heat["eR"], "eR_HIL": heat["eR_HIL"], "eT": eff["eT"],
                      "eT_norm_HIL": eff["eT_norm_HIL"],
                      "pose": heat["eR_HIL"] + eff["eT_norm_HIL"] if heat["eR_HIL"] is not None else None}
        accepted = bool(data["accepted"][index])
        for prefix, values, mask in (("heat", heat, accepted), ("effi", eff, True)):
            if values is None:
                continue
            for key, expected in values.items():
                actual = float(np.asarray(data["per"][prefix + "_" + key]).reshape(-1)[index])
                if mask:
                    require(expected is not None, "Original accepted H pose has no numerical value")
                    same_numeric(actual, expected, prefix + "_" + key + "/" + ident)
                else:
                    require(math.isnan(actual), "Original rejected metric no longer NaN")
        original = hybrid if hybrid is not None else (heat if h else eff)
        actual_final = float(np.asarray(data["per"]["final_pose"]).reshape(-1)[index])
        if accepted:
            same_numeric(actual_final, original["pose"], "original_final/" + ident)
        else:
            require(math.isnan(actual_final), "Original rejected final score no longer NaN")
        rows.append({
            "index": index, "sample_id": ident, "original_accepted": accepted,
            "original_reject": bool(data["records"][index]["reject"]) if h else None,
            "H_numeric_rotation": heat is not None and heat["eR"] is not None,
            "H_numeric_translation": heat is not None and heat["eT"] is not None,
            "H_numeric_pose": heat is not None and heat["pose"] is not None,
            "H_E_intersection": hybrid is not None and hybrid["pose"] is not None,
            "geometry_rejected_numeric_H": h and not accepted and heat["pose"] is not None,
            "extended_H": heat, "original_E": eff, "extended_hybrid": hybrid,
        })
    protocols = {}
    for label, key, applicable, original_subset in (
        ("original_accepted_H", "extended_H", h, True),
        ("original_E", "original_E", e, False),
        ("original_H_E_intersection", "extended_hybrid", h and e, True),
        ("extended_H", "extended_H", h, False),
        ("extended_hybrid", "extended_hybrid", h and e, False),
    ):
        if not applicable:
            protocols[label] = {"applicable": False, "reason": "required_head_not_present", "metrics": None}
            continue
        metrics = {}
        for metric, unit in UNITS.items():
            values = [r[key][metric] if not original_subset or r["original_accepted"] else None for r in rows]
            summary = statistics(values, n, mask_name=label + ":" + metric, original=original_subset)
            summary["unit"] = unit
            metrics[metric] = summary
        protocols[label] = {"applicable": True, "metrics": metrics}
    for label, prefix, mask in (("original_accepted_H", "heat", data["accepted"]),
                                ("original_E", "effi", np.ones(n, dtype=bool))):
        if not protocols[label]["applicable"]:
            continue
        for metric in UNITS:
            values = np.asarray(data["per"][prefix + "_" + metric]).reshape(-1)
            expected = float(data["mean"][prefix + "_" + metric])
            if mask.any():
                same_numeric(values[mask].mean(dtype=np.float64), expected, "original_mean_" + prefix + "_" + metric)
            else:
                require(np.isnan(expected), "Zero original denominator must stay missing")
    baseline = np.asarray(data['per']['final_pose']).reshape(-1)
    if data['accepted'].any():
        same_numeric(baseline[data['accepted']].mean(dtype=np.float64), float(data['mean']['final_pose']), 'original_mean_final_pose')
    else:
        require(np.isnan(float(data['mean']['final_pose'])), 'Zero original primary denominator must stay missing')
    require(float(np.asarray(data['mean']['final_pose_valid']).item()) == float(data['accepted'].sum()), 'Original primary mean denominator changed')
    mode = "H_rotation_E_translation" if h and e else ("H_rotation_H_translation" if h else "E_only_original")
    primary = protocols["extended_hybrid"] if h and e else (protocols["extended_H"] if h else protocols["original_E"])
    counts = {key: sum(bool(r[key]) for r in rows) for key in
              ("original_accepted", "H_numeric_rotation", "H_numeric_translation", "H_numeric_pose",
               "H_E_intersection", "geometry_rejected_numeric_H")}
    counts["original_accepted_H_E_intersection"] = sum(r["original_accepted"] and r["H_E_intersection"] for r in rows)
    counts.update(total=n, missing_H_rotation=n - counts["H_numeric_rotation"] if h else None,
                  missing_H_translation=n - counts["H_numeric_translation"] if h else None,
                  heatmap_applicable=h, efficientpose_applicable=e)
    return rows, protocols, primary, mode, counts


def write_idempotent(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    if path.exists():
        require(path.read_text() == text, "Refuse overwrite of different accepted extension/proof")
    else:
        atomic(path, value)


def build_extension(*, capture_dir, result_dir, csv_path, cfg_path, output_dir, context=None):
    data = load_inputs(capture_dir=capture_dir, result_dir=result_dir, csv_path=csv_path, cfg_path=cfg_path)
    require(context is None or context == data["cap"]["context"], "Producer context differs from capture")
    rows, protocols, primary, mode, counts = calculate(data)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_path = output / "extended_per_sample.jsonl"
    raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    if per_path.exists():
        require(per_path.read_text() == raw, "Refuse overwrite of different extended per-sample evidence")
    else:
        with per_path.open("x") as stream:
            stream.write(raw)
    path = output / "extended_report.json"
    report = {
        "schema": "spnv2-extended-evaluation-v1", "status": "produced_pending_independent_cpu_verification",
        "report_path": str(path), "per_sample_path": str(per_path), "per_sample_sha256": sha(per_path),
        "inputs": data["paths"], "heads": data["heads"], "domain": data["domain"],
        "sample_count": len(rows), "sample_id_sha256": digest(data["ids"]),
        "context": data["cap"]["context"], "context_sha256": data["cap"]["context_sha256"],
        "observer_code_sha256": implementation(), "counts": counts, "protocols": protocols,
        "primary_mode": mode, "extended_primary": primary,
        "original_accepted_subset_unchanged": True,
        "interpretation": "Numerical geometry-rejected H values are additional evidence only; None remains missing, no E replacement for H or zero filling; all_samples only for full N/N coverage",
    }
    write_idempotent(path, report)
    return report
