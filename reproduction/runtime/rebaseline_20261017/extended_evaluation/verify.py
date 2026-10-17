"""Independent CPU proof for immutable numerical extension evidence."""
import json
import math
from pathlib import Path
import numpy as np
from .capture import sha, digest, implementation
from .report import load_inputs, require, UNITS, write_idempotent


def independent_metrics(q, t, qgt, tgt, *, rotation_matrix=None, threshold, theta_q, theta_t):
    out = dict.fromkeys(UNITS)
    if rotation_matrix is not None:
        # Reproduce the frozen float32 GT DCM and matrix product, then scalar math.
        q0, q1, q2, q3 = np.asarray(qgt, dtype=np.float32) / np.linalg.norm(qgt)
        gt = np.zeros((3, 3), dtype=np.float32)
        gt[0, 0] = 2*q0**2-1+2*q1**2
        gt[1, 1] = 2*q0**2-1+2*q2**2
        gt[2, 2] = 2*q0**2-1+2*q3**2
        gt[0, 1], gt[0, 2] = 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2
        gt[1, 0], gt[1, 2] = 2*q1*q2-2*q0*q3, 2*q2*q3+2*q0*q1
        gt[2, 0], gt[2, 1] = 2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1
        product = np.dot(np.asarray(rotation_matrix, dtype=np.float32), gt.T)
        cosine = (float(np.trace(product)) - 1.0) / 2.0
        angle = math.acos(max(-1.0, min(1.0, cosine)))
    elif q is not None:
        cosine = abs(math.fsum(float(a)*float(b) for a, b in zip(q, qgt)))
        angle = 2*math.acos(max(0.0, min(1.0, cosine)))
    else:
        angle = None
    if angle is not None:
        out["eR"] = math.degrees(angle)
        out["eR_HIL"] = 0.0 if threshold and out["eR"] < theta_q else angle
    if t is not None:
        # Independently stage each operation at the frozen dtype, including the
        # float32 GT denominator. Scalar fsum/double norm changes HIL decisions.
        dtype = np.result_type(np.asarray(t).dtype, np.asarray(tgt).dtype).type
        terms = [dtype(dtype(a)-dtype(b)) for a,b in zip(tgt,t)]
        squares = [dtype(x*x) for x in terms]
        squared = dtype(dtype(squares[0]+squares[1])+squares[2])
        distance = dtype(np.sqrt(squared))
        gt_type = np.asarray(tgt).dtype.type
        gt_squares = [gt_type(gt_type(x)*gt_type(x)) for x in tgt]
        gt_squared = gt_type(gt_type(gt_squares[0]+gt_squares[1])+gt_squares[2])
        denominator = gt_type(np.sqrt(gt_squared))
        normalized = dtype(distance/denominator)
        out["eT"] = float(distance)
        out["eT_norm_HIL"] = 0.0 if threshold and normalized < theta_t else float(normalized)
    if out["eR_HIL"] is not None and out["eT_norm_HIL"] is not None:
        out["pose"] = out["eR_HIL"] + out["eT_norm_HIL"]
    return out


def equal(left, right, label):
    if isinstance(right, dict):
        require(isinstance(left, dict) and set(left) == set(right), "Object fields differ: " + label)
        for key in right:
            equal(left[key], right[key], label + "/" + key)
    elif isinstance(right, list):
        require(isinstance(left, list) and len(left) == len(right), "List differs: " + label)
        for i, value in enumerate(right):
            equal(left[i], value, label + "/" + str(i))
    elif isinstance(right, bool) or right is None or isinstance(right, (str, int)):
        require(type(left) is type(right) and left == right, "Exact evidence differs: " + label)
    else:
        require(isinstance(left, (int, float)) and not isinstance(left, bool) and math.isfinite(left)
                and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9),
                "Independent numerical recomputation differs: " + label)


def recompute(data):
    n, h, e = len(data["ids"]), data["h"], data["e"]
    cfg = data["cfg"]["TEST"]
    opts = dict(threshold=data["domain"] in ("lightbox", "sunlamp"),
                theta_q=cfg["SPEED_THRESHOLD_Q"], theta_t=cfg["SPEED_THRESHOLD_T"])
    rows = []
    for i in range(n):
        qgt, tgt = data["gt"][i]
        record = data["records"][i] if h else None
        hm = independent_metrics(
            np.asarray(record["q"], dtype=record["q_dtype"]) if record["q"] is not None else None,
            np.asarray(record["t"], dtype=record["t_dtype"]) if record["t"] is not None else None, qgt, tgt, **opts) if h else None
        em = independent_metrics(None, data["efficient"]["t"][i], qgt, tgt,
                                 rotation_matrix=data["efficient"]["R"][i], **opts) if e else None
        mix = None
        if h and e:
            mix = {k: hm[k] if k in ("eR", "eR_HIL") else em[k] for k in UNITS}
            mix["pose"] = None if hm["eR_HIL"] is None else hm["eR_HIL"] + em["eT_norm_HIL"]
        accepted = bool(data["accepted"][i])
        for prefix, vals, active in (("heat", hm, accepted), ("effi", em, True)):
            if vals is None:
                continue
            for key, value in vals.items():
                original = float(np.asarray(data["per"][prefix+"_"+key]).reshape(-1)[i])
                require((value is not None and math.isclose(original, value, rel_tol=2e-5, abs_tol=2e-5)) if active
                        else math.isnan(original), "Original per-ID metric changed")
        chosen = mix if h and e else (hm if h else em)
        original = float(np.asarray(data["per"]["final_pose"]).reshape(-1)[i])
        require(math.isclose(original, chosen["pose"], rel_tol=2e-5, abs_tol=2e-5) if accepted
                else math.isnan(original), "Original per-ID primary changed")
        rows.append(dict(index=i, sample_id=data["ids"][i], original_accepted=accepted,
                         original_reject=bool(data["records"][i]["reject"]) if h else None,
                         H_numeric_rotation=bool(h and hm["eR"] is not None),
                         H_numeric_translation=bool(h and hm["eT"] is not None),
                         H_numeric_pose=bool(h and hm["pose"] is not None),
                         H_E_intersection=bool(h and e and mix["pose"] is not None),
                         geometry_rejected_numeric_H=bool(h and not accepted and hm["pose"] is not None),
                         extended_H=hm, original_E=em, extended_hybrid=mix))
    protocols = {}
    definitions = [("original_accepted_H", "extended_H", h, True),
                   ("original_E", "original_E", e, False),
                   ("original_H_E_intersection", "extended_hybrid", h and e, True),
                   ("extended_H", "extended_H", h, False),
                   ("extended_hybrid", "extended_hybrid", h and e, False)]
    for label, field, active, subset in definitions:
        if not active:
            protocols[label] = dict(applicable=False, reason="required_head_not_present", metrics=None)
            continue
        metrics = {}
        for metric, unit in UNITS.items():
            vals = [row[field][metric] for row in rows if not subset or row["original_accepted"]]
            numeric = [float(x) for x in vals if x is not None]
            count = len(numeric)
            metrics[metric] = dict(mean=math.fsum(numeric)/count if count else None, count=count,
                                  total=n, missing=n-count, coverage=count/n,
                                  denominator_scope="original_accepted_subset" if subset else
                                  ("all_samples" if count == n else "numeric_pose_available_subset"),
                                  mask=label+":"+metric, unit=unit)
        protocols[label] = dict(applicable=True, metrics=metrics)
    for prefix, active, subset in (("heat", h, True), ("effi", e, False), ("", True, True)):
        if not active:
            continue
        keys = ["final_pose"] if not prefix else [prefix+"_"+key for key in UNITS]
        for key in keys:
            vals = np.asarray(data["per"][key]).reshape(-1)
            selected = [float(vals[i]) for i in range(n) if not subset or data["accepted"][i]]
            mean = float(data["mean"][key])
            require(math.isclose(mean, math.fsum(selected)/len(selected), rel_tol=2e-5, abs_tol=2e-5)
                    if selected else math.isnan(mean), "Original mean/subset denominator changed")
    require(float(np.asarray(data["mean"]["final_pose_valid"]).item()) == sum(data["accepted"]),
            "Original mean primary denominator changed")
    counts = {key: sum(row[key] for row in rows) for key in
              ("original_accepted", "H_numeric_rotation", "H_numeric_translation", "H_numeric_pose",
               "H_E_intersection", "geometry_rejected_numeric_H")}
    counts["original_accepted_H_E_intersection"] = sum(r["original_accepted"] and r["H_E_intersection"] for r in rows)
    counts.update(total=n, missing_H_rotation=n-counts["H_numeric_rotation"] if h else None,
                  missing_H_translation=n-counts["H_numeric_translation"] if h else None,
                  heatmap_applicable=h, efficientpose_applicable=e)
    mode = "H_rotation_E_translation" if h and e else ("H_rotation_H_translation" if h else "E_only_original")
    primary = protocols["extended_hybrid" if h and e else ("extended_H" if h else "original_E")]
    return rows, protocols, primary, mode, counts


def independent_cpu_verify(report_path, *, output_path):
    report_path, output_path = Path(report_path).resolve(), Path(output_path).resolve()
    report = json.loads(report_path.read_text())
    require(report["schema"] == "spnv2-extended-evaluation-v1"
            and report["status"] == "produced_pending_independent_cpu_verification", "Extension schema/status")
    require(report["report_path"] == str(report_path), "Extension report path changed")
    inputs = report["inputs"]
    for item in inputs.values():
        require(sha(item["path"]) == item["sha256"], "Bound input file hash changed")
    data = load_inputs(capture_dir=Path(inputs["capture_manifest"]["path"]).parent,
                       result_dir=Path(inputs["mean"]["path"]).parent,
                       csv_path=inputs["csv"]["path"], cfg_path=inputs["cfg"]["path"])
    equal(inputs, data["paths"], "inputs")
    equal(report["observer_code_sha256"], implementation(), "observer_code_sha256")
    equal(report["context"], data["cap"]["context"], "context")
    equal(report["context_sha256"], digest(report["context"]), "context_sha256")
    equal(report["heads"], data["heads"], "heads")
    equal(report["domain"], data["domain"], "domain")
    equal(report["sample_count"], len(data["ids"]), "sample_count")
    equal(report["sample_id_sha256"], digest(data["ids"]), "sample_id_sha256")
    require(report["original_accepted_subset_unchanged"] is True, "Original subset preservation claim changed")
    rows, protocols, primary, mode, counts = recompute(data)
    equal(report["counts"], counts, "counts")
    equal(report["protocols"], protocols, "protocols")
    equal(report["extended_primary"], primary, "extended_primary")
    equal(report["primary_mode"], mode, "primary_mode")
    raw_path = Path(report["per_sample_path"]).resolve()
    require(raw_path.parent == report_path.parent and raw_path.name == "extended_per_sample.jsonl",
            "Unexpected extended per-sample path")
    require(sha(raw_path) == report["per_sample_sha256"], "Extended per-sample SHA changed")
    actual = [json.loads(line) for line in raw_path.read_text().splitlines()]
    equal(actual, rows, "extended_per_sample")
    proof = dict(schema="spnv2-extended-evaluation-cpu-verification-v1", status="passed",
                 report_path=str(report_path), report_sha256=sha(report_path),
                 per_sample_sha256=sha(raw_path), inputs=inputs,
                 context_sha256=report["context_sha256"], observer_code_sha256=implementation(),
                 sample_count=len(data["ids"]), sample_id_sha256=digest(data["ids"]),
                 counts=counts, primary_mode=mode,
                 scope="Independent CPU scalar metrics/coverage, exact original sample IDs/accepted poses and immutable input/code/context hashes; no GPU inference or missing-pose substitution")
    require(output_path != report_path and output_path != raw_path
            and str(output_path) not in {item["path"] for item in inputs.values()}, "Proof cannot overwrite source evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_idempotent(output_path, proof)
    return proof
