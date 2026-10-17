# Extended numerical pose evidence v1

This package observes the frozen heatmap solver once per real image and reads the original three MAT exports. It does not launch training, evaluation, ODR or a GPU process. Source/ODR wrappers own execution, original artifact acceptance, expected context and alias mapping.

## Stable package API

Load the directory as a package (an isolated package name is supported). Internal imports are relative, so no module named `capture`, `report` or `common` is inserted at top level.

```python
from extended_evaluation.capture import observe_heatmap_poses
from extended_evaluation.report import build_extension
from extended_evaluation.verify import independent_cpu_verify

with observe_heatmap_poses(
    inference_module, csv_path=actual_csv, output_dir=new_capture_dir,
    heads=actual_cfg_heads, context=bounded_json_context,
):
    original_entrypoint()  # original inference_module.solve_pose_from_heatmaps resolves here

report = build_extension(
    capture_dir=new_capture_dir, result_dir=original_three_mat_directory,
    csv_path=actual_csv, cfg_path=actual_cfg, output_dir=new_extension_dir,
    context=bounded_json_context,
)
proof = independent_cpu_verify(
    report["report_path"], output_path=new_extension_dir / "extended_verified.json",
)
```

The observer object is a context manager. It calls the original solver exactly once and returns the identical tuple/list object; numerical copies do not modify the returned tensors or arrays. It has no model reference, forward call, random operation or CUDA call. It restores the original symbol on body exceptions. Record/write/count errors preserve the original result, then prevent successful capture at context exit. A body exception propagates unchanged and produces failed capture evidence where writable. Nested observation on one inference module is refused.

`context` is an optional finite JSON **object**, copied at entry, hashed and retained unchanged. There are no mandatory field names inside this low-level package. The integration wrapper must supply and validate its own bounded evidence: original/base entry SHA, actual config/final/source acceptance proof SHA, frozen source/line manifest, observer implementation and wrapper protocol manifest. Context is part of both extension and independent proof; it cannot replace checking original source/ODR acceptance. Passing context to `build_extension` requires exact equality with the capture context.

## Capture files

A capture directory must not exist. It contains `capture_manifest.json` (schema `spnv2-heatmap-pose-capture-v1`) and `heatmap_pose_capture.jsonl`. A heatmap record has exact CSV ordinal/ID, keypoints, q/t, numeric flags, reject flag and explicit missing reasons. The manifest binds CSV and ordered IDs, records, solver/inference source files, all package Python files and caller context. With H present, calls and records must equal actual N. Any missing, extra or reordered call/record prevents acceptance.

E-only still installs a guard to detect accidental solver calls; it creates no H records. Normal E-only capture has status `not_applicable`, `heatmap_applicable=false`, and zero calls/records. H captures have status `complete`; capture errors have status `failed`.

## Extension report and metrics

`build_extension` returns the JSON object also written to `output_dir/extended_report.json`; deterministic per-image data goes to `output_dir/extended_per_sample.jsonl`. Report schema is `spnv2-extended-evaluation-v1`, status `produced_pending_independent_cpu_verification`. A report does not claim independent acceptance.

The report binds all original MAT files, actual CSV/config, capture manifest/JSONL, context, implementation and ordered IDs. Original accepted q/t are reconciled exactly after the frozen float32 export conversion; rejected original predictions and scores must remain NaN. Original per-image metrics and accepted-subset means are reconciled within 2e-5 absolute/relative tolerance for floating-point arithmetic. No original file is written.

Five protocols remain separate:
- `original_accepted_H`: original H accepted subset.
- `original_E`: original E outputs over all actual images.
- `original_H_E_intersection`: original accepted H combined with E.
- `extended_H`: numeric H components including geometric rejects.
- `extended_hybrid`: numeric H rotation and original E translation.

Each applicable protocol has `metrics` keys `eR` (deg), `eT` (m), `eR_HIL` (rad), `eT_norm_HIL` (m/m), `pose` (sum score). HIL thresholds are actual config values and apply only to lightbox/sunlamp. Each metric explicitly reports mean/count/total/missing/coverage/denominator_scope/mask/unit. Count zero has null mean. `all_samples` is allowed only when the relevant count is N; otherwise the extension says `numeric_pose_available_subset`. Original accepted protocols always say `original_accepted_subset`.

`primary_mode` is `H_rotation_E_translation` for H+E, `H_rotation_H_translation` for H-only, or `E_only_original`. `extended_primary` selects the corresponding protocol. H absence is `applicable=false, reason=required_head_not_present, metrics=null`, and missing H counts are null. No H value is replaced with E or zero. Different components can have different numerical coverage; primary score requires both selected components.

Top-level counts include total, original_accepted, H_numeric_rotation, H_numeric_translation, H_numeric_pose, H_E_intersection, geometry_rejected_numeric_H, original_accepted_H_E_intersection, missing_H_rotation/translation, heatmap_applicable and efficientpose_applicable. Per-image JSONL records carry the original accept/reject and component availability flags plus separate H/E/hybrid metrics.

## Independent CPU proof and aliases

`independent_cpu_verify(report_path, *, output_path)` returns and writes schema `spnv2-extended-evaluation-cpu-verification-v1`, status `passed`. There is no `summary` field. Stable proof fields: `report_path`, `report_sha256`, `per_sample_sha256`, `inputs` (each path+sha256), `context_sha256`, `observer_code_sha256`, `sample_count`, `sample_id_sha256`, `counts`, `primary_mode`, `scope`. It independently recalculates scalar pose values, masks, counts and means, validates the producer at 1e-9 precision, and rechecks original evidence. Shared parsing only enforces input shape/order/hash invariants. Both numerical implementations preserve the frozen quaternion/DCM convention.

Producer/report/proof files have no generated timestamps. Existing identical outputs are returned without rewriting (SHA and mtime stay unchanged); different existing outputs are refused with ValueError. Proof cannot overwrite any input or its report/JSONL. Other invalid evidence also raises ValueError (malformed structures may additionally raise KeyError/TypeError). Capture contract errors raise CaptureError.

Alias wrappers should reference the canonical report and independently passed proof by path+SHA, rerun this verifier against the **canonical** report with the same canonical output path, and validate alias/canonical protocol/source identity themselves. Do not copy or mutate canonical context/paths to invent an alias extension, and do not rewrite proof timestamps. The frozen source/ODR verifier remains required separately.

## CPU validation scope

Run the test script only in a unique guarded screen with CUDA_VISIBLE_DEVICES empty and PYTHONNOUSERSITE=1; logs, exitcode and hash-bound JSON summary stay in this subtree. Tests use a real constructed heatmap/PnP and frozen do_valid with a tiny deterministic model plus controlled PnP branches. They prove exact MAT results/return identity, forward count/model state/CPU random state, exception restoration, H/E/EH denominators, missing values, idempotence and negative corruption contracts. They are engineering CPU evidence, not source accuracy, real ODR or GPU inference evidence. Initial failing runs remain preserved.

## Release v2: preserve frozen dtype staging at HIL boundaries

The previous 18-pass implementation is byte-for-byte archived under history/release_18pass_before_dtype_fix/. A read-only CPU probe found four true boundary counterexamples using real lightbox GT, at the actual normalized-translation threshold 0.002173 and at 0.005. Computing every norm in float64 can change a thresholded value by the entire threshold, causing false rejection of unchanged original MATs.

Capture records now retain q_dtype/t_dtype (float32 or float64, null for missing). H solver values retain their original precision; E is read from the original float32 MAT. Translation follows the frozen subtract → square → sum → sqrt staging at the promoted prediction/GT dtype, and the GT norm remains float32. Division and the original strict less-than threshold happen before conversion to JSON floats. The independent verifier models each scalar rounding stage separately. This changes no tolerance: original reconciliation remains 2e-5, independent producer agreement 1e-9. Four exact dtype/byte boundary cases are regression tested through both direct mathematics and frozen inference/MAT export, in addition to the original 18 tests.
