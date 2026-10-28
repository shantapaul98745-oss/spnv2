# Trainable BN affine acceptance, 2026-10-28

Explicit user authorization changes only cross-GPU acceptance of actually trainable backbone BatchNorm weight/bias toatol1e-5/rtol1e-5. Joint original/current requires_grad, model/buffer identities,1D FP32 shape and corresponding running statistics qualify exact learner.learning/window_base.model leaves. Missing context fails closed to old tolerance. No parameter, tensor, model/optimizer math, other tolerance or GT rule changes.

The runner/verifier use complete snapshots. Diagnostics may provide the already-read original complete snapshots as read-only qualification context for model subtrees; they do not replace or alter compared values. Effective-atol summaries use the same selector.

revision_v13 binds policy_v13.json, numeric_compare.py and unchanged saved-main recovery. resume_BN_affine_20261028.py requires the exact current sunlamp2022/w1 real failure, closed continuous budget and free original locks. It preserves complete64checkpoint/candidate/two128panels and old failure costs, finalizes zero-forward new main, then requires fresh independent proof.32accepted pairs retain original hashes. Evidence lives in D/revisions/numeric_acceptance_20261027/v13_trainable_BN_affine; versions/v12 preserves changed predecessor files.
