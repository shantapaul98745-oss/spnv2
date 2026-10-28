# Two authorized BN-mean tolerances, 2026-10-28

Only learner.learning.bn_mean and learner.window_base.bn_mean use atol1e-5. Their bn_features channel_mean uses atol2e-5. Both keep rtol1e-5. Mean-square, true running buffers, parameters/optimizer/gradients and every other numerical/exact/GT boundary remain unchanged. Preserve original raw tensors, failures, previous standards and costs.

Runtime mapping: revision_v9.policy ->policy_v9.json; comparator ->numeric_compare.py; unchanged recovery_code ->recover_saved_main.py. resume_BN_two_means_20261028.py verifies exact V8 baseline/realfailedexit/fourlocks/continuousledger, protects saved complete512state and both128panels, replaces partial recovery with complete saved_main_recovery, retains both actual failed attempts, and prepares fresh main/independent verification.

D/revisions/numeric_acceptance_20261027/v9_BN_two_mean_tolerances retains before snapshots, recovery manifest, targeted checks and actual preparation/validation/launch receipts. versions/v8 preserves the executed previous comparator/pair runner.29 accepted pairs retain their original bindings. Recovery and independent verification must add zero model forwards for w54; later windows run in original frozen order under the unchanged40000physical cap and no cumulative-time hard stop.
