# Heat-peaks vector acceptance, 2026-10-28

Explicit user authorization: only heat_peaks in typed complete pose predictions uses its11-vector L2 distance <=1e-6+1e-5*L2(expected). No raw data, individual gate threshold/decision, coefficient or other numerical standard changes. Existing shape/dtype/finiteness/availability/order requirements remain. Model tensors, BN, rotation, exact RNG/rollback and GT rules remain.

Runtime mapping: revision_v10.comparator ->numeric_compare.py; policy ->policy_v10.json. resume_heat_peaks_20261028.py binds the exact V9 blocked w84 task and actual failed attempt, keeps the continuous budget and original locks, snapshots complete704checkpoint via immutable hardlinks, and prepares the normal frozen queue. No w84 candidate/panel existed before continuation; all30accepted results are reused.

D/revisions/numeric_acceptance_20261027/v10_heat_peaks_vector_norm contains prior index/state/failure, partial recovery manifest, targeted checks and actual metadata validation/launch receipts. versions/v9 preserves the old comparator. Prior77failedphysicalforwards stay charged. Normal main completion and independent zero-forward proof are still required; no newfullODR or scope/budget extension.
