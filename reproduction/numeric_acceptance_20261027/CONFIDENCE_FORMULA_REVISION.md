# Confidence original-formula self-consistency, 2026-10-28

Explicit user authorization applies only to confidence in full typed pose-prediction records. Verify all11 saved peaks are finite original float32 values and confidence exactly equals float(np.quantile(peaks_float32,.25,method=linear)). Retain None when peaks unavailable. Normalize only the comparison copy after validation; persisted raw values, all original input rules and actual thresholds/decisions remain unchanged. Original G50 self-consistency is checked before any nested confidence normalization.

Runtime mapping: revision_v11.policy ->policy_v11.json; comparator ->numeric_compare.py. resume_confidence_20261028.py verifies the exact V10 stopped w84 and original locks/continuous ledger, retains both actual failed attempts, protects complete704checkpoint and prepares the ordinary frozen queue.30accepted proofs remain untouched. Old comparator is versions/v10/numeric_compare.py.

Evidence: D/revisions/numeric_acceptance_20261027/v11_derived_confidence retains prior index/status/failure, partial recovery manifest, targeted checks and real validation/launch receipts. W84 had no saved candidate/panel before resumption. Previous77+222failedforwards stay charged; only unpersisted work is replayed. Main completion plus independent verification remain required.
