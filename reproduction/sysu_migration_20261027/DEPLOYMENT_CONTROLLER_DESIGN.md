# Versioned target deployment controller (review before execution)

`deployment_controller.py` is a candidate server controller. This change only
writes/tests code. No live transfer, rename, runtime install, screen, manager,
model, GT read, or scientific acceptance has been performed by its author.
Three synthetic filesystem tests and the affected dynamic-verifier fixture are
the bounded validation for this implementation.

## Explicit launch contract

The default CLI only inspects a bound launch spec. Execution requires
`--execute --launch-spec <native M path> --launch-sha256 <SHA>`, an independent
outer screen, UID1000, and `SPNV2_DEPLOYMENT_NONCE` equal to the spec nonce.
Spec schema is `spnv2-sysu-deployment-controller-spec-v1`, migration ID
`sysu_20261027`. It contains:

- `controller: {path, sha256}` naming the exact executing source file.
- `nonce` and distinct-role `manager_nonce`, each32 lowercase hex characters.
- `target_scientific_dispatch_authorized: true`, set only by the reviewed
  authorized launch publication.
- `inputs`, with exactly the17 immutable path/SHA bindings below. Native paths
  are fixed under M; no model/state/GT path can be supplied as a metadata input.

| Input key | Path relative to M |
|---|---|
| asset_manifest | asset_transfer_manifest.json |
| candidate_index | runtime_candidate/index_candidate.json |
| candidate_build_receipt | runtime_candidate/build_receipt.json |
| source_fence | source_fence_receipt.json |
| reconciled_state | source_handoff_snapshot_20261027/reconciled_state.json |
| source_index | source_handoff_snapshot_20261027/index.json |
| original_candidate_index | source_handoff_snapshot_20261027/index_candidate.json |
| source_budget_identity | source_handoff_snapshot_20261027/budget/identity.json |
| source_budget_journal | source_handoff_snapshot_20261027/budget/journal.jsonl |
| image_manifest | required_images_spec_20261027.json |
| image_cache_receipt | required_image_cache_receipt.json |
| storage_plan | storage_plan_20261027.json |
| screen_probe | namespace_screen_probe.json |
| metadata_transfer | metadata_transfer_verification.json |
| supplemental_closure | supplemental_asset_closure.json |
| source_controller_status | source_handoff_snapshot_20261027/controller_status.json |
| source_fence_intent | source_fence_intent.json |

Root must refresh the candidate index's static dispatcher hash and build receipt
after the actual `_verify_<indexSHA12>` suffix correction. The old034f candidate
remains historical evidence. The controller accepts only the exact refreshed
index/build bindings supplied in its spec; it never rewrites candidate hashes.
The resulting index binds static extensions only. Template/deployment/config
hashes are downstream, so no SHA cycle is introduced.

## Finite preparation and safe continuation

1. Verify the source dispatch fence/guardian, exact final source ledger, amended
   protocol/policy, reconciled27 mains/26 accepted and source pilot admission.
   Verify the existing zero-model fixed-device/SCREENDIR screen proof,47 required
   supplemental entries, initial3967 metadata acceptance and21-store8GiB plan.
   Compare all48 original row IDs and completed rows; verify each of the21 new
   task bodies differs only in its metadata-manager implementation binding.
   Reuse the resulting bound metadata-delta receipt on continuation. Check
   bound input/code hashes again after a long transfer wait.
2. Wait300 seconds between transfer checks. Require complete status matching the
   bound276-file manifest, every per-file verified SHA/size receipt, and the
   unchanged target device/inode/size/mtime. No large asset is rehashed/read.
   Acquire `asset_receive.lock` exclusively to establish receiver exit and hold
   it across handoff. Failed transfer status stops with evidence.
3. Acquire all four native project pipeline locks and diagnostic manager lock.
   Destination may initially contain those expected coordination locks and the
   separately verified derived-image cache only. Refuse foreign files, aliases,
   owners, differing destination files and cross-filesystem moves.
4. Rename/merge owned staging plans/scripts/frozen into native U without content
   copies. Durable per-path intents/completions retain source stat/inode. Existing
   equal coordination-lock files stay at their canonical inode; the staging
   lock inode moves to a rollback archive. Other existing files are never
   overwritten. Initial staging's remaining bounded metadata moves to an owned
   archive. Recheck all276 target stat receipts, then publish the canonical
   layout receipt. Required2171-image cache remains separate and unchanged.
5. Archive original index/status/registry/manager and budget metadata before any
   runtime replace. Reject a staged source index/status not matching the final
   fence. Clear only the old completed active main with actual exit0/outcome
   evidence; install the bound reconciled state without replay. Refuse target
   boot registry allocations and any nonempty supposedly unstarted store.
6. Copy the exact final source budget prefix/identity with atomic target file
   replacement only when the old staged journal is a prefix. Unknown suffixes
   block handoff. Preserve old metadata in rollback; never reset/refund costs.
   Install only the prepared metadata manager, exact new index and21 tasks.
   Original host_entry/budget/pair_runner/producer50 and historical proofs/tasks
   remain unchanged.
7. Create common config/template/deployment and final budget-handoff receipt from
   actual bound metadata. Store policies approve only the21 new exact output
   roots, latest two and8GiB; original endpoints remain separately protected.
   Bind exact initialization-cache transfer receipt/stat and2171-image receipt.
   Create only dedicated project temp/cache/screen directories; never relax an
   existing screen socket directory's ownership/mode checks.
8. Generate exactly the generic manager's current saved-w16 verifier task,
   including `_verify_<new index SHA12>`. Compose its single-task config using
   logical J to canonical U translation. In a separate guarded namespace screen,
   recompose/compare the config, require original gates and no Torch import, and
   obtain actual wrapper exit0 plus exact nonce/UID/STY metadata receipt.
9. Build only `ns.build_manager_bwrap`. Persist its launch intent before calling
   the original guarded wrapper, then release project locks so the original
   manager can acquire them. Launch only that manager and require its exact
   PID/start/boot/full-command/nonce record. Original300-second resource waits,
   two snapshots/four locks, budget and scientific/independent proof stops govern
   every worker. First target scientific/GPU proof remains pending.

## Identity, recovery and accounting

Each controller attempt records PID, UID, starttime, boot ID, full argv/hash,
nonce, STY, spec/code bindings, elapsed wall time and terminal outcome. Per-run
status and immutable move/validation/manager intents remain under
`M/deployment_controller/<nonce>`. Same-filesystem moves resume only through
their retained intents/completions. A known live manager with the same recorded
nonce is reused; a terminal manager is reported without automatic restart.
Unresolved launch intent or allocations block a second launch.

Source repair-controller completed resource waits are preserved separately in
`pending_costs.json`, with the source status binding. The tail between its last
update and the source fence intent is explicitly a wall-time upper bound, never
model time. Migration/source-IO/probe settlement remains pending target receipts;
controller attempt measurements and logs are retained for that settlement.
This controller never writes the source journal or charges the already-settled
source retention cost a second time. Historical physical/time costs remain in
the imported exact source prefix. No synthetic passed scientific flag is made.

The controller code, spec and inputs must remain version-bound while waiting.
Root publishes/reviews the concrete launch spec and launches it; this artifact
does not authorize a second controller, modify chat schedules, or change any
scientific/data/budget rule.
