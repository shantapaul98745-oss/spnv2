# SYSU namespace/bootstrap candidate (not deployed)

`namespace_runner.py` constructs a rootless bwrap argv preserving the historical
absolute paths. Its default CLI only prints the candidate command. The sole
execution option, `--run-probe`, runs bounded metadata code. It refuses scientific
execution; a future admitted controller must consume the explicitly gated argv.
No concrete scientific runtime config is included in this change.

`bootstrap.py --probe` checks its own/launcher hashes and all five extension
bindings, reports zero forwards, and installs no storage hook. A scientific
bootstrap requires both `source_dispatch_fenced: true` and
`target_scientific_dispatch_authorized: true`, an immutable source fence receipt,
and a final budget handoff receipt. Incomplete config fails before the original
entry can execute. These flags are rollout decisions, never inferred from a
healthy or stale controller status.

## Namespace boundaries

The root filesystem is read-only; a fixed `--dev-bind /dev /dev` immediately
after that root bind preserves access to the same host devices under the same
unprivileged UID, including /dev/null, /dev/pts and admitted GPU devices.
`/home` is a private tmpfs. The source
snapshot maps to `/home/DATA/Downloads/UsrSourceCode/LuXinghai`; the versioned
target repository, target environment and dataset retain their verified aliases
as read-only binds. A read-only `runtime_control` mount exposes immutable config
and receipts. Only project temp/cache are writable in probe mode.

After a verified owned staging rename, `paths.source_snapshot` may be canonical
U itself, exactly `/home/ps/Downloads/UsrSourceCode/LuXinghai`. Legacy S remains
supported. Science in canonical mode requires `asset_layout_receipt` and
`asset_transfer_manifest` bindings. Receipt schema is
`spnv2-canonical-asset-layout-v1`, status `verified_renamed_immutable_assets`;
target_root is exact U, source_staging is original S, transfer_manifest_sha256
equals the actual bound manifest hash, verified_file_count equals its `files`
length (276 for this transfer), and no_file_content_changes/source_dispatch_fenced
are both true. This code performs no move/copy or model-state rehash.

An optional `frozen_image_cache` explicitly names `source`, `target`, `manifest`
and `receipt`. It supports only the project-owned
`U/spnv2_runs/frozen/required_images_20261027/sunlamp/images_768x512_RGB` at the
original DATA/sunlamp/images_768x512_RGB path. After the dataset root RO bind,
the launcher overlays a private tmpfs on logical DATA/sunlamp, then rebinds the
existing native `sunlamp/images` and `test.json` RO and the exact-byte derived
cache RO. No mountpoint is created in actual shared dataset storage. Science
requires the bound `spnv2-required-image-spec-v1` manifest and a bound
`spnv2-required-image-cache-v1` receipt, status `verified_exact_frozen_bytes`,
matching `manifest_sha256`, `required_image_list_sha256`, `cache_root`,
`image_count: 2171`, `total_bytes: 61849173`, and `mismatch_count: 0`. Existing
source/target initial missing-image evidence remains separate and immutable.

Scientific mode adds a writable bind for the owned diagnostic D directory.
Read-only overlays protect `implementation`, all old `tasks`/`revisions`,
`panels`, `inputs`, `A`, `cost_A`, `validation`, `preflight`, `index.json`,
`frozen_inputs.json`, `protocol.json`, `budget/identity.json`, and every accepted
pair directory. Only `tasks/<new_task_revision>` is writable beneath old tasks.
Original N/source-A assets remain below the read-only snapshot mount.

The four historical pipeline lock files bind to canonical target
`U/spnv2_runs/plans/<same-plan>/pipeline.lock`, not private migration locks, so
future native project processes share coordination. Their source files and
snapshot destination placeholders must exist before rollout. Individual writable
control-file binds are not used: original `atomic_json` needs a writable parent
for temporary-file creation and `os.replace`.

## Runtime config contract

Schema is `spnv2-sysu-namespace-config-v1`, migration ID `sysu_20261027`. `paths`
contains exactly `target_root`, `source_snapshot`, `repository`, `environment`,
`logical_environment`, `dataset`, `logical_dataset`, `tmp`, `cache`,
`logical_root`, `control_root`, and `logical_control_root`. The target root is U;
snapshot/control roots are `U/spnv2_runs/migrations/sysu_20261027/source_snapshot`
and `.../runtime_control`. Config is placed at
`runtime_control/runtime_config.json`; its SHA256 is an explicit launch argument.

Top-level `screen_socket_dir` must equal `paths.tmp + "/screen"`. The directory
must already exist, be canonical, belong to UID1000 and have mode0700. Builder
exports `SCREENDIR` there on the RW project TMP bind, allowing independent nested
worker screens to create their sockets without writing /run. It preserves the
outer manager's inherited STY; the original screen wrapper creates each fresh
worker screen/STY and original host_entry still requires it.

`code_bindings` binds exact logical paths and SHA256 for `namespace_runner`,
`bootstrap`, `dispatch_config`, `storage_adapter` (`storage_adapter_v1.py`), and existing
`storage_policy`. Every other immutable binding is also an exact
`{"path": ..., "sha256": ...}` object. Paths inside config use the preserved
logical namespace; the launcher maps them to physical target files for checks.

Scientific configuration additionally needs:

- `source_dispatch_fenced`, `target_scientific_dispatch_authorized`,
  `source_fence_nonce`, `source_fence_receipt`, `budget_handoff_receipt`.
- `diagnostic_index`, `effective_protocol`, `budget_policy`, `budget_identity`,
  `budget_journal`, and `original_durable_state` bindings. Stable budget paths
  remain D/budget/{identity.json,journal.jsonl}; old14400 identity bytes are kept.
- `accepted_run_ids`, exactly the final source list; `new_task_revision`;
  exact `approved_tasks`; exact `approved_stores` containing `root`, complete
  original `identity`, and validated latest-two/pin/positive disk-peak `policy`.
  Identity contains the four original required fields and preserves every extra
  original nonempty-string field; factory comparison always uses the full dict.
  Each new run_pair task requires one matching isolated_durable store. Accepted
  pairs and completed A work cannot be replayed through these approvals.

The actual source fence schema is `spnv2-source-dispatch-fenced-v1`, status
`source_dispatch_fenced`; nonce and guardian locks must match. The final handoff
schema is `spnv2-sysu-budget-handoff-v1`, migration ID `sysu_20261027`, status
`final`. It binds `source_fence_receipt`, `budget_identity`, `budget_journal`,
`diagnostic_index`, `budget_journal_bytes`, `accepted_run_ids`, and `snapshot`.
The snapshot equals the fence's `source_budget`, has no unresolved attempts,
time stopping disabled/max seconds null, and forward upper <=40000. Journal
SHA/bytes equal the fence's final source journal SHA/bytes. Accepted count equals
the source receipt. This new final receipt is still a rollout deliverable.

The source journal is an immutable *prefix*: future target charges append while
the original final prefix SHA/bytes remain fixed. Bootstrap/launcher verify that
prefix and the entire current journal hash chain, reject unresolved current
attempts and physical reservations over40000, and retain all historical charges.
The frozen GlobalBudget repeats its authoritative admission checks under the
original locks. No costs are reset, refunded, or reinterpreted.

## Import and identity integration

The scientific argv contains the original logical `host_entry.py` path as the
value of `--execute-host-entry`, plus original `--task-json`, `--index`, and
`--log-path` arguments. The real `/proc` command therefore still contains the
entry/index/task paths required by original exact-identity/watchdog checks.
The config SHA in that exact argv makes extension bindings reviewable in the
original launch record; no existing identity comparison is weakened.

Before executing the entry, bootstrap checks bound metadata, imports only
stdlib storage extensions, verifies original budget/host-entry paths and hashes,
and installs a narrow `pair_runner` import finder. It calls original
`host_entry.execute` unchanged. The original four locks, two resource snapshots,
registration and ownership check, budget attempt, and watchdog remain in order.
Only when the original entry imports `pair_runner` after admission does the hook
wrap `load_producer`.

The wrapper calls original `load_producer(task)` (full frozen50 coverage), then
original `old(name)` (file path/SHA/shadow checks). Only a successful
`old("durable_state")` with the separately bound original module installs
`storage_adapter_v1.install_on_module`. Original scientific files, frozen
identity, state validator, and serializer bytes remain unchanged. Policy lookup
approves only exact output-store roots with the exact original identity.
Independent audits use the original routes and proofs.

## Validation and remaining rollout decisions

Eleven stdlib-only namespace/hook tests cover probe argv/read-only scope, missing-fence refusal,
accepted-pair overlays/shared locks, immutable budget-prefix continuation and
unresolved-attempt refusal, original-verifier-before-hook order/tamper refusal,
exact-root/full-original-identity policy lookup, and the affected exact-image
cache overlay/rejection of a wrong logical path, and exact initialization-cache
alias/versioned config location and the manager's fixed argv/config_versions-only
control-child write scope and owned private screen socket metadata. Dynamic-task composition is documented in
`DISPATCH_CONFIG_DESIGN.md`, as is the canonical asset-layout receipt gate.
These tests do not launch bwrap, import
Torch/model/GT, read checkpoint states, or establish scientific acceptance.

Before any target science: finish/verify asset closure, bind the final transferred
ledger and reconciled state without replay, create the reviewed target task/index
revision and runtime control bundle, freeze store peaks and pins, review the
versioned manager/wrapper integration that consumes this argv, and perform the
required admitted target engineering/independent acceptance. This candidate does
not create those approvals, deploy a controller, or enable target dispatch.
