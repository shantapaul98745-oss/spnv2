# Immutable per-task SYSU dispatch configuration

This candidate extension prepares metadata only. It does not edit tasks/index,
launch a controller/worker, deserialize state, or accept a scientific result.

## Binding graph and helper API

The external deployment manifest is explicitly passed to the metadata manager:
`--deployment-manifest <logical path> --deployment-sha256 <SHA>`. Its schema is
`spnv2-sysu-deployment-v1`, `migration_id: sysu_20261027`, with immutable
`template` and `index` path/SHA bindings. Index binds static extension files in
`migration_extensions`; it does not bind template/deployment/config hashes, so
the graph has no SHA cycle.

`dispatch_config.compose_single_task(deployment_binding, task_binding,
resolve=None)` returns one namespace config. Inside the existing namespace,
omit `resolve`. Native preparation may pass a logical-to-physical path mapper;
every file remains checked against its immutable hash. The returned config has
exactly `approved_tasks=[task_binding]`; `approved_stores` contains exactly the
current main's output store, or is empty for independent verification.

`write_config_version(config, resolve=None)` writes a complete version below
`runtime_control/config_versions/<taskSHA>_<configSHA>.json`, using a fresh
pending file, fsync, atomic no-replace hard-link publication, and directory
fsync. Existing identical versions are reused; different content or aliases
fail closed. It returns the version's logical `{path, sha256}` binding. Neither
config content nor index includes the config's own filename/hash.

`namespace_runner.build_bwrap` accepts an optional `config_path` for one exact
file in config_versions. Bootstrap checks that location and the explicit config
SHA. `build_manager_bwrap(config, deployment_binding, config_binding)` constructs
only the bound D/implementation/manager.py command with its fixed index and
deployment arguments; it accepts no arbitrary program. The current index binds
the manager bytes. The manager receives an RW config_versions child overlay
over the RO control bundle; static receipts/template/deployment remain RO.
Worker admission remains original host_entry's responsibility.

## Required template metadata

The template is a normal bound namespace-config object with empty
`approved_tasks: []` and `approved_stores: []`. Common source fence, final source
ledger-prefix, protocol/policy, cache and accepted-pair bindings are copied
unchanged. It additionally contains:

- `original_implementation_sha256`: full source implementation map. Current
  index must have exactly those keys and identical values except an explicitly
  revised metadata-only `manager.py`. Every current file is checked against
  the new index. Host entry, budget, pair runner and scientific code cannot vary.
- `original_producer`: exact unchanged frozen producer object.
- `original_run_ids`: exact forty-eight unique diagnostic window IDs.
- `frozen_inputs`: unchanged binding, equal to current index and tasks.
- `store_policies`: mapping from each authorized unfinished main run ID to its
  explicit latest-two policy, peak estimate and complete pin registry.
- `new_task_revision`: equal to `index.sysu_task_revision`.

`code_bindings` and `index.migration_extensions` are exactly equal, with these
five keys: `namespace_runner`, `bootstrap`, `dispatch_config`, `storage_adapter`,
`storage_policy`. The first three map to matching `.py` names below
J/spnv2/reproduction/sysu_migration_20261027; storage_adapter maps to
`storage_adapter_v1.py` there; storage_policy maps to the existing
J/spnv2/reproduction/storage_management_20261027/storage_policy.py. Manager is
bound separately by the ordinary implementation map.

## Single-task gates

Only `run_pair`, `verify_pair`, and the existing `verify_pilots`/
`summarize_diagnostic` metadata audits can be composed. No A replay, new action,
new window, or automatic acceptance is introduced. The current task must be inside
the new technical revision and match the new implementation/policy/frozen-input
bindings. An indexed main must be one exact unique index row. Live accepted
results must retain all initially protected results, and the selected run must
remain unaccepted. A run_pair with an existing saved report is refused rather
than replayed.

For a main, the helper reads only the explicitly SHA-bound parent generation's
metadata.json and original task JSON. Format/generation/state-hash/cursors and
the original task/source/manifest identity agree. The complete identity dict,
including original extra fields, passes unchanged to one exact isolated_durable
output approval with its explicit peak/pin policy. It never opens state.bin.

For verification, run_task must bind one exact current index row. Its historical
map may differ from the source-original map only in the metadata manager SHA;
the exact old task is already hash-bound by that deployed index row. Every other
key/hash stays exact, and the current dispatched verifier map equals the new
index. This covers the saved w16 main preceding the source's later manager-only
candidate without rewriting completed evidence. The current verifier binds
that run, expected proof path and saved report; actual report/launch/outcome and
wrapper exit0 metadata agree. The report retains pending-independent status.
The original verifier still performs all scientific/identity/proof checks.

An audit must equal the exact bound index `pilot_admission_template` or
`summary_template`, with only `execution_receipts` and `index_path` added. Its
task path equals the explicit versioned `audit_task_paths` entry. Receipts bind
the fixed first two index IDs for pilots or all forty-eight IDs for summary,
including exact live main/accepted records and explicit missing lists. Both
pilots require actual acceptance; a summary may remain partial. No main store
is approved. Original audit modules retain scientific/statistical proof rules;
the composer does not infer GT decisions or accept results.

## Exact initialization cache

Science requires `initialization_cache={source,target,sha256,receipt}`. Source
is ROOT/spnv2_runs/frozen/sysu_external_cache/
efficientnet_b3_rwightman-cf984f9c.pth; target is its original
/home/ps/.cache/torch/hub/checkpoints path. Its frozen SHA is
`b3899882250c22946d0229d266049fcd133c169233530b36b9ffa7983988362f`.
The existing verified transfer receipt binds relative path, SHA,49,475,909bytes
and unchanged device/inode/size/mtime. Validation reads that small receipt and
stats the file; it does not re-read weights. A RO file bind plus
`TORCH_HOME=/home/ps/.cache/torch` exposes the exact existing cache in private
home, preventing an initialization download.

## Bounded evidence

Two composer metadata tests cover main identity/policy/version publication,
refusal to replay a saved main, and dynamic verifier creation after deployment
without changing its hashes, including refusal of a failed saved-main outcome.
One affected namespace test covers the exact frozen initialization alias,
TORCH_HOME, versioned config location and wrong-SHA refusal. One manager-argv
test checks its bound fixed command and sole writable control child. These use stdlib
fixtures only and make no model/state/GT imports or live dispatch.

The affected audit test covers both existing actions, exact-template acceptance,
and refusal of an extra task override or an expanded planned-ID cohort. The
canonical-layout test covers the bound rename receipt/manifest and refusal of
changed-content evidence or an arbitrary source root.
