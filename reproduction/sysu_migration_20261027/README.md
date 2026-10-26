# SYSU migration storage adapter v1

`storage_adapter_v1.install_on_module(frozen_module, policy_factory)` replaces
only that newly created process's `DurableRunStore` class with a subclass.
The original constructor arguments, identity, validator, serializer and save
implementation are preserved. Install before constructing approved output
stores, never into a healthy worker. The original 50 scientific files remain
unchanged.

The factory receives `(canonical_root: Path, identity: dict)` and must return
`WritableRootPolicy(canonical_root, explicit_policy)` for each exact authorized
new output store. It must return `None` or raise for source, read-only or unknown
roots. The adapter rejects these roots before the frozen constructor writes
anything; no implicit read-only fallback is provided. An ancestor approval does
not approve child paths. Paths must be absolute and free of symlink aliases.
Approvals for resumed stores must include the current explicit pin registry.

Every successful save calls the supplied serializer exactly once, checks its
actual byte length against the predeclared disk estimate, delegates publication
to the frozen writer, then calls the existing storage-policy retention routine
under the store writer lock. Publication errors retain pending evidence and do
not trigger retention. Pin references with the existing `storage_policy.pin`
before publishing them. Storage-policy initialization and retention do not alter
checkpoint state bytes or scientific identity.

Zero-model tests (stdlib only; no Torch import or forward) on the target:

```sh
cd /home/ps/Downloads/UsrSourceCode/LuXinghai/spnv2
SPNV2_FROZEN_DURABLE_STATE=/home/ps/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/migrations/sysu_20261027/source_snapshot/spnv2_runs/plans/odr003_round2_20261023/implementation/durable_state.py PYTHONDONTWRITEBYTECODE=1 python3 -m unittest reproduction.sysu_migration_20261027.test_storage_adapter_v1 -v
```

The tests use only temporary synthetic stores and the frozen stdlib writer.
