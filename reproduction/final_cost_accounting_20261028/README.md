# Final external-cost accounting — candidate only

No current index/guard/ledger is changed by these files. Root must publish and review the code/request/evidence, wait for real scientific and external-work termination, and explicitly invoke the entry inside the existing SYSU namespace. No model/scientific tests or summary replay are used.

## Pure test

`python -B -s entry.py --self-test` uses memory fixtures only.

## Actual entry, future only

Use the current existing namespace configuration, original four pipeline locks, manager lock, two fresh heavy-CPU resource snapshots and precise screen/nonce registration. Supply `SPNV2_RUN_NONCE`, a unique screen and the actual wrapper log path under D. The wrapper log and `--log-path` must match, because the next stage verifies its genuine `.exitcode`.

`python -B -s entry.py --action settle|publish --execute --request /absolute/bound/request.json --request-sha256 SHA --log-path D/logs/actual_unique.log`

Request schema is `spnv2-final-cost-accounting-request-v1`; fields: `action`, `entry` and `core` exact path/SHA bindings, and current `index` binding. Settle additionally needs `inventory`; publish needs `provisional`, `settlement`, and `settlement_success_nonce`. All evidence must be immutable bounded JSON/log/exit copies under this own D/M; no SSH, model or private-key reads occur.

## Required fee inventory

Schema `spnv2-final-external-cost-inventory-v1`, `migration_id=sysu_20261027`, exact logical `plan_path`, `all_external_work_closed=true`, and `pending_cost_items=[]`. `transfer_closure` binds real complete source sender JSON, its genuine successful exit file, and complete target receiver JSON. Both sides must show276/276 and98,724,602,862 raw bytes; an active transfer cannot be settled. Historical failed transfers remain separately accounted with their actual nonzero exit.

Each item has stable `item_id`/`coverage_key`, actual `host`, `kind=process|control_scope`, `closed=true`, nonempty immutable `evidence`, covered scope names from `core.SCOPES`, and `accounting=already_in_original_ledger|new_external_upper`. Already-paid items require their actual closed `existing_attempt_id`. New items require a known positive `conservative_debit_seconds_upper` and explicit grounded `upper_basis`; missing values are rejected, never replaced by0. Keep `measured_elapsed_wall_seconds` separate. Unavailable `measured_cpu_seconds` must be null with `cpu_measurement_status=unavailable`. Process items also require exact boot/PID/start/full-command/nonce identity, actual exit and bound terminal exit plus matching immutable JSON launch/status identity (normalize historical `nonce` to canonical `run_nonce`); multiple intervals of one process must be demonstrably nonoverlapping.

Stable ID ignores creation time and evidence relocation, but preserves fee/evidence meaning. A previous imported scope cannot be paid again under a different ID. Exact process boot/PID/start/full-command/nonce is compared independently of item/coverage/host labels; only explicit nonoverlapping monotonic intervals permit another phase. Control scopes reusing the same evidence SHA set and overlapping covered scopes cannot be relabelled and charged again. The same semantic checks apply inside one inventory. A retry of a successful settlement verifies/reuses its immutable import receipt, charges only the retry's real metadata execution, and never reimports external debit. An unclosed prior allocation still blocks via the original GlobalBudget. No old settle script is invoked.

## Recognized continuous ledger

The adapter creates one real `allow_model_forwards=False` attempt in the original GlobalBudget. It appends exactly one recognized `attempt_closed` with actual current wall, measured pre-budget wall, explicit retrospective upper and original2s publication tail separated. `external_cost_settlement_id` and coverage are persisted in that close; the original reader therefore counts the debit. The original source prefix/identity and every prior journal byte are preserved and checked. CPU-unavailable and overlapping-host wall remain explicit; no exact global elapsed-time or CPU-total claim is made.

## Candidate publisher and separately paid promotion

The `publish` action **only prepares a launch-local candidate**. It requires the genuine guard provisional, unchanged saved summary/proof, actual summary wrapper/outcome0 and its historical closed-budget receipt. It verifies the recognized external import event and a genuine successful settlement stage, then closes its own real zero-forward attempt. It never reruns summary/panels and never uses the old summary snapshot as final fees. The40000-forward cap/time-unlimited policy remains unchanged.

The only proposed manifest artifact is `D/launches/<nonce>/candidate_final_manifest.json`. It uses a candidate schema, has `candidate_only=true`, `promotion_costs_pending=true`, and contains `candidate_stage_closed_budget`, not a claimed `final_settled_budget`. A tail/publication failure leaves that failed launch's evidence intact, cannot create/block `D/delivery/final_manifest.json`, and can be retried with a fresh owned nonce. Root must verify the candidate publisher's **actual wrapper0, outcome, launch and closed ledger allocation**, not just a candidate file or a status string.

Actual promotion is a distinct, explicit metadata-only action; this candidate does **not** implement or perform it. Promotion must use the same original resource/identity/lock/single-writer boundaries, account for its own measured work and justified bounded tail, bind a fresh final ledger snapshot including that work, and only then atomically publish an immutable final manifest. Do not simply copy/rename this candidate into a final manifest or pretend its cutoff budget already covers later promotion. Both publisher and promotion failures remain real failed evidence/costs. Original scientific outputs are reused.

## Failure accounting before/after admission

One outer failure path covers argument/request/code validation, locks, resources, registry readback, source-prefix validation, budget construction, begin and action/output failures. It writes an own `entry_failure_receipt.json` and real failure outcome with requested nonce/request/log, precise available process identity, actual main start/wall, stage, known attempt ID/closed state, CPU unavailable and `accounting_pending`. A nonce-directory collision is never overwritten: an independently named unadmitted failure directory is used. Before-begin/unregistered work is never assigned cost0; its measured native wall remains pending external accounting. Existing original-ledger allocations are identified and must not be reimported; an unresolved/uncertain allocation remains pending. Own failed returned attempts are closed under the held original locks where possible; no past process/exit is fabricated.

The targeted pure tests exercise process/control label-renaming, permitted nonoverlapping phases, launch-local candidate survival after simulated tail failure, outer resource/registry/budget/prefix failures, unresolved begin, and already-ledgered failures. See `targeted_test_receipt.json` for the actual command/result and code hashes.

No actual admission, cost settlement, final publication or promotion has been run. No fee amount is supplied/estimated by this candidate. Missing evidence leaves the experiment costs-pending even with48 scientific windows complete.
