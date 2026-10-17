# New-source ODR and scans: prepared, not run

This subtree alone is owned by the ODR preparation work. Parent source/core/tools, active training, queue/status, prior experiments and papers are untouched. No GPU ODR, evaluation, scan, or scientific acceptance has been performed here.

## Frozen lines

- source_A: current parent frozen core/tools bytes, unmodified ODR algorithm.
- source_B1: only adapter writeback lookup switches model.named_modules() to model.backbone.named_modules().
- source_B2: B1 plus uniform backbone BN momentum=0.1 in tools/odr.py, following the existing own bscan repair evidence.

A/B1/B2_sha256.json inventory every source file. B1/B2_only_change.patch are minimal complete differences from A. All other frozen training/inference code is common across lines. Historical code was read for evidence; no old manager was executed.

## Requests

manifest.json has 40 fixed, 120 B-scan, and 240 N-scan requests. Fixed comprises 20 A (both architectures) plus 20 phi3 B1/B2. B scan is phi3 A/B2, N1024, B=1,2,4,8,16,32. N scan is A for both architectures, B4, N=2**2 through 2**13. Every combination has both target domains and seeds 2021 through 2025.

There are 370 unique executions and 30 explicit reuse requests: 20 B-scan fixed points reuse fixed phi3 A/B2; 10 phi3 A N1024/B4 N-scan points reuse fixed A. Reuse is accepted only after independently verifying the canonical full output, with identical new source final SHA, line, domain, seed, N/B, and protocol. Old source results never enter this manifest.

ODR is FP32, TRAIN.REPRODUCIBLE=False, TRAIN.ACCUM_STEPS=1, one physical observation per forward, original ODR B gradient/statistics window. Original AdamW lr=1e-4, wd=0; all target augment probabilities=0 and texture disabled. CUDNN settings preserve original ODR config (benchmark true, deterministic false); saved RNG is provenance, not a promise of bitwise GPU determinism. Five seeds measure observation-stream randomness from one source model, not independent source training.

## Parent queue integration

Only parent-reviewed serial scheduling may invoke run_task.py --task TASK_ID --execute, inside a unique screen using the own spnv2_runs/scripts/run_in_screen_guarded.sh. Without --execute it only prints the task. No manager or GPU task is started by importing these files or generating the list. Parent must wait for accepted phi3/phi6 source finals and their three-domain evaluations, then perform its ordinary GPU resource/process check. run_task calls the parent pipeline.runner.gpu_available function with the identical parent queue limits (currently free>=14500MiB and used<=1200MiB), preserving its bounded desktop whitelist; it prevents repeated output/report writes and uses its own advisory serial lock. It never controls GPU processes. Parent retains the global queue and GPU serialization responsibility.

Example command inside parent guarded-screen launch:

/home/ps/anaconda3/envs/SPNv2/bin/python -B -s -u /home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/plans/rebaseline_20261017/odr/run_task.py --task fixed_phi3_A_lightbox_s2021_N1024_B4 --execute

The parent source proofs are reports/source_phi3_verified.json and source_phi6_verified.json; final paths are taken from the existing parent source configs and frozen in every generated task config. Verify the parent's chosen names before enqueuing. A missing proof is a hard stop, never a fallback to historical final weights.

run_task launches the exact per-line CLI through audited_entry.py, captures full stdout/stderr and return code, then runs verify_result.py independently on CPU. Failed adaptation/evaluation/verification stops the task and must stop the parent queue. There is no automatic retry, source selection by target score, parameter change or seed replacement. Existing failed output is preserved and blocks reuse.

## Audit and output

The wrapper emits standard SPNV2_EVENT records (phase=odr) for observation/update, checkpoint, evaluation callback progress and failures. It validates initial weights against the accepted new source, only backbone BN/GN affine trainability, finite losses/gradients, unchanged heads, intended BN buffer behavior, actual observation IDs/N_seen/N_unique, every step boundary, and optimizer state step counts. The adapted weights, optimizer and initial/final RNG checkpoint, and observations are persisted BEFORE post-audit/evaluation; evaluation failure cannot discard a completed adaptation. Failed audits retain these as unaccepted evidence.

Independent verification checks source/config/snapshot/helper hashes, all adapted/CLI final/checkpoint weights exactly equal, strict architecture reload, frozen heads/update scope, optimizer counts, stable observation CSV membership/cycles, and full-domain H/E predictions, denominators, rejection masks and NaN preservation. This verification must pass before an alias is accepted.

No continuation/restart of an interrupted ODR task is implemented: adaptation RNG/optimizer/checkpoint evidence is retained, but automatic recovery would require a separately reviewed protocol and a new run ID. No scientific aggregation or case visualization is implemented here; parent analysis owns those later artifacts.

## CPU acceptance

reports/cpu_acceptance.json is engineering evidence only. It tests exact frozen adapter code with tiny BN/GN networks against an independent pooled-moment/gradient reference at B1/B4/B32; checks unmodified-A versus repaired BN buffers, affine-only updates and unchanged heads; validates seed-reproducible cyclic random observation streams and exact truncation through N8192; and checks all configs, counts and source diffs. Real CUDA model/dataloader/evaluation integration remains unvalidated until parent review and later authorized GPU execution.

CPU and dry-list runs use their own guarded screen logs with .exitcode; dry_list_*.log lists every requested task without executing it. implementation_manifest.json binds helper hashes and the complete task manifest; changing helpers requires review and refreshing it before queue use.

## Reviewed parent-queue integration revision

The original generated manifest, implementation manifest, helpers, configurations and CPU evidence are preserved under history/preintegration_20261017. The current manifest adds queue_command (run_task --execute), verify_command (independent CPU verifier), a fixed40 dependency list on every scan and the unchanged parent GPU gate specification. Source three-domain acceptance remains a parent dependency. run_task also refuses any scan until all 40 fixed acceptance reports exist for the same reviewed implementation, protocol and new source finals.

TORCH_HOME is uniformly spnv2_runs/env/cache/torch. Independent verification forcibly sets CUDA_VISIBLE_DEVICES='' before importing torch/model code; run_task passes that environment to its verifier subprocess. The GPU gate uses the parent's actual function; it adds no PID/name exception of its own.

At target-loader creation, before the first observation, input_evidence.json records the full target CSV byte SHA, camera/keypoint byte SHAs, and every target image's actual path, size and mtime with a canonical inventory SHA. This is metadata integrity for images, explicitly not raw-image content hashing. The independent verifier rereads every item and checks it again.

Adapted weights and optimizer/RNG evidence remain saved before post-adaptation audit and evaluation; an SPNV2_EVENT checkpoint names the saved path. Independent CPU verification strictly reloads and compares the original source and adapted states, validates every optimizer moment's finiteness/shape/step and configuration, compares early/final optimizer payloads, validates the full Python/NumPy/CPU/CUDA RNG schema without restoring CUDA, and rechecks the exact affine/head/buffer update scope. The copied root verify_heads checks all physical metrics and explicitly preserves NaN predictions and null scores for zero accepted H samples.

The integration CPU suite exercises the real frozen adapter math plus bounded desktop gate mock cases, input corruption, optimizer/RNG/state corruption, standard event and error emission, safe default CLI, zero-H denominators, missing physical metrics and rejected zero-filled missing predictions. It uses only tiny/mock data and no real ODR or GPU query. Its current acceptance is reports/integration_cpu_acceptance.json, bound to implementation_manifest.json. Real CUDA integration remains pending.
