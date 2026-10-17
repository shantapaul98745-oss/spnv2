# Remaining-configuration engineering pilots

The four configurations in `configs/` copy the formal source settings. Only the
experiment name, output root and training-log root change. Frozen `source/` is
used directly and never edited. Pilot weights are never initialization for formal
runs.

`gpu_smoke.py` accepts `--source`, `--cfg`, `--report`, `--artifacts`,
`--attempts 32` and `--checkpoint 16`. Only the specified 32/16 engineering
protocol is accepted. Each invocation requires a new report and empty artifact
directory under this `pilots/` subtree.

The manager must launch this through the project's guarded screen wrapper, after
the prior task and resource checks succeed. Required environment includes
`PYTHONNOUSERSITE=1`, `TORCH_HOME=/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/env/cache/torch`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, `MPLBACKEND=Agg`, and the manager's selected GPU.
Use a separate cache directory for Matplotlib/XDG.

The supported configurations are:

| Configuration | Actual heads and losses | Physical batch | Accumulation | Subset rows |
|---|---|---:|---:|---:|
| phi6_gn_ehs.yaml | heatmap, efficientpose, segmentation | 1 | 4 | 128 |
| table2_h.yaml | heatmap | 4 | 4 | 512 |
| table2_e.yaml | efficientpose | 4 | 4 | 512 |
| table2_eh.yaml | heatmap, efficientpose | 4 | 4 | 512 |

These are the first physical-batch × accumulation × 32 rows of the original CSV,
then the production epoch-keyed permutation. The original dataset, augmentation,
heads, loss normalization, AdamW, AMP, clipping, full 768×512 resolution and seed
2021 remain active. The forward hook checks every actual batch. The report records
actual emitted sample IDs and checks them against the selected CSV/sampler order.

The continuous run saves its complete boundary checkpoint at attempt 16 and
finishes at 32. The CUDA model, optimizer and loader are deleted before a fresh
same-seed reconstruction. Its actual initial weights must match before restoring
the boundary. The resumed suffix must match every final model, optimizer, scaler,
RNG and counter entry and its actual sample-ID stream. Every active head and the
backbone must change trainable parameters, with at least three successful updates.
NaN/Inf, counters, resume or component-update failures fail the pilot.

BN initialization only loads the official weight file already present in the
fixed local torch cache; a missing cache fails without a download. The phi6 GN
configuration follows the public code's random initialization and residual GN
zero initialization. The author's GN initialization remains unverified.

Run the independent verifier in another guarded CPU screen:

```text
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 python -B -s pilots/verify_pilot.py
    --report <new GPU report>
    --output <new CPU verification report>
```

All paths above must be absolute for queue commands. The verifier reads the
configuration independently, checks artifact hashes and the frozen source,
recomputes the complete CSV/input inventory and sample order, inspects the saved
initial/boundary/final counters, checks actual AdamW step counts, verifies every
active component changed, and fingerprints every tensor byte and typed scalar in
both full final checkpoints. Standalone model snapshots must match them too.
Verification can be memory intensive for phi6 because it compares real saved
model and optimizer state on CPU.

CPU tests are `tests/test_pilot_evidence.py --summary <new JSON>`, also requiring
an independent guarded screen and `CUDA_VISIBLE_DEVICES=''`. Tiny synthetic
fixtures exercise the verifier and deliberate rehashed model, optimizer, scaler,
RNG, sampler, boundary, shape, ID, source/config hash and input-asset corruption.
Those CPU fixtures are not evidence of a real GPU pilot.

As of preparation, the CPU suite passed 16/16, recorded in
`reports/pilot_evidence_cpu_20261017_145127_9d81cb.json`, with stdout/stderr and
exit code 0 in the matching `logs/` files. No remaining-configuration GPU pilot
has been run by this preparation task. A GPU pilot and successful independent
CPU verification must precede its corresponding formal training. These are
engineering checks, not source accuracy results.
