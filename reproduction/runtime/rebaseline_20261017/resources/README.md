# Table6 PARTIAL — static parameters only

All work in this subtree is CPU only, uses the parent's unchanged frozen source and actual configuration files, and runs through unique guarded screen sessions. ODR, training, parent queue/docs and prior cache entries are untouched.

`scripts/download_b0.py` reads the B0 URL from frozen `core/nets/backbones/efficientnet.py` and downloads only that exact official download.pytorch.org URL. The first attempt failed the required filename SHA256-prefix check. Its payload remains quarantined in resources/cache and was never installed in the shared Torch cache. `reports/b0_download.json` preserves the exact URL, code-origin hash, received full SHA, failure, and identical before/after hashes for both existing B3 cache copies. No alternate initialization or automatic retry was used.

Consequently phi0-BN counts remain unavailable. The other three actual models are constructed sequentially on CPU, seed 2021, from their original parent configurations using the existing own B3 cache or the public GN random/zero-residual construction. Author GN initialization is explicitly unverified.

`scripts/parameter_stats.py` exports `reports/table6_parameters_partial.json` and CSV plus a full inventory per available model. Counts cover whole SPNv2, EfficientDet backbone, EfficientNet encoder, BiFPN, each head, and ODR-updated backbone BN/GN affine parameters. Full ODR parameter names, shapes/counts, normalization/buffer inventories and configuration/source/cache provenance are included. It validates disjoint sums/subsets and zero ODR trainability in all prediction heads.

`scripts/verify_parameters.py` independently checks inventory arithmetic, scope partitions, normalization-derived ODR membership, provenance, cache preservation and explicit null unavailable counts. Its acceptance is parameter-only and retains the phi0 limitation.

No model forward, GPU latency, allocated/peak VRAM, training throughput or accuracy is measured here. Tensor storage bytes are static CPU storage arithmetic, not GPU memory measurements. GPU portions of Table6 remain pending. Existing phi3 short-smoke resources belong to their own workload/conditions and are not extrapolated to the configurations here.

Runtime environment: CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE=1, TORCH_HOME=spnv2_runs/env/cache/torch, OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2. Logs retain full stdout/stderr, exact commands and .exitcode evidence.
