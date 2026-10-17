# Current Table6 status: four parameter scopes verified; GPU measurements pending

The current API is `reports/table6_success_20261017/table6_parameters.json`, matching CSV, per-model inventories and `parameter_acceptance.json`. These are new success reports. The earlier partial reports, failure reports, original manifest and scripts remain preserved as historical evidence.

## Official B0 identity and cache addition

The [torchvision0.20 EfficientNet source](https://docs.pytorch.org/vision/0.20/_modules/torchvision/models/efficientnet.html) identifies EfficientNet_B0_Weights.IMAGENET1K_V1 with `efficientnet_b0_rwightman-7f5810bc.pth` and B3 V1 with `efficientnet_b3_rwightman-b3899882.pth`. [Official PR7898](https://github.com/pytorch/vision/pull/7898) explains that incorrect hash URLs were corrected when hash checks exposed them. Its linked [official issue7744](https://github.com/pytorch/vision/issues/7744) records the same complete B0 hash received from the legacy URL: `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`.

A new download from the corrected official B0 URL passed both official filename-prefix and full SHA checks. Every tensor was compared against the quarantined old-URL payload: keys, shapes, dtypes and values match exactly; the files are byte-identical as well. `reports/b0_official_v1_success_20261017.json` records saved official HTML evidence hashes and the complete tensor comparison inventory.

Only the previously missing `spnv2_runs/env/cache/torch/hub/checkpoints/efficientnet_b0_rwightman-3dd342df.pth` was added, through an exclusive atomic new hardlink, to satisfy the unchanged frozen loader's legacy filename. This aliases the independently verified same official ImageNet V1 payload; it changes neither source nor initialization identity. Both old B3 cache copies and all earlier reports remain hash-identical.

## Parameter-only result

All four actual configurations were constructed sequentially on CPU with seed2021. Counts cover whole SPNv2, EfficientDet backbone, EfficientNet encoder, BiFPN, each head and ODR-updated backbone BN/GN affine parameters. Each report includes all names, shapes, normalization/buffer inventories and source/config/cache provenance. The independent verifier checks disjoint sums, scope membership, frozen heads (zero ODR parameters), identical counts/names for the earlier three models and the B0 identity chain.

GN remains public random construction with zero residual weights; author initialization is still unverified. No model forward, accuracy evaluation, GPU latency, peak GPU memory or throughput experiment occurred. Table6 remains PARTIAL because its GPU resource portion is pending. Static parameter storage bytes must not be described as measured GPU memory.

Unique guarded screen logs are `b0_identity_20261017_a.log`, `parameters_v2_20261017_a.log`, and `verify_parameters_v2_20261017_a.log`, each with full stdout/stderr, exact command and exit-code file. New provenance is bound by `manifest_success_20261017.json`; the earlier `manifest.json` is historical and unchanged.
