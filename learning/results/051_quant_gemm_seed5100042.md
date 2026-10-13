# Experiment 051 raw quantized GEMM results

- Timestamp: `2026-10-13T10:33:09.031168+00:00`
- Git revision: `8f40b56`
- GPU: `NVIDIA L20` (SM `8.9`)
- PyTorch/CUDA: `2.9.1+cu128` / `12.8`
- sgl-kernel: `0.3.21`
- Warmup/repetitions/samples: `20` / `100` / `5`
- L2 cache flush before each timed call: `128` MiB (flush excluded from timing)

Times are kernel-only medians. FP8/INT8 dynamic activation-quantization timings are retained in the
JSON artifact. Weight quantization and Marlin repacking are offline and excluded.

| Layer (TP4 local) | M | Path | Median ms | vs BF16 | Weight MiB | Quant rel-L2 | Kernel rel-L2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| qkv_proj | 1 | BF16 | 0.0577 | 1.00x | 25.00 | - | - |
| qkv_proj | 1 | FP8 W8A8 | 0.0320 | 1.80x | 12.51 | 0.03774 | 0.00339 |
| qkv_proj | 1 | INT8 W8A8 | 0.0322 | 1.79x | 12.51 | 0.01837 | 0.00313 |
| qkv_proj | 1 | INT4 W4A16 | 0.0211 | 2.74x | 6.25 | 0.15156 | 0.00397 |
| qkv_proj | 8 | BF16 | 0.0637 | 1.00x | 25.00 | - | - |
| qkv_proj | 8 | FP8 W8A8 | 0.0332 | 1.92x | 12.51 | 0.03777 | 0.00300 |
| qkv_proj | 8 | INT8 W8A8 | 0.0322 | 1.98x | 12.51 | 0.01991 | 0.00325 |
| qkv_proj | 8 | INT4 W4A16 | 0.0243 | 2.62x | 6.25 | 0.15413 | 0.00395 |
| qkv_proj | 32 | BF16 | 0.0618 | 1.00x | 25.00 | - | - |
| qkv_proj | 32 | FP8 W8A8 | 0.0323 | 1.91x | 12.51 | 0.03758 | 0.00363 |
| qkv_proj | 32 | INT8 W8A8 | 0.0326 | 1.90x | 12.51 | 0.01977 | 0.00379 |
| qkv_proj | 32 | INT4 W4A16 | 0.0286 | 2.16x | 6.25 | 0.15387 | 0.00501 |
| o_proj | 1 | BF16 | 0.0460 | 1.00x | 20.00 | - | - |
| o_proj | 1 | FP8 W8A8 | 0.0280 | 1.64x | 10.02 | 0.03743 | 0.00306 |
| o_proj | 1 | INT8 W8A8 | 0.0281 | 1.64x | 10.02 | 0.01860 | 0.00315 |
| o_proj | 1 | INT4 W4A16 | 0.0187 | 2.46x | 5.01 | 0.14577 | 0.00346 |
| o_proj | 8 | BF16 | 0.0462 | 1.00x | 20.00 | - | - |
| o_proj | 8 | FP8 W8A8 | 0.0282 | 1.64x | 10.02 | 0.03723 | 0.00319 |
| o_proj | 8 | INT8 W8A8 | 0.0282 | 1.63x | 10.02 | 0.01883 | 0.00318 |
| o_proj | 8 | INT4 W4A16 | 0.0202 | 2.28x | 5.01 | 0.14486 | 0.00357 |
| o_proj | 32 | BF16 | 0.0482 | 1.00x | 20.00 | - | - |
| o_proj | 32 | FP8 W8A8 | 0.0284 | 1.70x | 10.02 | 0.03748 | 0.00309 |
| o_proj | 32 | INT8 W8A8 | 0.0281 | 1.71x | 10.02 | 0.01854 | 0.00325 |
| o_proj | 32 | INT4 W4A16 | 0.0213 | 2.26x | 5.01 | 0.14509 | 0.00402 |
| gate_up_proj | 1 | BF16 | 0.2185 | 1.00x | 125.00 | - | - |
| gate_up_proj | 1 | FP8 W8A8 | 0.1357 | 1.61x | 62.55 | 0.03763 | 0.00350 |
| gate_up_proj | 1 | INT8 W8A8 | 0.1443 | 1.51x | 62.55 | 0.02021 | 0.00322 |
| gate_up_proj | 1 | INT4 W4A16 | 0.0720 | 3.04x | 31.27 | 0.15634 | 0.00325 |
| gate_up_proj | 8 | BF16 | 0.2570 | 1.00x | 125.00 | - | - |
| gate_up_proj | 8 | FP8 W8A8 | 0.1360 | 1.89x | 62.55 | 0.03760 | 0.00304 |
| gate_up_proj | 8 | INT8 W8A8 | 0.1444 | 1.78x | 62.55 | 0.01952 | 0.00327 |
| gate_up_proj | 8 | INT4 W4A16 | 0.0718 | 3.58x | 31.27 | 0.15403 | 0.00327 |
| gate_up_proj | 32 | BF16 | 0.2871 | 1.00x | 125.00 | - | - |
| gate_up_proj | 32 | FP8 W8A8 | 0.1346 | 2.13x | 62.55 | 0.03764 | 0.00418 |
| gate_up_proj | 32 | INT8 W8A8 | 0.1474 | 1.95x | 62.55 | 0.01938 | 0.00430 |
| gate_up_proj | 32 | INT4 W4A16 | 0.0719 | 3.99x | 31.27 | 0.15410 | 0.00440 |
| down_proj | 1 | BF16 | 0.1294 | 1.00x | 62.50 | - | - |
| down_proj | 1 | FP8 W8A8 | 0.0709 | 1.82x | 31.27 | 0.03766 | 0.00312 |
| down_proj | 1 | INT8 W8A8 | 0.0700 | 1.85x | 31.27 | 0.01885 | 0.00301 |
| down_proj | 1 | INT4 W4A16 | 0.0385 | 3.36x | 15.63 | 0.15780 | 0.00351 |
| down_proj | 8 | BF16 | 0.1316 | 1.00x | 62.50 | - | - |
| down_proj | 8 | FP8 W8A8 | 0.0718 | 1.83x | 31.27 | 0.03744 | 0.00322 |
| down_proj | 8 | INT8 W8A8 | 0.0703 | 1.87x | 31.27 | 0.02002 | 0.00325 |
| down_proj | 8 | INT4 W4A16 | 0.0378 | 3.48x | 15.63 | 0.15578 | 0.00354 |
| down_proj | 32 | BF16 | 0.1352 | 1.00x | 62.50 | - | - |
| down_proj | 32 | FP8 W8A8 | 0.0710 | 1.90x | 31.27 | 0.03760 | 0.00310 |
| down_proj | 32 | INT8 W8A8 | 0.0703 | 1.92x | 31.27 | 0.01936 | 0.00326 |
| down_proj | 32 | INT4 W4A16 | 0.0423 | 3.19x | 15.63 | 0.15558 | 0.00396 |
