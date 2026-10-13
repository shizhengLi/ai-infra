# Experiment 053 Raw Result Index

The accepted target comparison is Qwen3-32B TP4, FlashInfer attention, Direct PyNCCL, CUDA Graph
shapes 1/2/4/8/16/24/32, 4,096 fixed KV pages, 256 input tokens, 64 output tokens, and three online
repeats at C=1/8/32.

| C | BF16 tok/s | FP8 tok/s | Throughput change | BF16 avg TPOT ms | FP8 avg TPOT ms | Peak MiB/card change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.73 | 61.58 | +55.00% | 24.20 | 15.23 | 18,229 -> 15,407 |
| 8 | 228.14 | 341.03 | +49.48% | 27.87 | 17.94 | 18,229 -> 15,407 |
| 32 | 273.00 | 402.24 | +47.34% | 29.48 | 19.25 | 18,229 -> 15,407 |

Final 570-question MMLU: BF16 458 correct (80.35%), FP8 454 correct (79.65%), both parse rates
100%, delta -0.70 percentage points. The exact commands, principles, invalid evaluator attempts,
and interpretation are in `learning/experiments/053_fp8_end_to_end.md`.
