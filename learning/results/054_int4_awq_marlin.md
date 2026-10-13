# Experiment 054 Raw Result Index

The accepted performance matrix uses Qwen3-32B-AWQ, BF16 runtime tensors, FlashInfer attention,
CUDA Graph shapes 1/2/4/8/16/24/32, 4,096 fixed KV pages, 256 input tokens, 64 output tokens, and
three repeats after one warmup at C=1/8/32.

| C | AWQ TP1 tok/s | AWQ TP2 tok/s | TP2 scaling efficiency | TP1 peak MiB | TP2 peak MiB/card |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.69 +/- 0.02 | 60.64 +/- 0.04 | 87.41% | 21,037 | 11,283 |
| 8 | 169.66 +/- 0.13 | 278.69 +/- 0.11 | 82.13% | 21,295 | 11,463 |
| 32 | 197.74 +/- 0.07 | 316.09 +/- 0.12 | 79.92% | 21,755 | 11,617 |

The 570-question quality gate parsed 100% of answers. BF16 scored 458/570 (80.35%), FP8 scored
454/570 (79.65%), checkpoint-default FP16 AWQ scored 451/570 (79.12%), and controlled BF16 AWQ
scored 448/570 (78.60%). Both AWQ rows fail the one-point gate.

The exact commands, implementation principle, dtype-control correction, failed Stage A attempt,
and interpretation are in `learning/experiments/054_int4_awq_marlin.md`. Machine-readable JSON is
kept locally and ignored by Git under the repository's existing result policy.
