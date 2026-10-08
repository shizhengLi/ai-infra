# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T10:38:49.753553+00:00`
- Git revision: `0983cb1`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `0`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.5, 0.9, 1.5, 2.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 99.94 | 9.73 | 1429.18 | 2034.61 | 299.31 | 880.05 | 2518.88 | 11.692 | 98.6 | 94.4 | 95.97 |
| 0.9 | 182.92 | 45.24 | 2702.61 | 3866.38 | 916.00 | 1884.11 | 2746.78 | 19.632 | 76.4 | 33.3 | 96.30 |
| 1.5 | 240.22 | 2.23 | 4239.27 | 5006.61 | 997.14 | 2410.74 | 2956.88 | 22.298 | 38.9 | 15.3 | 95.40 |
| 2.5 | 260.80 | 3.69 | 7507.70 | 8120.69 | 477.09 | 2815.02 | 3044.21 | 22.719 | 9.7 | 23.6 | 94.61 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 166.67 | 329.60 | 11.574 |
| 0.5 | 1024 | 636.57 | 2034.61 | 11.685 |
| 0.5 | 4096 | 1442.07 | 1619.62 | 12.611 |
| 0.9 | 256 | 896.22 | 2251.15 | 19.403 |
| 0.9 | 1024 | 969.53 | 2837.72 | 19.657 |
| 0.9 | 4096 | 2028.40 | 3527.65 | 20.393 |
| 1.5 | 256 | 2592.10 | 4354.46 | 22.344 |
| 1.5 | 1024 | 2116.89 | 4109.30 | 22.546 |
| 1.5 | 4096 | 2795.01 | 5006.61 | 22.310 |
| 2.5 | 256 | 5164.66 | 7321.82 | 22.651 |
| 2.5 | 1024 | 5854.37 | 7747.23 | 23.007 |
| 2.5 | 4096 | 5534.23 | 8120.69 | 22.950 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
