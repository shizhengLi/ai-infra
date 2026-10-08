# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T12:13:46.130849+00:00`
- Git revision: `be24e68`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `2304`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode result before prefill: `False`
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.5, 0.9, 1.5, 2.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 114.91 | 20.17 | 2006.94 | 2559.53 | 468.54 | 819.65 | 1119.69 | 14.159 | 94.4 | 95.8 | 91.84 |
| 0.9 | 183.48 | 8.08 | 2065.59 | 2915.04 | 764.69 | 825.74 | 830.86 | 19.498 | 90.3 | 100.0 | 96.28 |
| 1.5 | 219.69 | 19.10 | 3492.29 | 4445.97 | 766.17 | 815.22 | 835.34 | 21.784 | 66.7 | 100.0 | 95.64 |
| 2.5 | 253.73 | 6.23 | 5599.04 | 6469.09 | 768.91 | 791.63 | 797.46 | 22.010 | 27.8 | 100.0 | 94.90 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 422.27 | 1223.47 | 14.270 |
| 0.5 | 1024 | 766.87 | 1996.24 | 14.319 |
| 0.5 | 4096 | 1680.19 | 2559.53 | 14.801 |
| 0.9 | 256 | 400.97 | 1388.79 | 19.525 |
| 0.9 | 1024 | 820.68 | 1937.93 | 19.597 |
| 0.9 | 4096 | 1832.21 | 2915.04 | 19.806 |
| 1.5 | 256 | 930.13 | 3326.03 | 21.268 |
| 1.5 | 1024 | 1448.25 | 2839.04 | 21.499 |
| 1.5 | 4096 | 2534.90 | 4445.97 | 22.050 |
| 2.5 | 256 | 3742.25 | 5768.92 | 21.280 |
| 2.5 | 1024 | 3330.91 | 5084.56 | 22.206 |
| 2.5 | 4096 | 4243.32 | 6469.09 | 22.412 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
