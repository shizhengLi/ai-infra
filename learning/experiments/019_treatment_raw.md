# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T12:22:15.992691+00:00`
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
- Decode result before prefill: `True`
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.5, 0.9, 1.5, 2.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 114.91 | 20.14 | 2026.52 | 2572.74 | 580.60 | 764.69 | 773.70 | 14.191 | 94.4 | 100.0 | 91.62 |
| 0.9 | 183.35 | 8.08 | 2101.60 | 2934.69 | 758.72 | 777.48 | 781.89 | 19.535 | 90.3 | 100.0 | 96.37 |
| 1.5 | 219.44 | 19.11 | 3510.09 | 4468.54 | 771.11 | 781.59 | 786.56 | 21.827 | 66.7 | 100.0 | 95.10 |
| 2.5 | 253.41 | 6.25 | 5620.23 | 6492.28 | 771.52 | 779.72 | 781.54 | 22.047 | 27.8 | 100.0 | 94.66 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 429.03 | 1232.99 | 14.297 |
| 0.5 | 1024 | 773.82 | 2007.82 | 14.343 |
| 0.5 | 4096 | 1690.16 | 2572.74 | 14.826 |
| 0.9 | 256 | 409.79 | 1411.99 | 19.574 |
| 0.9 | 1024 | 831.92 | 1950.49 | 19.631 |
| 0.9 | 4096 | 1856.60 | 2934.69 | 19.864 |
| 1.5 | 256 | 942.78 | 3346.96 | 21.301 |
| 1.5 | 1024 | 1471.70 | 2854.44 | 21.555 |
| 1.5 | 4096 | 2558.57 | 4468.54 | 22.108 |
| 2.5 | 256 | 3758.67 | 5794.42 | 21.318 |
| 2.5 | 1024 | 3345.95 | 5103.97 | 22.243 |
| 2.5 | 4096 | 4259.09 | 6492.28 | 22.452 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
