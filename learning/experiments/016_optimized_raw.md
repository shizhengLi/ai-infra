# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T10:47:43.295225+00:00`
- Git revision: `0983cb1`
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
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.5, 0.9, 1.5, 2.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 99.92 | 9.71 | 1497.43 | 1833.58 | 269.62 | 819.10 | 1339.39 | 11.662 | 97.2 | 97.2 | 96.15 |
| 0.9 | 182.66 | 45.08 | 2406.88 | 3393.19 | 650.59 | 820.54 | 834.68 | 19.529 | 81.9 | 100.0 | 96.29 |
| 1.5 | 239.55 | 1.75 | 3881.12 | 4570.73 | 767.99 | 829.72 | 875.92 | 22.100 | 38.9 | 100.0 | 94.93 |
| 2.5 | 258.97 | 3.70 | 7112.33 | 7565.01 | 763.28 | 778.53 | 789.03 | 22.530 | 12.5 | 100.0 | 94.61 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 162.11 | 330.36 | 11.531 |
| 0.5 | 1024 | 600.93 | 1787.98 | 11.653 |
| 0.5 | 4096 | 1473.25 | 1748.36 | 12.583 |
| 0.9 | 256 | 752.84 | 1907.28 | 19.288 |
| 0.9 | 1024 | 815.54 | 2096.05 | 19.544 |
| 0.9 | 4096 | 2071.63 | 3393.19 | 20.268 |
| 1.5 | 256 | 2301.00 | 3650.34 | 22.147 |
| 1.5 | 1024 | 2132.96 | 4136.18 | 22.347 |
| 1.5 | 4096 | 2677.71 | 4408.71 | 22.150 |
| 2.5 | 256 | 4278.48 | 7093.52 | 22.405 |
| 2.5 | 1024 | 5560.34 | 7563.45 | 22.821 |
| 2.5 | 4096 | 4758.75 | 7528.67 | 22.764 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
