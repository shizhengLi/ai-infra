# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T10:16:56.452114+00:00`
- Git revision: `6516af1`
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
- Arrival rates: `[0.9, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 209.98 | 21.74 | 2877.98 | 3397.88 | 763.92 | 808.73 | 828.39 | 20.864 | 75.0 | 100.0 | 95.74 |
| 1.5 | 232.63 | 21.45 | 4718.77 | 5094.56 | 773.44 | 801.39 | 828.25 | 21.801 | 40.3 | 100.0 | 95.36 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 807.29 | 2227.49 | 21.062 |
| 0.9 | 1024 | 1462.60 | 2794.68 | 21.047 |
| 0.9 | 4096 | 2211.41 | 3397.88 | 21.199 |
| 1.5 | 256 | 2246.73 | 4194.43 | 21.932 |
| 1.5 | 1024 | 2471.58 | 4519.27 | 21.874 |
| 1.5 | 4096 | 3170.26 | 5094.56 | 21.987 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
