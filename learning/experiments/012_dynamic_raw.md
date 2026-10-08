# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T09:39:49.419821+00:00`
- Git revision: `d72a8f6`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `4096`
- Decode-overload prefill budget: `2048`
- Decode-overload threshold: `4096` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.9, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 210.05 | 21.71 | 2959.66 | 3624.08 | 749.17 | 960.98 | 1002.50 | 20.854 | 69.4 | 79.2 | 95.92 |
| 1.5 | 232.68 | 21.35 | 4718.40 | 5266.47 | 731.12 | 990.94 | 1082.92 | 21.769 | 38.9 | 43.1 | 95.16 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 827.19 | 2406.64 | 21.052 |
| 0.9 | 1024 | 1444.00 | 2741.46 | 21.050 |
| 0.9 | 4096 | 2313.60 | 3624.08 | 21.212 |
| 1.5 | 256 | 2238.44 | 4475.31 | 21.901 |
| 1.5 | 1024 | 2476.46 | 4633.57 | 21.831 |
| 1.5 | 4096 | 3304.31 | 5266.47 | 21.970 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
