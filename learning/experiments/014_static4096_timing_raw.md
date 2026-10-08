# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T10:04:31.813682+00:00`
- Git revision: `6516af1`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `4096`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.9, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 210.07 | 21.89 | 3276.29 | 3618.90 | 1074.88 | 1408.96 | 1550.36 | 20.859 | 59.7 | 23.6 | 95.79 |
| 1.5 | 233.36 | 21.45 | 4667.00 | 5400.89 | 1327.82 | 1389.93 | 1453.87 | 21.828 | 33.3 | 12.5 | 95.33 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 931.57 | 2916.69 | 21.058 |
| 0.9 | 1024 | 1527.44 | 3010.84 | 21.054 |
| 0.9 | 4096 | 2577.15 | 3618.90 | 21.206 |
| 1.5 | 256 | 2245.96 | 4371.19 | 21.972 |
| 1.5 | 1024 | 2527.37 | 4380.91 | 21.889 |
| 1.5 | 4096 | 3602.14 | 5188.42 | 22.029 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
