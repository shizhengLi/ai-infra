# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T09:35:44.987345+00:00`
- Git revision: `d72a8f6`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `2048`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.9, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 209.98 | 21.79 | 3076.90 | 3515.05 | 682.29 | 711.38 | 756.99 | 20.847 | 65.3 | 100.0 | 95.32 |
| 1.5 | 232.32 | 21.34 | 4715.70 | 5058.30 | 686.16 | 707.40 | 732.19 | 21.791 | 31.9 | 100.0 | 95.19 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 875.00 | 2384.34 | 21.044 |
| 0.9 | 1024 | 1495.81 | 2856.46 | 21.029 |
| 0.9 | 4096 | 2401.64 | 3411.75 | 21.191 |
| 1.5 | 256 | 2239.42 | 4377.01 | 21.897 |
| 1.5 | 1024 | 2496.63 | 4481.38 | 21.852 |
| 1.5 | 4096 | 3301.62 | 4983.40 | 21.989 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
