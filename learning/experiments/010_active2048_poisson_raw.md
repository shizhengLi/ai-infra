# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T09:04:33.707051+00:00`
- Git revision: `1f0b104`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `2048`
- Workload: `16 x 1024 input / 256 output tokens`
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 157.11 | 28.62 | 674.42 | 788.80 | 229.05 | 416.02 | 657.31 | 10.314 | 100.0 | 100.0 | 94.60 |
| 1.0 | 200.29 | 8.67 | 845.00 | 913.94 | 273.34 | 475.30 | 655.88 | 11.905 | 100.0 | 100.0 | 93.71 |
| 1.5 | 215.58 | 4.19 | 1023.74 | 1118.14 | 316.40 | 560.58 | 661.09 | 12.191 | 100.0 | 100.0 | 94.44 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
