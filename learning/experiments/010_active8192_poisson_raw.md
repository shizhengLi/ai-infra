# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T08:51:22.192716+00:00`
- Git revision: `1f0b104`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `8192`
- Workload: `16 x 1024 input / 256 output tokens`
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 157.13 | 28.65 | 661.16 | 799.40 | 228.37 | 462.47 | 871.47 | 10.313 | 100.0 | 100.0 | 95.45 |
| 1.0 | 200.44 | 8.65 | 826.72 | 839.37 | 230.03 | 489.18 | 621.92 | 11.888 | 100.0 | 100.0 | 94.68 |
| 1.5 | 215.39 | 3.98 | 947.25 | 995.15 | 316.61 | 539.52 | 848.88 | 12.207 | 100.0 | 100.0 | 94.11 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
