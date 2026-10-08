# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T09:55:38.467780+00:00`
- Git revision: `b648b82`
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
- Decode-overload threshold: `2048` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.9, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 209.97 | 21.84 | 3079.37 | 3519.72 | 681.80 | 711.06 | 756.18 | 20.848 | 65.3 | 100.0 | 95.78 |
| 1.5 | 232.38 | 21.39 | 4709.17 | 5050.41 | 686.06 | 707.00 | 731.78 | 21.782 | 31.9 | 100.0 | 95.15 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 872.72 | 2387.72 | 21.034 |
| 0.9 | 1024 | 1493.25 | 2857.56 | 21.020 |
| 0.9 | 4096 | 2400.55 | 3417.15 | 21.193 |
| 1.5 | 256 | 2236.16 | 4372.48 | 21.890 |
| 1.5 | 1024 | 2493.35 | 4475.89 | 21.842 |
| 1.5 | 4096 | 3296.47 | 4975.61 | 21.982 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
