# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T09:53:06.149066+00:00`
- Git revision: `f0ee64b`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- Normal prefill budget: `4096`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `1024`
- Decode-overload prefill budget: `768`
- Decode-overload threshold: `4096` pending tokens
- Decode-overload active threshold: `4` active decode requests
- Decode result before prefill: `False`
- Workload: `12` requests, balanced input lengths `[256, 1024, 4096]`, `128` output tokens
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 70.31 | 14.68 | 1082.06 | 1281.87 | 216.09 | 240.87 | 270.22 | 4.432 | 100.0 | 100.0 | 73.60 |
| 1.0 | 98.61 | 8.77 | 994.84 | 1177.08 | 167.93 | 230.41 | 237.90 | 4.284 | 100.0 | 100.0 | 88.30 |
| 1.5 | 152.97 | 8.30 | 2505.13 | 3014.35 | 222.60 | 243.92 | 266.06 | 5.489 | 80.6 | 100.0 | 88.94 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 184.60 | 468.32 | 3.763 |
| 0.5 | 1024 | 369.64 | 688.76 | 4.417 |
| 0.5 | 4096 | 1070.40 | 1281.87 | 4.609 |
| 1.0 | 256 | 124.33 | 214.41 | 3.480 |
| 1.0 | 1024 | 355.70 | 553.29 | 3.591 |
| 1.0 | 4096 | 992.55 | 1177.08 | 4.723 |
| 1.5 | 256 | 753.57 | 2230.62 | 5.398 |
| 1.5 | 1024 | 739.44 | 1185.52 | 5.350 |
| 1.5 | 4096 | 1967.67 | 3014.35 | 5.803 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
