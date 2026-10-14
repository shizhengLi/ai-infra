# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T09:12:47.900794+00:00`
- Git revision: `8542c51`
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
- Decode-overload prefill budget: `512`
- Decode-overload threshold: `2048` pending tokens
- Decode result before prefill: `False`
- Workload: `12` requests, balanced input lengths `[256, 1024, 4096]`, `128` output tokens
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 70.26 | 14.65 | 1112.07 | 1349.14 | 134.42 | 272.63 | 371.82 | 4.493 | 100.0 | 100.0 | 76.64 |
| 1.0 | 98.37 | 8.74 | 1036.98 | 1220.89 | 154.73 | 222.13 | 230.39 | 4.357 | 100.0 | 100.0 | 86.30 |
| 1.5 | 149.74 | 7.48 | 2687.47 | 3208.62 | 165.51 | 225.93 | 231.03 | 5.668 | 80.6 | 100.0 | 89.16 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 176.77 | 442.07 | 3.799 |
| 0.5 | 1024 | 359.28 | 686.45 | 4.479 |
| 0.5 | 4096 | 1107.27 | 1349.14 | 4.694 |
| 1.0 | 256 | 126.92 | 219.26 | 3.516 |
| 1.0 | 1024 | 349.55 | 538.13 | 3.635 |
| 1.0 | 4096 | 1026.48 | 1220.89 | 4.796 |
| 1.5 | 256 | 786.90 | 2304.55 | 5.573 |
| 1.5 | 1024 | 808.83 | 1295.63 | 5.527 |
| 1.5 | 4096 | 2091.20 | 3208.62 | 5.983 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
