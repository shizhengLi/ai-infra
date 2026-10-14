# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T07:44:43.232059+00:00`
- Git revision: `7465362`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- Normal prefill budget: `2048`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `1024`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode result before prefill: `False`
- Workload: `12` requests, balanced input lengths `[256, 1024, 4096]`, `128` output tokens
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 70.31 | 14.69 | 1075.05 | 1283.11 | 216.83 | 241.22 | 270.14 | 4.429 | 100.0 | 100.0 | 75.38 |
| 1.0 | 98.52 | 8.83 | 1012.65 | 1176.46 | 168.32 | 231.17 | 238.00 | 4.300 | 100.0 | 100.0 | 87.68 |
| 1.5 | 153.76 | 8.57 | 2489.49 | 2990.17 | 227.51 | 238.21 | 240.54 | 5.445 | 80.6 | 100.0 | 88.91 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 170.16 | 413.47 | 3.762 |
| 0.5 | 1024 | 373.53 | 706.04 | 4.419 |
| 0.5 | 4096 | 1075.69 | 1283.11 | 4.611 |
| 1.0 | 256 | 120.61 | 208.97 | 3.472 |
| 1.0 | 1024 | 362.22 | 565.79 | 3.605 |
| 1.0 | 4096 | 1005.02 | 1176.46 | 4.746 |
| 1.5 | 256 | 737.20 | 2178.09 | 5.354 |
| 1.5 | 1024 | 737.37 | 1188.14 | 5.327 |
| 1.5 | 4096 | 1958.96 | 2990.17 | 5.762 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
