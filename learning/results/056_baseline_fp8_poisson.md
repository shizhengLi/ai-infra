# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T07:39:46.051709+00:00`
- Git revision: `7465362`
- Git worktree dirty: `False`
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
| 0.5 | 70.32 | 14.69 | 1071.98 | 1279.27 | 216.34 | 240.52 | 269.07 | 4.441 | 100.0 | 100.0 | 74.57 |
| 1.0 | 98.64 | 8.78 | 1005.68 | 1182.53 | 166.15 | 237.51 | 247.01 | 4.314 | 100.0 | 100.0 | 87.19 |
| 1.5 | 153.81 | 8.59 | 2489.72 | 2977.62 | 230.46 | 244.49 | 260.23 | 5.429 | 80.6 | 100.0 | 88.65 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 182.43 | 459.50 | 3.768 |
| 0.5 | 1024 | 374.30 | 690.19 | 4.426 |
| 0.5 | 4096 | 1065.35 | 1279.27 | 4.619 |
| 1.0 | 256 | 123.53 | 212.44 | 3.496 |
| 1.0 | 1024 | 360.20 | 562.08 | 3.611 |
| 1.0 | 4096 | 1001.59 | 1182.53 | 4.759 |
| 1.5 | 256 | 734.83 | 2161.31 | 5.337 |
| 1.5 | 1024 | 745.00 | 1195.73 | 5.332 |
| 1.5 | 4096 | 1954.65 | 2977.62 | 5.749 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
