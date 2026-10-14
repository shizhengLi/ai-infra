# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T07:48:04.847089+00:00`
- Git revision: `7465362`
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
- Decode-active prefill budget: `512`
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
| 0.5 | 70.15 | 14.58 | 1165.23 | 1428.92 | 128.00 | 148.44 | 181.75 | 4.573 | 100.0 | 100.0 | 75.99 |
| 1.0 | 98.22 | 8.79 | 1076.36 | 1301.94 | 126.16 | 131.81 | 139.88 | 4.426 | 100.0 | 100.0 | 87.14 |
| 1.5 | 148.62 | 7.71 | 2725.45 | 3290.59 | 131.17 | 158.90 | 174.23 | 5.722 | 80.6 | 100.0 | 89.34 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 186.19 | 479.78 | 3.841 |
| 0.5 | 1024 | 387.31 | 746.40 | 4.550 |
| 0.5 | 4096 | 1167.83 | 1428.92 | 4.803 |
| 1.0 | 256 | 132.45 | 247.29 | 3.552 |
| 1.0 | 1024 | 372.66 | 574.21 | 3.670 |
| 1.0 | 4096 | 1065.43 | 1301.94 | 4.859 |
| 1.5 | 256 | 801.93 | 2335.92 | 5.625 |
| 1.5 | 1024 | 825.90 | 1312.86 | 5.567 |
| 1.5 | 4096 | 2119.50 | 3290.59 | 6.033 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
