# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T07:51:26.840686+00:00`
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
| 0.5 | 70.15 | 14.57 | 1188.28 | 1448.95 | 127.81 | 147.94 | 181.03 | 4.618 | 100.0 | 100.0 | 75.86 |
| 1.0 | 98.19 | 8.85 | 1095.38 | 1307.28 | 125.64 | 132.09 | 135.43 | 4.453 | 100.0 | 100.0 | 85.11 |
| 1.5 | 148.70 | 7.69 | 2729.03 | 3287.92 | 131.41 | 153.10 | 167.41 | 5.721 | 80.6 | 100.0 | 88.58 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 181.58 | 455.78 | 3.834 |
| 0.5 | 1024 | 391.11 | 762.47 | 4.595 |
| 0.5 | 4096 | 1182.24 | 1448.95 | 4.827 |
| 1.0 | 256 | 129.29 | 238.36 | 3.549 |
| 1.0 | 1024 | 371.77 | 561.07 | 3.689 |
| 1.0 | 4096 | 1071.87 | 1307.28 | 4.890 |
| 1.5 | 256 | 803.95 | 2334.11 | 5.624 |
| 1.5 | 1024 | 829.48 | 1314.29 | 5.565 |
| 1.5 | 4096 | 2121.55 | 3287.92 | 6.030 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
