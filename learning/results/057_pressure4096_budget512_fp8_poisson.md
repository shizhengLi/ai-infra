# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T09:09:24.984950+00:00`
- Git revision: `8542c51`
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
- Decode-overload prefill budget: `512`
- Decode-overload threshold: `4096` pending tokens
- Decode result before prefill: `False`
- Workload: `12` requests, balanced input lengths `[256, 1024, 4096]`, `128` output tokens
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 70.30 | 14.68 | 1055.08 | 1276.64 | 213.86 | 273.85 | 371.31 | 4.451 | 100.0 | 100.0 | 74.96 |
| 1.0 | 98.68 | 8.77 | 1004.46 | 1169.85 | 186.57 | 226.17 | 229.73 | 4.298 | 100.0 | 100.0 | 87.71 |
| 1.5 | 151.68 | 7.63 | 2608.29 | 3102.79 | 221.35 | 231.09 | 232.54 | 5.577 | 80.6 | 100.0 | 88.33 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 175.96 | 435.46 | 3.777 |
| 0.5 | 1024 | 353.92 | 651.04 | 4.440 |
| 0.5 | 4096 | 1065.36 | 1276.64 | 4.634 |
| 1.0 | 256 | 126.55 | 224.42 | 3.479 |
| 1.0 | 1024 | 342.67 | 519.36 | 3.620 |
| 1.0 | 4096 | 996.76 | 1169.85 | 4.741 |
| 1.5 | 256 | 752.94 | 2234.23 | 5.485 |
| 1.5 | 1024 | 748.10 | 1205.79 | 5.396 |
| 1.5 | 4096 | 2015.33 | 3102.79 | 5.896 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
