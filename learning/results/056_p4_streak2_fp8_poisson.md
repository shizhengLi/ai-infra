# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T07:54:51.370637+00:00`
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
- Maximum prefill streak: `2`
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
| 0.5 | 70.35 | 14.72 | 1064.00 | 1250.31 | 176.60 | 445.66 | 480.83 | 4.428 | 100.0 | 100.0 | 75.80 |
| 1.0 | 98.83 | 8.78 | 982.60 | 1147.80 | 164.62 | 435.63 | 448.10 | 4.287 | 100.0 | 100.0 | 88.04 |
| 1.5 | 154.68 | 9.10 | 2441.38 | 2949.93 | 423.23 | 447.27 | 448.75 | 5.401 | 80.6 | 100.0 | 88.74 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 185.73 | 470.94 | 3.774 |
| 0.5 | 1024 | 339.38 | 656.80 | 4.407 |
| 0.5 | 4096 | 1049.50 | 1250.31 | 4.587 |
| 1.0 | 256 | 124.97 | 222.24 | 3.480 |
| 1.0 | 1024 | 305.04 | 480.69 | 3.580 |
| 1.0 | 4096 | 985.23 | 1147.80 | 4.736 |
| 1.5 | 256 | 727.46 | 2140.09 | 5.310 |
| 1.5 | 1024 | 687.29 | 1142.91 | 5.281 |
| 1.5 | 4096 | 1928.43 | 2949.93 | 5.723 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
