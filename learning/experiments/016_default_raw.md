# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T10:30:07.456255+00:00`
- Git revision: `0983cb1`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `0`
- Decode-active prefill budget: `0`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.5, 0.9, 1.5, 2.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 99.95 | 9.73 | 1471.54 | 1772.12 | 299.74 | 881.19 | 2524.07 | 11.670 | 98.6 | 83.3 | 95.97 |
| 0.9 | 183.03 | 45.32 | 2421.70 | 3171.30 | 717.07 | 3130.98 | 5512.10 | 19.647 | 76.4 | 30.6 | 96.47 |
| 1.5 | 240.75 | 2.17 | 4204.90 | 4534.31 | 639.37 | 5491.63 | 9304.89 | 22.324 | 38.9 | 19.4 | 95.21 |
| 2.5 | 261.95 | 3.91 | 8057.32 | 8578.99 | 38.44 | 7404.22 | 13318.34 | 22.727 | 12.5 | 40.3 | 94.59 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 176.72 | 327.52 | 11.572 |
| 0.5 | 1024 | 595.28 | 1725.81 | 11.664 |
| 0.5 | 4096 | 1458.19 | 1688.67 | 12.606 |
| 0.9 | 256 | 833.78 | 1872.03 | 19.432 |
| 0.9 | 1024 | 994.44 | 2705.81 | 19.694 |
| 0.9 | 4096 | 2029.03 | 3111.30 | 20.407 |
| 1.5 | 256 | 2648.14 | 4145.35 | 22.370 |
| 1.5 | 1024 | 2099.52 | 4103.11 | 22.570 |
| 1.5 | 4096 | 2771.12 | 4470.53 | 22.349 |
| 2.5 | 256 | 5204.77 | 8185.72 | 22.657 |
| 2.5 | 1024 | 6106.99 | 8448.65 | 23.039 |
| 2.5 | 4096 | 5595.28 | 8345.24 | 22.972 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
