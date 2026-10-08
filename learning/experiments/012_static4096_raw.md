# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T09:31:25.579615+00:00`
- Git revision: `d72a8f6`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `4096`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[0.9, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 210.05 | 21.87 | 3284.22 | 3626.05 | 1073.53 | 1409.27 | 1553.35 | 20.863 | 58.3 | 23.6 | 95.24 |
| 1.5 | 233.39 | 21.64 | 4673.56 | 5406.20 | 1328.08 | 1390.14 | 1455.00 | 21.821 | 33.3 | 12.5 | 95.52 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 936.46 | 2923.31 | 21.061 |
| 0.9 | 1024 | 1537.12 | 3017.22 | 21.058 |
| 0.9 | 4096 | 2583.41 | 3626.05 | 21.209 |
| 1.5 | 256 | 2250.16 | 4376.60 | 21.967 |
| 1.5 | 1024 | 2527.94 | 4387.61 | 21.882 |
| 1.5 | 4096 | 3605.45 | 5194.49 | 22.022 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
