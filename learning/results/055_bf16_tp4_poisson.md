# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T12:57:56.474449+00:00`
- Git revision: `17518c8`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `none`
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
| 0.5 | 59.54 | 9.49 | 1534.47 | 1768.81 | 294.14 | 317.88 | 548.80 | 5.840 | 97.2 | 100.0 | 88.60 |
| 1.0 | 100.39 | 10.35 | 3355.19 | 3730.62 | 297.72 | 319.94 | 320.84 | 7.882 | 66.7 | 100.0 | 92.58 |
| 1.5 | 104.68 | 1.82 | 2792.49 | 3348.98 | 301.53 | 319.58 | 320.54 | 7.758 | 72.2 | 100.0 | 92.08 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 211.62 | 491.18 | 5.838 |
| 0.5 | 1024 | 407.94 | 685.27 | 5.660 |
| 0.5 | 4096 | 1465.54 | 1768.81 | 6.026 |
| 1.0 | 256 | 884.58 | 2026.15 | 7.233 |
| 1.0 | 1024 | 1770.94 | 3423.96 | 8.354 |
| 1.0 | 4096 | 2160.22 | 3422.82 | 8.015 |
| 1.5 | 256 | 1133.25 | 1714.58 | 7.771 |
| 1.5 | 1024 | 1452.62 | 2365.60 | 7.354 |
| 1.5 | 4096 | 2242.28 | 3348.98 | 7.908 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
