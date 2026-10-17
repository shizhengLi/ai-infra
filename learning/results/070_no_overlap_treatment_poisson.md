# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T18:15:56.713064+00:00`
- Git revision: `9f8e00b`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- CUDA Graph shape policy: `coarse`
- Normal prefill budget: `2048`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `512`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode-overload active threshold: `0` active decode requests
- Decode result before prefill: `True`
- Workload: `8` requests, balanced input lengths `[256, 1024, 2048]`, `64` output tokens
- Arrival rates: `[1.5, 2.5]` requests/s
- Repeats: `2`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 69.52 | 13.34 | 1396.08 | 1396.08 | 154.44 | 156.95 | 157.53 | 2.979 | 93.8 | 100.0 | 78.94 |
| 2.5 | 84.21 | 3.79 | 3185.09 | 3185.09 | 134.14 | 156.54 | 156.85 | 4.364 | 68.8 | 100.0 | 80.25 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 256 | 129.22 | 189.29 | 2.158 |
| 1.5 | 1024 | 608.18 | 1178.33 | 2.896 |
| 1.5 | 2048 | 1162.68 | 1396.08 | 2.739 |
| 2.5 | 256 | 789.53 | 1469.88 | 2.819 |
| 2.5 | 1024 | 1549.86 | 2998.75 | 4.178 |
| 2.5 | 2048 | 2186.44 | 3050.77 | 4.272 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
