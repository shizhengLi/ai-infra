# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T08:36:55.361464+00:00`
- Git revision: `13593d2`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- Host append mode: `cat` (legacy A/B only)
- Normal prefill budget: `4096`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `1024`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode-overload active threshold: `0` active decode requests
- Decode result before prefill: `False`
- Workload: `6` requests, balanced input lengths `[256, 1024, 4096]`, `512` output tokens
- Arrival rates: `[1.5]` requests/s
- Repeats: `1`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 210.10 | 0.00 | 2468.68 | 2468.68 | 56.60 | 291.34 | 488.86 | 12.775 | 50.0 | 100.0 | 91.67 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 256 | 114.68 | 115.45 | 12.669 |
| 1.5 | 1024 | 2385.53 | 2468.68 | 12.567 |
| 1.5 | 4096 | 1867.11 | 2324.46 | 12.775 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
