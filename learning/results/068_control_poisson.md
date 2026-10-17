# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T16:44:05.679399+00:00`
- Git revision: `6396a2c`
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
- Workload: `12` requests, balanced input lengths `[256, 1024, 2048]`, `64` output tokens
- Arrival rates: `[1.5, 2.5]` requests/s
- Repeats: `1`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 66.91 | 0.00 | 1234.77 | 1647.38 | 150.63 | 154.37 | 154.37 | 2.668 | 100.0 | 100.0 | 86.30 |
| 2.5 | 72.30 | 0.00 | 3766.34 | 4566.55 | 153.04 | 156.36 | 156.36 | 5.311 | 50.0 | 100.0 | 85.74 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 256 | 650.62 | 1234.77 | 2.668 |
| 1.5 | 1024 | 492.76 | 765.07 | 2.064 |
| 1.5 | 2048 | 1020.02 | 1647.38 | 3.081 |
| 2.5 | 256 | 1259.22 | 3093.72 | 4.639 |
| 2.5 | 1024 | 2579.64 | 4566.55 | 5.632 |
| 2.5 | 2048 | 2371.96 | 3766.34 | 5.311 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
