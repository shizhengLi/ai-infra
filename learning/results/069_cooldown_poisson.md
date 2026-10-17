# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T17:09:29.399033+00:00`
- Git revision: `6fe6d7f`
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
| 1.5 | 66.95 | 0.00 | 1235.23 | 1648.82 | 151.37 | 160.08 | 160.08 | 2.658 | 100.0 | 100.0 | 87.45 |
| 2.5 | 73.23 | 0.00 | 3678.23 | 4445.97 | 151.17 | 153.78 | 153.78 | 5.186 | 50.0 | 100.0 | 87.28 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 256 | 655.20 | 1235.23 | 2.658 |
| 1.5 | 1024 | 493.34 | 764.64 | 2.053 |
| 1.5 | 2048 | 1026.72 | 1648.82 | 3.072 |
| 2.5 | 256 | 1223.91 | 3007.74 | 4.516 |
| 2.5 | 1024 | 2510.77 | 4445.97 | 5.505 |
| 2.5 | 2048 | 2320.39 | 3678.23 | 5.186 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
