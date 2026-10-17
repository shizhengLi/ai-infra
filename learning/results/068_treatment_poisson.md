# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T16:48:19.823317+00:00`
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
| 1.5 | 15.97 | 0.00 | 1670.08 | 1983.71 | 2338.20 | 2764.36 | 2764.36 | 40.327 | 100.0 | 50.0 | 93.95 |
| 2.5 | 10.42 | 0.00 | 3479.94 | 3801.44 | 3524.69 | 3861.33 | 3861.33 | 69.474 | 58.3 | 33.3 | 97.03 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 256 | 917.60 | 1670.08 | 23.067 |
| 1.5 | 1024 | 1336.74 | 1983.71 | 27.354 |
| 1.5 | 2048 | 819.63 | 1442.48 | 40.939 |
| 2.5 | 256 | 1394.31 | 3479.94 | 23.529 |
| 2.5 | 1024 | 2318.01 | 3801.44 | 69.009 |
| 2.5 | 2048 | 1785.75 | 3440.10 | 71.562 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
