# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T15:17:25.731373+00:00`
- Git revision: `8407b33`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- CUDA Graph shape policy: `coarse`
- Normal prefill budget: `20000`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `1024`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode-overload active threshold: `0` active decode requests
- Decode result before prefill: `False`
- Workload: `32` requests, balanced input lengths `[128, 256, 512]`, `64` output tokens
- Arrival rates: `[4.0, 8.0, 12.0]` requests/s
- Repeats: `2`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0 | 223.38 | 11.98 | 328.62 | 381.30 | 142.52 | 365.67 | 543.61 | 2.377 | 100.0 | 100.0 | 91.74 |
| 8.0 | 372.41 | 12.10 | 378.05 | 458.79 | 334.37 | 535.95 | 973.37 | 4.300 | 100.0 | 100.0 | 81.90 |
| 12.0 | 463.62 | 13.74 | 961.77 | 1076.64 | 428.39 | 1874.42 | 2237.23 | 4.110 | 100.0 | 57.8 | 87.33 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 4.0 | 128 | 161.29 | 244.32 | 2.351 |
| 4.0 | 256 | 197.50 | 286.54 | 2.342 |
| 4.0 | 512 | 230.68 | 276.55 | 2.325 |
| 8.0 | 128 | 204.90 | 356.10 | 4.307 |
| 8.0 | 256 | 206.23 | 323.09 | 4.220 |
| 8.0 | 512 | 312.79 | 424.16 | 4.134 |
| 12.0 | 128 | 501.01 | 741.29 | 4.066 |
| 12.0 | 256 | 548.21 | 927.23 | 4.166 |
| 12.0 | 512 | 723.23 | 997.14 | 3.798 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
