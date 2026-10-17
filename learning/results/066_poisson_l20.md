# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T15:19:59.830658+00:00`
- Git revision: `8407b33`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- CUDA Graph shape policy: `l20`
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
| 4.0 | 224.66 | 11.76 | 298.51 | 357.61 | 156.01 | 392.45 | 543.68 | 2.292 | 100.0 | 100.0 | 90.55 |
| 8.0 | 372.53 | 11.54 | 368.01 | 462.38 | 474.89 | 835.72 | 1140.78 | 4.289 | 100.0 | 98.4 | 88.16 |
| 12.0 | 463.46 | 12.51 | 941.52 | 1048.06 | 421.27 | 1905.70 | 2184.77 | 4.113 | 100.0 | 57.8 | 81.79 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 4.0 | 128 | 139.87 | 208.80 | 2.289 |
| 4.0 | 256 | 198.75 | 287.13 | 2.292 |
| 4.0 | 512 | 226.57 | 273.92 | 2.257 |
| 8.0 | 128 | 201.24 | 344.44 | 4.298 |
| 8.0 | 256 | 218.95 | 370.77 | 4.202 |
| 8.0 | 512 | 313.69 | 412.90 | 4.123 |
| 12.0 | 128 | 500.33 | 781.69 | 4.068 |
| 12.0 | 256 | 544.17 | 852.06 | 4.171 |
| 12.0 | 512 | 715.56 | 1010.69 | 3.802 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
