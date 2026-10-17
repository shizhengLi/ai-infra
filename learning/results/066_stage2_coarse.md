# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T15:28:45.324170+00:00`
- Git revision: `042d0d7`
- Git worktree dirty: `False`
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
- Workload: `24` requests, balanced input lengths `[128, 256, 512]`, `128` output tokens
- Arrival rates: `[4.0, 8.0, 12.0]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0 | 361.75 | 42.00 | 283.27 | 342.41 | 226.60 | 547.26 | 1034.48 | 4.656 | 100.0 | 90.3 | 90.78 |
| 8.0 | 484.36 | 16.37 | 469.78 | 493.43 | 314.11 | 705.52 | 1201.08 | 5.159 | 100.0 | 97.2 | 87.21 |
| 12.0 | 543.22 | 30.27 | 583.18 | 631.91 | 261.09 | 1045.30 | 1698.15 | 5.160 | 100.0 | 79.2 | 86.54 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 4.0 | 128 | 145.73 | 256.29 | 4.695 |
| 4.0 | 256 | 177.03 | 292.89 | 4.588 |
| 4.0 | 512 | 243.82 | 335.10 | 4.690 |
| 8.0 | 128 | 252.21 | 475.56 | 5.145 |
| 8.0 | 256 | 265.62 | 400.97 | 5.211 |
| 8.0 | 512 | 323.83 | 487.49 | 5.171 |
| 12.0 | 128 | 370.47 | 585.13 | 5.118 |
| 12.0 | 256 | 350.40 | 544.96 | 5.148 |
| 12.0 | 512 | 433.43 | 627.14 | 5.239 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
