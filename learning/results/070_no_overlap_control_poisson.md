# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T18:07:23.679285+00:00`
- Git revision: `9f8e00b`
- Git worktree dirty: `False`
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
| 1.5 | 70.69 | 14.47 | 1322.42 | 1322.42 | 154.00 | 366.03 | 573.40 | 2.968 | 100.0 | 100.0 | 78.06 |
| 2.5 | 84.08 | 3.79 | 3187.29 | 3187.29 | 134.15 | 155.81 | 157.75 | 4.372 | 68.8 | 100.0 | 80.64 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 1.5 | 256 | 135.52 | 192.64 | 2.158 |
| 1.5 | 1024 | 601.45 | 1161.54 | 2.874 |
| 1.5 | 2048 | 1081.46 | 1310.83 | 2.660 |
| 2.5 | 256 | 783.21 | 1460.83 | 2.810 |
| 2.5 | 1024 | 1548.44 | 3002.27 | 4.186 |
| 2.5 | 2048 | 2182.57 | 3049.19 | 4.275 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
