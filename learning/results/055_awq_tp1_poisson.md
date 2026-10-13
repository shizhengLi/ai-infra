# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T13:08:38.888861+00:00`
- Git revision: `17518c8`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`

## Configuration

- Tensor parallelism: `1`
- Runtime dtype: `bfloat16`
- Quantization: `awq`
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
| 0.5 | 56.44 | 7.68 | 3481.09 | 4807.42 | 687.73 | 914.08 | 1316.32 | 12.046 | 61.1 | 94.4 | 95.69 |
| 1.0 | 67.20 | 3.83 | 9327.50 | 10370.65 | 692.85 | 723.32 | 724.68 | 15.547 | 27.8 | 100.0 | 94.90 |
| 1.5 | 68.33 | 1.88 | 9958.94 | 10555.36 | 688.23 | 714.16 | 727.34 | 15.043 | 22.2 | 100.0 | 94.14 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 723.35 | 2217.73 | 11.094 |
| 0.5 | 1024 | 1030.97 | 1695.00 | 10.479 |
| 0.5 | 4096 | 3493.86 | 4807.42 | 12.025 |
| 1.0 | 256 | 3925.48 | 8244.33 | 14.161 |
| 1.0 | 1024 | 5334.45 | 9186.91 | 16.434 |
| 1.0 | 4096 | 6465.93 | 10370.65 | 15.501 |
| 1.5 | 256 | 4179.11 | 6938.22 | 14.306 |
| 1.5 | 1024 | 5627.03 | 9805.40 | 14.692 |
| 1.5 | 4096 | 6484.61 | 10555.36 | 15.450 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
