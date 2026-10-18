# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-18T11:30:46.687930+00:00`
- Git revision: `9d66f74`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `float16`
- Quantization: `awq`
- KV cache page size: `8`
- KV cache pages: `16` (128 token total capacity)
- CUDA Graph shape policy: `coarse`
- Normal prefill budget: `64`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `32`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode-overload active threshold: `0` active decode requests
- Decode result before prefill: `False`
- Workload: `8` requests, balanced input lengths `[17]`, `16` output tokens
- Arrival rates: `[20.0]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `120000.0` ms; per-request maximum TPOT <= `120000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20.0 | 132.04 | 6.64 | 403.79 | 403.79 | 36.58 | 43.16 | 44.81 | 0.585 | 100.0 | 100.0 | 66.83 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 20.0 | 17 | 221.83 | 403.79 | 0.585 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
