# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-18T13:53:22.132238+00:00`
- Git revision: `25de0bb`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `float16`
- Quantization: `awq`
- KV cache page size: `16`
- KV cache pages: `16` (256 token total capacity)
- Prefill reservation policy: `page-aware`
- CUDA Graph shape policy: `coarse`
- Normal prefill budget: `64`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `32`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode-overload active threshold: `0` active decode requests
- Decode result before prefill: `False`
- Workload: `16` requests, balanced input lengths `[17]`, `128` output tokens
- Arrival rates: `[80.0]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `120000.0` ms; per-request maximum TPOT <= `120000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80.0 | 90.14 | 0.58 | 19718.82 | 21123.13 | 11.73 | 14.28 | 16.52 | 21.089 | 100.0 | 100.0 | 96.62 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 80.0 | 17 | 10621.33 | 19718.82 | 21.089 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
