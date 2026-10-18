# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-18T03:13:59.649969+00:00`
- Git revision: `4aeab22`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-4B-AWQ/snapshots/master`

## Configuration

- Tensor parallelism: `1`
- Runtime dtype: `float16`
- Quantization: `awq`
- CUDA Graph shape policy: `coarse`
- Normal prefill budget: `64`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `32`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Decode-overload active threshold: `0` active decode requests
- Decode result before prefill: `False`
- Workload: `6` requests, balanced input lengths `[8, 16, 24]`, `4` output tokens
- Arrival rates: `[2.0]` requests/s
- Repeats: `1`
- SLOs: TTFT <= `10000.0` ms; per-request maximum TPOT <= `10000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 6.99 | 0.00 | 596.91 | 596.91 | 5.41 | 5.41 | 5.41 | 0.608 | 100.0 | 100.0 | 11.25 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 2.0 | 8 | 23.04 | 23.63 | 0.032 |
| 2.0 | 16 | 20.36 | 20.72 | 0.030 |
| 2.0 | 24 | 310.70 | 596.91 | 0.608 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
