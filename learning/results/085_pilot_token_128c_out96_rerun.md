# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-18T15:02:03.631359+00:00`
- Git revision: `c251c20`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`

## Configuration

- Tensor parallelism: `4`
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
- Workload: `16` requests, balanced input lengths `[17]`, `96` output tokens
- Arrival rates: `[80.0]` requests/s
- Repeats: `1`
- SLOs: TTFT <= `120000.0` ms; per-request maximum TPOT <= `120000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80.0 | 87.87 | 0.00 | 15183.72 | 16268.17 | 12.86 | 20.54 | 21.17 | 16.226 | 100.0 | 100.0 | 95.41 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 80.0 | 17 | 8188.77 | 15183.72 | 16.226 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
