# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T11:30:01.417252+00:00`
- Git revision: `82482c1`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `2304`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `0`
- Decode-overload prefill budget: `0`
- Decode-overload threshold: `0` pending tokens
- Workload: `24` requests, balanced input lengths `[256, 1024, 4096]`, `256` output tokens
- Arrival rates: `[2.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 | 258.87 | 3.87 | 7119.57 | 7572.15 | 764.78 | 778.78 | 803.59 | 22.539 | 12.5 | 100.0 | 95.04 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 2.5 | 256 | 4282.70 | 7100.64 | 22.403 |
| 2.5 | 1024 | 5566.62 | 7570.76 | 22.817 |
| 2.5 | 4096 | 4763.59 | 7536.43 | 22.772 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
