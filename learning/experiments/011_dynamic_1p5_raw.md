# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T09:18:49.582166+00:00`
- Git revision: `98764aa`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Normal prefill budget: `8192`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `4096`
- Decode-overload prefill budget: `2048`
- Decode-overload threshold: `4096` pending tokens
- Workload: `16 x 1024 input / 256 output tokens`
- Arrival rates: `[1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 215.54 | 4.07 | 935.87 | 989.67 | 317.91 | 609.88 | 870.02 | 12.183 | 100.0 | 100.0 | 93.87 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
