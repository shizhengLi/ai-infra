# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T09:16:09.998369+00:00`
- Git revision: `8542c51`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Configuration

- Tensor parallelism: `4`
- Runtime dtype: `bfloat16`
- Quantization: `fp8`
- Normal prefill budget: `4096`
- Maximum prefill streak: `1`
- Decode-active prefill budget: `1024`
- Decode-overload prefill budget: `256`
- Decode-overload threshold: `4096` pending tokens
- Decode result before prefill: `False`
- Workload: `12` requests, balanced input lengths `[256, 1024, 4096]`, `128` output tokens
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 70.30 | 14.68 | 1031.55 | 1295.54 | 213.88 | 286.95 | 412.76 | 4.473 | 100.0 | 100.0 | 74.66 |
| 1.0 | 98.52 | 8.71 | 997.39 | 1144.03 | 173.84 | 224.02 | 224.86 | 4.304 | 100.0 | 100.0 | 87.11 |
| 1.5 | 148.02 | 6.38 | 2884.95 | 3353.30 | 220.74 | 229.77 | 230.46 | 5.827 | 80.6 | 100.0 | 89.76 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 180.34 | 450.96 | 3.795 |
| 0.5 | 1024 | 354.67 | 666.22 | 4.473 |
| 0.5 | 4096 | 1058.83 | 1295.54 | 4.644 |
| 1.0 | 256 | 126.86 | 227.55 | 3.491 |
| 1.0 | 1024 | 334.93 | 509.42 | 3.611 |
| 1.0 | 4096 | 990.05 | 1144.03 | 4.746 |
| 1.5 | 256 | 859.39 | 2545.08 | 5.776 |
| 1.5 | 1024 | 864.14 | 1345.64 | 5.577 |
| 1.5 | 4096 | 2158.43 | 3353.30 | 6.136 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
