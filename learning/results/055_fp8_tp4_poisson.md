# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T13:02:26.197636+00:00`
- Git revision: `17518c8`
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
| 0.5 | 63.05 | 10.91 | 1099.33 | 1284.48 | 170.36 | 239.25 | 397.37 | 3.757 | 100.0 | 100.0 | 72.02 |
| 1.0 | 125.05 | 23.02 | 1553.18 | 1940.93 | 226.09 | 246.54 | 260.74 | 5.113 | 97.2 | 100.0 | 85.92 |
| 1.5 | 133.14 | 11.27 | 1352.11 | 1720.27 | 224.33 | 250.77 | 276.82 | 4.975 | 97.2 | 100.0 | 90.33 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 134.63 | 265.54 | 3.764 |
| 0.5 | 1024 | 298.14 | 478.17 | 3.585 |
| 0.5 | 4096 | 1072.56 | 1284.48 | 3.749 |
| 1.0 | 256 | 298.28 | 757.63 | 4.970 |
| 1.0 | 1024 | 838.28 | 1621.23 | 5.002 |
| 1.0 | 4096 | 1345.33 | 1872.88 | 5.458 |
| 1.5 | 256 | 413.63 | 720.26 | 4.660 |
| 1.5 | 1024 | 522.44 | 790.55 | 4.528 |
| 1.5 | 4096 | 1344.45 | 1720.27 | 5.065 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
