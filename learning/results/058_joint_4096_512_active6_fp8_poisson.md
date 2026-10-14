# L20 Poisson-arrival benchmark result

## Principle

Independent exponential inter-arrival times approximate a memoryless online request stream. The
same seeded arrival schedules and prompts are reused across prefill budgets. TTFT ends at the first
streamed token; TPOT excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-14T09:58:03.211568+00:00`
- Git revision: `f0ee64b`
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
- Decode-overload prefill budget: `512`
- Decode-overload threshold: `4096` pending tokens
- Decode-overload active threshold: `6` active decode requests
- Decode result before prefill: `False`
- Workload: `12` requests, balanced input lengths `[256, 1024, 4096]`, `128` output tokens
- Arrival rates: `[0.5, 1.0, 1.5]` requests/s
- Repeats: `3`
- SLOs: TTFT <= `2000.0` ms; per-request maximum TPOT <= `1000.0` ms

## Aggregate results

| Rate req/s | Output tok/s | Stddev | P90 TTFT ms | P99 TTFT ms | P99 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | TTFT SLO % | TPOT SLO % | GPU util % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 70.30 | 14.68 | 1084.28 | 1289.06 | 215.85 | 241.67 | 269.37 | 4.447 | 100.0 | 100.0 | 75.38 |
| 1.0 | 98.64 | 8.76 | 1005.02 | 1170.47 | 168.84 | 231.01 | 237.38 | 4.292 | 100.0 | 100.0 | 86.49 |
| 1.5 | 153.47 | 8.72 | 2496.96 | 2981.49 | 224.65 | 235.16 | 238.58 | 5.456 | 80.6 | 100.0 | 88.45 |

## Results by input length

| Rate req/s | Input tokens | Average TTFT ms | P90 TTFT ms | P90 E2E s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 256 | 183.61 | 464.70 | 3.768 |
| 0.5 | 1024 | 372.30 | 694.88 | 4.433 |
| 0.5 | 4096 | 1074.55 | 1289.06 | 4.623 |
| 1.0 | 256 | 121.68 | 211.03 | 3.476 |
| 1.0 | 1024 | 353.32 | 557.34 | 3.591 |
| 1.0 | 4096 | 997.70 | 1170.47 | 4.736 |
| 1.5 | 256 | 731.57 | 2148.75 | 5.364 |
| 1.5 | 1024 | 720.83 | 1160.39 | 5.311 |
| 1.5 | 4096 | 1952.27 | 2981.49 | 5.773 |

Per-repeat measurements, scheduled arrival offsets, and normalized raw stream timestamps are stored
in the machine-readable JSON result.
