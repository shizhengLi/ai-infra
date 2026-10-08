# L20 Radix Cache workload result

## Configuration

- Scenario: `unique`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:34:00.372063+00:00`
- Git revision: `f27eb8d`
- Tensor parallelism: `4`
- Memory ratio / KV pages: `0.8` / `305066`
- CUDA Graph maximum batch size: `64`
- Scheduling profile: prefill streak `1`, active prefill budget `2304`, result-before-prefill enabled
- Output length: `32` tokens
- Seed: `3000042`
- Expected measured server UIDs: `1-12` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 12 |
| Average input tokens | 1032.00 |
| Total input tokens | 12384 |
| Sequential output throughput | 28.78 token/s |
| Average TTFT | 370.13 ms |
| P50 TTFT | 349.56 ms |
| P90 TTFT | 350.65 ms |
| Average TPOT | 23.93 ms |
| Average E2E | 1.112 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1032 | 596.80 | 23.92 | 1.338 |
| 2 | 1032 | 349.98 | 23.95 | 1.092 |
| 3 | 1032 | 350.65 | 23.92 | 1.092 |
| 4 | 1032 | 349.37 | 23.93 | 1.091 |
| 5 | 1032 | 349.15 | 23.92 | 1.091 |
| 6 | 1032 | 349.18 | 23.93 | 1.091 |
| 7 | 1032 | 348.63 | 23.92 | 1.090 |
| 8 | 1032 | 348.23 | 23.93 | 1.090 |
| 9 | 1032 | 349.86 | 23.93 | 1.092 |
| 10 | 1032 | 350.33 | 23.92 | 1.092 |
| 11 | 1032 | 349.80 | 23.93 | 1.092 |
| 12 | 1032 | 349.56 | 23.93 | 1.091 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
