# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T14:08:43.748679+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Pressure output lengths A-I: `[16, 64, 128, 16, 64, 128, 16, 64, 128]`
- Output length: `16` tokens
- Seed: `3102042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 38.72 token/s |
| Average TTFT | 127.32 ms |
| P50 TTFT | 79.67 ms |
| P90 TTFT | 186.88 ms |
| Average TPOT | 23.89 ms |
| Average E2E | 1.715 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 415.81 | 22.99 | 0.761 |
| 2 (fill:B) | 520 | 194.35 | 24.26 | 1.723 |
| 3 (fill:C) | 520 | 186.88 | 24.46 | 3.293 |
| 4 (fill:D) | 520 | 185.93 | 23.03 | 0.531 |
| 5 (fill:E) | 520 | 185.41 | 24.26 | 1.714 |
| 6 (fill:F) | 520 | 185.85 | 24.46 | 3.292 |
| 7 (refresh:A) | 520 | 44.39 | 23.02 | 0.390 |
| 8 (refresh:B) | 520 | 44.19 | 24.27 | 1.573 |
| 9 (pressure:G) | 520 | 184.14 | 23.01 | 0.529 |
| 10 (pressure:H) | 520 | 185.53 | 24.27 | 1.715 |
| 11 (pressure:I) | 520 | 185.96 | 24.46 | 3.293 |
| 12 (probe-survivor:I) | 520 | 42.86 | 24.45 | 3.149 |
| 13 (probe-survivor:H) | 520 | 39.04 | 24.26 | 1.567 |
| 14 (probe-survivor:G) | 520 | 39.90 | 23.02 | 0.385 |
| 15 (probe-survivor:B) | 520 | 40.62 | 24.25 | 1.569 |
| 16 (probe-survivor:A) | 520 | 41.63 | 23.02 | 0.387 |
| 17 (probe-survivor:F) | 520 | 41.26 | 24.46 | 3.148 |
| 18 (probe-survivor:E) | 520 | 37.97 | 24.25 | 1.566 |
| 19 (probe-evicted:D) | 520 | 79.67 | 23.05 | 0.425 |
| 20 (probe-evicted:C) | 520 | 184.94 | 24.46 | 3.292 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
