# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T14:06:52.903533+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `16` pages
- Adaptive reserve maximum: `0` pages
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
| Sequential output throughput | 38.77 token/s |
| Average TTFT | 124.87 ms |
| P50 TTFT | 79.46 ms |
| P90 TTFT | 186.06 ms |
| Average TPOT | 23.89 ms |
| Average E2E | 1.713 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 372.35 | 23.05 | 0.718 |
| 2 (fill:B) | 520 | 185.24 | 24.26 | 1.714 |
| 3 (fill:C) | 520 | 185.13 | 24.46 | 3.292 |
| 4 (fill:D) | 520 | 185.34 | 23.00 | 0.530 |
| 5 (fill:E) | 520 | 184.92 | 24.26 | 1.714 |
| 6 (fill:F) | 520 | 185.55 | 24.46 | 3.292 |
| 7 (refresh:A) | 520 | 39.52 | 23.01 | 0.385 |
| 8 (refresh:B) | 520 | 43.38 | 24.26 | 1.571 |
| 9 (pressure:G) | 520 | 185.46 | 23.04 | 0.531 |
| 10 (pressure:H) | 520 | 186.12 | 24.27 | 1.715 |
| 11 (pressure:I) | 520 | 186.06 | 24.46 | 3.293 |
| 12 (probe-survivor:I) | 520 | 38.62 | 24.46 | 3.145 |
| 13 (probe-survivor:H) | 520 | 40.26 | 24.26 | 1.569 |
| 14 (probe-survivor:G) | 520 | 40.79 | 22.98 | 0.385 |
| 15 (probe-survivor:B) | 520 | 49.06 | 24.27 | 1.578 |
| 16 (probe-survivor:A) | 520 | 41.79 | 23.03 | 0.387 |
| 17 (probe-survivor:F) | 520 | 42.12 | 24.46 | 3.148 |
| 18 (probe-survivor:E) | 520 | 40.78 | 24.25 | 1.569 |
| 19 (probe-evicted:D) | 520 | 79.46 | 23.04 | 0.425 |
| 20 (probe-evicted:C) | 520 | 185.55 | 24.47 | 3.293 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
