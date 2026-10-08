# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T14:04:55.892255+00:00`
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
- Seed: `3101042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 38.75 token/s |
| Average TTFT | 126.48 ms |
| P50 TTFT | 95.71 ms |
| P90 TTFT | 185.79 ms |
| Average TPOT | 23.88 ms |
| Average E2E | 1.714 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 401.92 | 23.04 | 0.747 |
| 2 (fill:B) | 520 | 184.74 | 24.25 | 1.713 |
| 3 (fill:C) | 520 | 185.51 | 24.46 | 3.291 |
| 4 (fill:D) | 520 | 184.65 | 23.00 | 0.530 |
| 5 (fill:E) | 520 | 185.79 | 24.26 | 1.714 |
| 6 (fill:F) | 520 | 185.83 | 24.45 | 3.291 |
| 7 (refresh:A) | 520 | 38.92 | 23.00 | 0.384 |
| 8 (refresh:B) | 520 | 40.15 | 24.26 | 1.568 |
| 9 (pressure:G) | 520 | 184.00 | 23.01 | 0.529 |
| 10 (pressure:H) | 520 | 184.31 | 24.26 | 1.713 |
| 11 (pressure:I) | 520 | 185.22 | 24.45 | 3.290 |
| 12 (probe-survivor:I) | 520 | 41.31 | 24.45 | 3.147 |
| 13 (probe-survivor:H) | 520 | 43.01 | 24.25 | 1.571 |
| 14 (probe-survivor:G) | 520 | 38.14 | 22.99 | 0.383 |
| 15 (probe-survivor:B) | 520 | 36.53 | 24.25 | 1.564 |
| 16 (probe-survivor:A) | 520 | 43.37 | 22.99 | 0.388 |
| 17 (probe-survivor:F) | 520 | 42.07 | 24.45 | 3.147 |
| 18 (probe-survivor:E) | 520 | 42.84 | 24.25 | 1.571 |
| 19 (probe-evicted:D) | 520 | 95.71 | 23.04 | 0.441 |
| 20 (probe-evicted:C) | 520 | 185.66 | 24.45 | 3.291 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
