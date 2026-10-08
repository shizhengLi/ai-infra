# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T14:00:41.566497+00:00`
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
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 38.71 token/s |
| Average TTFT | 127.90 ms |
| P50 TTFT | 80.70 ms |
| P90 TTFT | 186.29 ms |
| Average TPOT | 23.89 ms |
| Average E2E | 1.715 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 414.51 | 23.03 | 0.760 |
| 2 (fill:B) | 520 | 185.25 | 24.25 | 1.713 |
| 3 (fill:C) | 520 | 186.68 | 24.46 | 3.293 |
| 4 (fill:D) | 520 | 185.62 | 23.01 | 0.531 |
| 5 (fill:E) | 520 | 185.94 | 24.26 | 1.714 |
| 6 (fill:F) | 520 | 185.82 | 24.46 | 3.292 |
| 7 (refresh:A) | 520 | 43.21 | 23.07 | 0.389 |
| 8 (refresh:B) | 520 | 42.92 | 24.26 | 1.571 |
| 9 (pressure:G) | 520 | 185.67 | 23.05 | 0.531 |
| 10 (pressure:H) | 520 | 186.11 | 24.26 | 1.715 |
| 11 (pressure:I) | 520 | 185.57 | 24.46 | 3.291 |
| 12 (probe-survivor:I) | 520 | 43.63 | 24.46 | 3.149 |
| 13 (probe-survivor:H) | 520 | 43.34 | 24.26 | 1.572 |
| 14 (probe-survivor:G) | 520 | 42.64 | 23.05 | 0.388 |
| 15 (probe-survivor:B) | 520 | 42.56 | 24.25 | 1.571 |
| 16 (probe-survivor:A) | 520 | 44.04 | 23.04 | 0.390 |
| 17 (probe-survivor:F) | 520 | 43.73 | 24.46 | 3.151 |
| 18 (probe-survivor:E) | 520 | 43.66 | 24.25 | 1.572 |
| 19 (probe-evicted:D) | 520 | 80.70 | 23.01 | 0.426 |
| 20 (probe-evicted:C) | 520 | 186.29 | 24.45 | 3.292 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
