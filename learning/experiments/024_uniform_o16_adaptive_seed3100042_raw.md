# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:48:21.339461+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `4096 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Pressure output lengths A-I: `None`
- Output length: `16` tokens
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 33.82 token/s |
| Average TTFT | 127.72 ms |
| P50 TTFT | 78.68 ms |
| P90 TTFT | 186.19 ms |
| Average TPOT | 23.02 ms |
| Average E2E | 0.473 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 406.03 | 22.97 | 0.751 |
| 2 (fill:B) | 520 | 192.95 | 23.00 | 0.538 |
| 3 (fill:C) | 520 | 186.19 | 23.02 | 0.531 |
| 4 (fill:D) | 520 | 185.86 | 23.00 | 0.531 |
| 5 (fill:E) | 520 | 185.60 | 23.05 | 0.531 |
| 6 (fill:F) | 520 | 185.36 | 23.04 | 0.531 |
| 7 (refresh:A) | 520 | 44.87 | 23.09 | 0.391 |
| 8 (refresh:B) | 520 | 42.54 | 23.05 | 0.388 |
| 9 (pressure:G) | 520 | 185.66 | 23.05 | 0.531 |
| 10 (pressure:H) | 520 | 185.89 | 23.06 | 0.532 |
| 11 (pressure:I) | 520 | 185.81 | 23.03 | 0.531 |
| 12 (probe-survivor:I) | 520 | 44.85 | 22.98 | 0.390 |
| 13 (probe-survivor:H) | 520 | 42.56 | 23.04 | 0.388 |
| 14 (probe-survivor:G) | 520 | 43.57 | 22.99 | 0.388 |
| 15 (probe-survivor:B) | 520 | 42.63 | 23.01 | 0.388 |
| 16 (probe-survivor:A) | 520 | 42.79 | 23.01 | 0.388 |
| 17 (probe-survivor:F) | 520 | 43.06 | 23.00 | 0.388 |
| 18 (probe-survivor:E) | 520 | 43.34 | 23.00 | 0.388 |
| 19 (probe-evicted:D) | 520 | 78.68 | 23.04 | 0.424 |
| 20 (probe-evicted:C) | 520 | 186.16 | 23.03 | 0.532 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
