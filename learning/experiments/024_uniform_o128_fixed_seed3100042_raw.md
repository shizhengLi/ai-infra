# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:54:28.333104+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `5104 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Pressure output lengths A-I: `None`
- Output length: `128` tokens
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 39.53 token/s |
| Average TTFT | 132.03 ms |
| P50 TTFT | 182.84 ms |
| P90 TTFT | 185.70 ms |
| Average TPOT | 24.45 ms |
| Average E2E | 3.238 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 413.81 | 24.46 | 3.520 |
| 2 (fill:B) | 520 | 185.59 | 24.45 | 3.291 |
| 3 (fill:C) | 520 | 184.74 | 24.45 | 3.290 |
| 4 (fill:D) | 520 | 185.39 | 24.46 | 3.291 |
| 5 (fill:E) | 520 | 184.42 | 24.45 | 3.290 |
| 6 (fill:F) | 520 | 185.70 | 24.45 | 3.291 |
| 7 (refresh:A) | 520 | 44.69 | 24.45 | 3.150 |
| 8 (refresh:B) | 520 | 39.37 | 24.45 | 3.145 |
| 9 (pressure:G) | 520 | 182.84 | 24.45 | 3.288 |
| 10 (pressure:H) | 520 | 184.12 | 24.48 | 3.293 |
| 11 (pressure:I) | 520 | 183.95 | 24.45 | 3.289 |
| 12 (probe-survivor:I) | 520 | 42.94 | 24.45 | 3.148 |
| 13 (probe-survivor:H) | 520 | 43.39 | 24.46 | 3.149 |
| 14 (probe-survivor:G) | 520 | 41.22 | 24.45 | 3.146 |
| 15 (probe-survivor:B) | 520 | 40.40 | 24.45 | 3.146 |
| 16 (probe-survivor:A) | 520 | 43.00 | 24.45 | 3.148 |
| 17 (probe-survivor:F) | 520 | 42.40 | 24.45 | 3.148 |
| 18 (probe-survivor:E) | 520 | 41.30 | 24.45 | 3.147 |
| 19 (probe-evicted:D) | 520 | 184.89 | 24.46 | 3.292 |
| 20 (probe-evicted:C) | 520 | 186.41 | 24.45 | 3.292 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
