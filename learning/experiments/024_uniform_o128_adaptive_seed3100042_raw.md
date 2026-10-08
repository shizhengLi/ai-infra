# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:56:48.756800+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `5104 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `0` pages
- Adaptive reserve maximum: `128` pages
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
| Average TTFT | 132.83 ms |
| P50 TTFT | 181.77 ms |
| P90 TTFT | 185.79 ms |
| Average TPOT | 24.45 ms |
| Average E2E | 3.238 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 412.76 | 24.46 | 3.519 |
| 2 (fill:B) | 520 | 185.78 | 24.45 | 3.291 |
| 3 (fill:C) | 520 | 185.71 | 24.45 | 3.291 |
| 4 (fill:D) | 520 | 185.79 | 24.46 | 3.292 |
| 5 (fill:E) | 520 | 184.91 | 24.45 | 3.291 |
| 6 (fill:F) | 520 | 185.69 | 24.45 | 3.291 |
| 7 (refresh:A) | 520 | 44.52 | 24.46 | 3.151 |
| 8 (refresh:B) | 520 | 43.15 | 24.45 | 3.148 |
| 9 (pressure:G) | 520 | 197.30 | 24.41 | 3.297 |
| 10 (pressure:H) | 520 | 184.39 | 24.46 | 3.291 |
| 11 (pressure:I) | 520 | 184.57 | 24.45 | 3.290 |
| 12 (probe-survivor:I) | 520 | 44.37 | 24.45 | 3.150 |
| 13 (probe-survivor:H) | 520 | 42.06 | 24.46 | 3.148 |
| 14 (probe-survivor:G) | 520 | 43.03 | 24.45 | 3.148 |
| 15 (probe-survivor:B) | 520 | 40.98 | 24.45 | 3.146 |
| 16 (probe-survivor:A) | 520 | 41.16 | 24.46 | 3.147 |
| 17 (probe-survivor:F) | 520 | 42.68 | 24.45 | 3.148 |
| 18 (probe-survivor:E) | 520 | 42.15 | 24.46 | 3.148 |
| 19 (probe-evicted:D) | 520 | 183.81 | 24.47 | 3.291 |
| 20 (probe-evicted:C) | 520 | 181.77 | 24.45 | 3.287 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
