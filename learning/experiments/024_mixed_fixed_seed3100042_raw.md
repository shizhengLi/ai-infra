# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:58:57.745137+00:00`
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
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 38.84 token/s |
| Average TTFT | 121.75 ms |
| P50 TTFT | 112.53 ms |
| P90 TTFT | 190.78 ms |
| Average TPOT | 23.89 ms |
| Average E2E | 1.710 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 198.50 | 23.00 | 0.544 |
| 2 (fill:B) | 520 | 195.73 | 24.27 | 1.725 |
| 3 (fill:C) | 520 | 186.73 | 24.47 | 3.294 |
| 4 (fill:D) | 520 | 190.78 | 23.04 | 0.536 |
| 5 (fill:E) | 520 | 187.09 | 24.26 | 1.716 |
| 6 (fill:F) | 520 | 186.66 | 24.46 | 3.293 |
| 7 (refresh:A) | 520 | 60.69 | 23.01 | 0.406 |
| 8 (refresh:B) | 520 | 55.67 | 24.25 | 1.583 |
| 9 (pressure:G) | 520 | 186.71 | 23.04 | 0.532 |
| 10 (pressure:H) | 520 | 186.81 | 24.27 | 1.716 |
| 11 (pressure:I) | 520 | 187.17 | 24.46 | 3.294 |
| 12 (probe-survivor:I) | 520 | 57.32 | 24.46 | 3.164 |
| 13 (probe-survivor:H) | 520 | 43.70 | 24.26 | 1.572 |
| 14 (probe-survivor:G) | 520 | 43.51 | 23.02 | 0.389 |
| 15 (probe-survivor:B) | 520 | 42.30 | 24.26 | 1.570 |
| 16 (probe-survivor:A) | 520 | 43.00 | 23.00 | 0.388 |
| 17 (probe-survivor:F) | 520 | 42.79 | 24.46 | 3.149 |
| 18 (probe-survivor:E) | 520 | 45.84 | 24.26 | 1.574 |
| 19 (probe-evicted:D) | 520 | 112.53 | 23.02 | 0.458 |
| 20 (probe-evicted:C) | 520 | 181.55 | 24.46 | 3.288 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
