# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:13:33.368395+00:00`
- Git revision: `d409d50`
- Tensor parallelism: `4`
- KV pages / page size: `4096` / `1`
- Partial-leaf eviction: `True`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Output length: `16` tokens
- Seed: `3102042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 33.38 token/s |
| Average TTFT | 134.19 ms |
| P50 TTFT | 78.22 ms |
| P90 TTFT | 187.63 ms |
| Average TPOT | 23.01 ms |
| Average E2E | 0.479 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 370.93 | 22.97 | 0.716 |
| 2 (fill:B) | 520 | 191.47 | 23.04 | 0.537 |
| 3 (fill:C) | 520 | 186.14 | 23.02 | 0.531 |
| 4 (fill:D) | 520 | 185.43 | 23.03 | 0.531 |
| 5 (fill:E) | 520 | 185.52 | 23.02 | 0.531 |
| 6 (fill:F) | 520 | 185.53 | 23.01 | 0.531 |
| 7 (refresh:A) | 520 | 62.90 | 23.03 | 0.408 |
| 8 (refresh:B) | 520 | 59.45 | 22.96 | 0.404 |
| 9 (pressure:G) | 520 | 187.42 | 23.00 | 0.532 |
| 10 (pressure:H) | 520 | 187.34 | 23.04 | 0.533 |
| 11 (pressure:I) | 520 | 186.73 | 23.06 | 0.533 |
| 12 (probe-survivor:I) | 520 | 59.04 | 22.97 | 0.404 |
| 13 (probe-survivor:H) | 520 | 58.53 | 22.98 | 0.403 |
| 14 (probe-survivor:G) | 520 | 59.00 | 22.98 | 0.404 |
| 15 (probe-survivor:B) | 520 | 58.38 | 22.98 | 0.403 |
| 16 (probe-survivor:A) | 520 | 59.34 | 23.00 | 0.404 |
| 17 (probe-survivor:F) | 520 | 63.71 | 23.02 | 0.409 |
| 18 (probe-survivor:E) | 520 | 71.09 | 23.07 | 0.417 |
| 19 (probe-evicted:D) | 520 | 78.22 | 23.01 | 0.423 |
| 20 (probe-evicted:C) | 520 | 187.63 | 22.98 | 0.532 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
