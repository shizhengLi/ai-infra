# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:10:47.606480+00:00`
- Git revision: `d409d50`
- Tensor parallelism: `4`
- KV pages / page size: `4096` / `1`
- Partial-leaf eviction: `True`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Output length: `16` tokens
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 34.34 token/s |
| Average TTFT | 120.72 ms |
| P50 TTFT | 81.77 ms |
| P90 TTFT | 187.30 ms |
| Average TPOT | 23.01 ms |
| Average E2E | 0.466 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 187.30 | 23.01 | 0.532 |
| 2 (fill:B) | 520 | 191.95 | 23.02 | 0.537 |
| 3 (fill:C) | 520 | 186.10 | 23.01 | 0.531 |
| 4 (fill:D) | 520 | 185.71 | 23.01 | 0.531 |
| 5 (fill:E) | 520 | 185.87 | 23.01 | 0.531 |
| 6 (fill:F) | 520 | 186.14 | 22.98 | 0.531 |
| 7 (refresh:A) | 520 | 52.46 | 23.03 | 0.398 |
| 8 (refresh:B) | 520 | 63.03 | 23.05 | 0.409 |
| 9 (pressure:G) | 520 | 187.49 | 22.99 | 0.532 |
| 10 (pressure:H) | 520 | 186.42 | 23.00 | 0.531 |
| 11 (pressure:I) | 520 | 186.92 | 23.02 | 0.532 |
| 12 (probe-survivor:I) | 520 | 58.84 | 23.01 | 0.404 |
| 13 (probe-survivor:H) | 520 | 57.06 | 23.00 | 0.402 |
| 14 (probe-survivor:G) | 520 | 56.17 | 23.01 | 0.401 |
| 15 (probe-survivor:B) | 520 | 43.10 | 23.01 | 0.388 |
| 16 (probe-survivor:A) | 520 | 43.36 | 22.99 | 0.388 |
| 17 (probe-survivor:F) | 520 | 43.37 | 22.99 | 0.388 |
| 18 (probe-survivor:E) | 520 | 45.58 | 23.01 | 0.391 |
| 19 (probe-evicted:D) | 520 | 81.77 | 23.02 | 0.427 |
| 20 (probe-evicted:C) | 520 | 185.76 | 23.02 | 0.531 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
