# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:12:12.577069+00:00`
- Git revision: `d409d50`
- Tensor parallelism: `4`
- KV pages / page size: `4096` / `1`
- Partial-leaf eviction: `True`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Output length: `16` tokens
- Seed: `3101042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 33.76 token/s |
| Average TTFT | 128.61 ms |
| P50 TTFT | 79.93 ms |
| P90 TTFT | 190.31 ms |
| Average TPOT | 23.02 ms |
| Average E2E | 0.474 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 416.29 | 23.03 | 0.762 |
| 2 (fill:B) | 520 | 191.35 | 23.03 | 0.537 |
| 3 (fill:C) | 520 | 186.83 | 23.03 | 0.532 |
| 4 (fill:D) | 520 | 190.31 | 23.02 | 0.536 |
| 5 (fill:E) | 520 | 186.65 | 23.06 | 0.533 |
| 6 (fill:F) | 520 | 186.03 | 23.02 | 0.531 |
| 7 (refresh:A) | 520 | 45.07 | 23.01 | 0.390 |
| 8 (refresh:B) | 520 | 44.84 | 22.99 | 0.390 |
| 9 (pressure:G) | 520 | 186.37 | 23.04 | 0.532 |
| 10 (pressure:H) | 520 | 186.31 | 23.02 | 0.532 |
| 11 (pressure:I) | 520 | 186.12 | 23.02 | 0.531 |
| 12 (probe-survivor:I) | 520 | 44.29 | 23.01 | 0.389 |
| 13 (probe-survivor:H) | 520 | 43.86 | 23.02 | 0.389 |
| 14 (probe-survivor:G) | 520 | 44.31 | 22.99 | 0.389 |
| 15 (probe-survivor:B) | 520 | 44.29 | 22.97 | 0.389 |
| 16 (probe-survivor:A) | 520 | 42.32 | 23.01 | 0.388 |
| 17 (probe-survivor:F) | 520 | 38.28 | 23.01 | 0.383 |
| 18 (probe-survivor:E) | 520 | 42.34 | 23.01 | 0.388 |
| 19 (probe-evicted:D) | 520 | 79.93 | 23.03 | 0.425 |
| 20 (probe-evicted:C) | 520 | 186.34 | 23.03 | 0.532 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
