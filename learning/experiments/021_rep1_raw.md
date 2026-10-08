# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:54:06.189713+00:00`
- Git revision: `fb72e3f`
- Tensor parallelism: `4`
- KV pages / page size: `4096` / `1`
- CUDA Graph maximum batch size: `64`
- Scheduling profile: prefill streak `1`, active prefill budget `2304`, result-before-prefill enabled
- Output length: `16` tokens
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 33.66 token/s |
| Average TTFT | 129.88 ms |
| P50 TTFT | 178.89 ms |
| P90 TTFT | 186.05 ms |
| Average TPOT | 23.03 ms |
| Average E2E | 0.475 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 371.81 | 23.01 | 0.717 |
| 2 (fill:B) | 520 | 185.32 | 23.01 | 0.530 |
| 3 (fill:C) | 520 | 185.64 | 23.03 | 0.531 |
| 4 (fill:D) | 520 | 185.68 | 23.01 | 0.531 |
| 5 (fill:E) | 520 | 185.53 | 23.07 | 0.532 |
| 6 (fill:F) | 520 | 178.89 | 23.09 | 0.525 |
| 7 (refresh:A) | 520 | 41.99 | 22.99 | 0.387 |
| 8 (refresh:B) | 520 | 41.59 | 23.04 | 0.387 |
| 9 (pressure:G) | 520 | 185.17 | 23.04 | 0.531 |
| 10 (pressure:H) | 520 | 186.95 | 23.00 | 0.532 |
| 11 (pressure:I) | 520 | 184.66 | 23.03 | 0.530 |
| 12 (probe-survivor:I) | 520 | 42.84 | 23.01 | 0.388 |
| 13 (probe-survivor:H) | 520 | 42.96 | 23.04 | 0.389 |
| 14 (probe-survivor:G) | 520 | 42.72 | 23.04 | 0.388 |
| 15 (probe-survivor:B) | 520 | 42.13 | 23.05 | 0.388 |
| 16 (probe-survivor:A) | 520 | 42.52 | 23.03 | 0.388 |
| 17 (probe-survivor:F) | 520 | 42.66 | 23.02 | 0.388 |
| 18 (probe-survivor:E) | 520 | 42.35 | 23.05 | 0.388 |
| 19 (probe-evicted:D) | 520 | 186.05 | 23.03 | 0.531 |
| 20 (probe-evicted:C) | 520 | 180.12 | 23.03 | 0.526 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
