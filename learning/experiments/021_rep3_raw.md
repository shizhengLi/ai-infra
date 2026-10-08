# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:57:42.463321+00:00`
- Git revision: `fb72e3f`
- Tensor parallelism: `4`
- KV pages / page size: `4096` / `1`
- CUDA Graph maximum batch size: `64`
- Scheduling profile: prefill streak `1`, active prefill budget `2304`, result-before-prefill enabled
- Output length: `16` tokens
- Seed: `3102042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 33.60 token/s |
| Average TTFT | 130.18 ms |
| P50 TTFT | 180.35 ms |
| P90 TTFT | 185.76 ms |
| Average TPOT | 23.07 ms |
| Average E2E | 0.476 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 369.84 | 23.04 | 0.715 |
| 2 (fill:B) | 520 | 184.22 | 22.98 | 0.529 |
| 3 (fill:C) | 520 | 189.38 | 23.04 | 0.535 |
| 4 (fill:D) | 520 | 184.57 | 23.00 | 0.530 |
| 5 (fill:E) | 520 | 185.76 | 23.01 | 0.531 |
| 6 (fill:F) | 520 | 184.86 | 23.00 | 0.530 |
| 7 (refresh:A) | 520 | 38.71 | 23.09 | 0.385 |
| 8 (refresh:B) | 520 | 40.98 | 23.09 | 0.387 |
| 9 (pressure:G) | 520 | 184.31 | 23.09 | 0.531 |
| 10 (pressure:H) | 520 | 183.96 | 23.09 | 0.530 |
| 11 (pressure:I) | 520 | 183.98 | 23.10 | 0.530 |
| 12 (probe-survivor:I) | 520 | 41.32 | 22.94 | 0.385 |
| 13 (probe-survivor:H) | 520 | 52.31 | 23.00 | 0.397 |
| 14 (probe-survivor:G) | 520 | 48.06 | 23.12 | 0.395 |
| 15 (probe-survivor:B) | 520 | 41.96 | 23.10 | 0.388 |
| 16 (probe-survivor:A) | 520 | 41.69 | 23.07 | 0.388 |
| 17 (probe-survivor:F) | 520 | 41.52 | 23.11 | 0.388 |
| 18 (probe-survivor:E) | 520 | 41.34 | 23.11 | 0.388 |
| 19 (probe-evicted:D) | 520 | 184.49 | 23.39 | 0.535 |
| 20 (probe-evicted:C) | 520 | 180.35 | 23.09 | 0.527 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
