# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:55:46.581267+00:00`
- Git revision: `fb72e3f`
- Tensor parallelism: `4`
- KV pages / page size: `4096` / `1`
- CUDA Graph maximum batch size: `64`
- Scheduling profile: prefill streak `1`, active prefill budget `2304`, result-before-prefill enabled
- Output length: `16` tokens
- Seed: `3101042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 33.38 token/s |
| Average TTFT | 133.83 ms |
| P50 TTFT | 184.94 ms |
| P90 TTFT | 186.50 ms |
| Average TPOT | 23.04 ms |
| Average E2E | 0.479 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 412.71 | 23.03 | 0.758 |
| 2 (fill:B) | 520 | 186.45 | 23.05 | 0.532 |
| 3 (fill:C) | 520 | 184.94 | 23.05 | 0.531 |
| 4 (fill:D) | 520 | 185.43 | 23.03 | 0.531 |
| 5 (fill:E) | 520 | 185.66 | 23.04 | 0.531 |
| 6 (fill:F) | 520 | 185.15 | 23.03 | 0.531 |
| 7 (refresh:A) | 520 | 42.87 | 23.02 | 0.388 |
| 8 (refresh:B) | 520 | 43.02 | 23.02 | 0.388 |
| 9 (pressure:G) | 520 | 186.52 | 23.04 | 0.532 |
| 10 (pressure:H) | 520 | 185.79 | 23.04 | 0.531 |
| 11 (pressure:I) | 520 | 185.70 | 23.04 | 0.531 |
| 12 (probe-survivor:I) | 520 | 42.59 | 22.99 | 0.387 |
| 13 (probe-survivor:H) | 520 | 49.59 | 23.03 | 0.395 |
| 14 (probe-survivor:G) | 520 | 49.88 | 23.05 | 0.396 |
| 15 (probe-survivor:B) | 520 | 48.80 | 23.06 | 0.395 |
| 16 (probe-survivor:A) | 520 | 42.61 | 23.04 | 0.388 |
| 17 (probe-survivor:F) | 520 | 43.30 | 23.05 | 0.389 |
| 18 (probe-survivor:E) | 520 | 42.85 | 23.05 | 0.389 |
| 19 (probe-evicted:D) | 520 | 186.50 | 23.03 | 0.532 |
| 20 (probe-evicted:C) | 520 | 186.18 | 23.03 | 0.532 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
