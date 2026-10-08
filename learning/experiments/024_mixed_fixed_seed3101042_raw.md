# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T14:02:51.043078+00:00`
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
- Seed: `3101042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 38.65 token/s |
| Average TTFT | 129.98 ms |
| P50 TTFT | 119.43 ms |
| P90 TTFT | 191.50 ms |
| Average TPOT | 23.89 ms |
| Average E2E | 1.718 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 374.07 | 23.01 | 0.719 |
| 2 (fill:B) | 520 | 186.20 | 24.26 | 1.715 |
| 3 (fill:C) | 520 | 185.83 | 24.46 | 3.292 |
| 4 (fill:D) | 520 | 191.50 | 23.01 | 0.537 |
| 5 (fill:E) | 520 | 190.89 | 24.26 | 1.719 |
| 6 (fill:F) | 520 | 190.93 | 24.46 | 3.297 |
| 7 (refresh:A) | 520 | 50.14 | 23.01 | 0.395 |
| 8 (refresh:B) | 520 | 48.92 | 24.26 | 1.577 |
| 9 (pressure:G) | 520 | 191.12 | 23.03 | 0.537 |
| 10 (pressure:H) | 520 | 191.13 | 24.27 | 1.720 |
| 11 (pressure:I) | 520 | 192.85 | 24.47 | 3.300 |
| 12 (probe-survivor:I) | 520 | 43.09 | 24.46 | 3.150 |
| 13 (probe-survivor:H) | 520 | 43.19 | 24.28 | 1.573 |
| 14 (probe-survivor:G) | 520 | 42.90 | 23.08 | 0.389 |
| 15 (probe-survivor:B) | 520 | 42.92 | 24.26 | 1.571 |
| 16 (probe-survivor:A) | 520 | 42.91 | 23.05 | 0.389 |
| 17 (probe-survivor:F) | 520 | 42.73 | 24.46 | 3.150 |
| 18 (probe-survivor:E) | 520 | 43.14 | 24.27 | 1.572 |
| 19 (probe-evicted:D) | 520 | 119.43 | 23.04 | 0.465 |
| 20 (probe-evicted:C) | 520 | 185.67 | 24.47 | 3.293 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
