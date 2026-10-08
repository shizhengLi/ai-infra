# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:50:14.384978+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `4528 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Pressure output lengths A-I: `None`
- Output length: `64` tokens
- Seed: `3100042`
- Expected measured server UIDs: `1-20` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Average input tokens | 520.00 |
| Total input tokens | 10400 |
| Sequential output throughput | 38.67 token/s |
| Average TTFT | 126.77 ms |
| P50 TTFT | 115.04 ms |
| P90 TTFT | 185.35 ms |
| Average TPOT | 24.26 ms |
| Average E2E | 1.655 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 379.21 | 24.26 | 1.908 |
| 2 (fill:B) | 520 | 184.83 | 24.26 | 1.713 |
| 3 (fill:C) | 520 | 185.35 | 24.26 | 1.714 |
| 4 (fill:D) | 520 | 184.71 | 24.27 | 1.713 |
| 5 (fill:E) | 520 | 184.91 | 24.27 | 1.714 |
| 6 (fill:F) | 520 | 184.65 | 24.25 | 1.713 |
| 7 (refresh:A) | 520 | 43.97 | 24.26 | 1.572 |
| 8 (refresh:B) | 520 | 43.24 | 24.26 | 1.572 |
| 9 (pressure:G) | 520 | 183.92 | 24.25 | 1.711 |
| 10 (pressure:H) | 520 | 186.11 | 24.26 | 1.714 |
| 11 (pressure:I) | 520 | 184.37 | 24.25 | 1.712 |
| 12 (probe-survivor:I) | 520 | 43.14 | 24.25 | 1.571 |
| 13 (probe-survivor:H) | 520 | 40.67 | 24.26 | 1.569 |
| 14 (probe-survivor:G) | 520 | 38.95 | 24.25 | 1.566 |
| 15 (probe-survivor:B) | 520 | 41.76 | 24.26 | 1.570 |
| 16 (probe-survivor:A) | 520 | 36.53 | 24.25 | 1.564 |
| 17 (probe-survivor:F) | 520 | 44.13 | 24.25 | 1.572 |
| 18 (probe-survivor:E) | 520 | 45.47 | 24.26 | 1.574 |
| 19 (probe-evicted:D) | 520 | 115.04 | 24.25 | 1.643 |
| 20 (probe-evicted:C) | 520 | 184.47 | 24.25 | 1.712 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
