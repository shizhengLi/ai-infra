# L20 Radix Cache workload result

## Configuration

- Scenario: `pressure-revisit`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T13:52:03.042122+00:00`
- Git revision: `e965692`
- Tensor parallelism: `4`
- KV pages / page size: `4528 / 1`
- CUDA Graph maximum batch size: `64`
- Scheduling: prefill streak `1`, active budget `2304`, result-before-prefill enabled
- Partial-leaf eviction: `True`
- Partial-eviction reserve: `0` pages
- Adaptive reserve maximum: `128` pages
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
| Sequential output throughput | 38.70 token/s |
| Average TTFT | 125.62 ms |
| P50 TTFT | 76.96 ms |
| P90 TTFT | 187.15 ms |
| Average TPOT | 24.26 ms |
| Average E2E | 1.654 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (fill:A) | 520 | 381.10 | 24.24 | 1.908 |
| 2 (fill:B) | 520 | 192.88 | 24.25 | 1.721 |
| 3 (fill:C) | 520 | 186.28 | 24.26 | 1.714 |
| 4 (fill:D) | 520 | 187.15 | 24.25 | 1.715 |
| 5 (fill:E) | 520 | 185.23 | 24.26 | 1.713 |
| 6 (fill:F) | 520 | 186.01 | 24.26 | 1.714 |
| 7 (refresh:A) | 520 | 44.34 | 24.26 | 1.572 |
| 8 (refresh:B) | 520 | 42.90 | 24.26 | 1.571 |
| 9 (pressure:G) | 520 | 185.37 | 24.26 | 1.714 |
| 10 (pressure:H) | 520 | 185.42 | 24.26 | 1.714 |
| 11 (pressure:I) | 520 | 187.06 | 24.26 | 1.715 |
| 12 (probe-survivor:I) | 520 | 43.12 | 24.27 | 1.572 |
| 13 (probe-survivor:H) | 520 | 43.67 | 24.26 | 1.572 |
| 14 (probe-survivor:G) | 520 | 41.14 | 24.26 | 1.570 |
| 15 (probe-survivor:B) | 520 | 42.31 | 24.26 | 1.571 |
| 16 (probe-survivor:A) | 520 | 37.28 | 24.26 | 1.566 |
| 17 (probe-survivor:F) | 520 | 40.24 | 24.25 | 1.568 |
| 18 (probe-survivor:E) | 520 | 40.39 | 24.26 | 1.569 |
| 19 (probe-evicted:D) | 520 | 76.96 | 24.26 | 1.606 |
| 20 (probe-evicted:C) | 520 | 183.53 | 24.26 | 1.712 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
