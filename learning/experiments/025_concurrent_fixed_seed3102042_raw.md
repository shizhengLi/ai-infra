# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:34:08.384444+00:00`
- Git revision: `90d5910`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `3102042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Concurrent output throughput | 77.48 token/s |
| Average TTFT | 519.14 ms |
| P90 TTFT | 1232.71 ms |
| Survivor-group average TTFT | 133.78 ms |
| Survivor-group average TPOT | 26.50 ms |
| P90 E2E | 3.779 s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | fill | A | 16 | 1 | 1234.30 | 25.30 | 1.614 |
| 2 | fill | B | 64 | 2 | 1233.37 | 26.16 | 2.881 |
| 3 | fill | C | 128 | 3 | 1232.71 | 26.44 | 4.591 |
| 4 | fill | D | 16 | 4 | 1232.11 | 25.28 | 1.611 |
| 5 | fill | E | 64 | 5 | 1231.55 | 26.15 | 2.879 |
| 6 | fill | F | 128 | 6 | 1230.82 | 26.44 | 4.589 |
| 7 | refresh | A | 16 | 7 | 56.34 | 26.94 | 0.460 |
| 8 | refresh | B | 64 | 8 | 107.24 | 24.74 | 1.666 |
| 9 | pressure | G | 16 | 11 | 554.11 | 24.67 | 0.924 |
| 10 | pressure | H | 64 | 9 | 196.91 | 31.42 | 2.176 |
| 11 | pressure | I | 128 | 10 | 547.47 | 25.44 | 3.779 |
| 12 | probe-survivor | I | 128 | 18 | 160.27 | 26.70 | 3.551 |
| 13 | probe-survivor | H | 64 | 14 | 151.53 | 26.73 | 1.835 |
| 14 | probe-survivor | G | 16 | 15 | 147.58 | 25.44 | 0.529 |
| 15 | probe-survivor | B | 64 | 12 | 63.87 | 27.55 | 1.799 |
| 16 | probe-survivor | A | 16 | 13 | 139.18 | 25.70 | 0.525 |
| 17 | probe-survivor | F | 128 | 16 | 138.14 | 26.70 | 3.529 |
| 18 | probe-survivor | E | 64 | 17 | 135.91 | 26.68 | 1.817 |
| 19 | probe-evicted | D | 16 | 19 | 197.29 | 36.47 | 0.744 |
| 20 | probe-evicted | C | 128 | 20 | 392.03 | 24.69 | 3.528 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
