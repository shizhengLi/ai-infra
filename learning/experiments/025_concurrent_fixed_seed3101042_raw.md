# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:23:52.708404+00:00`
- Git revision: `90d5910`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `3101042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Concurrent output throughput | 77.92 token/s |
| Average TTFT | 520.17 ms |
| P90 TTFT | 1234.37 ms |
| Survivor-group average TTFT | 132.24 ms |
| Survivor-group average TPOT | 26.41 ms |
| P90 E2E | 3.779 s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | fill | A | 16 | 1 | 1236.01 | 25.19 | 1.614 |
| 2 | fill | B | 64 | 2 | 1235.04 | 26.16 | 2.883 |
| 3 | fill | C | 128 | 3 | 1234.37 | 26.44 | 4.592 |
| 4 | fill | D | 16 | 4 | 1233.75 | 25.14 | 1.611 |
| 5 | fill | E | 64 | 5 | 1233.12 | 26.17 | 2.882 |
| 6 | fill | F | 128 | 6 | 1232.39 | 26.44 | 4.590 |
| 7 | refresh | A | 16 | 7 | 58.03 | 26.96 | 0.462 |
| 8 | refresh | B | 64 | 8 | 109.96 | 24.73 | 1.668 |
| 9 | pressure | G | 16 | 11 | 555.42 | 24.65 | 0.925 |
| 10 | pressure | H | 64 | 9 | 197.94 | 31.43 | 2.178 |
| 11 | pressure | I | 128 | 10 | 548.98 | 25.44 | 3.779 |
| 12 | probe-survivor | I | 128 | 12 | 72.52 | 27.14 | 3.519 |
| 13 | probe-survivor | H | 64 | 13 | 147.26 | 26.75 | 1.832 |
| 14 | probe-survivor | G | 16 | 14 | 145.29 | 25.42 | 0.527 |
| 15 | probe-survivor | B | 64 | 15 | 143.37 | 26.73 | 1.828 |
| 16 | probe-survivor | A | 16 | 16 | 141.48 | 25.43 | 0.523 |
| 17 | probe-survivor | F | 128 | 17 | 139.02 | 26.70 | 3.529 |
| 18 | probe-survivor | E | 64 | 18 | 136.73 | 26.71 | 1.820 |
| 19 | probe-evicted | D | 16 | 19 | 302.77 | 24.99 | 0.678 |
| 20 | probe-evicted | C | 128 | 20 | 300.04 | 24.70 | 3.438 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
