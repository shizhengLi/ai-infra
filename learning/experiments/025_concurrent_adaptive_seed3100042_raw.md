# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:18:25.576693+00:00`
- Git revision: `90d5910`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Concurrent output throughput | 77.55 token/s |
| Average TTFT | 518.00 ms |
| P90 TTFT | 1232.76 ms |
| Survivor-group average TTFT | 130.79 ms |
| Survivor-group average TPOT | 26.49 ms |
| P90 E2E | 3.777 s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | fill | A | 16 | 1 | 1234.53 | 25.44 | 1.616 |
| 2 | fill | B | 64 | 2 | 1233.52 | 26.17 | 2.882 |
| 3 | fill | C | 128 | 3 | 1232.76 | 26.45 | 4.592 |
| 4 | fill | D | 16 | 4 | 1232.01 | 25.24 | 1.611 |
| 5 | fill | E | 64 | 5 | 1231.27 | 26.18 | 2.880 |
| 6 | fill | F | 128 | 6 | 1230.42 | 26.44 | 4.588 |
| 7 | refresh | A | 16 | 7 | 56.18 | 26.79 | 0.458 |
| 8 | refresh | B | 64 | 8 | 105.59 | 24.72 | 1.663 |
| 9 | pressure | G | 16 | 11 | 554.81 | 24.69 | 0.925 |
| 10 | pressure | H | 64 | 9 | 197.36 | 31.43 | 2.177 |
| 11 | pressure | I | 128 | 10 | 548.18 | 25.43 | 3.777 |
| 12 | probe-survivor | I | 128 | 16 | 151.83 | 26.71 | 3.544 |
| 13 | probe-survivor | H | 64 | 12 | 69.13 | 27.52 | 1.803 |
| 14 | probe-survivor | G | 16 | 18 | 146.44 | 25.42 | 0.528 |
| 15 | probe-survivor | B | 64 | 17 | 142.97 | 26.69 | 1.825 |
| 16 | probe-survivor | A | 16 | 13 | 137.51 | 25.69 | 0.523 |
| 17 | probe-survivor | F | 128 | 14 | 134.97 | 26.71 | 3.528 |
| 18 | probe-survivor | E | 64 | 15 | 132.65 | 26.68 | 1.814 |
| 19 | probe-evicted | D | 16 | 19 | 196.63 | 36.64 | 0.746 |
| 20 | probe-evicted | C | 128 | 20 | 391.20 | 24.68 | 3.526 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
