# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:28:38.234527+00:00`
- Git revision: `90d5910`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `3101042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Concurrent output throughput | 77.61 token/s |
| Average TTFT | 518.10 ms |
| P90 TTFT | 1228.23 ms |
| Survivor-group average TTFT | 132.80 ms |
| Survivor-group average TPOT | 26.49 ms |
| P90 E2E | 3.757 s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | fill | A | 16 | 6 | 1231.87 | 25.43 | 1.613 |
| 2 | fill | B | 64 | 1 | 1228.95 | 26.20 | 2.879 |
| 3 | fill | C | 128 | 2 | 1228.23 | 26.45 | 4.587 |
| 4 | fill | D | 16 | 3 | 1227.49 | 25.28 | 1.607 |
| 5 | fill | E | 64 | 4 | 1226.81 | 26.16 | 2.875 |
| 6 | fill | F | 128 | 5 | 1226.02 | 26.44 | 4.584 |
| 7 | refresh | A | 16 | 7 | 58.72 | 27.13 | 0.466 |
| 8 | refresh | B | 64 | 8 | 112.27 | 24.74 | 1.671 |
| 9 | pressure | G | 16 | 11 | 555.55 | 24.63 | 0.925 |
| 10 | pressure | H | 64 | 10 | 551.50 | 26.24 | 2.205 |
| 11 | pressure | I | 128 | 9 | 194.80 | 28.05 | 3.757 |
| 12 | probe-survivor | I | 128 | 18 | 155.19 | 26.70 | 3.546 |
| 13 | probe-survivor | H | 64 | 12 | 69.67 | 27.55 | 1.805 |
| 14 | probe-survivor | G | 16 | 13 | 144.98 | 25.67 | 0.530 |
| 15 | probe-survivor | B | 64 | 14 | 143.14 | 26.72 | 1.827 |
| 16 | probe-survivor | A | 16 | 15 | 141.30 | 25.39 | 0.522 |
| 17 | probe-survivor | F | 128 | 16 | 138.76 | 26.70 | 3.530 |
| 18 | probe-survivor | E | 64 | 17 | 136.54 | 26.70 | 1.819 |
| 19 | probe-evicted | D | 16 | 19 | 197.91 | 36.48 | 0.745 |
| 20 | probe-evicted | C | 128 | 20 | 392.36 | 24.69 | 3.528 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
