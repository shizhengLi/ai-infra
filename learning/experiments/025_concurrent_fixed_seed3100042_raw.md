# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:08:25.955037+00:00`
- Git revision: `90d5910`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Concurrent output throughput | 78.25 token/s |
| Average TTFT | 400.33 ms |
| P90 TTFT | 935.74 ms |
| Survivor-group average TTFT | 133.17 ms |
| Survivor-group average TPOT | 26.52 ms |
| P90 E2E | 3.784 s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | fill | A | 16 | 1 | 200.50 | 79.39 | 1.391 |
| 2 | fill | B | 64 | 2 | 936.19 | 27.76 | 2.685 |
| 3 | fill | C | 128 | 3 | 935.74 | 27.22 | 4.393 |
| 4 | fill | D | 16 | 4 | 935.20 | 31.84 | 1.413 |
| 5 | fill | E | 64 | 5 | 934.77 | 27.75 | 2.683 |
| 6 | fill | F | 128 | 6 | 1054.69 | 26.47 | 4.416 |
| 7 | refresh | A | 16 | 7 | 57.15 | 26.91 | 0.461 |
| 8 | refresh | B | 64 | 8 | 109.07 | 24.72 | 1.666 |
| 9 | pressure | G | 16 | 11 | 563.32 | 24.61 | 0.932 |
| 10 | pressure | H | 64 | 9 | 205.39 | 31.42 | 2.185 |
| 11 | pressure | I | 128 | 10 | 552.08 | 25.44 | 3.784 |
| 12 | probe-survivor | I | 128 | 17 | 155.09 | 26.71 | 3.547 |
| 13 | probe-survivor | H | 64 | 12 | 71.26 | 27.54 | 1.806 |
| 14 | probe-survivor | G | 16 | 13 | 146.11 | 25.67 | 0.531 |
| 15 | probe-survivor | B | 64 | 18 | 146.68 | 26.68 | 1.828 |
| 16 | probe-survivor | A | 16 | 14 | 140.09 | 25.67 | 0.525 |
| 17 | probe-survivor | F | 128 | 15 | 137.57 | 26.71 | 3.530 |
| 18 | probe-survivor | E | 64 | 16 | 135.40 | 26.67 | 1.816 |
| 19 | probe-evicted | D | 16 | 19 | 197.88 | 36.48 | 0.745 |
| 20 | probe-evicted | C | 128 | 20 | 392.48 | 24.70 | 3.530 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
