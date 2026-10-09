# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:39:04.205126+00:00`
- Git revision: `90d5910`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `3102042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Concurrent output throughput | 78.41 token/s |
| Average TTFT | 494.54 ms |
| P90 TTFT | 1193.55 ms |
| Survivor-group average TTFT | 129.06 ms |
| Survivor-group average TPOT | 26.41 ms |
| P90 E2E | 3.781 s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | fill | A | 16 | 5 | 1196.50 | 25.27 | 1.576 |
| 2 | fill | B | 64 | 1 | 1193.55 | 26.18 | 2.843 |
| 3 | fill | C | 128 | 6 | 1193.80 | 26.43 | 4.550 |
| 4 | fill | D | 16 | 2 | 1191.55 | 25.13 | 1.568 |
| 5 | fill | E | 64 | 3 | 1190.74 | 26.12 | 2.836 |
| 6 | fill | F | 128 | 4 | 1189.77 | 26.44 | 4.547 |
| 7 | refresh | A | 16 | 7 | 59.98 | 25.01 | 0.435 |
| 8 | refresh | B | 64 | 8 | 57.56 | 24.82 | 1.621 |
| 9 | pressure | G | 16 | 9 | 201.03 | 46.53 | 0.899 |
| 10 | pressure | H | 64 | 10 | 550.60 | 26.25 | 2.204 |
| 11 | pressure | I | 128 | 11 | 548.58 | 25.45 | 3.781 |
| 12 | probe-survivor | I | 128 | 12 | 70.05 | 27.13 | 3.515 |
| 13 | probe-survivor | H | 64 | 13 | 144.11 | 26.71 | 1.827 |
| 14 | probe-survivor | G | 16 | 14 | 142.14 | 25.44 | 0.524 |
| 15 | probe-survivor | B | 64 | 15 | 140.10 | 26.73 | 1.824 |
| 16 | probe-survivor | A | 16 | 16 | 138.08 | 25.47 | 0.520 |
| 17 | probe-survivor | F | 128 | 17 | 135.59 | 26.71 | 3.527 |
| 18 | probe-survivor | E | 64 | 18 | 133.35 | 26.73 | 1.817 |
| 19 | probe-evicted | D | 16 | 20 | 307.73 | 25.06 | 0.684 |
| 20 | probe-evicted | C | 128 | 19 | 105.93 | 26.09 | 3.420 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
