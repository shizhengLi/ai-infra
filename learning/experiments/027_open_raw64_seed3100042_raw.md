# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T09:45:46.087904+00:00`
- Git revision: `70aeb02`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.83 token/s |
| Average TTFT | 471.32 ms |
| P90 TTFT | 937.21 ms |
| Survivor average TTFT | 215.81 ms |
| Survivor average TPOT | 29.97 ms |
| P90 E2E | 4.159 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1212.59 | 50.82 | 1.975 |
| 2 | 0.00 | fill | B | 64 | 5 | 936.28 | 39.16 | 3.403 |
| 3 | 0.00 | fill | C | 128 | 4 | 936.73 | 34.22 | 5.283 |
| 4 | 0.00 | fill | D | 16 | 3 | 937.21 | 55.81 | 1.774 |
| 5 | 0.00 | fill | E | 64 | 2 | 937.51 | 39.18 | 3.406 |
| 6 | 0.00 | fill | F | 128 | 1 | 200.89 | 39.82 | 5.258 |
| 7 | 0.75 | refresh | A | 16 | 8 | 467.16 | 50.81 | 1.229 |
| 8 | 0.75 | refresh | B | 64 | 7 | 469.61 | 35.11 | 2.682 |
| 9 | 1.50 | pressure | G | 16 | 11 | 651.61 | 25.80 | 1.039 |
| 10 | 1.50 | pressure | H | 64 | 10 | 446.09 | 29.42 | 2.300 |
| 11 | 1.50 | pressure | I | 128 | 9 | 239.55 | 30.86 | 4.159 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 125.48 | 28.92 | 3.798 |
| 13 | 4.50 | probe-survivor | H | 64 | 14 | 130.34 | 31.55 | 2.118 |
| 14 | 4.50 | probe-survivor | G | 16 | 18 | 660.32 | 26.11 | 1.052 |
| 15 | 4.50 | probe-survivor | B | 64 | 17 | 194.21 | 31.05 | 2.150 |
| 16 | 4.50 | probe-survivor | A | 16 | 15 | 127.75 | 31.68 | 0.603 |
| 17 | 4.50 | probe-survivor | F | 128 | 13 | 132.45 | 28.93 | 3.806 |
| 18 | 4.50 | probe-survivor | E | 64 | 12 | 140.13 | 31.56 | 2.129 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 241.39 | 25.87 | 0.629 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 239.07 | 25.63 | 3.494 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
