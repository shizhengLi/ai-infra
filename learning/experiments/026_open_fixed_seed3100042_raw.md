# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T07:58:54.334616+00:00`
- Git revision: `895cf91`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.94 token/s |
| Average TTFT | 470.43 ms |
| P90 TTFT | 935.36 ms |
| Survivor average TTFT | 217.31 ms |
| Survivor average TPOT | 30.72 ms |
| P90 E2E | 4.151 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1211.81 | 50.77 | 1.973 |
| 2 | 0.00 | fill | B | 64 | 5 | 934.45 | 39.16 | 3.402 |
| 3 | 0.00 | fill | C | 128 | 4 | 934.92 | 34.17 | 5.275 |
| 4 | 0.00 | fill | D | 16 | 2 | 935.71 | 55.87 | 1.774 |
| 5 | 0.00 | fill | E | 64 | 3 | 935.36 | 39.17 | 3.403 |
| 6 | 0.00 | fill | F | 128 | 1 | 200.20 | 39.78 | 5.252 |
| 7 | 0.75 | refresh | A | 16 | 8 | 467.24 | 50.76 | 1.229 |
| 8 | 0.75 | refresh | B | 64 | 7 | 469.50 | 35.11 | 2.681 |
| 9 | 1.50 | pressure | G | 16 | 11 | 650.26 | 25.80 | 1.037 |
| 10 | 1.50 | pressure | H | 64 | 10 | 444.48 | 29.43 | 2.298 |
| 11 | 1.50 | pressure | I | 128 | 9 | 238.48 | 30.81 | 4.151 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 128.30 | 28.86 | 3.794 |
| 13 | 4.50 | probe-survivor | H | 64 | 15 | 130.89 | 31.41 | 2.110 |
| 14 | 4.50 | probe-survivor | G | 16 | 17 | 208.86 | 32.33 | 0.694 |
| 15 | 4.50 | probe-survivor | B | 64 | 14 | 133.00 | 31.41 | 2.112 |
| 16 | 4.50 | probe-survivor | A | 16 | 13 | 140.80 | 32.76 | 0.632 |
| 17 | 4.50 | probe-survivor | F | 128 | 12 | 122.78 | 28.87 | 3.789 |
| 18 | 4.50 | probe-survivor | E | 64 | 18 | 656.53 | 29.38 | 2.508 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 233.60 | 25.87 | 0.622 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 231.34 | 25.63 | 3.487 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
