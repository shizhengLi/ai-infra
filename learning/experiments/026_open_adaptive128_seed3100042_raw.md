# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T08:12:52.433286+00:00`
- Git revision: `895cf91`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.81 token/s |
| Average TTFT | 473.85 ms |
| P90 TTFT | 935.88 ms |
| Survivor average TTFT | 227.18 ms |
| Survivor average TPOT | 30.12 ms |
| P90 E2E | 4.182 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1210.53 | 50.80 | 1.973 |
| 2 | 0.00 | fill | B | 64 | 5 | 934.99 | 39.14 | 3.401 |
| 3 | 0.00 | fill | C | 128 | 4 | 935.44 | 34.41 | 5.306 |
| 4 | 0.00 | fill | D | 16 | 3 | 935.88 | 55.75 | 1.772 |
| 5 | 0.00 | fill | E | 64 | 2 | 936.23 | 39.16 | 3.404 |
| 6 | 0.00 | fill | F | 128 | 1 | 197.90 | 40.03 | 5.282 |
| 7 | 0.75 | refresh | A | 16 | 7 | 464.67 | 50.79 | 1.227 |
| 8 | 0.75 | refresh | B | 64 | 8 | 468.55 | 35.11 | 2.680 |
| 9 | 1.50 | pressure | G | 16 | 10 | 443.57 | 37.51 | 1.006 |
| 10 | 1.50 | pressure | H | 64 | 11 | 649.52 | 26.63 | 2.327 |
| 11 | 1.50 | pressure | I | 128 | 9 | 237.36 | 31.06 | 4.182 |
| 12 | 4.50 | probe-survivor | I | 128 | 12 | 121.24 | 29.09 | 3.816 |
| 13 | 4.50 | probe-survivor | H | 64 | 13 | 143.93 | 31.82 | 2.149 |
| 14 | 4.50 | probe-survivor | G | 16 | 18 | 680.05 | 26.11 | 1.072 |
| 15 | 4.50 | probe-survivor | B | 64 | 14 | 132.79 | 31.87 | 2.140 |
| 16 | 4.50 | probe-survivor | A | 16 | 15 | 130.38 | 31.86 | 0.608 |
| 17 | 4.50 | probe-survivor | F | 128 | 16 | 184.12 | 28.82 | 3.844 |
| 18 | 4.50 | probe-survivor | E | 64 | 17 | 197.79 | 31.30 | 2.170 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 237.57 | 25.67 | 0.623 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 234.52 | 25.67 | 3.495 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
