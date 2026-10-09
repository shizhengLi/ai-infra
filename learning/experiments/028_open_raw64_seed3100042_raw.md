# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T11:01:17.008549+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `0`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.66 token/s |
| Average TTFT | 507.63 ms |
| P90 TTFT | 936.98 ms |
| Survivor average TTFT | 321.35 ms |
| Survivor average TPOT | 28.92 ms |
| P90 E2E | 4.156 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1213.13 | 50.95 | 1.977 |
| 2 | 0.00 | fill | B | 64 | 5 | 936.12 | 39.17 | 3.404 |
| 3 | 0.00 | fill | C | 128 | 4 | 936.60 | 33.72 | 5.219 |
| 4 | 0.00 | fill | D | 16 | 2 | 937.27 | 55.90 | 1.776 |
| 5 | 0.00 | fill | E | 64 | 3 | 936.98 | 39.18 | 3.405 |
| 6 | 0.00 | fill | F | 128 | 1 | 201.95 | 39.32 | 5.195 |
| 7 | 0.75 | refresh | A | 16 | 8 | 468.44 | 50.97 | 1.233 |
| 8 | 0.75 | refresh | B | 64 | 7 | 470.80 | 35.11 | 2.683 |
| 9 | 1.50 | pressure | G | 16 | 9 | 240.87 | 49.18 | 0.979 |
| 10 | 1.50 | pressure | H | 64 | 10 | 447.19 | 29.42 | 2.300 |
| 11 | 1.50 | pressure | I | 128 | 11 | 652.88 | 27.58 | 4.156 |
| 12 | 4.50 | probe-survivor | I | 128 | 18 | 594.08 | 27.91 | 4.138 |
| 13 | 4.50 | probe-survivor | H | 64 | 13 | 119.91 | 30.76 | 2.058 |
| 14 | 4.50 | probe-survivor | G | 16 | 15 | 111.44 | 28.35 | 0.537 |
| 15 | 4.50 | probe-survivor | B | 64 | 14 | 114.62 | 30.77 | 2.053 |
| 16 | 4.50 | probe-survivor | A | 16 | 16 | 583.35 | 26.01 | 0.973 |
| 17 | 4.50 | probe-survivor | F | 128 | 17 | 603.60 | 27.90 | 4.147 |
| 18 | 4.50 | probe-survivor | E | 64 | 12 | 122.42 | 30.75 | 2.060 |
| 19 | 6.00 | probe-evicted | D | 16 | 20 | 228.79 | 25.47 | 0.611 |
| 20 | 6.00 | probe-evicted | C | 128 | 19 | 232.11 | 25.80 | 3.509 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
