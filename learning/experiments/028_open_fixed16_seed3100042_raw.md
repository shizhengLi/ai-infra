# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T10:57:14.787626+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `0`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 140.02 token/s |
| Average TTFT | 468.95 ms |
| P90 TTFT | 933.41 ms |
| Survivor average TTFT | 219.10 ms |
| Survivor average TPOT | 30.46 ms |
| P90 E2E | 4.174 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1209.25 | 50.72 | 1.970 |
| 2 | 0.00 | fill | B | 64 | 5 | 932.65 | 39.10 | 3.396 |
| 3 | 0.00 | fill | C | 128 | 4 | 933.04 | 34.13 | 5.268 |
| 4 | 0.00 | fill | D | 16 | 2 | 933.68 | 55.81 | 1.771 |
| 5 | 0.00 | fill | E | 64 | 3 | 933.41 | 39.11 | 3.398 |
| 6 | 0.00 | fill | F | 128 | 1 | 197.19 | 39.74 | 5.244 |
| 7 | 0.75 | refresh | A | 16 | 8 | 462.76 | 50.69 | 1.223 |
| 8 | 0.75 | refresh | B | 64 | 7 | 465.12 | 35.09 | 2.676 |
| 9 | 1.50 | pressure | G | 16 | 9 | 241.80 | 49.01 | 0.977 |
| 10 | 1.50 | pressure | H | 64 | 11 | 640.52 | 26.60 | 2.316 |
| 11 | 1.50 | pressure | I | 128 | 10 | 441.27 | 29.39 | 4.174 |
| 12 | 4.50 | probe-survivor | I | 128 | 17 | 172.59 | 28.50 | 3.792 |
| 13 | 4.50 | probe-survivor | H | 64 | 13 | 103.65 | 31.38 | 2.081 |
| 14 | 4.50 | probe-survivor | G | 16 | 16 | 173.63 | 32.06 | 0.654 |
| 15 | 4.50 | probe-survivor | B | 64 | 14 | 175.78 | 30.65 | 2.107 |
| 16 | 4.50 | probe-survivor | A | 16 | 12 | 104.55 | 32.78 | 0.596 |
| 17 | 4.50 | probe-survivor | F | 128 | 15 | 174.94 | 28.50 | 3.794 |
| 18 | 4.50 | probe-survivor | E | 64 | 18 | 628.56 | 29.39 | 2.480 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 227.80 | 25.56 | 0.611 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 226.79 | 25.64 | 3.483 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
