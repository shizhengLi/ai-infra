# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T12:41:23.600360+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `0`
- Hotness decay: `0.99`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.33 token/s |
| Average TTFT | 469.21 ms |
| P90 TTFT | 937.55 ms |
| Survivor average TTFT | 206.46 ms |
| Survivor average TPOT | 29.84 ms |
| P90 E2E | 4.137 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1213.25 | 50.84 | 1.976 |
| 2 | 0.00 | fill | B | 64 | 5 | 936.58 | 39.16 | 3.404 |
| 3 | 0.00 | fill | C | 128 | 4 | 937.12 | 34.05 | 5.261 |
| 4 | 0.00 | fill | D | 16 | 3 | 937.55 | 55.88 | 1.776 |
| 5 | 0.00 | fill | E | 64 | 2 | 937.89 | 39.19 | 3.407 |
| 6 | 0.00 | fill | F | 128 | 1 | 202.74 | 39.64 | 5.237 |
| 7 | 0.75 | refresh | A | 16 | 8 | 468.07 | 50.83 | 1.231 |
| 8 | 0.75 | refresh | B | 64 | 7 | 471.12 | 35.12 | 2.684 |
| 9 | 1.50 | pressure | G | 16 | 11 | 652.73 | 25.80 | 1.040 |
| 10 | 1.50 | pressure | H | 64 | 10 | 447.04 | 29.42 | 2.301 |
| 11 | 1.50 | pressure | I | 128 | 9 | 240.66 | 30.68 | 4.137 |
| 12 | 4.50 | probe-survivor | I | 128 | 12 | 124.58 | 28.98 | 3.806 |
| 13 | 4.50 | probe-survivor | H | 64 | 14 | 134.81 | 31.69 | 2.131 |
| 14 | 4.50 | probe-survivor | G | 16 | 18 | 638.18 | 26.12 | 1.030 |
| 15 | 4.50 | probe-survivor | B | 64 | 13 | 142.71 | 31.69 | 2.139 |
| 16 | 4.50 | probe-survivor | A | 16 | 16 | 130.04 | 29.77 | 0.577 |
| 17 | 4.50 | probe-survivor | F | 128 | 15 | 132.74 | 28.97 | 3.812 |
| 18 | 4.50 | probe-survivor | E | 64 | 17 | 142.16 | 31.67 | 2.138 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 219.57 | 27.82 | 0.637 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 274.67 | 25.62 | 3.528 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
