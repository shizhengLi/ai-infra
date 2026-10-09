# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T12:35:08.722261+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `0`
- Hotness decay: `0.0`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.90 token/s |
| Average TTFT | 468.45 ms |
| P90 TTFT | 936.99 ms |
| Survivor average TTFT | 208.57 ms |
| Survivor average TPOT | 30.99 ms |
| P90 E2E | 4.152 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1213.34 | 50.83 | 1.976 |
| 2 | 0.00 | fill | B | 64 | 5 | 936.03 | 39.17 | 3.404 |
| 3 | 0.00 | fill | C | 128 | 4 | 936.54 | 34.18 | 5.277 |
| 4 | 0.00 | fill | D | 16 | 2 | 937.28 | 55.90 | 1.776 |
| 5 | 0.00 | fill | E | 64 | 3 | 936.99 | 39.17 | 3.405 |
| 6 | 0.00 | fill | F | 128 | 1 | 202.90 | 39.78 | 5.255 |
| 7 | 0.75 | refresh | A | 16 | 8 | 467.16 | 50.82 | 1.229 |
| 8 | 0.75 | refresh | B | 64 | 7 | 470.13 | 35.11 | 2.682 |
| 9 | 1.50 | pressure | G | 16 | 11 | 652.42 | 25.78 | 1.039 |
| 10 | 1.50 | pressure | H | 64 | 10 | 447.05 | 29.42 | 2.301 |
| 11 | 1.50 | pressure | I | 128 | 9 | 239.25 | 30.81 | 4.152 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 111.83 | 28.98 | 3.793 |
| 13 | 4.50 | probe-survivor | H | 64 | 12 | 127.57 | 31.70 | 2.124 |
| 14 | 4.50 | probe-survivor | G | 16 | 13 | 118.94 | 33.74 | 0.625 |
| 15 | 4.50 | probe-survivor | B | 64 | 15 | 114.09 | 31.68 | 2.110 |
| 16 | 4.50 | probe-survivor | A | 16 | 17 | 210.80 | 32.46 | 0.698 |
| 17 | 4.50 | probe-survivor | F | 128 | 14 | 116.90 | 28.99 | 3.798 |
| 18 | 4.50 | probe-survivor | E | 64 | 18 | 659.83 | 29.37 | 2.510 |
| 19 | 6.00 | probe-evicted | D | 16 | 20 | 236.94 | 25.70 | 0.622 |
| 20 | 6.00 | probe-evicted | C | 128 | 19 | 233.05 | 25.64 | 3.489 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
