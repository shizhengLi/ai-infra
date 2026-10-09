# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T11:06:54.893427+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `8`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.23 token/s |
| Average TTFT | 462.92 ms |
| P90 TTFT | 936.42 ms |
| Survivor average TTFT | 197.45 ms |
| Survivor average TPOT | 30.18 ms |
| P90 E2E | 4.107 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 5 | 935.53 | 31.86 | 1.414 |
| 2 | 0.00 | fill | B | 64 | 4 | 936.11 | 36.01 | 3.205 |
| 3 | 0.00 | fill | C | 128 | 3 | 936.42 | 32.20 | 5.026 |
| 4 | 0.00 | fill | D | 16 | 1 | 198.48 | 79.64 | 1.393 |
| 5 | 0.00 | fill | E | 64 | 2 | 936.61 | 36.03 | 3.206 |
| 6 | 0.00 | fill | F | 128 | 6 | 1063.42 | 31.94 | 5.119 |
| 7 | 0.75 | refresh | A | 16 | 8 | 311.82 | 25.89 | 0.700 |
| 8 | 0.75 | refresh | B | 64 | 7 | 313.91 | 34.44 | 2.484 |
| 9 | 1.50 | pressure | G | 16 | 11 | 563.62 | 25.67 | 0.949 |
| 10 | 1.50 | pressure | H | 64 | 9 | 555.62 | 26.56 | 2.229 |
| 11 | 1.50 | pressure | I | 128 | 10 | 558.99 | 27.94 | 4.107 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 142.31 | 29.01 | 3.826 |
| 13 | 4.50 | probe-survivor | H | 64 | 12 | 84.28 | 32.25 | 2.116 |
| 14 | 4.50 | probe-survivor | G | 16 | 17 | 141.42 | 31.66 | 0.616 |
| 15 | 4.50 | probe-survivor | B | 64 | 14 | 143.70 | 31.73 | 2.143 |
| 16 | 4.50 | probe-survivor | A | 16 | 18 | 583.00 | 25.86 | 0.971 |
| 17 | 4.50 | probe-survivor | F | 128 | 15 | 143.00 | 29.02 | 3.829 |
| 18 | 4.50 | probe-survivor | E | 64 | 13 | 144.41 | 31.71 | 2.142 |
| 19 | 6.00 | probe-evicted | D | 16 | 20 | 281.07 | 25.55 | 0.664 |
| 20 | 6.00 | probe-evicted | C | 128 | 19 | 284.76 | 25.62 | 3.538 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
