# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T08:06:50.953409+00:00`
- Git revision: `895cf91`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.23 token/s |
| Average TTFT | 462.09 ms |
| P90 TTFT | 933.92 ms |
| Survivor average TTFT | 198.26 ms |
| Survivor average TPOT | 30.24 ms |
| P90 E2E | 4.106 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 5 | 932.97 | 31.92 | 1.412 |
| 2 | 0.00 | fill | B | 64 | 4 | 933.57 | 36.03 | 3.203 |
| 3 | 0.00 | fill | C | 128 | 3 | 933.92 | 32.20 | 5.023 |
| 4 | 0.00 | fill | D | 16 | 1 | 198.04 | 79.56 | 1.391 |
| 5 | 0.00 | fill | E | 64 | 2 | 934.26 | 36.04 | 3.205 |
| 6 | 0.00 | fill | F | 128 | 6 | 1061.44 | 31.93 | 5.116 |
| 7 | 0.75 | refresh | A | 16 | 8 | 310.60 | 25.53 | 0.694 |
| 8 | 0.75 | refresh | B | 64 | 7 | 312.53 | 34.45 | 2.483 |
| 9 | 1.50 | pressure | G | 16 | 11 | 562.32 | 25.68 | 0.948 |
| 10 | 1.50 | pressure | H | 64 | 10 | 556.03 | 26.56 | 2.229 |
| 11 | 1.50 | pressure | I | 128 | 9 | 558.06 | 27.93 | 4.106 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 156.38 | 28.99 | 3.838 |
| 13 | 4.50 | probe-survivor | H | 64 | 14 | 102.95 | 32.19 | 2.131 |
| 14 | 4.50 | probe-survivor | G | 16 | 15 | 158.48 | 31.63 | 0.633 |
| 15 | 4.50 | probe-survivor | B | 64 | 12 | 112.76 | 32.20 | 2.141 |
| 16 | 4.50 | probe-survivor | A | 16 | 18 | 580.95 | 25.80 | 0.968 |
| 17 | 4.50 | probe-survivor | F | 128 | 13 | 105.02 | 29.26 | 3.820 |
| 18 | 4.50 | probe-survivor | E | 64 | 17 | 171.27 | 31.62 | 2.163 |
| 19 | 6.00 | probe-evicted | D | 16 | 20 | 279.01 | 25.69 | 0.664 |
| 20 | 6.00 | probe-evicted | C | 128 | 19 | 281.25 | 25.64 | 3.538 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
