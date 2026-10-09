# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T08:33:16.597333+00:00`
- Git revision: `895cf91`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `256` pages
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.18 token/s |
| Average TTFT | 527.55 ms |
| P90 TTFT | 939.82 ms |
| Survivor average TTFT | 388.37 ms |
| Survivor average TPOT | 29.45 ms |
| P90 E2E | 4.254 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1216.34 | 50.83 | 1.979 |
| 2 | 0.00 | fill | B | 64 | 5 | 938.90 | 39.13 | 3.404 |
| 3 | 0.00 | fill | C | 128 | 4 | 939.38 | 34.28 | 5.292 |
| 4 | 0.00 | fill | D | 16 | 2 | 940.14 | 55.90 | 1.779 |
| 5 | 0.00 | fill | E | 64 | 3 | 939.82 | 39.19 | 3.409 |
| 6 | 0.00 | fill | F | 128 | 1 | 203.22 | 39.89 | 5.270 |
| 7 | 0.75 | refresh | A | 16 | 8 | 471.62 | 50.80 | 1.234 |
| 8 | 0.75 | refresh | B | 64 | 7 | 473.69 | 35.12 | 2.686 |
| 9 | 1.50 | pressure | G | 16 | 9 | 243.83 | 49.46 | 0.986 |
| 10 | 1.50 | pressure | H | 64 | 11 | 656.29 | 26.63 | 2.334 |
| 11 | 1.50 | pressure | I | 128 | 10 | 450.24 | 29.52 | 4.200 |
| 12 | 4.50 | probe-survivor | I | 128 | 17 | 664.82 | 28.10 | 4.234 |
| 13 | 4.50 | probe-survivor | H | 64 | 14 | 201.34 | 31.07 | 2.159 |
| 14 | 4.50 | probe-survivor | G | 16 | 15 | 199.36 | 28.30 | 0.624 |
| 15 | 4.50 | probe-survivor | B | 64 | 12 | 99.09 | 32.36 | 2.138 |
| 16 | 4.50 | probe-survivor | A | 16 | 13 | 203.09 | 28.31 | 0.628 |
| 17 | 4.50 | probe-survivor | F | 128 | 18 | 684.04 | 28.11 | 4.254 |
| 18 | 4.50 | probe-survivor | E | 64 | 16 | 666.86 | 29.85 | 2.548 |
| 19 | 6.00 | probe-evicted | D | 16 | 20 | 277.84 | 25.80 | 0.665 |
| 20 | 6.00 | probe-evicted | C | 128 | 19 | 81.02 | 27.25 | 3.542 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
