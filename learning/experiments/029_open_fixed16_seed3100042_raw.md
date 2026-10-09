# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T12:31:46.389798+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
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
| Open-loop output throughput | 138.91 token/s |
| Average TTFT | 491.31 ms |
| P90 TTFT | 933.67 ms |
| Survivor average TTFT | 275.96 ms |
| Survivor average TPOT | 29.97 ms |
| P90 E2E | 4.179 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1209.67 | 50.82 | 1.972 |
| 2 | 0.00 | fill | B | 64 | 5 | 932.77 | 39.13 | 3.398 |
| 3 | 0.00 | fill | C | 128 | 4 | 933.25 | 33.86 | 5.233 |
| 4 | 0.00 | fill | D | 16 | 2 | 933.98 | 55.87 | 1.772 |
| 5 | 0.00 | fill | E | 64 | 3 | 933.67 | 39.11 | 3.398 |
| 6 | 0.00 | fill | F | 128 | 1 | 197.87 | 39.46 | 5.210 |
| 7 | 0.75 | refresh | A | 16 | 8 | 464.18 | 50.81 | 1.226 |
| 8 | 0.75 | refresh | B | 64 | 7 | 466.57 | 35.10 | 2.678 |
| 9 | 1.50 | pressure | G | 16 | 11 | 648.88 | 25.80 | 1.036 |
| 10 | 1.50 | pressure | H | 64 | 10 | 439.73 | 29.42 | 2.293 |
| 11 | 1.50 | pressure | I | 128 | 9 | 240.45 | 30.49 | 4.113 |
| 12 | 4.50 | probe-survivor | I | 128 | 18 | 606.86 | 28.12 | 4.179 |
| 13 | 4.50 | probe-survivor | H | 64 | 17 | 608.89 | 29.83 | 2.488 |
| 14 | 4.50 | probe-survivor | G | 16 | 13 | 134.21 | 30.35 | 0.589 |
| 15 | 4.50 | probe-survivor | B | 64 | 12 | 136.57 | 31.31 | 2.109 |
| 16 | 4.50 | probe-survivor | A | 16 | 14 | 132.11 | 30.38 | 0.588 |
| 17 | 4.50 | probe-survivor | F | 128 | 16 | 182.97 | 28.57 | 3.812 |
| 18 | 4.50 | probe-survivor | E | 64 | 15 | 130.08 | 31.22 | 2.097 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 219.25 | 27.66 | 0.634 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 274.17 | 25.85 | 3.557 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
