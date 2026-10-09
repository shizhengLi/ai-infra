# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T09:40:02.669453+00:00`
- Git revision: `70aeb02`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `16` pages
- Adaptive reserve maximum: `0` pages
- Adaptive reserve mode: `raw`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.88 token/s |
| Average TTFT | 472.80 ms |
| P90 TTFT | 938.65 ms |
| Survivor average TTFT | 217.92 ms |
| Survivor average TPOT | 29.92 ms |
| P90 E2E | 4.186 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1215.00 | 50.81 | 1.977 |
| 2 | 0.00 | fill | B | 64 | 5 | 937.86 | 39.16 | 3.405 |
| 3 | 0.00 | fill | C | 128 | 4 | 938.26 | 34.18 | 5.279 |
| 4 | 0.00 | fill | D | 16 | 2 | 938.95 | 55.88 | 1.777 |
| 5 | 0.00 | fill | E | 64 | 3 | 938.65 | 39.17 | 3.406 |
| 6 | 0.00 | fill | F | 128 | 1 | 203.46 | 39.77 | 5.255 |
| 7 | 0.75 | refresh | A | 16 | 8 | 469.86 | 50.80 | 1.232 |
| 8 | 0.75 | refresh | B | 64 | 7 | 472.27 | 35.11 | 2.684 |
| 9 | 1.50 | pressure | G | 16 | 11 | 653.28 | 25.80 | 1.040 |
| 10 | 1.50 | pressure | H | 64 | 9 | 241.48 | 32.21 | 2.271 |
| 11 | 1.50 | pressure | I | 128 | 10 | 447.55 | 29.43 | 4.186 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 128.54 | 28.86 | 3.794 |
| 13 | 4.50 | probe-survivor | H | 64 | 17 | 197.53 | 30.93 | 2.146 |
| 14 | 4.50 | probe-survivor | G | 16 | 18 | 656.60 | 26.12 | 1.048 |
| 15 | 4.50 | probe-survivor | B | 64 | 15 | 130.74 | 31.43 | 2.111 |
| 16 | 4.50 | probe-survivor | A | 16 | 13 | 135.45 | 31.79 | 0.612 |
| 17 | 4.50 | probe-survivor | F | 128 | 14 | 133.38 | 28.87 | 3.800 |
| 18 | 4.50 | probe-survivor | E | 64 | 12 | 143.20 | 31.45 | 2.125 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 238.12 | 25.86 | 0.626 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 235.78 | 25.63 | 3.491 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
