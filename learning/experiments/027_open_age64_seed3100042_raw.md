# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T09:48:33.165617+00:00`
- Git revision: `70aeb02`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `age-aware`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.83 token/s |
| Average TTFT | 469.82 ms |
| P90 TTFT | 939.41 ms |
| Survivor average TTFT | 207.35 ms |
| Survivor average TPOT | 29.91 ms |
| P90 E2E | 4.189 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1216.04 | 50.79 | 1.978 |
| 2 | 0.00 | fill | B | 64 | 5 | 938.67 | 39.16 | 3.406 |
| 3 | 0.00 | fill | C | 128 | 4 | 939.03 | 34.20 | 5.283 |
| 4 | 0.00 | fill | D | 16 | 2 | 939.60 | 55.87 | 1.778 |
| 5 | 0.00 | fill | E | 64 | 3 | 939.41 | 39.17 | 3.407 |
| 6 | 0.00 | fill | F | 128 | 1 | 203.98 | 39.80 | 5.259 |
| 7 | 0.75 | refresh | A | 16 | 8 | 470.61 | 50.78 | 1.232 |
| 8 | 0.75 | refresh | B | 64 | 7 | 472.94 | 35.11 | 2.685 |
| 9 | 1.50 | pressure | G | 16 | 11 | 654.30 | 25.78 | 1.041 |
| 10 | 1.50 | pressure | H | 64 | 9 | 242.29 | 32.32 | 2.279 |
| 11 | 1.50 | pressure | I | 128 | 10 | 448.38 | 29.46 | 4.189 |
| 12 | 4.50 | probe-survivor | I | 128 | 16 | 110.98 | 29.00 | 3.794 |
| 13 | 4.50 | probe-survivor | H | 64 | 14 | 115.97 | 31.75 | 2.116 |
| 14 | 4.50 | probe-survivor | G | 16 | 17 | 179.57 | 30.12 | 0.631 |
| 15 | 4.50 | probe-survivor | B | 64 | 13 | 118.10 | 31.75 | 2.119 |
| 16 | 4.50 | probe-survivor | A | 16 | 18 | 685.35 | 26.01 | 1.075 |
| 17 | 4.50 | probe-survivor | F | 128 | 15 | 113.29 | 29.00 | 3.796 |
| 18 | 4.50 | probe-survivor | E | 64 | 12 | 128.17 | 31.76 | 2.129 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 241.09 | 25.73 | 0.627 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 238.58 | 25.63 | 3.494 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
