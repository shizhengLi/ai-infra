# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T09:53:11.709636+00:00`
- Git revision: `70aeb02`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `128` pages
- Adaptive reserve mode: `age-aware`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.81 token/s |
| Average TTFT | 472.18 ms |
| P90 TTFT | 936.22 ms |
| Survivor average TTFT | 217.90 ms |
| Survivor average TPOT | 30.46 ms |
| P90 E2E | 4.189 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1212.36 | 50.80 | 1.974 |
| 2 | 0.00 | fill | B | 64 | 5 | 935.27 | 39.16 | 3.402 |
| 3 | 0.00 | fill | C | 128 | 4 | 935.73 | 34.22 | 5.282 |
| 4 | 0.00 | fill | D | 16 | 2 | 936.52 | 55.86 | 1.774 |
| 5 | 0.00 | fill | E | 64 | 3 | 936.22 | 39.16 | 3.404 |
| 6 | 0.00 | fill | F | 128 | 1 | 202.32 | 39.82 | 5.260 |
| 7 | 0.75 | refresh | A | 16 | 8 | 468.08 | 50.79 | 1.230 |
| 8 | 0.75 | refresh | B | 64 | 7 | 470.46 | 35.10 | 2.682 |
| 9 | 1.50 | pressure | G | 16 | 11 | 652.20 | 25.77 | 1.039 |
| 10 | 1.50 | pressure | H | 64 | 9 | 240.58 | 32.21 | 2.270 |
| 11 | 1.50 | pressure | I | 128 | 10 | 446.45 | 29.47 | 4.189 |
| 12 | 4.50 | probe-survivor | I | 128 | 12 | 124.29 | 28.93 | 3.798 |
| 13 | 4.50 | probe-survivor | H | 64 | 17 | 195.59 | 31.02 | 2.150 |
| 14 | 4.50 | probe-survivor | G | 16 | 15 | 132.37 | 31.75 | 0.609 |
| 15 | 4.50 | probe-survivor | B | 64 | 18 | 666.54 | 29.37 | 2.517 |
| 16 | 4.50 | probe-survivor | A | 16 | 14 | 134.46 | 31.71 | 0.610 |
| 17 | 4.50 | probe-survivor | F | 128 | 16 | 129.75 | 28.91 | 3.802 |
| 18 | 4.50 | probe-survivor | E | 64 | 13 | 142.30 | 31.52 | 2.128 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 242.15 | 25.80 | 0.629 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 239.86 | 25.64 | 3.496 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
