# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T11:03:44.481542+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `4`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.76 token/s |
| Average TTFT | 471.17 ms |
| P90 TTFT | 936.84 ms |
| Survivor average TTFT | 214.35 ms |
| Survivor average TPOT | 30.05 ms |
| P90 E2E | 4.192 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1212.98 | 50.81 | 1.975 |
| 2 | 0.00 | fill | B | 64 | 5 | 935.91 | 39.16 | 3.403 |
| 3 | 0.00 | fill | C | 128 | 4 | 936.41 | 34.24 | 5.285 |
| 4 | 0.00 | fill | D | 16 | 2 | 937.18 | 55.87 | 1.775 |
| 5 | 0.00 | fill | E | 64 | 3 | 936.84 | 39.17 | 3.405 |
| 6 | 0.00 | fill | F | 128 | 1 | 201.44 | 39.83 | 5.260 |
| 7 | 0.75 | refresh | A | 16 | 8 | 468.00 | 50.80 | 1.230 |
| 8 | 0.75 | refresh | B | 64 | 7 | 470.46 | 35.11 | 2.683 |
| 9 | 1.50 | pressure | G | 16 | 11 | 652.30 | 25.79 | 1.039 |
| 10 | 1.50 | pressure | H | 64 | 9 | 239.92 | 32.21 | 2.269 |
| 11 | 1.50 | pressure | I | 128 | 10 | 446.01 | 29.50 | 4.192 |
| 12 | 4.50 | probe-survivor | I | 128 | 17 | 182.65 | 28.69 | 3.826 |
| 13 | 4.50 | probe-survivor | H | 64 | 12 | 140.10 | 31.61 | 2.131 |
| 14 | 4.50 | probe-survivor | G | 16 | 18 | 647.98 | 26.12 | 1.040 |
| 15 | 4.50 | probe-survivor | B | 64 | 15 | 131.36 | 31.61 | 2.123 |
| 16 | 4.50 | probe-survivor | A | 16 | 16 | 129.15 | 31.78 | 0.606 |
| 17 | 4.50 | probe-survivor | F | 128 | 14 | 133.49 | 28.94 | 3.809 |
| 18 | 4.50 | probe-survivor | E | 64 | 13 | 135.71 | 31.62 | 2.128 |
| 19 | 6.00 | probe-evicted | D | 16 | 20 | 241.25 | 25.69 | 0.627 |
| 20 | 6.00 | probe-evicted | C | 128 | 19 | 244.26 | 25.65 | 3.502 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
