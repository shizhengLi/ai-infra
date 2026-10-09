# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-09T12:38:26.760286+00:00`
- Git revision: `1683a96`
- Tensor parallelism: `4`
- KV pages / page size: `4576 / 1`
- Partial-leaf eviction: `True`
- Fixed reserve: `0` pages
- Adaptive reserve maximum: `64` pages
- Adaptive reserve mode: `raw`
- Protected recent matches: `0`
- Hotness decay: `0.9`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `3100042`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 20 |
| Output tokens | 1328 |
| Open-loop output throughput | 139.48 token/s |
| Average TTFT | 486.04 ms |
| P90 TTFT | 936.94 ms |
| Survivor average TTFT | 258.53 ms |
| Survivor average TPOT | 29.48 ms |
| P90 E2E | 4.155 s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00 | fill | A | 16 | 6 | 1213.22 | 50.82 | 1.976 |
| 2 | 0.00 | fill | B | 64 | 5 | 935.95 | 39.12 | 3.401 |
| 3 | 0.00 | fill | C | 128 | 4 | 936.47 | 33.98 | 5.252 |
| 4 | 0.00 | fill | D | 16 | 2 | 937.29 | 55.90 | 1.776 |
| 5 | 0.00 | fill | E | 64 | 3 | 936.94 | 39.13 | 3.402 |
| 6 | 0.00 | fill | F | 128 | 1 | 201.90 | 39.57 | 5.227 |
| 7 | 0.75 | refresh | A | 16 | 8 | 467.44 | 50.81 | 1.230 |
| 8 | 0.75 | refresh | B | 64 | 7 | 470.26 | 35.10 | 2.682 |
| 9 | 1.50 | pressure | G | 16 | 9 | 244.07 | 48.95 | 0.978 |
| 10 | 1.50 | pressure | H | 64 | 11 | 652.83 | 26.60 | 2.329 |
| 11 | 1.50 | pressure | I | 128 | 10 | 443.26 | 29.23 | 4.155 |
| 12 | 4.50 | probe-survivor | I | 128 | 18 | 600.09 | 27.95 | 4.149 |
| 13 | 4.50 | probe-survivor | H | 64 | 15 | 155.10 | 30.84 | 2.098 |
| 14 | 4.50 | probe-survivor | G | 16 | 14 | 99.13 | 30.39 | 0.555 |
| 15 | 4.50 | probe-survivor | B | 64 | 12 | 100.52 | 31.26 | 2.070 |
| 16 | 4.50 | probe-survivor | A | 16 | 17 | 600.46 | 26.15 | 0.993 |
| 17 | 4.50 | probe-survivor | F | 128 | 16 | 154.60 | 28.54 | 3.779 |
| 18 | 4.50 | probe-survivor | E | 64 | 13 | 99.78 | 31.26 | 2.069 |
| 19 | 6.00 | probe-evicted | D | 16 | 19 | 237.40 | 25.55 | 0.621 |
| 20 | 6.00 | probe-evicted | C | 128 | 20 | 234.14 | 25.85 | 3.517 |

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
