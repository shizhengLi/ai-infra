# L20 Radix Cache workload result

## Configuration

- Scenario: `shared-system`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:40:28.062129+00:00`
- Git revision: `f27eb8d`
- Tensor parallelism: `4`
- Memory ratio / KV pages: `0.8` / `305066`
- CUDA Graph maximum batch size: `64`
- Scheduling profile: prefill streak `1`, active prefill budget `2304`, result-before-prefill enabled
- Output length: `32` tokens
- Seed: `3000042`
- Expected measured server UIDs: `1-12` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 12 |
| Average input tokens | 1037.00 |
| Total input tokens | 12444 |
| Sequential output throughput | 36.07 token/s |
| Average TTFT | 144.95 ms |
| P50 TTFT | 108.14 ms |
| P90 TTFT | 108.71 ms |
| Average TPOT | 23.94 ms |
| Average E2E | 0.887 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1037 | 555.57 | 23.94 | 1.298 |
| 2 | 1037 | 101.44 | 23.95 | 0.844 |
| 3 | 1037 | 108.14 | 23.95 | 0.851 |
| 4 | 1037 | 108.52 | 23.93 | 0.850 |
| 5 | 1037 | 108.71 | 23.92 | 0.850 |
| 6 | 1037 | 108.37 | 23.93 | 0.850 |
| 7 | 1037 | 108.12 | 23.94 | 0.850 |
| 8 | 1037 | 107.95 | 23.95 | 0.850 |
| 9 | 1037 | 108.40 | 23.95 | 0.851 |
| 10 | 1037 | 108.34 | 23.94 | 0.850 |
| 11 | 1037 | 108.08 | 23.94 | 0.850 |
| 12 | 1037 | 107.71 | 23.96 | 0.850 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
