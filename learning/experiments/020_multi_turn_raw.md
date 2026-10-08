# L20 Radix Cache workload result

## Configuration

- Scenario: `multi-turn`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:42:15.902764+00:00`
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
| Average input tokens | 566.00 |
| Total input tokens | 6792 |
| Sequential output throughput | 37.33 token/s |
| Average TTFT | 117.41 ms |
| P50 TTFT | 70.79 ms |
| P90 TTFT | 145.89 ms |
| Average TPOT | 23.87 ms |
| Average E2E | 0.857 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (c1t1) | 397 | 414.44 | 23.85 | 1.154 |
| 2 (c1t2) | 566 | 69.35 | 23.86 | 0.809 |
| 3 (c1t3) | 735 | 70.79 | 23.89 | 0.812 |
| 4 (c2t1) | 397 | 144.98 | 23.85 | 0.884 |
| 5 (c2t2) | 566 | 68.48 | 23.88 | 0.809 |
| 6 (c2t3) | 735 | 70.81 | 23.88 | 0.811 |
| 7 (c3t1) | 397 | 145.89 | 23.84 | 0.885 |
| 8 (c3t2) | 566 | 68.70 | 23.87 | 0.809 |
| 9 (c3t3) | 735 | 70.50 | 23.89 | 0.811 |
| 10 (c4t1) | 397 | 144.99 | 23.85 | 0.884 |
| 11 (c4t2) | 566 | 68.47 | 23.88 | 0.809 |
| 12 (c4t3) | 735 | 71.47 | 23.88 | 0.812 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
