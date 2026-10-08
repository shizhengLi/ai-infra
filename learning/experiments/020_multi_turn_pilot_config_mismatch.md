# L20 Radix Cache pilot: excluded configuration mismatch

This run is retained for auditability but excluded from experiment 020's formal comparison. The
server accidentally used `memory_ratio=0.9` and CUDA Graph maximum batch size 160 instead of the
preregistered `0.8/64`. The scenario was rerun from a fresh server in `020_multi_turn_raw.md`.

## Configuration

- Scenario: `multi-turn`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:38:00.895133+00:00`
- Git revision: `f27eb8d`
- Tensor parallelism: `4`
- Output length: `32` tokens
- Seed: `3000042`
- Expected measured server UIDs: `1-12` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 12 |
| Average input tokens | 566.00 |
| Total input tokens | 6792 |
| Sequential output throughput | 38.20 token/s |
| Average TTFT | 97.10 ms |
| P50 TTFT | 70.61 ms |
| P90 TTFT | 144.60 ms |
| Average TPOT | 23.89 ms |
| Average E2E | 0.838 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 (c1t1) | 397 | 170.91 | 23.87 | 0.911 |
| 2 (c1t2) | 566 | 72.76 | 23.87 | 0.813 |
| 3 (c1t3) | 735 | 71.54 | 23.93 | 0.813 |
| 4 (c2t1) | 397 | 144.60 | 23.88 | 0.885 |
| 5 (c2t2) | 566 | 68.87 | 23.89 | 0.809 |
| 6 (c2t3) | 735 | 70.30 | 23.91 | 0.812 |
| 7 (c3t1) | 397 | 144.51 | 23.88 | 0.885 |
| 8 (c3t2) | 566 | 67.91 | 23.89 | 0.809 |
| 9 (c3t3) | 735 | 70.14 | 23.90 | 0.811 |
| 10 (c4t1) | 397 | 144.42 | 23.87 | 0.884 |
| 11 (c4t2) | 566 | 68.66 | 23.89 | 0.809 |
| 12 (c4t3) | 735 | 70.61 | 23.90 | 0.811 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
