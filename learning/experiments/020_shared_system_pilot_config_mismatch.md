# L20 Radix Cache pilot: excluded configuration mismatch

This run is retained for auditability but excluded from experiment 020's formal comparison. The
server accidentally used `memory_ratio=0.9` and CUDA Graph maximum batch size 160 instead of the
preregistered `0.8/64`. The scenario was rerun from a fresh server in
`020_shared_system_raw.md`.

## Configuration

- Scenario: `shared-system`
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T12:36:08.974225+00:00`
- Git revision: `f27eb8d`
- Tensor parallelism: `4`
- Output length: `32` tokens
- Seed: `3000042`
- Expected measured server UIDs: `1-12` (`0` is warmup)

## Summary

| Metric | Value |
| --- | ---: |
| Requests | 12 |
| Average input tokens | 1037.00 |
| Total input tokens | 12444 |
| Sequential output throughput | 36.75 token/s |
| Average TTFT | 128.61 ms |
| P50 TTFT | 108.43 ms |
| P90 TTFT | 109.16 ms |
| Average TPOT | 23.94 ms |
| Average E2E | 0.871 s |

## Requests

| Request | Input tokens | TTFT ms | Average TPOT ms | E2E s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1037 | 350.44 | 23.95 | 1.093 |
| 2 | 1037 | 109.04 | 23.94 | 0.851 |
| 3 | 1037 | 108.74 | 23.94 | 0.851 |
| 4 | 1037 | 109.16 | 23.94 | 0.851 |
| 5 | 1037 | 108.61 | 23.94 | 0.851 |
| 6 | 1037 | 108.13 | 23.95 | 0.851 |
| 7 | 1037 | 108.16 | 23.93 | 0.850 |
| 8 | 1037 | 108.08 | 23.94 | 0.850 |
| 9 | 1037 | 107.98 | 23.94 | 0.850 |
| 10 | 1037 | 108.43 | 23.95 | 0.851 |
| 11 | 1037 | 108.02 | 23.94 | 0.850 |
| 12 | 1037 | 108.49 | 23.94 | 0.851 |

Cache matches, insertions, evictions, and resident-token high-water marks are stored in the paired
rank-0 cache telemetry JSONL file.
