# L20 mixed-arrival benchmark result

## Principle

Anchor requests begin decoding before a timed burst of long-prefill requests arrives. Anchor TPOT
after burst dispatch measures decode starvation directly. Stream finish events are excluded.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T08:18:12.841350+00:00`
- Git revision: `1e1d7fa`
- Tensor parallelism: `4`
- Prefill budget: `8192`
- Maximum consecutive prefill batches: `0`
- Anchors: `8 x 128 input / 512 output tokens`
- Burst: `24 x 1024 input / 64 output tokens at 2.0 seconds`
- Repeats: `3`

## Aggregate results

| Metric | Value |
| --- | ---: |
| Output throughput | 250.03 token/s |
| Throughput stddev | 0.12 |
| Anchor average TTFT | 345.08 ms |
| Anchor P99 TPOT | 39.20 ms |
| Anchor P99.9 TPOT | 7655.96 ms |
| Anchor worst TPOT | 7674.46 ms |
| Post-burst anchor P99.9 TPOT | 7655.98 ms |
| Post-burst anchor worst TPOT | 7674.46 ms |
| Burst average TTFT | 5643.11 ms |
| Burst P90 TTFT | 7799.19 ms |
| Burst P90 E2E | 10.045 s |
| Average GPU utilization | 94.16% |
| Peak GPU memory | 37833 MiB |
| Average power per GPU | 221.22 W |

Per-repeat measurements and normalized raw stream timestamps are stored in the JSON result.
