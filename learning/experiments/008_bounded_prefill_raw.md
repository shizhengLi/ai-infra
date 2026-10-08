# L20 mixed-arrival benchmark result

## Principle

Anchor requests begin decoding before a timed burst of long-prefill requests arrives. Anchor TPOT
after burst dispatch measures decode starvation directly. Stream finish events are excluded.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T08:22:00.118437+00:00`
- Git revision: `1e1d7fa`
- Tensor parallelism: `4`
- Prefill budget: `8192`
- Maximum consecutive prefill batches: `1`
- Anchors: `8 x 128 input / 512 output tokens`
- Burst: `24 x 1024 input / 64 output tokens at 2.0 seconds`
- Repeats: `3`

## Aggregate results

| Metric | Value |
| --- | ---: |
| Output throughput | 249.71 token/s |
| Throughput stddev | 0.10 |
| Anchor average TTFT | 346.35 ms |
| Anchor P99 TPOT | 153.29 ms |
| Anchor P99.9 TPOT | 2799.49 ms |
| Anchor worst TPOT | 2895.48 ms |
| Post-burst anchor P99.9 TPOT | 2799.50 ms |
| Post-burst anchor worst TPOT | 2895.48 ms |
| Burst average TTFT | 5336.88 ms |
| Burst P90 TTFT | 7880.41 ms |
| Burst P90 E2E | 10.103 s |
| Average GPU utilization | 94.55% |
| Peak GPU memory | 37833 MiB |
| Average power per GPU | 223.52 W |

Per-repeat measurements and normalized raw stream timestamps are stored in the JSON result.
