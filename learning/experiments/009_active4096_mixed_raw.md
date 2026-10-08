# L20 mixed-arrival benchmark result

## Principle

Anchor requests begin decoding before a timed burst of long-prefill requests arrives. Anchor TPOT
after burst dispatch measures decode starvation directly. Stream finish events are excluded.

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`
- Timestamp: `2026-10-08T08:34:57.564879+00:00`
- Git revision: `1f0b104`
- Tensor parallelism: `4`
- Prefill budget: `8192`
- Maximum consecutive prefill batches: `1`
- Decode-active prefill tokens: `4096`
- Anchors: `8 x 128 input / 512 output tokens`
- Burst: `24 x 1024 input / 64 output tokens at 2.0 seconds`
- Repeats: `3`

## Aggregate results

| Metric | Value |
| --- | ---: |
| Output throughput | 250.85 token/s |
| Throughput stddev | 0.12 |
| Anchor average TTFT | 344.60 ms |
| Anchor P99 TPOT | 692.99 ms |
| Anchor P99.9 TPOT | 1385.93 ms |
| Anchor worst TPOT | 1427.81 ms |
| Post-burst anchor P99.9 TPOT | 1385.94 ms |
| Post-burst anchor worst TPOT | 1427.81 ms |
| Burst average TTFT | 4788.49 ms |
| Burst P90 TTFT | 7866.61 ms |
| Burst P90 E2E | 10.079 s |
| Average GPU utilization | 94.50% |
| Peak GPU memory | 37391 MiB |
| Average power per GPU | 220.68 W |

Per-repeat measurements and normalized raw stream timestamps are stored in the JSON result.
