# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T14:55:48.755470+00:00`
- Git revision: `2ac5c0e`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.7`
- CUDA Graph max batch: `0`
- Maximum prefill tokens: `2048`
- Maximum prefill streak: `1`
- Decode-active prefill tokens: `1024`
- GPUs monitored: `0,1,2,3`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `32` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31.50 | 2.30 | 161.09 | 161.09 | 27.76 | 29.24 | 30.59 | 31.71 | 1.022 | 81.88 | 32193 |
| 8 | 158.22 | 0.58 | 543.55 | 722.80 | 33.92 | 30.56 | 354.72 | 364.43 | 1.611 | 88.53 | 32193 |
| 32 | 158.61 | 0.45 | 2868.44 | 5377.46 | 39.26 | 45.49 | 363.82 | 364.06 | 6.369 | 94.12 | 32193 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
