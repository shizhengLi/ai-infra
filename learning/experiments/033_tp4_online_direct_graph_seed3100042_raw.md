# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T15:31:37.922744+00:00`
- Git revision: `62b9e0e`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.7`
- CUDA Graph max batch: `64`
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
| 1 | 35.79 | 3.77 | 167.07 | 167.07 | 23.81 | 24.70 | 25.24 | 26.01 | 0.905 | 99.92 | 32493 |
| 8 | 179.43 | 0.16 | 443.78 | 593.70 | 30.97 | 28.89 | 285.93 | 287.84 | 1.421 | 98.78 | 32493 |
| 32 | 303.65 | 0.62 | 1320.30 | 2303.08 | 62.50 | 241.94 | 301.38 | 302.35 | 3.349 | 94.99 | 32493 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
