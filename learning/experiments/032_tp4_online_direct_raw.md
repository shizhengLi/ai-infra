# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T14:44:34.768718+00:00`
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
- Repeats: `1`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 35.08 | 0.00 | 118.64 | 118.64 | 25.60 | 26.46 | 32.18 | 32.18 | 0.912 | 48.62 | 32185 |
| 8 | 177.68 | 0.00 | 456.88 | 597.92 | 31.02 | 28.42 | 285.21 | 285.21 | 1.434 | 94.75 | 32185 |
| 32 | 177.21 | 0.00 | 2533.33 | 4783.13 | 35.97 | 40.20 | 295.37 | 295.37 | 5.699 | 94.78 | 32185 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
