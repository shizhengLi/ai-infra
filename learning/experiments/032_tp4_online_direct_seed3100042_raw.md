# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T14:49:04.534914+00:00`
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
| 1 | 34.50 | 1.07 | 100.63 | 100.63 | 26.70 | 27.90 | 33.54 | 42.48 | 0.928 | 78.29 | 32185 |
| 8 | 176.92 | 0.64 | 447.72 | 594.95 | 31.54 | 29.02 | 285.45 | 288.52 | 1.442 | 93.89 | 32185 |
| 32 | 176.59 | 0.31 | 2544.79 | 4799.24 | 36.12 | 35.30 | 290.63 | 294.99 | 5.726 | 95.41 | 32185 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
