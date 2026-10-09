# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T14:31:47.291009+00:00`
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
| 1 | 18.77 | 0.00 | 419.32 | 419.32 | 41.48 | 43.92 | 56.19 | 56.19 | 1.705 | 80.00 | 32201 |
| 8 | 120.92 | 0.00 | 581.20 | 763.98 | 48.48 | 57.07 | 338.87 | 338.87 | 2.109 | 85.75 | 32201 |
| 32 | 124.45 | 0.00 | 3533.31 | 6738.45 | 53.06 | 71.91 | 363.00 | 363.00 | 8.117 | 87.77 | 32205 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
