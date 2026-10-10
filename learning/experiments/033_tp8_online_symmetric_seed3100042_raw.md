# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T15:22:34.211017+00:00`
- Git revision: `62b9e0e`
- Git worktree dirty: `False`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `8`
- Memory ratio: `0.7`
- CUDA Graph max batch: `0`
- Maximum prefill tokens: `2048`
- Maximum prefill streak: `1`
- Decode-active prefill tokens: `1024`
- GPUs monitored: `0,1,2,3,4,5,6,7`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `32` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.70 | 4.57 | 228.62 | 228.62 | 28.33 | 30.09 | 33.01 | 33.84 | 1.107 | 78.42 | 32053 |
| 8 | 167.90 | 1.59 | 451.47 | 598.69 | 33.80 | 31.36 | 285.61 | 289.58 | 1.517 | 85.40 | 32053 |
| 32 | 172.80 | 0.27 | 2589.91 | 4871.37 | 37.41 | 36.53 | 289.85 | 290.31 | 5.842 | 91.55 | 32053 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
