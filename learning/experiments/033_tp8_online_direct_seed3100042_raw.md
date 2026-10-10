# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T15:24:52.187478+00:00`
- Git revision: `62b9e0e`
- Git worktree dirty: `True`
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
| 1 | 32.72 | 4.41 | 189.78 | 189.78 | 26.07 | 27.58 | 30.86 | 35.35 | 0.998 | 73.88 | 32047 |
| 8 | 177.17 | 1.66 | 445.63 | 591.84 | 31.57 | 29.65 | 272.06 | 279.46 | 1.443 | 86.15 | 32047 |
| 32 | 181.89 | 0.31 | 2465.07 | 4638.64 | 35.24 | 31.22 | 268.30 | 268.46 | 5.549 | 88.97 | 32047 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
