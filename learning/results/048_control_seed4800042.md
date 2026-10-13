# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T12:30:53.011895+00:00`
- Git revision: `8f40b56`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.7`
- CUDA Graph max batch: `0`
- CUDA Graph explicit batches: `None`
- Maximum prefill tokens: `2048`
- Maximum prefill streak: `1`
- Decode-active prefill tokens: `1024`
- Decode Graph-tail prefill guard batch size: `0`
- Decode Graph-tail prefill priority batch size: `0`
- GPUs monitored: `0,1,2,3`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `32` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `2`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.29 | 0.02 | 85.93 | 85.93 | 25.68 | 26.72 | 27.63 | 28.12 | 0.882 | 95.33 | 32225 |
| 8 | 178.20 | 0.22 | 442.23 | 591.98 | 31.40 | 29.07 | 288.40 | 288.55 | 1.432 | 95.36 | 32225 |
| 32 | 303.15 | 0.20 | 1308.32 | 2290.54 | 62.84 | 244.68 | 300.66 | 300.82 | 3.344 | 96.28 | 32227 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
