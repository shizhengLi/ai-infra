# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T15:49:27.631954+00:00`
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
- Repeats: `5`
- Unrecorded steady-state warmup repeats per concurrency: `2`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.59 | 0.15 | 91.75 | 91.75 | 23.79 | 24.72 | 25.05 | 25.68 | 0.829 | 100.00 | 32493 |
| 8 | 179.65 | 0.32 | 443.54 | 593.66 | 31.00 | 28.07 | 288.12 | 288.49 | 1.421 | 95.65 | 32493 |
| 32 | 303.66 | 0.31 | 1316.95 | 2299.90 | 62.54 | 242.85 | 300.74 | 301.04 | 3.346 | 95.69 | 32493 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
