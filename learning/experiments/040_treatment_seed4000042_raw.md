# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T09:03:15.293698+00:00`
- Git revision: `b23d3a2`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.7`
- CUDA Graph max batch: `64`
- CUDA Graph explicit batches: `[1, 2, 4, 8, 16, 20, 24, 28, 32, 40, 48, 56, 64]`
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
| 1 | 37.92 | 0.21 | 104.50 | 104.50 | 23.85 | 24.72 | 25.00 | 25.07 | 0.844 | 100.00 | 32565 |
| 8 | 166.35 | 0.08 | 535.34 | 715.80 | 31.67 | 27.58 | 361.99 | 362.28 | 1.532 | 97.33 | 32565 |
| 32 | 267.98 | 0.11 | 1596.72 | 2787.26 | 67.99 | 220.97 | 372.72 | 383.54 | 3.792 | 95.27 | 32565 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
