# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T09:00:21.584139+00:00`
- Git revision: `b23d3a2`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.7`
- CUDA Graph max batch: `64`
- CUDA Graph explicit batches: `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]`
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
| 1 | 38.05 | 0.01 | 101.30 | 101.30 | 23.86 | 24.84 | 25.46 | 26.47 | 0.841 | 100.00 | 32517 |
| 8 | 166.74 | 0.08 | 531.75 | 711.99 | 31.66 | 28.06 | 361.49 | 361.84 | 1.528 | 95.06 | 32517 |
| 32 | 267.83 | 0.05 | 1599.67 | 2789.99 | 68.02 | 220.97 | 373.32 | 384.20 | 3.796 | 93.76 | 32517 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
