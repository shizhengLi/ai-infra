# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-09T15:47:04.637604+00:00`
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
| 1 | 38.08 | 0.04 | 101.20 | 101.20 | 23.84 | 24.74 | 24.97 | 25.06 | 0.840 | 99.95 | 32523 |
| 8 | 166.83 | 0.32 | 533.20 | 714.72 | 31.65 | 27.35 | 362.86 | 363.33 | 1.530 | 97.40 | 32523 |
| 32 | 267.67 | 0.05 | 1598.59 | 2790.33 | 68.05 | 217.00 | 378.92 | 385.59 | 3.796 | 98.37 | 32523 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
