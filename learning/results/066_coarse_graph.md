# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-17T15:13:57.591988+00:00`
- Git revision: `8407b33`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Quantization: `fp8`
- Memory ratio: `0.8`
- CUDA Graph max batch: `32`
- CUDA Graph explicit batches: `None`
- CUDA Graph shape policy: `coarse`
- Maximum prefill tokens: `20000`
- Maximum prefill streak: `0`
- Decode-active prefill tokens: `0`
- Decode Graph-tail prefill guard batch size: `0`
- Decode Graph-tail prefill priority batch size: `0`
- GPUs monitored: `2,3,6,7`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `64` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `1`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 54.42 | 0.13 | 98.39 | 98.39 | 17.11 | 17.59 | 18.63 | 19.83 | 1.176 | 99.58 | 36767 |
| 8 | 291.67 | 0.29 | 504.10 | 535.26 | 19.81 | 20.41 | 265.40 | 266.05 | 1.753 | 89.58 | 36767 |
| 32 | 546.30 | 3.70 | 1751.17 | 1988.32 | 31.31 | 30.21 | 476.74 | 1661.31 | 3.737 | 94.32 | 36767 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
