# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T10:40:24.818041+00:00`
- Git revision: `05680d2`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.7`
- CUDA Graph max batch: `64`
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
| 1 | 37.96 | 0.04 | 103.38 | 103.38 | 23.86 | 24.74 | 24.82 | 24.93 | 0.843 | 99.88 | 32547 |
| 8 | 166.42 | 0.10 | 537.48 | 719.80 | 31.64 | 27.37 | 361.73 | 362.43 | 1.535 | 97.94 | 32547 |
| 32 | 267.11 | 0.78 | 1608.86 | 2799.85 | 68.04 | 221.71 | 373.11 | 385.02 | 3.806 | 96.58 | 32547 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
