# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T10:11:06.403796+00:00`
- Git revision: `05680d2`
- Git worktree dirty: `False`
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
| 1 | 38.01 | 0.03 | 102.65 | 102.65 | 23.85 | 24.72 | 25.17 | 25.82 | 0.842 | 100.00 | 32517 |
| 8 | 166.87 | 0.10 | 532.79 | 714.42 | 31.66 | 27.71 | 361.68 | 362.04 | 1.530 | 99.17 | 32517 |
| 32 | 267.70 | 0.03 | 1597.92 | 2788.93 | 68.03 | 210.15 | 384.46 | 384.69 | 3.794 | 99.35 | 32517 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
