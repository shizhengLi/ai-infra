# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T07:30:00.517812+00:00`
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
| 1 | 38.06 | 0.03 | 101.61 | 101.61 | 23.85 | 24.75 | 25.05 | 25.17 | 0.841 | 99.92 | 32517 |
| 8 | 166.66 | 0.21 | 532.91 | 713.77 | 31.64 | 27.59 | 362.71 | 362.77 | 1.529 | 95.56 | 32517 |
| 32 | 267.60 | 0.40 | 1603.39 | 2793.55 | 68.04 | 221.15 | 374.68 | 385.51 | 3.799 | 97.90 | 32517 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
