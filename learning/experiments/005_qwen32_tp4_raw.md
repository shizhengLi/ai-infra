# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `256` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T07:34:56.870605+00:00`
- Git revision: `951fc2e`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.8`
- CUDA Graph max batch: `64`
- GPUs monitored: `0,1,2,3`

## Workload

- Prompt content length: `1024` tokens plus chat-template tokens
- Requested output: `256` tokens with EOS ignored
- Concurrency: `[1, 8, 32, 64]`
- Repeats: `3`

## Aggregate results

| C | Output tok/s | Stddev | P90 TTFT ms | P90 TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.08 | 0.72 | 438.90 | 24.84 | 6.725 | 98.88 | 37059 |
| 8 | 212.16 | 0.09 | 2610.39 | 27.88 | 9.652 | 98.46 | 37715 |
| 32 | 415.30 | 0.25 | 10346.97 | 37.13 | 19.708 | 96.88 | 38117 |
| 64 | 503.35 | 0.33 | 20701.20 | 46.95 | 32.512 | 98.11 | 38117 |

Per-repeat measurements are stored in `learning/results/005_qwen32_tp4.json`.
