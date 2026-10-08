# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `256` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-08T07:59:30.236778+00:00`
- Git revision: `129c869`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `2`
- Memory ratio: `0.8`
- CUDA Graph max batch: `64`
- Maximum prefill tokens: `8192`
- GPUs monitored: `0,1`

## Workload

- Prompt content length: `1024` tokens plus chat-template tokens
- Requested output: `256` tokens with EOS ignored
- Concurrency: `[32]`
- Repeats: `3`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 159.09 | 0.10 | 15934.23 | 24966.77 | 56.12 | 53.37 | 1820.77 | 4782.50 | 38.449 | 98.82 | 37955 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
