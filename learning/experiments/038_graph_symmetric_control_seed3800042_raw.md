# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `128` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T08:03:55.695099+00:00`
- Git revision: `b23d3a2`
- Git worktree dirty: `False`
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
- Requested output: `128` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `2`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.93 | 0.01 | 101.48 | 101.48 | 24.44 | 24.71 | 25.58 | 26.56 | 3.206 | 99.97 | 32517 |
| 8 | 247.61 | 0.05 | 533.36 | 715.17 | 28.21 | 27.36 | 181.05 | 362.85 | 4.132 | 96.76 | 32517 |
| 32 | 575.29 | 0.29 | 1604.10 | 2794.77 | 42.52 | 34.74 | 385.26 | 385.34 | 7.092 | 97.52 | 32517 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
