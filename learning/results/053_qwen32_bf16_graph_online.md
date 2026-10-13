# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T11:31:38.411738+00:00`
- Git revision: `4a61602`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master`

## Server configuration

- Tensor parallelism: `4`
- Memory ratio: `0.9`
- CUDA Graph max batch: `32`
- CUDA Graph explicit batches: `None`
- Maximum prefill tokens: `4096`
- Maximum prefill streak: `0`
- Decode-active prefill tokens: `0`
- Decode Graph-tail prefill guard batch size: `0`
- Decode Graph-tail prefill priority batch size: `0`
- GPUs monitored: `4,5,6,7`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `64` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `1`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.73 | 0.03 | 86.36 | 86.36 | 24.20 | 24.78 | 26.01 | 27.04 | 1.611 | 99.94 | 18229 |
| 8 | 228.14 | 0.11 | 483.40 | 537.41 | 27.87 | 27.78 | 450.25 | 450.35 | 2.243 | 95.06 | 18229 |
| 32 | 273.00 | 0.17 | 2958.80 | 5770.53 | 29.48 | 29.60 | 186.19 | 741.70 | 7.475 | 96.73 | 18229 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
