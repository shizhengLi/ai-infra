# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T11:34:39.126524+00:00`
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
| 1 | 61.58 | 0.01 | 79.58 | 79.58 | 15.23 | 15.56 | 16.01 | 16.71 | 1.039 | 100.00 | 15407 |
| 8 | 341.03 | 1.34 | 369.35 | 427.61 | 17.94 | 17.47 | 318.15 | 346.77 | 1.500 | 98.75 | 15407 |
| 32 | 402.24 | 0.59 | 2091.53 | 4000.46 | 19.25 | 19.40 | 140.77 | 504.02 | 5.075 | 96.79 | 15407 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
