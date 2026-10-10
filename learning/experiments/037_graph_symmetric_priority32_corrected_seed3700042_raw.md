# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `32` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-10T07:45:42.218542+00:00`
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
- Decode Graph-tail prefill priority batch size: `32`
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
| 1 | 38.08 | 0.05 | 100.93 | 100.93 | 23.85 | 24.82 | 25.31 | 25.65 | 0.840 | 100.00 | 32517 |
| 8 | 166.89 | 0.16 | 532.45 | 713.72 | 31.67 | 27.40 | 362.80 | 362.91 | 1.529 | 92.83 | 32517 |
| 32 | 265.91 | 0.04 | 1603.69 | 2824.95 | 68.02 | 220.23 | 348.22 | 348.34 | 3.822 | 92.29 | 32517 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
