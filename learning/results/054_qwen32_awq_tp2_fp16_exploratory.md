# L20 online benchmark result

This exploratory run inherited the checkpoint's FP16 runtime dtype and is excluded from the
dtype-controlled performance table.

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T12:29:07.873411+00:00`
- Git revision: `1d8d8de`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`

## Server configuration

- Tensor parallelism: `2`
- Quantization: `awq`
- Memory ratio: `0.9`
- CUDA Graph max batch: `32`
- CUDA Graph explicit batches: `None`
- Maximum prefill tokens: `4096`
- Maximum prefill streak: `0`
- Decode-active prefill tokens: `0`
- Decode Graph-tail prefill guard batch size: `0`
- Decode Graph-tail prefill priority batch size: `0`
- GPUs monitored: `4,5`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `64` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `1`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.84 | 0.09 | 109.66 | 109.66 | 14.96 | 15.25 | 16.39 | 18.01 | 1.052 | 99.67 | 11329 |
| 8 | 278.95 | 0.08 | 721.37 | 780.24 | 17.60 | 17.09 | 477.27 | 477.77 | 1.834 | 93.96 | 11419 |
| 32 | 316.05 | 0.33 | 2989.89 | 5399.28 | 19.54 | 18.95 | 407.03 | 1020.93 | 6.451 | 97.24 | 11573 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
