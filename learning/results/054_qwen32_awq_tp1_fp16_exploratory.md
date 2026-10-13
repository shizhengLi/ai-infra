# L20 online benchmark result

This exploratory run inherited the checkpoint's FP16 runtime dtype and is excluded from the
dtype-controlled performance table.

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T12:26:38.015910+00:00`
- Git revision: `1d8d8de`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`

## Server configuration

- Tensor parallelism: `1`
- Quantization: `awq`
- Memory ratio: `0.9`
- CUDA Graph max batch: `32`
- CUDA Graph explicit batches: `None`
- Maximum prefill tokens: `4096`
- Maximum prefill streak: `0`
- Decode-active prefill tokens: `0`
- Decode Graph-tail prefill guard batch size: `0`
- Decode Graph-tail prefill priority batch size: `0`
- GPUs monitored: `4`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `64` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `1`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.66 | 0.01 | 177.39 | 177.39 | 26.49 | 27.17 | 29.04 | 30.07 | 1.846 | 90.50 | 21031 |
| 8 | 169.74 | 0.08 | 1030.40 | 1250.23 | 31.43 | 28.62 | 887.33 | 887.63 | 3.015 | 95.33 | 21289 |
| 32 | 198.07 | 0.16 | 4784.41 | 8563.56 | 30.47 | 29.70 | 493.72 | 1676.33 | 10.314 | 100.00 | 21749 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
