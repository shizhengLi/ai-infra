# L20 online benchmark result

This accepted comparison run explicitly used BF16 runtime dtype with the AWQ checkpoint.

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T12:34:22.839623+00:00`
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
| 1 | 34.69 | 0.02 | 177.28 | 177.28 | 26.47 | 27.07 | 27.32 | 27.36 | 1.845 | 100.00 | 21037 |
| 8 | 169.66 | 0.13 | 1008.23 | 1253.25 | 31.81 | 28.48 | 901.10 | 925.78 | 3.016 | 95.73 | 21295 |
| 32 | 197.74 | 0.07 | 4796.97 | 8568.41 | 30.31 | 29.55 | 646.58 | 1689.25 | 10.324 | 95.71 | 21755 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
