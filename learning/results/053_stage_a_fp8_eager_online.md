# L20 online benchmark result

## Principle

All requests in a concurrency group start together. Prompts are generated from distinct seeds for
every repeat to avoid radix-cache reuse. TTFT ends at the first streamed token; TPOT uses intervals
between the first `64` token events and excludes the final OpenAI finish event.

## Environment

- Timestamp: `2026-10-13T11:16:11.433170+00:00`
- Git revision: `4a61602`
- Git worktree dirty: `True`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128` (`cu12.8`)
- Model: `/data2/lszlsz/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

## Server configuration

- Tensor parallelism: `1`
- Memory ratio: `0.9`
- CUDA Graph max batch: `0`
- CUDA Graph explicit batches: `None`
- Maximum prefill tokens: `4096`
- Maximum prefill streak: `0`
- Decode-active prefill tokens: `0`
- Decode Graph-tail prefill guard batch size: `0`
- Decode Graph-tail prefill priority batch size: `0`
- GPUs monitored: `6`

## Workload

- Prompt content length: `256` tokens plus chat-template tokens
- Requested output: `64` tokens with EOS ignored
- Concurrency: `[1, 8, 32]`
- Repeats: `3`
- Unrecorded steady-state warmup repeats per concurrency: `1`

## Aggregate results

| C | Output tok/s | Stddev | Avg TTFT ms | P90 TTFT ms | Avg TPOT ms | P90 TPOT ms | P99.9 TPOT ms | Max TPOT ms | P90 E2E s | GPU util % | Peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76.90 | 0.15 | 33.34 | 33.34 | 12.68 | 12.98 | 14.10 | 15.72 | 0.832 | 18.33 | 2005 |
| 8 | 572.32 | 0.38 | 53.60 | 68.81 | 13.31 | 13.28 | 51.44 | 51.76 | 0.893 | 19.83 | 2005 |
| 32 | 781.85 | 3.44 | 822.13 | 1773.68 | 13.06 | 13.51 | 16.22 | 17.22 | 2.592 | 18.67 | 2069 |

Per-repeat measurements and raw stream timestamps are stored in the machine-readable JSON result.
