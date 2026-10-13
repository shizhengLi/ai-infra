# L20 offline benchmark

## Purpose

Measure end-to-end offline decode throughput with deterministic, non-shared-prefix inputs. Each
repeat uses a different seed so the radix cache cannot turn later repeats into prefix-cache tests.

## Principle

The workload keeps input and requested output lengths fixed. `ignore_eos=True` guarantees the same
number of generated tokens in every run. Initialization time is reported separately because CUDA
Graph capture and kernel JIT affect startup but not steady-state throughput.

## Environment

- Timestamp (UTC): `2026-10-13T12:22:56.187255+00:00`
- Git revision: `1d8d8de`
- Git worktree dirty: `True`
- Tracked diff SHA-256: `6b3fc0c8a61d`
- GPU: `NVIDIA L20` (SM `8.9`)
- GPU memory: `44.39 GiB`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128`
- PyTorch CUDA: `12.8`

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-4B-AWQ/snapshots/master`
- Attention backend: `fi`
- Quantization: `awq`
- Requests: `32`
- Input/output tokens per request: `128/64`
- Repeats: `3`
- Max running requests: `32`
- CUDA Graph max batch size: `32`
- Resolved CUDA Graph sizes: `[1, 2, 4, 8, 16, 24, 32]`
- Memory ratio: `0.8`
- Fixed KV pages: `4096`
- Page size/cache: `1` / `radix`

## Results

- Engine initialization: **5.3532 s**
- CUDA allocated after initialization: **3311.72 MiB**
- CUDA reserved after initialization: **3502.00 MiB**
- Peak CUDA allocated: **3503.81 MiB**
- Mean throughput: **1885.22 token/s**
- Median throughput: **1885.08 token/s**
- Population standard deviation: **0.76 token/s**

| Repeat | Duration (s) | Throughput (token/s) |
| ---: | ---: | ---: |
| 1 | 1.0868 | 1884.36 |
| 2 | 1.0864 | 1885.08 |
| 3 | 1.0858 | 1886.21 |

## Conclusion

This file records raw benchmark facts. Interpret the comparison and next action in
`learning/progress.md` and the corresponding numbered experiment note.
