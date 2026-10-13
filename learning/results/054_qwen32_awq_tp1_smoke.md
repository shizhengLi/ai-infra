# L20 offline benchmark

## Purpose

Measure end-to-end offline decode throughput with deterministic, non-shared-prefix inputs. Each
repeat uses a different seed so the radix cache cannot turn later repeats into prefix-cache tests.

## Principle

The workload keeps input and requested output lengths fixed. `ignore_eos=True` guarantees the same
number of generated tokens in every run. Initialization time is reported separately because CUDA
Graph capture and kernel JIT affect startup but not steady-state throughput.

## Environment

- Timestamp (UTC): `2026-10-13T12:19:57.599514+00:00`
- Git revision: `1d8d8de`
- Git worktree dirty: `True`
- Tracked diff SHA-256: `19ce5ef7873e`
- GPU: `NVIDIA L20` (SM `8.9`)
- GPU memory: `44.39 GiB`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128`
- PyTorch CUDA: `12.8`

## Configuration

- Model: `/data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master`
- Attention backend: `fi`
- Quantization: `awq`
- Requests: `1`
- Input/output tokens per request: `16/4`
- Repeats: `1`
- Max running requests: `1`
- CUDA Graph max batch size: `1`
- Resolved CUDA Graph sizes: `[1]`
- Memory ratio: `0.8`
- Fixed KV pages: `1024`
- Page size/cache: `1` / `radix`

## Results

- Engine initialization: **9.4455 s**
- CUDA allocated after initialization: **18911.08 MiB**
- CUDA reserved after initialization: **19824.00 MiB**
- Peak CUDA allocated: **18914.80 MiB**
- Mean throughput: **36.27 token/s**
- Median throughput: **36.27 token/s**
- Population standard deviation: **0.00 token/s**

| Repeat | Duration (s) | Throughput (token/s) |
| ---: | ---: | ---: |
| 1 | 0.1103 | 36.27 |

## Conclusion

This file records raw benchmark facts. Interpret the comparison and next action in
`learning/progress.md` and the corresponding numbered experiment note.
