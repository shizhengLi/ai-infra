# L20 offline benchmark

## Purpose

Measure end-to-end offline decode throughput with deterministic, non-shared-prefix inputs. Each
repeat uses a different seed so the radix cache cannot turn later repeats into prefix-cache tests.

## Principle

The workload keeps input and requested output lengths fixed. `ignore_eos=True` guarantees the same
number of generated tokens in every run. Initialization time is reported separately because CUDA
Graph capture and kernel JIT affect startup but not steady-state throughput.

## Environment

- Timestamp (UTC): `2026-10-13T11:02:31.986528+00:00`
- Git revision: `4a61602`
- Git worktree dirty: `True`
- Tracked diff SHA-256: `d7ce5ef91c88`
- GPU: `NVIDIA L20` (SM `8.9`)
- GPU memory: `44.39 GiB`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128`
- PyTorch CUDA: `12.8`

## Configuration

- Model: `/data2/lszlsz/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
- Attention backend: `fi`
- Quantization: `none`
- Requests: `32`
- Input/output tokens per request: `128/64`
- Repeats: `1`
- Max running requests: `32`
- CUDA Graph max batch size: `32`
- Resolved CUDA Graph sizes: `[1, 2, 4, 8, 16, 24, 32]`
- Memory ratio: `0.8`
- Fixed KV pages: `4096`
- Page size/cache: `1` / `radix`

## Results

- Engine initialization: **4.2376 s**
- CUDA allocated after initialization: **1768.01 MiB**
- CUDA reserved after initialization: **1862.00 MiB**
- Peak CUDA allocated: **1833.34 MiB**
- Mean throughput: **4681.88 token/s**
- Median throughput: **4681.88 token/s**
- Population standard deviation: **0.00 token/s**

| Repeat | Duration (s) | Throughput (token/s) |
| ---: | ---: | ---: |
| 1 | 0.4374 | 4681.88 |

## Conclusion

This file records raw benchmark facts. Interpret the comparison and next action in
`learning/progress.md` and the corresponding numbered experiment note.
