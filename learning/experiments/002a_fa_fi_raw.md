# L20 offline benchmark

## Purpose

Measure end-to-end offline decode throughput with deterministic, non-shared-prefix inputs. Each
repeat uses a different seed so the radix cache cannot turn later repeats into prefix-cache tests.

## Principle

The workload keeps input and requested output lengths fixed. `ignore_eos=True` guarantees the same
number of generated tokens in every run. Initialization time is reported separately because CUDA
Graph capture and kernel JIT affect startup but not steady-state throughput.

## Environment

- Timestamp (UTC): `2026-10-08T11:16:10.108628+00:00`
- Git revision: `6d757c7`
- Git worktree dirty: `True`
- Tracked diff SHA-256: `67d96f3f384a`
- GPU: `NVIDIA L20` (SM `8.9`)
- GPU memory: `44.39 GiB`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128`
- PyTorch CUDA: `12.8`

## Configuration

- Model: `/data2/lszlsz/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
- Attention backend: `fa,fi`
- Requests: `128`
- Input/output tokens per request: `256/128`
- Repeats: `3`
- Max running requests: `128`
- CUDA Graph max batch size: `128`
- Resolved CUDA Graph sizes: `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]`
- Memory ratio: `0.8`
- Page size/cache: `1` / `radix`

## Results

- Engine initialization: **4.4744 s**
- Mean throughput: **10130.52 token/s**
- Median throughput: **10131.53 token/s**
- Population standard deviation: **3.13 token/s**

| Repeat | Duration (s) | Throughput (token/s) |
| ---: | ---: | ---: |
| 1 | 1.6180 | 10126.28 |
| 2 | 1.6171 | 10131.53 |
| 3 | 1.6168 | 10133.76 |

## Conclusion

This file records raw benchmark facts. Interpret the comparison and next action in
`learning/progress.md` and the corresponding numbered experiment note.
