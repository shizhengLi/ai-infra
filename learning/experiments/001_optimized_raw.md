# L20 offline benchmark

## Purpose

Measure end-to-end offline decode throughput with deterministic, non-shared-prefix inputs. Each
repeat uses a different seed so the radix cache cannot turn later repeats into prefix-cache tests.

## Principle

The workload keeps input and requested output lengths fixed. `ignore_eos=True` guarantees the same
number of generated tokens in every run. Initialization time is reported separately because CUDA
Graph capture and kernel JIT affect startup but not steady-state throughput.

## Environment

- Timestamp (UTC): `2026-10-08T07:06:55.364691+00:00`
- Git revision: `b831cd9`
- GPU: `NVIDIA L20` (SM `8.9`)
- GPU memory: `44.39 GiB`
- Python: `3.12.14`
- PyTorch: `2.9.1+cu128`
- PyTorch CUDA: `12.8`

## Configuration

- Model: `Qwen/Qwen3-0.6B`
- Attention backend: `fi`
- Requests: `128`
- Input/output tokens per request: `256/128`
- Repeats: `3`
- Max running requests: `128`
- CUDA Graph max batch size: `auto`
- Memory ratio: `0.8`
- Page size/cache: `1` / `radix`

## Results

- Engine initialization: **7.3553 s**
- Mean throughput: **10142.23 token/s**
- Median throughput: **10145.02 token/s**
- Population standard deviation: **11.44 token/s**

| Repeat | Duration (s) | Throughput (token/s) |
| ---: | ---: | ---: |
| 1 | 1.6178 | 10127.02 |
| 2 | 1.6150 | 10145.02 |
| 3 | 1.6135 | 10154.64 |

## Conclusion

This file records raw benchmark facts. Interpret the comparison and next action in
`learning/progress.md` and the corresponding numbered experiment note.
