# Experiment 000: CUDA toolchain validation

## Hypothesis

The FlashInfer JIT failure is caused by an old `nvcc`, not by Mini-SGLang, the model, proxychains,
or L20 compatibility.

## Principle

PyTorch ships CUDA runtime libraries but FlashInfer also JIT-compiles kernels. Its host CUDA toolkit
must understand the emitted compiler flags and should match PyTorch's CUDA build. The original PATH
selected `/usr/bin/nvcc` 11.5, which does not support `--compress-mode=size`; PyTorch is built for
CUDA 12.8.

## Environment

- Date: 2026-10-08
- GPUs: 8 x NVIDIA L20, SM89, 45.47 GiB visible memory each
- Driver: 580.178.04
- PyTorch: 2.9.1+cu128
- FlashInfer: 0.6.18.post1
- Failing compiler: `/usr/bin/nvcc`, CUDA 11.5
- Selected compiler: `/usr/local/cuda-12.8/bin/nvcc`, CUDA 12.8

## Commands

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
proxychains python -m minisgl --model Qwen/Qwen3-0.6B
```

Validation used GPU 0, port 1930, `--memory-ratio 0.3`, `--cuda-graph-max-bs 1`, and
`--max-running-requests 1` to minimize resource usage.

## Results

- FlashInfer reported CUDA path `/usr/local/cuda-12.8` and version 12.8.
- Its generated decode kernel compiled successfully.
- CUDA Graph capture completed.
- Scheduler and API server reached ready state.
- The test server was then shut down cleanly.

## Interpretation

The failure was a compiler-selection problem. `/usr/local/cuda` currently points to CUDA 13.0, so
using that generic symlink would introduce a second mismatch; CUDA 12.8 must be selected explicitly.

## Decision

Keep CUDA 12.8 exports in every experiment command. No Mini-SGLang source change is needed for this
failure.

## Next step

Establish a deterministic single-L20 baseline before changing engine behavior.
