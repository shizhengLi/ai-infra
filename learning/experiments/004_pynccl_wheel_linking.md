# Experiment 004: link PyNCCL against the wheel runtime

## Hypothesis

TP=4 startup fails because Mini-SGLang links with `-lnccl`, which requires an unversioned
`libnccl.so` normally supplied by an NCCL development package. The active Python environment has a
compatible NCCL runtime, but its wheel intentionally contains only `libnccl.so.2`.

## Principle

The static linker resolves `-lnccl` to `libnccl.so`; it does not automatically use
`libnccl.so.2`. Linking the versioned library by absolute path fixes build-time resolution. Because
the library declares SONAME `libnccl.so.2`, the generated extension also needs an rpath to the wheel
library directory so `dlopen` can resolve it at runtime.

The wheel reports NCCL 2.27.5 (`ncclGetVersion=22705`), matching Mini-SGLang's bundled NCCL 2.27
interface header.

## Environment

- Model command: Qwen3-32B, TP=4, four L20 GPUs
- PyTorch: 2.9.1+cu128
- NCCL wheel: `nvidia-nccl-cu12 2.27.5`
- Runtime library:
  `.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`
- System linker cache: no NCCL entry
- Original failure: `/usr/bin/ld: cannot find -lnccl`

## Implementation

`python/minisgl/kernel/pynccl.py` now searches, in order:

1. `NCCL_HOME` / `NCCL_ROOT` library directories;
2. the active `nvidia.nccl` Python namespace package;
3. `LD_LIBRARY_PATH`.

When it finds a runtime library, it links its absolute path and adds its parent as rpath. If none is
found, it preserves `-lnccl` for conventional system development installations.

## Commands

Compile/load validation:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python -c 'from minisgl.kernel.pynccl import _load_nccl_module; _load_nccl_module()'
```

Server retry after stopping the failed parent process:

```bash
python -m minisgl --model Qwen/Qwen3-32B --tp 4 --model-source modelscope
```

Temporary fallback that bypasses the custom PyNCCL extension:

```bash
python -m minisgl --model Qwen/Qwen3-32B --tp 4 \
  --model-source modelscope --disable-pynccl
```

## Results

All validation layers passed:

1. Two resolver unit tests found the active wheel library and verified absolute-link/rpath flags.
2. The PyNCCL extension compiled and loaded in one process; `create_nccl_uid()` returned 128 bytes.
3. Qwen3-0.6B TP=4 on GPUs 4-7 connected all four Gloo ranks, initialized PyNCCL, loaded weights,
   allocated KV caches, and reached API ready on port 1940.
4. The actual Qwen3-32B ModelScope snapshot passed TP=4 initialization on GPUs 4-7: all 17 weight
   shards loaded, each rank allocated a 14.23 GiB KV cache, the scheduler became idle, and the API
   reached ready state on port 1940.

The 32B validation used `--memory-ratio 0.7 --cuda-graph-max-bs 0 --max-running-requests 8` to
isolate model loading and communication from CUDA Graph capture. Both validation servers were shut
down cleanly after reaching ready state.

The original failed parent PID 2675009 did not respond to one `SIGINT` and remained alive without GPU
allocation during validation. It must exit before port 1919 can be reused.

## Decision

Accept. The default PyNCCL path now works with the NCCL runtime already installed alongside PyTorch;
installing a system-wide NCCL development package is no longer required for this environment.

## Next step

After stopping the stale failed parent, rerun the normal Qwen3-32B command on port 1919 and begin the
TP=1/2/4/8 communication scaling experiment.
