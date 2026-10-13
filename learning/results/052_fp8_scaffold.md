# Experiment 052 raw validation summary

## Focused tests

```text
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest -q \
  tests/core/test_fp8_quantization.py --no-cov

10 passed in 4.22s
```

The focused suite covers default-disabled CLI/config behavior, per-output-channel FP8 weight
quantization, pre-quantized state reload, missing-scale rejection, TP scale sharding, the unchanged
BF16 path, L20 kernel correctness, and FP8 dispatch through O/row-parallel wrappers.

## Regression tests

```text
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest -q \
  tests/core/test_fp8_quantization.py tests/core tests/misc tests/benchmark --no-cov

78 passed in 5.24s
```

The unrestricted repository run reached 81 passes, but retained three unrelated JIT failures and
one collection error. The JIT failures use `/usr/bin/nvcc`, which rejects `-std=c++20`; the collection
error treats `python/minisgl/kernel/tensor.py::test_tensor` as a pytest test requiring nonexistent
fixtures. These failures are outside the quantization files and are not counted as experiment 052
regressions.

## Qwen3-32B TP4 structure check

A meta-device construction using the local Qwen3-32B config produced:

```text
fp8_weights=256 scales=256 total_tensors=771
qkv torch.Size([2560, 5120])
gate_up torch.Size([12800, 5120])
```

The 256 pairs correspond to 64 layers times QKV, O, gate/up, and down projections. Embedding and LM
head tensors remained BF16. First-layer O and down weights were also checked as `[5120, 2048]` and
`[5120, 6400]`.

## Real-checkpoint smoke test

The local Qwen3-0.6B safetensors checkpoint was loaded twice on physical L20 GPU 6 with CUDA Graphs
disabled, a 256-token KV pool, and a four-token deterministic request.

| Mode | Generated token IDs | Dense linear dtype | Result |
| --- | --- | --- | --- |
| BF16 default | `[3, 4, 3, 4]` | `torch.bfloat16` | Pass |
| `quantization='fp8'` | `[3, 4, 3, 4]` | `torch.float8_e4m3fn` | Pass |

The FP8 gate/up scale had shape `[6144]`. This is a smoke-level token agreement result, not a model
accuracy or performance claim.
