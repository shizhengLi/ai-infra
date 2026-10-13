# Experiment 052: FP8 Quantization Configuration, Loading, and Dispatch

## Hypothesis

Mini-SGLang can introduce a default-disabled FP8 W8A8 linear path without changing the existing
BF16 behavior. The boundary is correct if configuration propagates to every dense transformer
projection, checkpoint weights and scales follow TP sharding semantics, both BF16 and pre-quantized
weights load, and all linear wrappers reach the same FP8 kernel dispatch.

## Principle

The execution format has three distinct pieces:

1. Weights use E4M3 FP8 with one FP32 dequantization scale per output channel. BF16 checkpoints are
   quantized after TP sharding; pre-quantized checkpoints provide both `weight` and `weight_scale`.
2. Activations remain BF16 between operators and are dynamically quantized per token immediately
   before each scaled GEMM. The GEMM emits BF16 so norms, attention, activation functions, KV cache,
   and TP collectives keep their existing contracts.
3. Column-parallel scale vectors are sharded with output channels. QKV/gate-up scales are merged in
   the same order as weights. Row-parallel weights are sharded along input K, so their per-output
   scale vectors are replicated on every rank before the local outputs are all-reduced.

This design isolates quantization inside the linear layer rather than teaching every model forward
method about FP8. It also keeps embedding and LM-head weights BF16, avoiding vocabulary projection
shape and accuracy concerns in the first integration.

## Controlled variables

- Quantization is disabled unless `--quantization fp8` is supplied.
- Supported activation/model dtype: BF16 or FP16; FP32 is rejected.
- Dense Llama, Mistral, Qwen2, and Qwen3 transformer projections use the shared path.
- MoE is rejected for now because expert kernels require a separate quantization design.
- Embeddings, LM head, norms, attention, KV cache, and collectives retain the model dtype.
- FP8 kernel validation uses physical L20 GPU 6 (SM89).
- Real-checkpoint smoke model: local Qwen3-0.6B, one deterministic four-token request.
- Qwen3-32B validation is structure-only on a meta device; no 32B weights are loaded in this
  experiment.

## Implementation

- Added `QuantizationConfig` and the FP8 weight quantization/validation/linear functions under
  `python/minisgl/quantization/`.
- Added `EngineConfig.quantization` and CLI `--quantization fp8`; the default remains `None`.
- Added the runtime quantization config to `ModelConfig`, allowing shared model building code to pass
  it to QKV, O, gate/up, down, and replicated dense linears.
- Extended the linear state loader to quantize floating-point checkpoint weights or validate an FP8
  `weight` plus `weight_scale` pair.
- Changed O and row-parallel wrappers to call the shared linear dispatch before their existing
  all-reduce. This prevents those layers from silently falling back to `F.linear`.
- Extended TP checkpoint loading so column-parallel scales shard on output channels, GQA KV scales
  follow replicated-head ownership, and row-parallel scales are replicated.
- Kept dummy-weight mode functional by generating floating weights and allowing the layer loader to
  quantize them.

## Commands

Focused and regression tests:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest -q \
  tests/core/test_fp8_quantization.py --no-cov

CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest -q \
  tests/core/test_fp8_quantization.py tests/core tests/misc tests/benchmark --no-cov
```

The real checkpoint smoke tests used `LLM` with `cuda_graph_max_bs=0`, `num_page_override=256`,
`max_seq_len_override=256`, and the local Qwen3-0.6B snapshot. The FP8 run additionally passed
`quantization='fp8'`. Both requested four tokens with temperature zero and EOS ignored.

## Results

- Focused quantization suite: 10/10 passed.
- Core/misc/benchmark regression selection: 78/78 passed.
- Qwen3-32B TP4 meta structure: 256 FP8 weights plus 256 scale vectors, with all four local layer
  shapes matching experiment 051.
- Real Qwen3-0.6B BF16 and FP8 loads both completed one four-token request and returned identical
  token IDs `[3, 4, 3, 4]`.
- Both smoke-test processes shut down cleanly; physical GPUs 6 and 7 returned to zero memory use.

The unrestricted test command reported 81 passes plus three failures and one collection error. The
three failures are existing CUDA JIT builds using an older `/usr/bin/nvcc` that rejects C++20; the
collection error is an existing kernel helper named like a pytest test. Neither failure path imports
or exercises the new quantization code.

## Interpretation

The configuration, state representation, TP metadata rules, and FP8 kernel dispatch now form a
working end-to-end skeleton. The identical four-token smoke output is useful for catching gross
loading or layout errors, but it is not enough to claim model accuracy. Likewise, experiment 051's
GEMM speedups cannot yet be presented as server speedups.

On-load conversion from BF16 is practical for TP4, where each rank first holds a BF16 shard and then
replaces it with FP8. It is not a valid route to Qwen3-32B TP1 because the full BF16 checkpoint does
not fit one L20 before conversion. TP1 will require an offline FP8 checkpoint containing FP8 weights
and scales, which this loader can now consume.

Dynamic activation tensors are allocated inside each FP8 linear call. CUDA Graph capture may absorb
those allocations into a graph pool, but eager allocation overhead and graph compatibility remain
unmeasured and belong to experiment 053.

## Decision

Accept the FP8 integration scaffold. Keep it experimental and default-disabled. No deployment or
performance recommendation changes at this stage.

## Next step

Experiment 053 will run paired BF16/FP8 end-to-end measurements. Start with Qwen3-0.6B to validate
logit/token agreement, startup memory, eager/CUDA-Graph compatibility, and C=1/8/32 latency. Then run
Qwen3-32B TP4 only after those gates pass. The acceptance decision must use task/output quality plus
TTFT, TPOT, throughput, and peak memory, not GEMM microbenchmarks alone.

## Artifacts

- `python/minisgl/quantization/`
- `python/minisgl/layers/linear.py`
- `python/minisgl/models/weight.py`
- `tests/core/test_fp8_quantization.py`
- `learning/results/052_fp8_scaffold.md`
