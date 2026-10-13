# Experiment 051: L20 Quantized GEMM Capability and Shape Sweep

## Hypothesis

The installed `sgl-kernel 0.3.21` FP8, INT8, and GPTQ-Marlin INT4 paths execute correctly on the
L20 (SM89). On cold-cache Qwen3-32B TP4 layer shapes, at least one path retains a material speedup
after including dynamic activation quantization and is suitable for end-to-end integration.

Capability acceptance requires every shape to execute, maximum kernel relative-L2 error below 1%,
and timing coefficient of variation below 5%. Quantization error is reported separately and is not
treated as model-quality evidence in this synthetic-weight screen.

## Principle

Decode GEMMs at small token batches are often weight-bandwidth bound. BF16 stores two bytes per
weight, FP8/INT8 store one, and INT4 stores half a byte. Lower precision can therefore reduce the
bytes streamed for each layer, but only if scale handling, activation quantization, and kernel launch
overhead do not erase the gain.

FP8 and INT8 use W8A8 scaled GEMMs. Weights are quantized offline per output channel; activations are
quantized at runtime per token. INT4 uses W4A16 GPTQ-Marlin: BF16 activations remain unquantized,
while packed weights and permuted channel scales are consumed directly by Marlin.

Repeated microbenchmarks can incorrectly leave a small layer's weights resident in L2. The harness
therefore writes a 128 MiB flush buffer before every timed GEMM. The flush is ordered before the CUDA
start event and excluded from the measured interval. This better represents a 64-layer model that
streams different weights, although only an end-to-end run can establish serving impact.

## Controlled variables

- GPU: physical GPU 6, NVIDIA L20, SM89; other users' GPU processes were left untouched.
- Software: PyTorch `2.9.1+cu128`, CUDA runtime 12.8, `sgl-kernel 0.3.21`.
- Model shapes: Qwen3-32B TP4 local QKV `(K=5120,N=2560)`, O projection
  `(2048,5120)`, merged gate/up `(5120,12800)`, and down projection `(6400,5120)`.
- Logical token batches: `M=1,8,32`.
- Dtype: BF16 baseline/output; FP8 E4M3 W8A8; symmetric INT8 W8A8; symmetric
  channelwise INT4 W4A16 with GPTQ packing and Marlin scale permutation.
- Seed: `5100042`; warmup 20; 100 calls per sample; five samples.
- L2 flush: 128 MiB before each timed call, excluded from timing.
- Offline weight quantization and Marlin repacking are excluded from latency.

## Commands

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python \
  benchmark/kernels/bench_quant_gemm_l20.py \
  --warmup 20 --repetitions 100 --samples 5 --cache-flush-mib 128 \
  --json-out learning/results/051_quant_gemm_seed5100042.json \
  --markdown-out learning/results/051_quant_gemm_seed5100042.md
```

The script was also checked with `python -m py_compile`. The environment does not contain the
optional `black` or `ruff` executables, so those checks were unavailable.

## Results

All 36 low-precision shape/path combinations executed on SM89. Maximum timing coefficient of
variation was 0.46%, below the 5% gate. Kernel relative-L2 error versus an explicitly dequantized
reference was 0.300%-0.501%, below the 1% gate.

The table reports speedup over the matching cold-cache BF16 GEMM. FP8 and INT8 include dynamic
activation quantization; INT4 is already W4A16 and needs no activation conversion.

| TP4 local layer | M | FP8 dynamic W8A8 | INT8 dynamic W8A8 | INT4 W4A16 |
| --- | ---: | ---: | ---: | ---: |
| QKV | 1 | 1.64x | 1.15x | 2.74x |
| QKV | 8 | 1.74x | 1.19x | 2.62x |
| QKV | 32 | 1.70x | 1.09x | 2.16x |
| O projection | 1 | 1.49x | 1.26x | 2.46x |
| O projection | 8 | 1.47x | 1.25x | 2.28x |
| O projection | 32 | 1.53x | 1.30x | 2.26x |
| Gate/up | 1 | 1.57x | 1.35x | 3.04x |
| Gate/up | 8 | 1.84x | 1.55x | 3.58x |
| Gate/up | 32 | 2.07x | 1.68x | 3.99x |
| Down | 1 | 1.74x | 1.39x | 3.36x |
| Down | 8 | 1.75x | 1.33x | 3.48x |
| Down | 32 | 1.80x | 1.35x | 3.19x |

Across the 12 shapes, dynamic FP8 speedup ranged from 1.47x to 2.07x with a 1.70x arithmetic mean.
INT8 ranged from 1.09x to 1.68x with a 1.32x mean. INT4 kernel speedup ranged from 2.16x to 3.99x
with a 2.93x mean. FP8/INT8 reduce linear weight storage by about 50%; INT4 reduces it by about 75%.

Synthetic output relative-L2 error versus BF16 was 3.72%-3.78% for FP8, 1.84%-2.02% for INT8, and
14.49%-15.78% for naive channelwise INT4 round-to-nearest. These values measure one random linear
operation, not task accuracy. The high naive INT4 error specifically shows why GPTQ/AWQ calibration
and an end-to-end quality gate are required.

## Interpretation

FP8 is the best first integration target. It accelerated every cold-cache shape after runtime
activation quantization, maps directly to L20 Tensor Cores, and avoids the checkpoint/calibration
complexity of INT4. INT8's GEMM-only performance was similar to FP8, but its installed activation
quantizer consumed more of the gain.

INT4 Marlin has the strongest bandwidth result and the smallest weight footprint. Its kernel and
packing path are correct on SM89, but the synthetic RTN error is not acceptable evidence for model
quality. It remains a later GPTQ/AWQ experiment, not the immediate engine implementation.

This benchmark does not include attention, norms, collectives, CUDA Graph capture, or a real
checkpoint. Weight-only compression also does not reduce BF16 TP all-reduce bytes. It may reduce
communication indirectly only if the memory saving permits TP4 to become TP2 or TP1.

## Decision

Accept the low-precision kernel capability gate and select FP8 W8A8 for experiment 052's integration
scaffolding. Do not change Mini-SGLang defaults. Keep INT8 as a measured fallback and defer INT4
end-to-end work until a calibrated GPTQ/AWQ checkpoint path and task-accuracy harness exist.

## Next step

Experiment 052 will introduce a default-disabled quantization configuration, FP8 weight/scale
metadata, TP-aware loading, and a `QuantLinear` dispatch boundary. It must preserve the existing BF16
state-dict behavior and add focused correctness tests before any full-model benchmark.

## Artifacts

- `benchmark/kernels/bench_quant_gemm_l20.py`
- `learning/results/051_quant_gemm_seed5100042.json`
- `learning/results/051_quant_gemm_seed5100042.md`
