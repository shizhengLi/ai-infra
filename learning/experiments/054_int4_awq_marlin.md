# Experiment 054: INT4 AWQ-Marlin End-to-End Integration

## Hypothesis

A calibrated INT4 checkpoint can turn experiment 051's Marlin kernel result into an end-to-end
Qwen3-32B serving path that fits on one L20, captures CUDA Graphs, and supports TP1/TP2. The strict
quality gate is at most a one-percentage-point MMLU drop from the experiment 053 BF16 baseline.
Throughput, latency, total GPU memory, and tokens/s/GPU are reported separately so fewer-GPU
capacity is not confused with maximum aggregate throughput.

## Why AWQ instead of the planned GPTQ checkpoint

The roadmap originally named GPTQ-Marlin. The implemented runtime is still the shared Marlin
W4A16 kernel, but the checkpoint format is AWQ because Qwen publishes an official calibrated
`Qwen/Qwen3-32B-AWQ` artifact with 4-bit weights, group size 128, zero points, and GEMM layout. This
avoids making the main quality claim with a community GPTQ artifact. AWQ and GPTQ need different
checkpoint unpacking/reordering, then both reach `gptq_marlin_gemm` after Marlin repacking.

## Principle

AWQ stores eight unsigned 4-bit output-channel values in each INT32 word. Every group of 128 input
channels has an output-channel scale and packed zero point. Runtime loading therefore has four
steps:

1. shard the checkpoint's packed tensors using TP semantics;
2. merge Q/K/V and gate/up along packed output channels;
3. undo AWQ's nibble interleave and apply Marlin's weight, scale, and zero-point permutations;
4. execute W4A16 GEMM while embeddings, LM head, norms, attention, KV cache, and collectives retain
   the selected 16-bit model dtype.

Column-parallel projections shard `qweight`, `qzeros`, and `scales` along output channels. Row
parallel projections shard them along input groups before the existing all-reduce. Repacking occurs
once at load time, not on every forward pass. Each layer owns a persistent Marlin workspace, which
is also required for CUDA Graph capture.

## Acceptance gates

- Official calibrated Qwen3-32B-AWQ checkpoint loads without unsupported projection shapes.
- TP1 and TP2 both capture Graph shapes 1/2/4/8/16/24/32 and complete C=1/8/32 serving.
- Each throughput point uses three recorded repeats after one unrecorded warmup.
- MMLU parses all 570 fixed questions and drops no more than one point from BF16.
- Existing BF16 and FP8 behavior remains unchanged; focused and selected regressions pass.

## Controlled variables

- Hardware: NVIDIA L20, SM89. Stage A used physical GPU 6; main runs used GPUs 4 and 4-5.
- Main model: official Qwen3-32B-AWQ. Stage A: official Qwen3-4B-AWQ.
- Main controlled dtype: BF16 activations, scales, KV cache, collectives, and outputs. The official
  checkpoint-default FP16 path is retained as a separately labeled deployment probe.
- Attention: FlashInfer. TP2 uses Direct PyNCCL with `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`.
- Load: unique 256-token prompts, 64 generated tokens, C=1/8/32, seed `5302042`.
- Graph shapes: 1/2/4/8/16/24/32; fixed 4,096-token KV capacity.
- Quality: the exact 570 questions and seed `5300042` used by experiment 053.

## Implementation

- Added `--quantization awq` and validation of checkpoint metadata: 4 bits, GEMM format, supported
  group size, and zero points are mandatory. Dummy weights and MoE are rejected.
- Added AWQ unpack/pack helpers, zero-point conversion, Marlin scale permutation, load-time
  `awq_marlin_repack`, a persistent workspace, and `gptq_marlin_gemm` dispatch.
- Added AWQ state to the shared dense linear layer without changing the default BF16 state or FP8
  state.
- Extended checkpoint loading for packed column/row TP sharding, GQA KV-head replication, and
  fused QKV/gate-up metadata merging.
- Extended offline/online benchmark metadata and MMLU mode choices to record AWQ runs.
- Added focused tests for metadata rejection, INT4 packing, TP sharding/merge dimensions, and L20
  numerical agreement against explicit dequantization.

## Reproduction

Start the controlled TP1 server. Change `CUDA_VISIBLE_DEVICES` to `4,5`, add
`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`, and set `--tp-size 2` for TP2.

```bash
CUDA_VISIBLE_DEVICES=4 python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B-AWQ/snapshots/master \
  --dtype bfloat16 --quantization awq --tp-size 1 \
  --attention-backend fi --memory-ratio 0.9 \
  --cuda-graph-max-bs 32 --num-pages 4096 \
  --max-seq-len-override 2048 --max-prefill-length 4096 --port 1953
```

Run the serving matrix:

```bash
python benchmark/online/bench_l20.py \
  --base-url http://127.0.0.1:1953/v1 \
  --input-len 256 --output-len 64 --concurrency 1 8 32 \
  --repeats 3 --steady-warmup-repeats 1 --seed 5302042 \
  --gpu-indices 4 --server-tp 1 --server-quantization awq \
  --server-memory-ratio 0.9 --server-graph-max-bs 32 \
  --server-max-extend-tokens 4096 \
  --json-out learning/results/054_qwen32_awq_tp1_graph_online.json \
  --markdown-out learning/results/054_qwen32_awq_tp1_graph_online.md
```

Run the task-quality gate against TP1:

```bash
python benchmark/online/bench_mmlu_quantization.py \
  --base-url http://127.0.0.1:1953/v1 \
  --data-tar /data2/lszlsz/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe/data.tar \
  --questions-per-subject 10 --concurrency 32 --max-tokens 64 \
  --seed 5300042 --mode awq \
  --json-out learning/results/054_qwen32_awq_mmlu.json
```

## Debugging record

The first Qwen3-4B-AWQ Graph run failed with `expected scalar type BFloat16 but found Half`.
Checkpoint scales were FP16 while the model activations were BF16. The loader originally preserved
all AWQ tensor dtypes. The fix preserves only packed INT32 `qweight`/`qzeros` and converts floating
scales to the selected model dtype before Marlin preparation. The repeated 4B run captured all seven
Graph shapes and sustained 1,885.22 +/- 0.76 tok/s over three repeats.

The first 32B target pass inherited the checkpoint's FP16 dtype. It was valid as an official
deployment probe but invalid as a dtype-controlled comparison to experiment 053 BF16. Those files
were retained with `fp16_exploratory` names. The main TP1/TP2 matrices were repeated with explicit
BF16 and the same throughput seed as experiment 053.

## Main performance result

Both TP degrees loaded every dense projection and captured all planned Graph shapes. Results below
are the BF16-controlled runs; standard deviation is the population standard deviation over three
throughput repeats.

| C | AWQ TP1 tok/s | AWQ TP2 tok/s | TP2 speedup | TP2 efficiency | TP1 avg TPOT ms | TP2 avg TPOT ms | Peak MiB/card TP1/TP2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.69 +/- 0.02 | 60.64 +/- 0.04 | +74.82% | 87.41% | 26.47 | 14.95 | 21,037 / 11,283 |
| 8 | 169.66 +/- 0.13 | 278.69 +/- 0.11 | +64.26% | 82.13% | 31.81 | 17.60 | 21,295 / 11,463 |
| 32 | 197.74 +/- 0.07 | 316.09 +/- 0.12 | +59.85% | 79.92% | 30.31 | 19.09 | 21,755 / 11,617 |

TP2 cuts average TPOT by 43.53%/44.67%/37.03% relative to TP1, but its tokens/s/GPU is
12.59%/17.87%/20.08% lower. Adding the second L20 is useful for latency and aggregate throughput,
not for cost efficiency.

Experiment 053's FP8 TP4 reached 61.58/341.03/402.24 tok/s. AWQ TP2 is 1.52%/18.28%/21.42%
slower in aggregate, but 96.96%/63.44%/57.16% faster per GPU. AWQ TP1 is about 97-125% faster per
GPU than FP8 TP4. Total sampled GPU memory is 62-66% lower for AWQ TP1/TP2 than FP8 TP4 because the
32B model no longer needs four cards. This is a capacity and cost result, not a claim that Marlin
has higher absolute server throughput.

## Quality result

All variants parsed all 570 answers. The FP16 AWQ row is the checkpoint-default deployment probe;
the BF16 AWQ row matches experiment 053's activation/KV/collective dtype.

| Mode | Correct | Accuracy | Delta from BF16 | One-point gate |
| --- | ---: | ---: | ---: | --- |
| BF16 TP4 baseline | 458/570 | 80.35% | - | Pass |
| FP8 TP4 | 454/570 | 79.65% | -0.70 point | Pass |
| AWQ TP1, checkpoint-default FP16 | 451/570 | 79.12% | -1.23 points | Fail |
| AWQ TP1, controlled BF16 | 448/570 | 78.60% | -1.75 points | Fail |

BF16 and controlled AWQ agreed on 528/570 predictions (92.63%); BF16 and default-FP16 AWQ agreed
on 531/570 (93.16%). The FP16/BF16 AWQ difference shows that runtime dtype is a material quality
variable for this checkpoint. Neither AWQ configuration passes the pre-registered one-point gate.

## Decision

Accept the AWQ-Marlin implementation and the TP1/TP2 feasibility result, but reject AWQ as the
current quality-gated deployment default. Its strongest use case is fitting Qwen3-32B on one L20 or
nearly matching FP8 TP4 single-request throughput on two L20s with much lower total GPU memory. FP8
remains the leading quality-qualified performance candidate.

Experiment 055 should retain AWQ as a conditional low-memory profile, use fresh mixed/long prompts,
and expand the quality sample before deciding whether the observed 1.23-point default-FP16 loss is
sampling noise or a stable deployment tradeoff. No generic quantization default changes.

## Verification

- `4/4` focused AWQ tests passed during development.
- `84/84` selected core, misc, and benchmark regressions passed on L20 after the final changes.
- Qwen3-4B-AWQ Graph Stage A and Qwen3-32B-AWQ TP1/TP2 completed end to end.
- `git diff --check` passed; all eight GPUs returned to 0 MiB and port 1953 had no listener.

## Artifacts

- `python/minisgl/quantization/awq.py`
- `python/minisgl/layers/linear.py`
- `python/minisgl/models/weight.py`
- `tests/core/test_awq_quantization.py`
- `learning/results/054_stage_a_qwen4b_awq_graph.{json,md}`
- `learning/results/054_qwen32_awq_tp{1,2}_graph_online.{json,md}`
- `learning/results/054_qwen32_awq_mmlu.json`
- `learning/results/054_qwen32_awq_*fp16_exploratory*`
