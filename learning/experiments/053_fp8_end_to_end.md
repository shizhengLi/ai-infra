# Experiment 053: FP8 W8A8 End-to-End Validation

## Hypothesis

The default-disabled FP8 path from experiment 052 can preserve task quality while converting the
L20 kernel gains from experiment 051 into lower memory use and better serving latency/throughput on
Qwen3-32B TP4. The experiment is accepted only if CUDA Graph capture works, MMLU accuracy drops by
no more than one percentage point, and the three-repeat C=1/8/32 matrix improves throughput without
regressing average or P90 TPOT.

## Principle

FP8 reduces the bytes read by dense transformer GEMMs, but dynamic per-token activation
quantization adds allocations and kernel launches. Eager and CUDA Graph execution must therefore be
measured separately. A layer microbenchmark is insufficient because attention, collectives,
scheduling, and output quality remain outside the GEMM.

Quality uses deterministic greedy multiple-choice generation over 570 MMLU questions: ten seeded
test questions from each of 57 subjects. Qwen3 thinking is disabled in the rendered chat template,
unparsed answers count as wrong, and BF16/FP8 use identical questions. The final run allows 64
output tokens so a formatting violation is not confused with an accuracy error.

## Controlled variables

- Hardware: NVIDIA L20, SM89. Stage A used physical GPU 6; target runs used GPUs 4-7.
- Models: Qwen3-0.6B for compatibility screening and Qwen3-32B TP4 for the main result.
- Dtype: BF16 activations/KV/collectives/output; either BF16 or FP8 E4M3 dense linear weights.
- FP8 activations: dynamic per-token E4M3 quantization immediately before each dense GEMM.
- Attention: FlashInfer; Direct PyNCCL with `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` for TP4.
- Serving load: unique 256-token prompts, 64 output tokens, C=1/8/32, three repeats, one unrecorded
  warmup per concurrency, seed `5302042`.
- Graph shapes: 1/2/4/8/16/24/32. KV capacity was fixed at 4,096 pages for memory comparability.
- Quality seed: `5300042`; 570 questions; online concurrency 32.

## Implementation

- Extended `benchmark/offline/bench_l20.py` with opt-in quantization, fixed KV pages, CUDA memory
  metrics, and output token IDs. Its existing defaults are unchanged.
- Added an offline MMLU gate for single-GPU compatibility testing and an online MMLU gate for TP
  servers. Both read the official cached `cais/mmlu` `data.tar` directly and persist per-question
  predictions.
- Added focused tests for answer parsing and deterministic balanced sampling.

## Reproduction

Start the BF16 target server as follows. Append `--quantization fp8` for the paired FP8 run; all
other arguments stay unchanged.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MINISGL_PYNCCL_MAX_BUFFER_SIZE=0 \
python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp-size 4 --attention-backend fi --memory-ratio 0.9 \
  --cuda-graph-max-bs 32 --num-pages 4096 --max-seq-len-override 2048 \
  --max-prefill-length 4096 --port 1953
```

Run the serving matrix against that server:

```bash
python benchmark/online/bench_l20.py \
  --base-url http://127.0.0.1:1953/v1 \
  --input-len 256 --output-len 64 --concurrency 1 8 32 \
  --repeats 3 --steady-warmup-repeats 1 --seed 5302042 \
  --gpu-indices 4,5,6,7 --server-tp 4 --server-memory-ratio 0.9 \
  --server-graph-max-bs 32 --server-max-extend-tokens 4096 \
  --json-out learning/results/053_qwen32_bf16_graph_online.json \
  --markdown-out learning/results/053_qwen32_bf16_graph_online.md
```

For FP8, change the two output filenames above and run the final task-quality gate with the matching
`--mode` value:

```bash
python benchmark/online/bench_mmlu_quantization.py \
  --base-url http://127.0.0.1:1953/v1 \
  --data-tar /data2/lszlsz/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe/data.tar \
  --questions-per-subject 10 --concurrency 32 --max-tokens 64 \
  --seed 5300042 --mode bf16 \
  --json-out learning/results/053_qwen32_bf16_mmlu_final.json
```

## Stage A: Qwen3-0.6B

FP8 captured and replayed all seven CUDA Graph shapes. With a fixed 4,096-page KV pool, allocated
CUDA memory after Graph initialization fell from 1,768.01 to 1,365.32 MiB.

The random-token fidelity probe showed why exact autoregressive agreement is not an accuracy metric:
26/32 first tokens and 8/32 complete 64-token sequences matched, while position-level agreement was
46.48%. On natural-language MMLU, both modes parsed all 570 answers; BF16 scored 226/570 (39.65%)
and FP8 scored 232/570 (40.70%). This is a non-regression result, not evidence that FP8 improves the
model.

The small-model online matrix exposed the Graph dependency:

| Mode | C=1 tok/s | C=8 tok/s | C=32 tok/s |
| --- | ---: | ---: | ---: |
| BF16 eager | 98.28 | 706.91 | 971.25 |
| FP8 eager | 76.90 | 572.32 | 781.85 |
| BF16 Graph | 388.70 | 2,179.74 | 2,886.63 |
| FP8 Graph | 427.01 | 2,445.01 | 3,495.91 |

FP8 eager regressed throughput by 21.75%/19.04%/19.50%, while FP8 Graph improved it by
9.86%/12.17%/21.11%. Graph capture amortizes the dynamic quantization allocation and launch
overhead; FP8 is not a universal eager-mode win.

## Main result: Qwen3-32B TP4

The table compares BF16 and FP8 under the same Graph and Direct PyNCCL configuration. Standard
deviation is the population standard deviation over three throughput repeats.

| C | BF16 tok/s | FP8 tok/s | Change | BF16 avg/P90 TPOT ms | FP8 avg/P90 TPOT ms | Avg TTFT change | Peak MiB/card |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.73 +/- 0.03 | 61.58 +/- 0.01 | +55.00% | 24.20 / 24.78 | 15.23 / 15.56 | -7.86% | 18,229 -> 15,407 |
| 8 | 228.14 +/- 0.11 | 341.03 +/- 1.34 | +49.48% | 27.87 / 27.78 | 17.94 / 17.47 | -23.59% | 18,229 -> 15,407 |
| 32 | 273.00 +/- 0.17 | 402.24 +/- 0.59 | +47.34% | 29.48 / 29.60 | 19.25 / 19.40 | -29.31% | 18,229 -> 15,407 |

FP8 reduced sampled peak memory by 2,822 MiB/card (15.48%). C=1/8/32 average TPOT improved by
37.05%/35.65%/34.69%, and P90 TPOT improved by 37.23%/37.13%/34.47%. C=32 P90 TTFT improved by
30.67%. P99.9 TPOT also improved from 186.19 to 140.77 ms at C=32, although this closed-loop test
does not replace an open-loop tail study.

Final MMLU quality with a 64-token answer ceiling was:

| Mode | Parsed | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| BF16 | 570/570 | 458/570 | 80.35% |
| FP8 | 570/570 | 454/570 | 79.65% |

The FP8 change was -0.70 percentage points, passing the one-point gate. Predictions agreed on
96.49% of questions; 450 were correct in both modes, eight only in BF16, four only in FP8, and 108
wrong in both.

## Invalid evaluator attempts

The first online run used the API's default thinking template and a four-token ceiling. Every output
was truncated near `<think>`, so the result was invalid. Rendering a non-thinking template fixed
that. Four and 16 output tokens still left one formatting violation unparsed; pre-filling
`The answer is ` made the model emit answer text rather than option letters and reduced parse rate
to 90.88%. That variant was rejected. The final symmetric 64-token run reached 100% parse rate.

Cold versus warm startup time is also invalid as a quantization comparison. The first BF16 TP4 load
took 8m27s while later FP8/BF16 loads took 17-19s after Linux page-cache warmup. Only steady-state
memory and serving measurements are used.

## Decision

Accept FP8 W8A8 as an end-to-end candidate on L20 for the measured Qwen3-32B TP4 Graph workload.
Keep it default-disabled. The result passes task quality, memory, throughput, average latency, P90
latency, Graph compatibility, and repeatability gates, but eager execution regresses on the small
model and the target run does not cover long prompts or open-loop arrivals.

Experiment 054 remains the calibrated INT4 integration phase. Experiment 055 must use a fresh seed,
longer/mixed prompts, and open-loop serving before recommending FP8 as the deployment default.

## Verification

- `80/80` selected core, misc, quantization, and benchmark tests passed on L20.
- The three modified/new benchmark entry points passed Python bytecode compilation; both MMLU CLI
  help paths loaded successfully.
- `git diff --check` passed. All eight GPUs returned to 0 MiB used, and experiment ports 1953/1954
  had no listeners after teardown.

## Artifacts

- `benchmark/offline/bench_l20.py`
- `benchmark/offline/bench_mmlu_quantization.py`
- `benchmark/online/bench_mmlu_quantization.py`
- `tests/benchmark/test_mmlu_quantization.py`
- `learning/results/053_stage_a_*`
- `learning/results/053_qwen32_bf16_graph_online.{json,md}`
- `learning/results/053_qwen32_fp8_graph_online.{json,md}`
- `learning/results/053_qwen32_{bf16,fp8}_mmlu_final.json`
