# Experiment 043: BF16 reduction mode in CUDA Graph replay

## Question

Experiment 041 attributed padded-32 Graph replay to NCCL plus three BF16 GEMM classes. Experiment
042 showed that changing PyNCCL can improve throughput while worsening the Graph C=32 tail. This
experiment isolates PyTorch's BF16 reduced-precision reduction setting.

## Hypothesis

Disabling `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` might select a
different GEMM path, but is not expected to improve throughput. It is useful as a controlled GEMM
experiment because it changes one setting before model construction and CUDA Graph capture.

## Implementation

The server has an explicit opt-in flag:

```text
--disable-bf16-reduced-precision-reduction
```

The default remains `True`, matching existing PyTorch behavior. The setting is applied only for
BF16 models; FP16 and FP32 are unaffected. Raw reports are in
`learning/results/043_control_seed4300042.json` and
`learning/results/043_treatment_seed4300042.json`.

## Measurement

- Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Automatic attention (`fi`), CUDA Graph maximum batch 64
- Input/output 256/32 tokens; concurrency 1/8/32
- Two warmups and three steady-state repeats per concurrency
- Seed 4300042 with unique prompts per repeat
- Memory ratio 0.7, max prefill 2048, decode-active prefill 1024, max prefill streak 1
- Control: default reduction setting
- Treatment: `--disable-bf16-reduced-precision-reduction`

## Results

| Concurrency | Control tok/s | Treatment tok/s | Throughput change | Control P90 TPOT ms | Treatment P90 TPOT ms | P90 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.965 | 37.826 | -0.37% | 24.736 | 24.779 | +0.17% |
| 8 | 166.418 | 165.691 | -0.44% | 27.370 | 27.422 | +0.19% |
| 32 | 267.107 | 266.539 | -0.21% | 221.709 | 236.642 | +6.73% |

Treatment P99.9 TPOT at C=32 also increased from 373.110 ms to 386.516 ms (+3.59%). Throughput
standard deviations were small: 0.777 tok/s control and 0.275 tok/s treatment at C=32. All 123
benchmark requests in each group completed with 32 streamed tokens; this is a completion check,
not a token-by-token accuracy evaluation.

## Decision

Reject disabling reduced-precision BF16 reductions as a Graph-mode optimization. It failed the
strict C=32 P90 TPOT gate and slightly reduced throughput at every tested concurrency. Keep the
switch default-disabled as a reproducible diagnostic control only; do not include it in the L20
deployment profile.

## Next step

Experiment 044 should measure Graph replay timing by padded batch shape with the existing
CUDA-event telemetry and correlate it with the Nsight kernel classes, then target a scheduler or
kernel change only if it reduces the padded-32 tail without changing collective semantics.
