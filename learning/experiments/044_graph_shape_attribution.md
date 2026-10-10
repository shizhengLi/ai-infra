# Experiment 044: Graph replay attribution by padded batch shape

## Question

Experiment 039 showed that Graph replay dominates sampling and token-copy time. Experiment 043
provided fresh CUDA-event telemetry under the same Qwen3-32B TP4 workload. This experiment groups
that telemetry by padded decode batch shape before changing the scheduler or Graph implementation.

## Method

This is an attribution follow-up over the fresh three-repeat control and treatment traces from
experiment 043. Reusing those traces keeps the model, seed, warmups, request distribution, and
runtime configuration identical while avoiding a second JIT/capture confounder. Only events with
`phase=decode`, `gpu_execution_kind=graph`, and a valid `padded_batch_size` are included.

- Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Symmetric PyNCCL, CUDA Graph maximum batch 64, automatic attention (`fi`)
- Input/output: 256/32 tokens; concurrency 1/8/32
- Seed 4300042; three measured repeats after two warmups
- 530 decode Graph events in each trace

Input telemetry:

- `learning/results/043_control_telemetry.jsonl`
- `learning/results/043_treatment_telemetry.jsonl`

## Shape results

| Padded shape | Control events | Control model mean ms | Control P90 ms | Treatment model mean ms | Treatment P90 ms | Logical batches represented |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 180 | 24.617 | 24.618 | 24.620 | 24.620 | 1 |
| 4 | 20 | 26.346 | 26.340 | 26.347 | 26.341 | 4 |
| 8 | 160 | 27.027 | 27.043 | 27.030 | 27.043 | 7, 8 |
| 16 | 20 | 28.625 | 28.722 | 28.629 | 28.723 | 12, 16 |
| 24 | 20 | 32.541 | 32.659 | 32.753 | 32.851 | 20, 24 |
| 32 | 130 | 34.082 | 34.105 | 34.268 | 34.292 | 28, 31, 32 |

The shape distribution was identical between groups. Every padded-24 and padded-32 event exceeded
30 ms in the control trace (20/20 and 130/130 respectively); no smaller shape did. Thus 150 of
530 decode Graph events, or 28.3%, account for all model intervals above 30 ms.

## Interpretation

The step from padded 16 to 24 adds 3.916 ms (+13.7%) of Graph model time, and padded 32 adds a
further 1.541 ms (+4.7%). This is a shape-dependent GPU execution cost, not sampler or host-copy
overhead. Logical batches 20/24 map to padded 24, while logical batches 28/31/32 map to padded 32.
Experiment 040 already showed that adding explicit shapes 20 and 28 did not improve C=32 P90 TPOT.

The BF16 reduction treatment shifted padded-32 model time from 34.082 to 34.268 ms and worsened
C=32 P90 TPOT by 6.73%, confirming that a generic GEMM precision switch is not a safe fix.

## Decision

Do not change Graph padding, sampler, token copy, or collective semantics based on this attribution
alone. The next candidate must operate inside the padded-24/32 Graph path or reduce scheduler work
around those transitions, and must pass the strict C=32 P90 gate.

## Next step

Experiment 045 should compare a narrowly scoped Graph replay implementation change for padded 24/32
against the unchanged symmetric control, with per-shape CUDA-event telemetry retained.
