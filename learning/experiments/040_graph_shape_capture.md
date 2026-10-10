# Experiment 040: explicit Graph capture shapes for the padded-32 tail

## Status

Complete and rejected as a deployment optimization. The explicit shape list is useful as a profiling
control, but adding 20 and 28 did not improve the C=32 P90 gate.

## Hypothesis

The default Graph capture list is `1, 2, 4, 8, 16, 24, 32, ...`. Logical decode batches around 20
and 28 therefore round up to padded 24 and 32. Capturing 20 and 28 explicitly should reduce padding,
lower model work for those batches, and reduce the number of padded-32 replays.

## Implementation

Added an opt-in `--cuda-graph-bs` argument to the server. When present, it supplies the existing
`EngineConfig.cuda_graph_bs` list and overrides generated shapes from `--cuda-graph-max-bs`. The
online benchmark now records `--server-graph-bs` in its JSON and Markdown metadata. Defaults are
unchanged.

Treatment shape list:

```text
1 2 4 8 16 20 24 28 32 40 48 56 64
```

Control used the existing max-batch policy with max batch 64:

```text
1 2 4 8 16 24 32 40 48 56 64
```

Both used Qwen3-32B, TP=4, four L20 GPUs, symmetric PyNCCL, input/output 256/32, concurrency 1/8/32,
three measured repeats after two warmups, seed `4000042`, max prefill streak 1, and decode-active
prefill budget 1024.

## Results

| C | Control tok/s | Treatment tok/s | Throughput change | Control P90 TPOT ms | Treatment P90 TPOT ms | P90 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.048 | 37.917 | -0.34% | 24.837 | 24.725 | -0.45% |
| 8 | 166.740 | 166.350 | -0.23% | 28.062 | 27.581 | -1.71% |
| 32 | 267.827 | 267.976 | +0.06% | 220.974 | 220.968 | -0.003% |

The C=32 P90 improvement is statistically negligible and does not meet the optimization gate. The
small C=8 P90 difference is not sufficient to offset the absence of a C=32 tail improvement.

## Telemetry validation

The shape change was real: control had 130 padded-32 decode batches, while treatment had 120 padded-32
batches plus 10 padded-20 and 10 padded-28 batches. However, padded-32 Graph model time remained
34.073 ms in control versus 34.106 ms in treatment. Completion intervals over 30 ms changed from
125/130 padded-32 samples to 120/120, so the tail was redistributed only minimally.

This demonstrates that reducing padding at the 20/28 boundaries is not enough: the remaining padded-32
Graph replay itself is the dominant cost identified by experiment 039.

## Decision and next step

- Reject the explicit 20/28 shape list as a default L20 profile.
- Keep `--cuda-graph-bs` as an opt-in profiling and future tuning interface; it has no default behavior
  change.
- Do not spend another experiment on sampler, copy, or small padding-shape variants without a new
  measured hypothesis.
- Experiment 041 should use NVTX/kernel-level attribution for padded-32 Graph replay and identify the
  specific attention, collective, or projection kernels responsible for the roughly 34 ms execution.

## Artifacts

- `learning/results/040_control_seed4000042.json`
- `learning/results/040_treatment_seed4000042.json`
- `learning/results/040_control_telemetry.jsonl`
- `learning/results/040_treatment_telemetry.jsonl`
- `learning/experiments/040_control_seed4000042_raw.md`
- `learning/experiments/040_treatment_seed4000042_raw.md`
- `python/minisgl/server/args.py`
- `benchmark/online/bench_l20.py`

