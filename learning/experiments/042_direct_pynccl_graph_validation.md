# Experiment 042: direct PyNCCL under the padded-32 Graph workload

## Status

Complete. Direct PyNCCL is rejected for the default CUDA Graph profile because it fails the C=32 P90
TPOT gate, despite a large throughput gain and a better extreme tail. Keep it as an explicit
graph-disabled profile only.

## Question

Experiment 041 attributed 52.0% of padded-32 Graph kernel time to NCCL all-reduce. This experiment
retests whether the direct PyNCCL buffer path (`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`) can reduce that
cost without the Graph-mode latency regression observed in experiment 033.

## Method

Qwen3-32B BF16, TP=4, four L20 GPUs, CUDA Graph max batch 64, symmetric scheduler settings,
input/output lengths 256/32, concurrency 1/8/32, three measured repeats after two steady-state
warmups, seed `4200042`. The only treatment variable was the PyNCCL buffer mode:

- Control: default symmetric PyNCCL buffer.
- Treatment: `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` direct path.

Both runs recorded the same per-batch Graph/model timing telemetry.

## Results

| C | Control tok/s | Direct tok/s | Throughput change | Control P90 TPOT ms | Direct P90 TPOT ms | P90 change | P99.9 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.006 | 38.449 | +1.17% | 24.717 | 24.684 | -0.13% | +1.71% |
| 8 | 166.866 | 180.222 | +8.00% | 27.710 | 27.619 | -0.33% | -20.29% |
| 32 | 267.696 | 303.536 | +13.39% | 210.150 | 241.596 | +14.96% | -21.62% |

The direct path substantially raises throughput and improves P99.9, but the deployment metric is C=32
P90 TPOT and it regresses by 14.96%. This fails the pre-registered acceptance gate.

## Telemetry attribution

The Graph shape distribution remained identical: 130 padded-32 decode batches in each run. At padded
32, direct PyNCCL increased the measured model/Graph interval from 34.073 ms to 35.582 ms; sampling
remained 0.020 -> 0.021 ms and copy remained 0.007 ms. Scheduler completion intervals above 30 ms
remained 125/130 in both runs, while their mean increased from 30.892 to 32.277 ms.

This explains the result: direct communication removes enough overhead to improve throughput and the
extreme tail, but its Graph replay synchronization behavior makes the normal P90 tail worse.

## Decision and next step

- Reject direct PyNCCL as a default CUDA Graph setting.
- Preserve `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0` as an explicit graph-disabled TP4/TP8 profile, where
  earlier experiments showed consistent throughput gains.
- Do not combine direct PyNCCL with the Graph-tail priority or synchronous guard experiments.
- The next experiment should target the dominant BF16 GEMM path inside Graph replay, or compare Graph
  disabled/direct communication only if the serving objective explicitly prioritizes throughput over
  P90 latency.

## Artifacts

- `learning/results/042_symmetric_seed4200042.json`
- `learning/results/042_direct_seed4200042.json`
- `learning/results/042_symmetric_telemetry.jsonl`
- `learning/results/042_direct_telemetry.jsonl`
- `learning/experiments/042_symmetric_seed4200042_raw.md`
- `learning/experiments/042_direct_seed4200042_raw.md`

