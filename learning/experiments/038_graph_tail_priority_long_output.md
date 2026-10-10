# Experiment 038: independent long-output validation of Graph-tail priority

## Status

Complete, rejected as a deployment optimization. The one-turn priority behavior is reproducible at
longer output length, but its benefit remains confined to P99.9; the pre-registered P90 criterion did
not improve. Keep the flag for research and profiling only.

## Method

This is an independent validation of experiment 037. Both runs used Qwen3-32B, TP=4 on four L20
GPUs, symmetric PyNCCL, CUDA Graph max batch 64, `--max-prefill-streak 1`, decode-active prefill
length 1024, input/output lengths 256/128, concurrency 1/8/32, seed `3800042`, two unrecorded
warmups and three measured repeats. Control used priority size `0`; treatment used priority size
`32`.

Acceptance required C=32 P90 TPOT improvement, less than 2% throughput loss, and no material C=1/8
regression. Telemetry also had to show a bounded one-turn policy.

## Results

| C | Control tok/s | Priority tok/s | Change | Control P90 TPOT ms | Priority P90 TPOT ms | Change | P99.9 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.93 | 39.91 | -0.06% | 24.71 | 24.78 | +0.29% | +0.79% |
| 8 | 247.61 | 247.61 | +0.00% | 27.36 | 27.34 | -0.06% | +0.03% |
| 32 | 575.29 | 573.48 | -0.32% | 34.74 | 34.77 | +0.07% | -9.68% |

C=32 P90 TTFT increased 1.13% (2794.77 -> 2826.39 ms). The P99.9 improvement is repeatable, but
it does not move the P90 metric that controls the normal serving tail.

## Telemetry

The control trace had 2,036 pipeline batches and 610 padded-32 decode batches. The treatment had
2,041 batches and the same 610 padded-32 batches, adding exactly five one-turn decode selections at
the five target boundaries. The first transition changed from `padded-32 decode -> prefill` to
`padded-32 decode -> one decode turn -> prefill`; no unbounded decode loop occurred.

## Decision

- Reject priority=32 as a general L20 optimization because the independent P90 criterion failed.
- Keep the option default-disabled for profiling and as an interview example of tail redistribution.
- Do not include it in the recommended Graph or direct-PyNCCL deployment command.
- Keep experiment 036's synchronous guard rejected as well; neither policy fixes the P90 tail.

The next optimization should move below the scheduler policy layer and inspect the Graph execution or
sampling path for the remaining P90 bottleneck. Any new candidate needs a paired control and an
explicit P90 gate.

## Artifacts

- `learning/results/038_graph_symmetric_control_seed3800042.json`
- `learning/results/038_graph_symmetric_priority32_seed3800042.json`
- `learning/results/038_graph_symmetric_control_telemetry.jsonl`
- `learning/results/038_graph_symmetric_priority32_telemetry.jsonl`
- `learning/experiments/038_graph_symmetric_control_seed3800042_raw.md`
- `learning/experiments/038_graph_symmetric_priority32_seed3800042_raw.md`

