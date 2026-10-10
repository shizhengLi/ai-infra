# Experiment 037: non-blocking padded-32 decode priority

## Status

Complete. The corrected one-turn priority policy is accepted as an explicit, workload-calibrated
profile, but remains disabled by default. It avoids the synchronization regression from experiment
036 and modestly improves the C=32 tail.

## Hypothesis

Experiment 036 proved that processing the previous padded-32 decode result before prefill destroys
useful overlap. This experiment instead changes only the next-batch choice: after a padded-32 Graph
decode with pending prefill, schedule one additional decode batch first, while leaving
`copy_done.synchronize()` and result processing in the normal overlap path.

The policy is one-shot. A consumed state prevents repeated decode preference while the same prefill
queue remains pending. The new option is `--decode-graph-tail-prefill-priority-batch-size`, default
`0`.

## Method

Both runs used Qwen3-32B, TP=4 on four L20 GPUs, symmetric PyNCCL, CUDA Graph max batch 64,
`--max-prefill-streak 1`, decode-active prefill length 1024, input/output lengths 256/32,
concurrency 1/8/32, seed `3700042`, two unrecorded warmups and three measured repeats. The only
treatment difference was priority size `0` versus `32`.

Acceptance rule: at C=32, P90 TPOT should improve, throughput loss must stay below 2%, and C=1/8
must not materially regress. The policy must not create an unbounded decode-only loop.

## Results

| C | Control tok/s | Priority tok/s | Change | Control P90 TPOT ms | Priority P90 TPOT ms | Change | P99.9 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.06 | 38.08 | +0.05% | 24.75 | 24.82 | +0.31% | +1.04% |
| 8 | 166.66 | 166.89 | +0.14% | 27.59 | 27.40 | -0.70% | +0.02% |
| 32 | 267.60 | 265.91 | -0.63% | 221.15 | 220.23 | -0.41% | -7.06% |

C=32 average TTFT changed from 1603.39 ms to 1603.69 ms (+0.02%), while P90 TTFT changed from
2793.55 ms to 2824.95 ms (+1.12%). The tail improvement is small and comes with a modest
throughput cost, not a broad latency improvement.

## Telemetry validation

The control trace contained 596 pipeline batches, including 130 padded-32 decode batches. The
corrected treatment contained 601 batches and the same 130 padded-32 batches: exactly five extra
one-turn decode selections corresponded to the five observed padded-32/prefill boundaries. A sample
transition changed from `padded-32 decode -> prefill` to `padded-32 decode -> one decode turn -> prefill`.

The original implementation before the one-shot latch created 655 decode batches and delayed C=32
TTFT by 30.19%; that diagnostic run is retained but is not used as the experiment result.

## Decision

- Keep the corrected one-turn policy behind the explicit priority-size option.
- Do not enable it by default: the C=32 gain is only 0.41% at P90 TPOT and throughput falls 0.63%.
- Do not combine it automatically with direct PyNCCL or experiment 036's synchronous guard.
- Preserve the one-shot latch and unit tests to prevent starvation regressions.

A future acceptance run should use a second independent seed and a longer output length before
considering a default policy.

## Artifacts

- `learning/results/037_graph_symmetric_control_seed3700042.json`
- `learning/results/037_graph_symmetric_priority32_corrected_seed3700042.json`
- `learning/results/037_graph_symmetric_control_telemetry.jsonl`
- `learning/results/037_graph_symmetric_priority32_corrected_telemetry.jsonl`
- `learning/experiments/037_graph_symmetric_control_seed3700042_raw.md`
- `learning/experiments/037_graph_symmetric_priority32_corrected_seed3700042_raw.md`
- Diagnostic superseded run: `learning/results/037_graph_symmetric_priority32_seed3700042.json`

