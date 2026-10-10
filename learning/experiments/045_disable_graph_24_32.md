# Experiment 045: Eager fallback for padded-24/32

## Question

Experiment 044 identified padded-24 and padded-32 Graph replays as the only model intervals above
30 ms. This experiment tests the direct hypothesis that CUDA Graph replay itself is slower than an
eager forward for those shapes.

## Implementation

Added a default-disabled diagnostic option:

```text
--cuda-graph-disable-bs 24 32
```

When a decode batch would map to padded 24 or 32, `GraphRunner.pad_batch` leaves it unpadded and
`can_use_cuda_graph` routes it through eager execution. Other shapes continue to use their existing
CUDA Graphs. The setting does not change PyNCCL, attention backend, scheduler priorities, or the
default Graph shape list.

Changed files:

- `python/minisgl/engine/config.py`
- `python/minisgl/server/args.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/engine/graph.py`

## Measurement

The control is experiment 043's fresh default-reduction trace. The treatment used the same model,
seed, workload, warmups, repeats, and server settings, with `--cuda-graph-disable-bs 24 32`.

- Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Symmetric PyNCCL, automatic attention (`fi`), Graph max batch 64
- Input/output 256/32 tokens; concurrency 1/8/32
- Seed 4300042; two warmups and three measured repeats
- Memory ratio 0.7; max prefill 2048; decode-active prefill 1024; max prefill streak 1

Treatment telemetry and benchmark output:

- `learning/results/045_eager2432_telemetry.jsonl`
- `learning/results/045_eager2432_seed4300042.json`

## Online results

| Concurrency | Control tok/s | Treatment tok/s | Throughput change | Control P90 TPOT ms | Treatment P90 TPOT ms | P90 change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.965 | 37.782 | -0.48% | 24.736 | 24.757 | +0.08% |
| 8 | 166.418 | 166.638 | +0.13% | 27.370 | 27.340 | -0.11% |
| 32 | 267.107 | 266.809 | -0.11% | 221.709 | 229.953 | +3.72% |

At C=32, P99.9 TPOT changed from 373.110 ms to 374.638 ms (+0.41%). All 123 requests in the
treatment completed with the requested 32 streamed tokens.

## Shape telemetry

The treatment produced 380 Graph decode events for shapes 1/4/8/16 and 150 eager events for the
logical batches that previously mapped to padded 24/32. Mean model times were:

| Execution | Logical batch | Mean model ms |
| --- | ---: | ---: |
| eager | 20 | 32.229 |
| eager | 24 | 33.196 |
| eager | 28 | 33.700 |
| eager | 31 | 34.772 |
| eager | 32 | 34.681 |

These small per-shape differences did not translate into a lower online tail. The C=32 P90 gate
failed, and C=1 throughput also regressed by 0.48%.

## Decision

Reject eager fallback for padded-24/32 as a deployment optimization. Keep the option default-disabled
for future diagnostics. The hypothesis that Graph replay selection is the primary cause of the
C=32 tail is not supported; the remaining target is the work inside the Graph/kernel or the
scheduler transition around those batches.

## Next step

Experiment 046 should use an Nsight trace with separate markers for the padded-24/32 replay and
their surrounding scheduler interval, then target a concrete kernel or synchronization mechanism.
