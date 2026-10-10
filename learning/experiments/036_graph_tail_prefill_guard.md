# Experiment 036: padded-32 decode/prefill boundary guard

## Status

Complete, rejected. The targeted guard changed the intended scheduler ordering, but it increased
C=32 P90 TPOT by 28.69%. It remains disabled by default and is not a deployment recommendation.

## Hypothesis

Experiment 035 found that Graph-mode tail activity concentrates in padded-32 decode batches. The
overlap loop can submit pending prefill before it processes the previous decode result. The tested
policy finishes the previous decode result first only when all of these conditions hold:

- pending prefill work exists;
- the previous batch is a CUDA Graph decode batch;
- its padded batch size equals the configured guard size (`32` in this experiment).

This is narrower than `--decode-result-before-prefill`, which applies to every decode batch. The new
`--decode-graph-tail-prefill-batch-size` option defaults to `0`, preserving the existing scheduler.

## Method

Both runs used Qwen3-32B, TP=4 on four L20 GPUs, symmetric PyNCCL, CUDA Graph max batch 64,
`--max-prefill-streak 1`, decode-active prefill length 1024, input/output lengths 256/32,
concurrency 1/8/32, seed `3600042`, two unrecorded warmups and three measured repeats. The only
treatment difference was guard size `0` versus `32`.

Acceptance rule: C=32 P90 TPOT must improve without more than 2% throughput loss. Lower-load results
must not show a material regression.

## Results

| C | Control tok/s | Guard tok/s | Change | Control P90 TPOT ms | Guard P90 TPOT ms | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.73 | 38.05 | +0.84% | 24.82 | 24.77 | -0.20% |
| 8 | 166.71 | 166.54 | -0.10% | 27.59 | 27.59 | +0.02% |
| 32 | 267.89 | 267.30 | -0.22% | 210.55 | 270.96 | +28.69% |

At C=32 the P99.9 TPOT improved from 385.43 ms to 348.65 ms (-9.54%), but the pre-registered P90
target failed. This is a tail-redistribution effect, not an overall latency win.

## Telemetry validation

The trace contained five padded-32 decode-to-prefill transitions in each treatment:

| Measurement across five transitions | Control | Guard 32 |
| --- | ---: | ---: |
| Decode `forward_return -> result_sent` | 320.15 ms | 206.14 ms |
| Next prefill `selected - decode result_sent` | -147.13 ms | +0.52 ms |
| Padded-32 decode `process_start -> result_sent` | 30.82 ms | 32.26 ms |

A negative selection offset means prefill was selected before the previous decode result was sent.
The treatment moved it after result delivery exactly as designed. However, synchronizing
`copy_done` before launching prefill removed useful CPU/GPU overlap and moved the visible stall into
more token intervals, raising P90 even though P99.9 fell.

## Decision

- Reject guard 32 for the L20 Graph deployment profile.
- Keep its generic option default-disabled so the experiment is reproducible.
- Do not combine it with direct PyNCCL; the symmetric control already fails the latency gate.
- Do not infer that earlier response visibility always improves token latency. The position of a
  synchronization boundary matters as much as its duration.

Experiment 037 should test a non-blocking alternative: when the previous batch is padded-32 decode
and prefill is pending, prefer one additional decode scheduling turn without synchronizing the
previous result. Measure both decode P90 TPOT and prefill TTFT because this trades response tail
against admission fairness.

## Artifacts

- `learning/results/036_graph_symmetric_control_seed3600042.json`
- `learning/results/036_graph_symmetric_guard32_seed3600042.json`
- `learning/results/036_graph_symmetric_control_telemetry.jsonl`
- `learning/results/036_graph_symmetric_guard32_telemetry.jsonl`
- `learning/experiments/036_graph_symmetric_control_seed3600042_raw.md`
- `learning/experiments/036_graph_symmetric_guard32_seed3600042_raw.md`

