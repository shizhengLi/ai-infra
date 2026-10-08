# Experiment 016: main L20 scheduling experiment

## Status

Complete. The final profile substantially improves tail latency, but fails two pre-registered
acceptance criteria and is not accepted as a strict one-second-TPOT deployment profile.

## Objective

Measure the complete scheduling optimization on held-out traces after experiments 008-015 selected
the mechanism and parameters. This is the main experiment, not another tuning sweep. It compares
the original scheduling behavior, fairness alone, and the final latency-bounded policy under one
fixed representative workload matrix.

## Hypothesis

Limiting consecutive prefill batches prevents decode starvation, while the model-derived
2304-token decode-active prefill budget bounds the duration of each remaining prefill interruption.
Together they should satisfy the one-second maximum-TPOT SLO across light through overloaded
arrival rates without reducing output throughput by more than 1% relative to the scheduling
control.

Fairness alone should reduce the most severe starvation events but cannot bound the execution time
of one 8192-token prefill turn. The complete policy should therefore provide the decisive tail-TPOT
improvement.

## Principle

The two scheduler controls address different failure modes:

1. `max_prefill_streak=1` bounds how many prefill batches can run while decode work is runnable.
2. `decode_active_prefill_tokens=2304` bounds how much prefill work one such batch can admit.

The three profiles form an incremental comparison, so the main result can distinguish the value of
fairness from the value of bounding one prefill stall:

| Profile | Maximum prefill streak | Decode-active budget |
| --- | ---: | ---: |
| Scheduling control | 0 (unlimited) | 0 (reuse 8192) |
| Fairness only | 1 | 0 (reuse 8192) |
| Final optimized | 1 | 2304 |

All other model, engine, and workload settings are pinned. The experiment uses seed 1600042 rather
than the 800042 tuning seed from experiments 012-015. These are held-out traces: no policy parameter
will be changed in response to their results.

## Experiment matrix

- Model: Qwen3-32B BF16
- Hardware: 4 x NVIDIA L20, GPUs 0-3
- Tensor parallelism: 4, PyNCCL enabled
- CUDA/compiler: 12.8, SM89
- Attention backend: FlashInfer selected by `auto`
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Normal/decode-idle prefill budget: 8192 tokens
- Requests per trace: 24
- Input lengths: balanced 256/1024/4096 tokens, shuffled per trace
- Output length: 256 tokens
- Arrival rates: 0.5, 0.9, 1.5, and 2.5 requests/s
- Repeats: 3 per profile and arrival rate
- Held-out seed: 1600042
- TTFT SLO: 2000 ms
- Per-request maximum-TPOT SLO: 1000 ms
- Total measured traces: 3 profiles x 4 rates x 3 repeats = 36
- Total measured requests: 864

The 0.5 rate covers light load, 0.9 and 1.5 cover the previously observed latency knee, and 2.5 is
an intentional overload point. Each profile reuses exactly the same trace seeds, input-length
orders, generated prompts, and scheduled arrival offsets.

## Pre-registered acceptance criteria

The final optimized profile is accepted for the declared one-second TPOT objective only if:

1. TPOT SLO attainment is 100% at every arrival-rate point.
2. No measured optimized token gap exceeds one second.
3. Mean output throughput at each rate is no more than 1% below the scheduling control.
4. All three repetitions complete without correctness or server failures.

TTFT, P99/P99.9 TPOT, maximum TPOT, GPU utilization, power, and per-input-length TTFT are reported as
secondary metrics. Throughput claims include the standard deviation across the three repetitions;
paired trace results are used to interpret changes smaller than between-trace dispersion.

## Planned outputs

- `learning/experiments/016_default_raw.md`
- `learning/experiments/016_fairness_raw.md`
- `learning/experiments/016_optimized_raw.md`
- Corresponding machine-readable JSON under `learning/results/`

## Results

All 36 measured traces and 864 requests completed. Aggregate results use three repeats per profile
and arrival-rate point.

| Rate | Profile | Output tok/s | Stddev | Average TTFT ms | P99.9 TPOT ms | Max TPOT ms | TPOT SLO % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | Control | 99.95 | 9.73 | 743.40 | 881.19 | 2524.07 | 83.3 |
| 0.5 | Fairness | 99.94 | 9.73 | 748.44 | 880.05 | 2518.88 | 94.4 |
| 0.5 | Optimized | 99.92 | 9.71 | 745.43 | 819.10 | 1339.39 | 97.2 |
| 0.9 | Control | 183.03 | 45.32 | 1285.75 | 3130.98 | 5512.10 | 30.6 |
| 0.9 | Fairness | 182.92 | 45.24 | 1298.05 | 1884.11 | 2746.78 | 33.3 |
| 0.9 | Optimized | 182.66 | 45.08 | 1213.33 | 820.54 | 834.68 | 100.0 |
| 1.5 | Control | 240.75 | 2.17 | 2506.26 | 5491.63 | 9304.89 | 19.4 |
| 1.5 | Fairness | 240.22 | 2.23 | 2501.33 | 2410.74 | 2956.88 | 15.3 |
| 1.5 | Optimized | 239.55 | 1.75 | 2370.55 | 829.72 | 875.92 | 100.0 |
| 2.5 | Control | 261.95 | 3.91 | 5635.68 | 7404.22 | 13318.34 | 40.3 |
| 2.5 | Fairness | 260.80 | 3.69 | 5517.75 | 2815.02 | 3044.21 | 23.6 |
| 2.5 | Optimized | 258.97 | 3.70 | 4865.86 | 778.53 | 789.03 | 100.0 |

Relative change from the scheduling control to the final optimized profile:

| Rate | Output tok/s | Average TTFT | P99.9 TPOT | Maximum TPOT | TPOT SLO change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | -0.03% | +0.27% | -7.05% | -46.94% | +13.9 pp |
| 0.9 | -0.20% | -5.63% | -73.79% | -84.86% | +69.4 pp |
| 1.5 | -0.50% | -5.41% | -84.89% | -90.59% | +80.6 pp |
| 2.5 | -1.14% | -13.66% | -89.49% | -94.08% | +59.7 pp |

The median output-throughput and per-run average-TTFT values are:

| Rate | Control median tok/s | Optimized median tok/s | Control median TTFT ms | Optimized median TTFT ms |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 105.61 | 105.51 | 737.95 | 731.87 |
| 0.9 | 214.24 | 213.45 | 1304.51 | 1239.30 |
| 1.5 | 240.57 | 238.66 | 2513.24 | 2300.11 |
| 2.5 | 264.18 | 261.03 | 5920.01 | 5140.57 |

Across all four rates, the scheduling control satisfies the per-request TPOT SLO for 125 of 288
requests and produces 294 token gaps over one second. Fairness alone satisfies 120 of 288 and
produces 454 gaps, but reduces the global maximum from 13.32 seconds to 3.04 seconds. The final
profile satisfies 286 of 288 and produces only two gaps over one second; both occur in the first
0.5 requests/s repeat and share the same 1339 ms stall.

Power is neutral: mean per-GPU power averaged across the four rates is 238.26 W for the control,
238.69 W for fairness, and 238.37 W for the final profile. GPU utilization also remains within
approximately one percentage point at each load.

Raw reports:

- `learning/experiments/016_default_raw.md`
- `learning/experiments/016_fairness_raw.md`
- `learning/experiments/016_optimized_raw.md`

Machine-readable results use the corresponding `016_*.json` names under `learning/results/`.

## Interpretation

The hypothesis is strongly supported at moderate and high load but rejected as a universal strict
SLO claim. At 0.9, 1.5, and 2.5 requests/s, the complete policy reaches 100% TPOT SLO and cuts the
maximum TPOT by 85-94%. It also improves average TTFT by 5-14% at those rates. The result generalizes
the tuning experiments to new seeds and a wider arrival-rate range.

The incremental comparison confirms that both controls are necessary. Fairness alone caps long
runs of prefill and lowers the worst starvation event from 13.32 to 3.04 seconds, but an 8192-token
prefill batch still exceeds the SLO. It can also spread shorter violations across more requests,
which explains why its total SLO pass count does not improve. The 2304 active budget then bounds
the moderate/high-load stalls below 876 ms.

The light-load failure exposes a coverage gap in the active-only policy. In the failing trace, two
existing decode requests stall simultaneously from approximately 34.393 to 35.733 seconds. A
4096-token request and a 1024-token request arrived shortly before the gap, and their first tokens
appear when the gap ends. No request arrives during the gap. The external trace proves that the
current overlap/chunk scheduling path can still expose a 1339 ms gap even with an active budget of
2304. This main run did not enable scheduler telemetry, so the exact sequence of selected budgets
and chunk execution times is not observable; it must be reproduced with the experiment 014
instrumentation before attributing the failure to one internal transition.

The overload throughput result is also real rather than a single-repeat outlier. At 2.5 requests/s,
the three paired throughput changes are -1.05%, -1.17%, and -1.19%. More frequent chunking improves
average TTFT by 13.66% but costs 1.14% output throughput, narrowly exceeding the pre-registered 1%
limit.

## Decision

Do not accept the active-only 2304 policy as the final strict one-second-TPOT profile. It misses two
of 288 requests and slightly exceeds the allowed overload throughput regression.

Retain the implementation and results: the policy is an effective balanced profile for the tested
0.9-2.5 requests/s range, where it reaches 100% SLO with large TTFT and tail-TPOT improvements. Do
not claim that it guarantees a one-second gap for arbitrary arrivals.

Experiment 017 should first replay the failing 0.5 requests/s trace with rank-0 prefill telemetry to
identify the selected budget and execution sequence. It should then test a globally bounded 2304
normal prefill budget, rather than limiting only decode-active batches, at the two failed boundary
conditions: light-load SLO coverage and 2.5 requests/s throughput. A new held-out seed is required
for any final acceptance claim.

## Validation

- All 36 planned traces completed: three profiles, four arrival rates, and three repeats.
- All profiles use identical trace seeds, input-length schedules, and arrival offsets.
- Each profile processes 288 measured requests; all 864 requests complete successfully.
- The optimized profile passes the throughput criterion at 0.5, 0.9, and 1.5 requests/s, but fails
  it at 2.5 requests/s with a 1.14% regression.
- The optimized profile passes the TPOT criterion at 0.9, 1.5, and 2.5 requests/s, but fails it at
  0.5 requests/s with two gaps of approximately 1339 ms.
- Every optimized paired overload trace shows the same 1.05-1.19% throughput regression.
- All servers exit cleanly and all eight GPUs return to zero allocated memory.
