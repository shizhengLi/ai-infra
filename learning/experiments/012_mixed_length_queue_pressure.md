# Experiment 012: mixed-length queue-pressure validation

## Status

Complete.

## Hypothesis

The 4096 pending-token threshold from experiment 011 should react to actual queued prefill work,
not merely request count. With a balanced mix of 256, 1024, and 4096-token prompts, the adaptive
policy should retain the TTFT advantage of static 4096 under moderate pressure while approaching
the TPOT protection of static 2048 under overload.

## Principle

Request count is a weak pressure signal when prompt lengths vary: one 4096-token prompt represents
the same prefill work as sixteen 256-token prompts. The scheduler therefore sums uncached pending
input tokens and compares that work to the configured threshold. A mixed-length trace is needed to
test whether this token-domain control signal remains meaningful when queue depth and queue work
diverge.

The benchmark assigns 24 requests evenly across the three input lengths, then applies a seeded
shuffle. Arrival offsets, length order, generated prompts, and output lengths are identical across
all policies. Per-length TTFT and E2E metrics expose whether protecting decode disproportionately
delays long prompts.

## Experiment matrix

- Model: Qwen3-32B BF16, TP=4 on GPUs 0-3
- Workload: 24 requests per trace; 8 each at 256, 1024, and 4096 input tokens
- Output length: 256 tokens
- Arrival rates: 0.9 and 1.5 requests/s
- Repeats: 3
- Seed: 800042
- Static baselines: decode-active budgets 4096 and 2048
- Treatment: active 4096, overload 2048, threshold 4096 pending tokens
- Decode-idle prefill budget: 8192
- Maximum prefill streak: 1

The mean input length is 1792 tokens. Rates 0.9 and 1.5 therefore offer approximately 1613 and
2688 arriving input tokens/s, comparable to the 1536 and 2560 input tokens/s regimes tested at
1.5 and 2.5 requests/s with fixed 1024-token prompts in experiments 010 and 011.

## Results

Aggregate results from three repeats per point:

| Rate | Policy | Output tok/s | P90 TTFT ms | P99.9 TPOT ms | Max TPOT ms | TTFT SLO % | TPOT SLO % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | Static 4096 | 210.05 | 3284.22 | 1409.27 | 1553.35 | 58.3 | 23.6 |
| 0.9 | Static 2048 | 209.98 | 3076.90 | 711.38 | 756.99 | 65.3 | 100.0 |
| 0.9 | Queue-adaptive | 210.05 | 2959.66 | 960.98 | 1002.50 | 69.4 | 79.2 |
| 1.5 | Static 4096 | 233.39 | 4673.56 | 1390.14 | 1455.00 | 33.3 | 12.5 |
| 1.5 | Static 2048 | 232.32 | 4715.70 | 707.40 | 732.19 | 31.9 | 100.0 |
| 1.5 | Queue-adaptive | 232.68 | 4718.40 | 990.94 | 1082.92 | 38.9 | 43.1 |

Relative to static 4096, queue-adaptive at 0.9 requests/s:

- Preserves throughput within +0.01%.
- Improves P90 TTFT by 9.88% and average TTFT by 9.34%.
- Improves P99.9 TPOT by 31.81% and maximum TPOT by 35.46%.
- Raises TPOT SLO attainment by 55.6 percentage points, but reaches only 79.2%.

At 1.5 requests/s, queue-adaptive preserves throughput within -0.31% and improves P99.9 TPOT by
28.72%, but only 43.1% of requests satisfy the per-request TPOT SLO. Static 2048 is the only policy
that reaches 100% TPOT SLO in both regimes; its throughput costs are 0.03% and 0.46% relative to
static 4096.

Average TTFT by input length:

| Rate | Policy | 256 tokens ms | 1024 tokens ms | 4096 tokens ms |
| ---: | --- | ---: | ---: | ---: |
| 0.9 | Static 4096 | 936.46 | 1537.12 | 2583.41 |
| 0.9 | Static 2048 | 875.00 | 1495.81 | 2401.64 |
| 0.9 | Queue-adaptive | 827.19 | 1444.00 | 2313.60 |
| 1.5 | Static 4096 | 2250.16 | 2527.94 | 3605.45 |
| 1.5 | Static 2048 | 2239.42 | 2496.63 | 3301.62 |
| 1.5 | Queue-adaptive | 2238.44 | 2476.46 | 3304.31 |

The adaptive policy does not impose a long-prompt TTFT penalty in this test. At 0.9 requests/s it
has the lowest average TTFT at every input length. At 1.5, it is effectively tied with static 2048.

Queue-adaptive scheduler telemetry, excluding warmup:

| Rate | 8192 selections | 4096 selections | 2048 selections | Prefill batches | Budget-limited batches | Maximum pending tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 3 | 40 | 32 | 75 | 32 | 10551 |
| 1.5 | 3 | 22 | 45 | 70 | 45 | 14959 |

Every 2048 selection is budget-limited, so overload switching is doing useful work rather than
selecting a lower budget for already-small batches. However, 55.6% of decode-active selections at
0.9 and 32.8% at 1.5 still use 4096.

Raw reports:

- `learning/experiments/012_static4096_raw.md`
- `learning/experiments/012_static2048_raw.md`
- `learning/experiments/012_dynamic_raw.md`

Machine-readable results and scheduler telemetry use the corresponding `012_*` names under
`learning/results/`.

## Interpretation

The hypothesis is rejected for the 1-second TPOT SLO. Pending tokens remain a valid pressure signal:
the policy switches more often at 1.5 requests/s, and every switch limits real work. The threshold
value does not generalize from homogeneous 1024-token prompts to the mixed workload.

The failure mode is threshold sensitivity rather than request-count sensitivity. With a 4096-token
threshold, a queue just below 4096 can still produce a nearly 4096-token prefill batch and stall
decode for about one second. Static 2048 also chunks these below-threshold batches, explaining why
it removes every token gap over one second while the adaptive policy does not.

Mixed prompt lengths also invalidate a simple arriving-token-rate comparison. Although the chosen
rates match experiments 010 and 011 in mean input tokens/s, 4096-token requests create bursty
prefill service times and saturate all four GPUs at roughly 95% utilization even at 0.9 requests/s.
Mean token rate is useful for choosing a starting range, but not for asserting equivalent queueing
behavior.

## Decision

Do not generalize the 4096 pending-token threshold as the balanced L20 default for mixed prompt
lengths. For this distribution and a strict 1-second per-request maximum TPOT SLO, use static 2048:

```bash
--max-prefill-length 8192 \
--max-prefill-streak 1 \
--decode-active-prefill-length 2048
```

Retain the queue-adaptive mechanism because it improves both TTFT and TPOT relative to static 4096,
but treat its threshold as workload-sensitive until a lower threshold or a direct next-batch cost
signal is validated.

## Validation

- Eighteen measured traces completed: three policies, two arrival rates, and three repeats.
- All policies used identical seeds, Poisson offsets, mixed-length orders, prompts, and output sizes.
- Each trace admitted 43128 input tokens including chat-template overhead.
- Each server wrote seven valid telemetry snapshots: one warmup and six measured traces.
- The mixed-length generator and per-length summary are covered by focused tests.
- All server processes exited cleanly and all eight GPUs returned to zero allocated memory.

## Next step

Experiment 013 should replay the identical traces with a 2048 pending-token overload threshold.
This isolates whether earlier switching recovers static 2048's TPOT SLO while retaining the
adaptive policy's moderate-load TTFT advantage. If 2048 remains insufficient, compare a direct
next-prefill-batch cost signal rather than continuing an unconstrained threshold sweep.
