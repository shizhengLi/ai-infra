# Experiment 013: lower overload threshold for mixed prompts

## Status

Complete.

## Hypothesis

Experiment 012 showed that a 4096-token overload threshold leaves some nearly 4096-token prefill
batches on the normal 4096 budget, causing decode gaps above one second. Lowering only the pending
token threshold to 2048 should switch those batches to the 2048 budget earlier and approach static
2048's 100% TPOT SLO, while retaining 4096 for genuinely small queues.

## Principle

The overload threshold controls when the scheduler selects the lower budget; it does not directly
limit a batch. A useful threshold must therefore activate before queued work can assemble a batch
large enough to violate the decode latency objective. With a 2048-token overload budget, using the
same value as the threshold tests a simple invariant: once pending work can fill one protected
batch, use the protected budget.

This is a one-variable replay of experiment 012. Seeds, arrival offsets, input-length schedules,
generated prompts, output lengths, model, and all other scheduler settings remain unchanged.

## Experiment matrix

- Model: Qwen3-32B BF16, TP=4 on GPUs 0-3
- Workload: 24 requests per trace; 8 each at 256, 1024, and 4096 input tokens
- Output length: 256 tokens
- Arrival rates: 0.9 and 1.5 requests/s
- Repeats: 3
- Seed: 800042
- Decode-idle prefill budget: 8192
- Decode-active normal budget: 4096
- Decode-overload budget: 2048
- Treatment threshold: 2048 pending tokens
- Experiment 012 comparison threshold: 4096 pending tokens
- Maximum prefill streak: 1

## Results

Aggregate results using the identical experiment 012 traces:

| Rate | Policy | Output tok/s | P90 TTFT ms | P99.9 TPOT ms | Max TPOT ms | TTFT SLO % | TPOT SLO % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | Threshold 4096 | 210.05 | 2959.66 | 960.98 | 1002.50 | 69.4 | 79.2 |
| 0.9 | Threshold 2048 | 209.97 | 3079.37 | 711.06 | 756.18 | 65.3 | 100.0 |
| 0.9 | Static 2048 | 209.98 | 3076.90 | 711.38 | 756.99 | 65.3 | 100.0 |
| 1.5 | Threshold 4096 | 232.68 | 4718.40 | 990.94 | 1082.92 | 38.9 | 43.1 |
| 1.5 | Threshold 2048 | 232.38 | 4709.17 | 707.00 | 731.78 | 31.9 | 100.0 |
| 1.5 | Static 2048 | 232.32 | 4715.70 | 707.40 | 732.19 | 31.9 | 100.0 |

Relative to threshold 4096, threshold 2048 at 0.9 requests/s:

- Restores TPOT SLO attainment from 79.2% to 100% and removes all gaps over one second.
- Improves P99.9 and maximum TPOT by 26.01% and 24.57%.
- Regresses average and P90 TTFT by 3.96% and 4.05%.
- Changes throughput by -0.04%.

At 1.5 requests/s, it restores TPOT SLO attainment from 43.1% to 100%, improves P99.9 and maximum
TPOT by 28.65% and 32.43%, and changes throughput by -0.13%. Average TTFT changes by +0.09%, while
P90 TTFT improves by 0.20%.

Threshold 2048 versus static 2048 differs by at most 0.15% on throughput and reported aggregate
latency metrics. These differences are measurement noise rather than evidence of a distinct
performance tradeoff.

Scheduler telemetry, excluding warmup:

| Rate | 8192 selections | 4096 selections | 2048 selections | Prefill batches | Budget-limited batches | Maximum pending tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 3 | 32 | 51 | 86 | 51 | 10551 |
| 1.5 | 3 | 16 | 58 | 77 | 58 | 14959 |

All 2048 selections are budget-limited. The remaining 4096 selections occur only while pending
work is below 2048 tokens, so selecting 4096 cannot admit more work than a static 2048 budget.

Raw report: `learning/experiments/013_dynamic_threshold2048_raw.md`.

Machine-readable results are `learning/results/013_dynamic_threshold2048.json` and
`learning/results/013_dynamic_threshold2048_telemetry.jsonl`.

## Interpretation

The SLO portion of the hypothesis is confirmed, but the claimed adaptive tradeoff is rejected.
Lowering the threshold to 2048 recovers static 2048's tail protection because it is behaviorally
equivalent to static 2048.

Let `P` be pending prefill tokens and `B=2048` the protected budget. When `P >= B`, the policy
selects `B`; when `P < B`, it selects 4096, but only `P` tokens exist to admit. In both cases the
maximum admitted work is `min(P, B)`, exactly the cap imposed by static 2048. Telemetry confirms
this equivalence: threshold 2048 and static 2048 create the same number of prefill batches and
budget-limited batches at both rates.

This establishes a lower bound on this threshold-policy family. A threshold at or below the
overload budget cannot produce a performance point different from the static overload budget. A
higher threshold can preserve TTFT in some traces, but experiment 012 showed that it misses the
strict one-second TPOT objective under mixed long prompts.

## Decision

Do not adopt threshold 2048 as a separate adaptive profile. Use the simpler static 2048 deployment
profile when the one-second maximum TPOT SLO is required:

```bash
--max-prefill-length 8192 \
--max-prefill-streak 1 \
--decode-active-prefill-length 2048
```

Retain configurable overload switching for workloads where a higher threshold has been validated,
but stop tuning the mixed workload through pending-token thresholds alone. A genuinely different
tradeoff requires a signal related to predicted prefill execution time or decode stall risk.

## Validation

- Six measured traces completed with the exact experiment 012 seeds, offsets, and length schedules.
- All traces contain eight requests at each input length and admit 43128 input tokens.
- Both rates reach 100% per-request TPOT SLO with zero token gaps over one second.
- Seven valid telemetry snapshots were written: one warmup and six measured traces.
- The server exited cleanly and all eight GPUs returned to zero allocated memory.

## Next step

Experiment 014 should instrument per-prefill-batch execution time together with admitted tokens and
active decode batch size. The measurements can calibrate an L20 stall-time model and determine the
largest prefill budget compatible with a target TPOT, instead of using queued tokens as an indirect
proxy for execution cost.
