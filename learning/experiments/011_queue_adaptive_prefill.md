# Experiment 011: queue-adaptive decode-active prefill budget

## Status

Complete.

## Hypothesis

Experiment 010 found that a static 2048-token active budget protects TPOT at 2.5 requests/s but
regresses P90 TTFT at 1.5 requests/s. Selecting 4096 normally and 2048 only when at least 4096
prefill tokens are queued should preserve the overload tail benefit while avoiding unnecessary
chunking at moderate load.

## Principle

Arrival rate is not directly available to the scheduler and would be a delayed control signal.
Pending prefill tokens are immediate local backpressure: they combine request count and prompt
length in the same unit as the prefill budget. A threshold policy can therefore react to the work
that would make the next prefill batch long, without estimating traffic rate externally.

The implementation keeps the existing 8192-token budget while decode is idle, uses 4096 while
decode is active under normal pressure, and uses 2048 when pending prefill input reaches 4096.
All new controls default to disabled.

## Experiment matrix

- Model: Qwen3-32B BF16, TP=4 on GPUs 0-3
- Workload: 16 requests, 1024 input and 256 output tokens
- Arrival rates: 1.5 and 2.5 requests/s
- Repeats: 3
- Baselines: static 4096 and static 2048 from experiment 010
- Treatment: active 4096, overload 2048, overload threshold 4096 pending tokens
- Maximum prefill streak: 1

The treatment reuses the exact experiment 010 seeds and arrival schedules. Scheduler JSONL
telemetry records selected budgets, admitted prefill tokens, budget-limited batches, and maximum
queue pressure.

## Results

Latency and throughput comparison using identical traces from experiment 010:

| Rate | Policy | Output tok/s | P90 TTFT ms | P99.9 TPOT ms | Max TPOT ms | TTFT SLO % | TPOT SLO % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | Static 4096 | 215.34 | 945.12 | 540.12 | 850.27 | 100.0 | 100.0 |
| 1.5 | Static 2048 | 215.58 | 1023.74 | 560.58 | 661.09 | 100.0 | 100.0 |
| 1.5 | Queue-adaptive | 215.54 | 935.87 | 609.88 | 870.02 | 100.0 | 100.0 |
| 2.5 | Static 4096 | 279.43 | 1561.64 | 1083.68 | 1416.00 | 95.8 | 66.7 |
| 2.5 | Static 2048 | 279.06 | 1455.32 | 659.99 | 662.55 | 97.9 | 100.0 |
| 2.5 | Queue-adaptive | 279.41 | 1546.29 | 730.74 | 861.03 | 95.8 | 100.0 |

At 1.5 requests/s, queue-adaptive versus static 2048:

- P90 TTFT improves by 8.58%.
- Output throughput changes by -0.02%.
- Both retain 100% TTFT and TPOT SLO attainment.

At 2.5 requests/s, queue-adaptive versus static 4096:

- P99.9 TPOT improves by 32.57%.
- Worst TPOT improves by 39.19%.
- TPOT SLO attainment improves from 66.7% to 100%.
- Output throughput changes by -0.01%.

The adaptive policy does not match static 2048's strongest high-load tail: at 2.5 requests/s its
P99.9 and maximum TPOT are 10.72% and 29.96% higher, respectively. It retains 0.13% more throughput
but has 2.1 percentage points lower TTFT SLO attainment.

Scheduler telemetry after excluding the two warmups:

| Rate | 8192 selections | 4096 selections | 2048 selections | Prefill batches | Budget-limited batches | Maximum pending tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5 | 3 | 40 | 0 | 43 | 0 | 3096 |
| 2.5 | 3 | 31 | 3 | 37 | 3 | 5160 |

At 1.5 requests/s, queue pressure never reaches the 4096-token threshold. At 2.5 requests/s, only
3 of 34 decode-active selections use 2048, and all three actually limit the assembled prefill
batch. Those three interventions are sufficient to remove every token gap over one second.

Raw reports:

- `learning/experiments/011_dynamic_1p5_raw.md`
- `learning/experiments/011_dynamic_2p5_raw.md`

Machine-readable results are `learning/results/011_dynamic_1p5.json`,
`learning/results/011_dynamic_2p5.json`, and `learning/results/011_dynamic_telemetry.jsonl`.

## Interpretation

Pending token pressure separates the tested regimes cleanly. The moderate trace peaks at three
queued prompts and remains on 4096. The overloaded trace reaches four to five queued prompts and
activates 2048 only around the densest arrival clusters. This validates queued tokens as a useful,
prompt-length-aware local signal rather than relying on a configured request rate.

The hypothesis is partially confirmed. Dynamic selection removes the moderate-load P90 TTFT
regression of static 2048 and restores the overloaded TPOT SLO without measurable throughput cost.
It does not preserve the entire static-2048 tail benefit because requests arriving below the
threshold still use 4096; the 861 ms maximum lies between the static 4096 and 2048 results.

The telemetry also explains the large effect from a small number of policy decisions. Most
prefill batches are short enough that either budget behaves identically. Tail violations originate
from the few batches assembled when four or more prompts are queued, so limiting only those batches
changes extreme TPOT without perturbing the common path.

## Decision

Accept the queue-adaptive policy as the balanced L20 profile:

```bash
--max-prefill-length 8192 \
--max-prefill-streak 1 \
--decode-active-prefill-length 4096 \
--decode-overload-prefill-length 2048 \
--decode-overload-prefill-threshold 4096
```

Keep overload switching disabled by default. Use static 2048 when the strictest TPOT bound is more
important than TTFT; use the adaptive profile when one deployment must cover both moderate and
overloaded traffic. More models and prompt-length distributions are required before changing the
global default.

## Validation

- Six treatment traces completed with the exact experiment 010 seeds and arrival offsets.
- Telemetry wrote eight valid JSONL snapshots: two warmups and six measured traces.
- Policy tests cover normal, overloaded, disabled, and invalid configurations.
- The production default remains unchanged when overload options are zero.
- The server exited cleanly and all GPU memory was released.

## Next step

Experiment 012 should validate token-pressure switching with mixed 256/1024/4096-token prompts and
a longer trace. This will test whether the fixed 4096-token threshold generalizes when request
count is no longer a proxy for queued prompt work.
