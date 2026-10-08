# Experiment 009: decode-active prefill budget

## Status

Complete.

## Hypothesis

Experiment 008 bounded starvation to one 8192-token prefill batch, approximately 2.9 seconds.
Reducing the prefill budget to 4096 only while decode is runnable should roughly halve the maximum
anchor token gap. Retaining 8192 while decode is idle avoids globally reducing prefill batch size.

## Principle

Prefill batch duration determines the minimum interruption bound when prefill and decode execute on
the same stream. The scheduler can use workload state to select its token budget: full-size prefill
when no decode latency is at risk, and smaller chunks when decode requests are already running.

This experiment retains `max_prefill_streak=1`, changes only the decode-active budget from the
default 8192 to 4096, and reuses the mixed-arrival and closed-batch workloads from experiment 008.

## Controlled variables

- Model: Qwen3-32B BF16
- Hardware: GPUs 0-3, four NVIDIA L20 cards
- Tensor parallelism: 4, PyNCCL
- Normal prefill budget: 8192 tokens
- Maximum prefill streak: 1
- Baseline decode-active budget: 8192 tokens
- Treatment decode-active budget: 4096 tokens
- Mixed workload: 8 anchors, then 24 long-prefill requests after 2 seconds
- Closed workload: 32 requests with 1024 input and 256 output tokens
- Repeats: 3

## Implementation

The scheduler configuration now has an optional decode-active prefill budget:

```bash
python -m minisgl ... \
  --max-prefill-length 8192 \
  --max-prefill-streak 1 \
  --decode-active-prefill-length 4096
```

`PrefillBudgetPolicy` selects 4096 only when the decode manager has runnable requests. It falls
back to the normal 8192-token budget while decode is idle. A value of `0`, the default, disables
the adaptive limit and preserves the previous scheduling behavior. Invalid non-positive normal
budgets and decode-active budgets larger than the normal budget are rejected.

The two benchmark drivers record the server-side decode-active budget in both Markdown and JSON,
so a result cannot be confused with a run using the old scheduling configuration.

## Results

Mixed-arrival comparison against experiment 008 bounded scheduling:

| Metric | Active budget 8192 | Active budget 4096 | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 249.71 token/s | 250.85 token/s | +0.46% |
| Anchor average TTFT | 346.35 ms | 344.60 ms | -0.50% |
| Anchor P99 TPOT | 153.29 ms | 692.99 ms | +352.07% |
| Anchor P99.9 TPOT | 2799.49 ms | 1385.93 ms | -50.49% |
| Anchor worst TPOT | 2895.48 ms | 1427.81 ms | -50.69% |
| Burst average TTFT | 5336.88 ms | 4788.49 ms | -10.28% |
| Burst P90 TTFT | 7880.41 ms | 7866.61 ms | -0.18% |
| Burst P90 E2E | 10.103 s | 10.079 s | -0.24% |
| Average GPU utilization | 94.55% | 94.50% | -0.05% |
| Peak memory per GPU | 37833 MiB | 37391 MiB | -1.17% |
| Average power per GPU | 223.52 W | 220.68 W | -1.27% |

Closed C32 regression check:

| Metric | Active budget 8192 | Active budget 4096 | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 413.64 token/s | 414.92 token/s | +0.31% |
| Average TTFT | 6691.53 ms | 6076.11 ms | -9.20% |
| P90 TTFT | 10472.70 ms | 10427.12 ms | -0.44% |
| Average TPOT | 51.20 ms | 53.13 ms | +3.76% |
| P90 TPOT | 37.29 ms | 37.24 ms | -0.12% |
| P99.9 TPOT | 2901.86 ms | 1488.24 ms | -48.71% |
| Worst TPOT | 2903.49 ms | 1492.60 ms | -48.59% |
| P90 E2E | 19.780 s | 19.715 s | -0.33% |

The treatment was stable across all three repeats. Mixed throughput ranged from 250.68 to 250.94
token/s, and closed throughput ranged from 414.86 to 414.97 token/s.

Raw Markdown reports:

- `learning/experiments/009_active4096_mixed_raw.md`
- `learning/experiments/009_active4096_closed_raw.md`

Machine-readable traces are stored in `learning/results/009_active4096_mixed.json` and
`learning/results/009_active4096_closed.json`.

## Interpretation

With one prefill batch permitted between decode steps, the active prefill budget directly controls
the longest time that decode can be interrupted. Halving that budget from 8192 to 4096 halves both
the mixed-arrival P99.9 and worst TPOT. The 1.43-second observed maximum is close to half of the
2.90-second baseline, supporting the causal hypothesis rather than merely showing run-to-run noise.

The optimization redistributes prefill work into more, shorter scheduling intervals. Each mixed
repeat now has 40 post-burst gaps over one second across eight anchors, compared with 24 under the
8192-token bounded policy. This explains why P99 TPOT increases even though P99.9 and the maximum
improve: the interference is more frequent, but its upper bound is much smaller.

There is no measured throughput penalty in these workloads. Smaller active chunks improve average
burst TTFT by 10.28% and average closed-batch TTFT by 9.20%, while total throughput changes by less
than 0.5%. This is plausible because idle-decode prefill still uses the full 8192 budget and the GPU
remains saturated; however, it should not be generalized beyond the tested model and arrival trace.

## Decision

Accept the adaptive budget as an opt-in control. The measured L20 interactive profile is:

```bash
--max-prefill-length 8192 \
--max-prefill-streak 1 \
--decode-active-prefill-length 4096
```

Keep the decode-active default at `0` for compatibility and throughput-oriented deployments. Use
4096 when bounding extreme streaming stalls matters more than minimizing the number of moderate
token gaps. Do not reduce it further without an arrival-rate sweep, because smaller chunks may
eventually add scheduler and kernel-launch overhead.

## Validation

- Installed `pytest 9.1.1` and `pytest-cov 7.1.0` in the project virtual environment.
- Eight CPU-only policy tests pass, including disabled, selected, and invalid budget cases.
- Both treatment workloads completed three repetitions with the expected server metadata.
- The server exited cleanly after all measurements.

## Next step

Experiment 010 should sweep decode-active budgets 8192, 4096, and 2048 under Poisson arrivals at
multiple request rates. The purpose is to locate the throughput/latency knee under sustained online
traffic rather than a single synchronized burst.
