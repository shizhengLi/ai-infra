# Experiment 008: bounded prefill scheduling under mixed arrivals

## Status

Complete.

## Hypothesis

Strict prefill priority causes already-running decode requests to stop whenever a multi-batch
prefill burst arrives. Limiting the number of consecutive prefill batches should bound decode
starvation and reduce anchor-request TPOT tails, with a small cost to burst TTFT and total
throughput.

## Principle

The current scheduler evaluates prefill before decode on every iteration. Pending prefill therefore
drains completely before decode resumes. A bounded policy permits at most one consecutive prefill
batch while decode is runnable, then schedules one decode batch before accepting more prefill.

This experiment uses an open-loop timed trace rather than a closed batch. Eight anchor requests
start first and generate long outputs. After two seconds, 24 long-prefill requests arrive together,
independent of server completion. Anchor token gaps after that dispatch directly measure decode
starvation.

## Controlled variables

- Model: Qwen3-32B BF16
- Hardware: GPUs 0-3, four NVIDIA L20 cards in one `PIX` group
- Tensor parallelism: 4, PyNCCL
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Prefill budget: 8192 tokens
- Anchor workload: 8 requests, 128 input tokens, 512 output tokens
- Timed burst: 24 requests after 2 seconds, 1024 input tokens, 64 output tokens
- Repeats: 3 with unique deterministic prompts
- Independent variable: unlimited versus one consecutive prefill batch

## Implementation

The server now accepts an optional scheduling limit:

```bash
python -m minisgl ... --max-prefill-streak 1
```

`0` is the default and preserves unlimited prefill priority. Positive values bound the consecutive
prefill batches only when both prefill and decode are runnable. The policy state is isolated in
`python/minisgl/scheduler/policy.py` and the scheduler falls back to its existing behavior when the
limit is disabled.

The mixed-arrival workload is implemented in `benchmark/online/bench_mixed_l20.py`. Baseline and
treatment use identical deterministic traces:

```bash
python benchmark/online/bench_mixed_l20.py \
  --anchor-count 8 --anchor-input-len 128 --anchor-output-len 512 \
  --burst-count 24 --burst-input-len 1024 --burst-output-len 64 \
  --burst-delay 2 --repeats 3 --gpu-indices 0,1,2,3 \
  --server-tp 4 --server-max-extend-tokens 8192 \
  --server-max-prefill-streak STREAK
```

## Results

Mixed-arrival results:

| Metric | Strict priority (0) | Bounded (1) | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 250.03 token/s | 249.71 token/s | -0.13% |
| Anchor average TTFT | 345.08 ms | 346.35 ms | +0.37% |
| Anchor P99 TPOT | 39.20 ms | 153.29 ms | +291.10% |
| Anchor P99.9 TPOT | 7655.96 ms | 2799.49 ms | -63.43% |
| Anchor worst TPOT | 7674.46 ms | 2895.48 ms | -62.27% |
| Burst average TTFT | 5643.11 ms | 5336.88 ms | -5.43% |
| Burst P90 TTFT | 7799.19 ms | 7880.41 ms | +1.04% |
| Burst P90 E2E | 10.045 s | 10.103 s | +0.58% |
| Average GPU utilization | 94.16% | 94.55% | +0.42% |
| Average power per GPU | 221.22 W | 223.52 W | +1.04% |

Each strict-priority repeat produced eight token gaps over one second: one long stall for each
anchor. Each bounded repeat produced 24: three shorter stalls for each anchor, corresponding to the
three 8192-token prefill batches.

Closed-batch regression check against experiment 006:

| Metric | Strict priority (0) | Bounded (1) | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 415.34 token/s | 413.64 token/s | -0.41% |
| P90 TTFT | 10344.71 ms | 10472.70 ms | +1.24% |
| P90 TPOT | 37.23 ms | 37.29 ms | +0.15% |
| P99.9 TPOT | 4490.44 ms | 2901.86 ms | -35.38% |
| Worst TPOT | 9637.84 ms | 2903.49 ms | -69.87% |
| P90 E2E | 19.709 s | 19.780 s | +0.36% |

Raw Markdown reports:

- `learning/experiments/008_strict_prefill_raw.md`
- `learning/experiments/008_bounded_prefill_raw.md`
- `learning/experiments/008_bounded_closed_batch_raw.md`

Machine-readable traces are stored in the corresponding `learning/results/008_*.json` files.

## Interpretation

Strict priority drains all three burst prefill batches before resuming decode, creating one stable
7.6-second stall. Bounded scheduling inserts one decode step between prefill batches, reducing the
worst interruption to the duration of one prefill batch, about 2.9 seconds on this workload.

This changes the shape rather than the total amount of prefill interference. More intervals move
above 100 ms and one second, so P99 TPOT regresses while P99.9 and maximum TPOT improve sharply.
The policy is therefore a tail-bound/fairness control, not a way to remove prefill cost.

The throughput costs are small and repeatable: 0.13% on mixed arrivals and 0.41% on the closed
batch. Burst P90 TTFT and E2E regress by about 1% or less. The improvement is not caused by lower
GPU load; utilization and memory remain effectively unchanged.

## Decision

Accept `max_prefill_streak` as an opt-in scheduling control. Use `--max-prefill-streak 1` for the
measured TP=4 L20 interactive profile because it cuts the worst decode stall by 62-70% at less than
0.5% throughput cost. Keep the global default at `0` until more models and arrival distributions
are tested, preserving existing behavior for throughput-oriented deployments.

## Validation

- Four CPU-only policy state tests pass.
- Server argument parsing was checked for default `0` and explicit `1`.
- Both policies completed three mixed-arrival repetitions with 32 requests each.
- The bounded policy completed three closed-batch regression repetitions.
- `pytest` is not installed in the current virtual environment, so the new pytest-compatible test
  file was invoked directly.

## Next step

Experiment 009 will reduce the active-decode prefill batch size while retaining the normal 8192
budget when decode is idle. The goal is to lower the remaining 2.9-second bound without globally
reducing prefill throughput.
