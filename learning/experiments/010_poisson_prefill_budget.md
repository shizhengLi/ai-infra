# Experiment 010: decode-active prefill budget under Poisson arrivals

## Status

Complete.

## Hypothesis

Experiment 009 showed that reducing the decode-active prefill budget from 8192 to 4096 halves the
worst decode interruption under a synchronized burst without reducing throughput. Under sustained
online arrivals, 4096 should retain this tail-latency benefit. Reducing the budget again to 2048 may
further bound TPOT, but could expose scheduling and kernel-launch overhead near saturation.

## Principle

A Poisson process uses independent exponential inter-arrival times and avoids the artificial phase
alignment of a closed batch or synchronized burst. As offered load approaches serving capacity,
queueing delay grows non-linearly; this makes an arrival-rate sweep suitable for locating the
throughput/latency knee.

The decode-active budget controls the maximum amount of new prompt work admitted between decode
steps. Smaller chunks reduce any one interruption but increase how often prefill competes with
decode. The useful operating point must therefore consider TTFT, TPOT, end-to-end latency, SLO
attainment, and throughput together.

## Experiment matrix

- Model: Qwen3-32B BF16
- Hardware: GPUs 0-3, four NVIDIA L20 cards in one `PIX` group
- Tensor parallelism: 4, PyNCCL
- Normal prefill budget: 8192 tokens
- Maximum prefill streak: 1
- Decode-active budgets: 8192, 4096, 2048 tokens
- Arrival rates: 0.5, 1.0, 1.5, 2.5 requests/s
- Workload per trace: 16 requests, 1024 input and 256 output tokens
- Repeats: 3 per matrix point
- TTFT SLO: 2000 ms
- Per-request maximum TPOT SLO: 1000 ms

Each rate/repeat pair uses the same seeded exponential arrival offsets and unique prompt tokens
under every budget. GPU placement, model, graph coverage, memory ratio, and all other scheduler
settings remain fixed.

The initial 0.5-1.5 requests/s sweep retained 100% TTFT and TPOT SLO attainment, so it did not
cross the latency knee. A 2.5 requests/s point was added symmetrically for all three budgets. It
uses seed 500042 rather than 42 so its prompts do not overlap the low-rate traces.

## Implementation

`benchmark/online/bench_poisson_l20.py` generates deterministic open-loop traces and records raw
stream timestamps, scheduled offsets, throughput, latency percentiles, SLO attainment, GPU
utilization, memory, and power. Finish events are excluded from TPOT.

One 2.5 requests/s pilot was discarded before producing a report: starting a separate benchmark
with the default seed 42 regenerated the 0.5 requests/s prompts already held by the baseline
server's Radix Cache. The resulting 79 ms average TTFT was a cache hit, not a scheduling gain. The
valid high-rate runs use seed 500042; each budget begins from a separately restarted server.

## Results

All values are means of three repeats except maximum TPOT, which is the maximum across repeats.

| Active budget | Rate req/s | Output tok/s | P90 TTFT ms | P99.9 TPOT ms | Max TPOT ms | TTFT SLO % | TPOT SLO % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.5 | 157.13 | 661.16 | 462.47 | 871.47 | 100.0 | 100.0 |
| 4096 | 0.5 | 157.13 | 662.34 | 401.00 | 670.95 | 100.0 | 100.0 |
| 2048 | 0.5 | 157.11 | 674.42 | 416.02 | 657.31 | 100.0 | 100.0 |
| 8192 | 1.0 | 200.44 | 826.72 | 489.18 | 621.92 | 100.0 | 100.0 |
| 4096 | 1.0 | 200.37 | 827.45 | 489.47 | 621.75 | 100.0 | 100.0 |
| 2048 | 1.0 | 200.29 | 845.00 | 475.30 | 655.88 | 100.0 | 100.0 |
| 8192 | 1.5 | 215.39 | 947.25 | 539.52 | 848.88 | 100.0 | 100.0 |
| 4096 | 1.5 | 215.34 | 945.12 | 540.12 | 850.27 | 100.0 | 100.0 |
| 2048 | 1.5 | 215.58 | 1023.74 | 560.58 | 661.09 | 100.0 | 100.0 |
| 8192 | 2.5 | 279.38 | 1481.79 | 1007.08 | 1275.17 | 95.8 | 66.7 |
| 4096 | 2.5 | 279.43 | 1561.64 | 1083.68 | 1416.00 | 95.8 | 66.7 |
| 2048 | 2.5 | 279.06 | 1455.32 | 659.99 | 662.55 | 97.9 | 100.0 |

At 2.5 requests/s, 2048 versus 8192 changes:

| Metric | 8192 | 2048 | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 279.38 token/s | 279.06 token/s | -0.12% |
| P90 TTFT | 1481.79 ms | 1455.32 ms | -1.79% |
| P99.9 TPOT | 1007.08 ms | 659.99 ms | -34.46% |
| Worst TPOT | 1275.17 ms | 662.55 ms | -48.04% |
| TTFT SLO attainment | 95.8% | 97.9% | +2.1 points |
| TPOT SLO attainment | 66.7% | 100.0% | +33.3 points |
| Mean token gaps over 1 second | 7.33 | 0.00 | -100.0% |

The 2048 throughput difference remains within 0.12% of 8192 at every tested rate. At 1.5
requests/s its P90 TTFT is 8.08% higher, showing that the smaller budget is not uniformly better
before the SLO boundary.

Raw Markdown reports:

- `learning/experiments/010_active8192_poisson_raw.md`
- `learning/experiments/010_active4096_poisson_raw.md`
- `learning/experiments/010_active2048_poisson_raw.md`
- `learning/experiments/010_active8192_poisson_high_raw.md`
- `learning/experiments/010_active4096_poisson_high_raw.md`
- `learning/experiments/010_active2048_poisson_high_raw.md`

Machine-readable runs are stored in the corresponding `learning/results/010_*.json` files.

## Interpretation

At 0.5-1.5 requests/s, all budgets meet both SLOs and deliver effectively identical throughput.
The 4096 cap is mostly inactive for this workload: a prompt is about 1024 tokens and Poisson
arrivals rarely leave four prompts waiting at one scheduling instant. Its high-rate tail therefore
does not improve over 8192; the observed differences are run-to-run scheduling variation.

The 2048 cap becomes active when roughly two prompts accumulate. At 2.5 requests/s it splits
prefill often enough to bound every token gap below one second, cutting the worst gap by 48% while
preserving throughput. GPU utilization decreases only from 92.67% to 91.54%, and P90 end-to-end
latency changes from 12.532 to 12.504 seconds, so the improvement is not caused by doing less work.

Smaller active chunks do not eliminate prefill cost; they redistribute it across more decode
opportunities. At 1.5 requests/s, 2048 increases P90 TTFT while TPOT already meets the SLO. A static
2048 setting is therefore useful for overload protection but is unnecessarily aggressive for the
entire operating range.

## Decision

Accept 2048 as the measured SLO-focused overload profile for 1024-token prompts:

```bash
--max-prefill-length 8192 \
--max-prefill-streak 1 \
--decode-active-prefill-length 2048
```

Retain 4096 as the balanced interactive profile established by experiment 009, and keep the
software default at `0`. The current static control cannot select both the lower moderate-load TTFT
and the better overload TPOT tail.

## Validation

- All 36 valid traces completed: 3 budgets x 4 rates x 3 repeats.
- Seeds and every scheduled arrival offset match exactly across budgets.
- Each report records the server-side active budget and contains three repeats.
- Two CPU-only tests cover deterministic, increasing arrivals and invalid inputs.
- The server exited cleanly and all GPUs were released after measurement.

## Next step

Experiment 011 should instrument admitted prefill tokens and budget-binding frequency, then use
that signal to select 4096 under normal load and 2048 only when the prefill queue is building. This
targets the 2048 overload tail benefit without its moderate-load TTFT regression.
