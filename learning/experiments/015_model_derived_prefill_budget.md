# Experiment 015: model-derived 2304-token prefill budget

## Status

Complete.

## Hypothesis

Experiment 014 predicts that a 2304-token decode-active prefill batch needs 737.62 ms of GPU time.
Adding the conservative 250 ms host, scheduling, and adjacent-work reserve gives a 987.62 ms
end-to-end envelope. Therefore, 2304 should retain the one-second per-request maximum TPOT SLO of
the 2048 baseline while using the remaining latency headroom to reduce prefill waiting time.

## Principle

The decode-active prefill budget controls a direct latency-throughput tradeoff. A smaller budget
returns to decode sooner but splits queued prompts into more chunks. A larger budget admits more
prefill work per scheduling turn, reducing chunking and queue delay at the cost of a longer decode
stall. Experiment 014 turns this parameter from an unconstrained sweep into a model-derived choice:

```text
prefill_execution_ms = 11.1947 + 0.315291 * admitted_tokens
```

This experiment changes only the active budget from 2048 to 2304. It reuses the experiment 012
static-2048 result because the seeds, generated prompts, arrival offsets, model, server settings,
and benchmark implementation are unchanged. Experiment 014 separately showed that the new opt-in
timing instrumentation perturbs end-to-end metrics by at most 0.41%; timing is disabled here.

## Experiment matrix

- Model: Qwen3-32B BF16, TP=4 on GPUs 0-3
- Attention backend: FlashInfer selected by `auto`
- CUDA: 12.8, SM89
- KV-cache memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Workload: 24 requests per trace; 8 each at 256, 1024, and 4096 input tokens
- Output length: 256 tokens
- Arrival rates: 0.9 and 1.5 requests/s
- Repeats: 3
- Seed: 800042
- Decode-idle prefill budget: 8192
- Maximum prefill streak: 1
- Control: static decode-active budget 2048 from experiment 012
- Treatment: static decode-active budget 2304
- SLO: each request's maximum TPOT must not exceed 1000 ms

The six treatment traces have exactly the same trace seeds, input-length order, and scheduled
arrival offsets as the six control traces.

## Execution note

The first server start inherited `/usr/bin/nvcc` (CUDA 11.5) and failed while rebuilding PyNCCL
because that compiler does not support `-std=c++20`. No benchmark request was issued and no result
was recorded from that attempt. The server was restarted with the planned toolchain:

```bash
CUDA_HOME=/usr/local/cuda-12.8
PATH=/usr/local/cuda-12.8/bin:$PATH
```

CUDA 12.8 compiled the extension successfully and all measured traces then completed normally.

## Results

Aggregate results from three repeats per point:

| Rate | Budget | Output tok/s | Stddev | Average TTFT ms | P90 TTFT ms | P99.9 TPOT ms | Max TPOT ms | TTFT SLO % | TPOT SLO % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 2048 | 209.98 | 21.79 | 1590.81 | 3076.90 | 711.38 | 756.99 | 65.3 | 100.0 |
| 0.9 | 2304 | 209.98 | 21.74 | 1493.77 | 2877.98 | 808.73 | 828.39 | 75.0 | 100.0 |
| 1.5 | 2048 | 232.32 | 21.34 | 2679.23 | 4715.70 | 707.40 | 732.19 | 31.9 | 100.0 |
| 1.5 | 2304 | 232.63 | 21.45 | 2629.53 | 4718.77 | 801.39 | 828.25 | 40.3 | 100.0 |

Relative change from 2048 to 2304:

| Rate | Output tok/s | Average TTFT | P90 TTFT | P99.9 TPOT | Max TPOT | TTFT SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | -0.002% | -6.10% | -6.46% | +13.68% | +9.43% | +9.7 pp |
| 1.5 | +0.134% | -1.85% | +0.07% | +13.29% | +13.12% | +8.3 pp |

Every treatment request satisfies the TPOT SLO. Across 144 measured treatment requests there are
zero token gaps above one second. The observed worst case is 828.39 ms, leaving 171.61 ms of SLO
headroom. Throughput changes by at most 0.14%, well below the approximately 21 token/s between-run
standard deviation.

Average TTFT by input length:

| Rate | Input tokens | 2048 ms | 2304 ms | Change |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 256 | 875.00 | 807.29 | -7.74% |
| 0.9 | 1024 | 1495.81 | 1462.60 | -2.22% |
| 0.9 | 4096 | 2401.64 | 2211.41 | -7.92% |
| 1.5 | 256 | 2239.42 | 2246.73 | +0.33% |
| 1.5 | 1024 | 2496.63 | 2471.58 | -1.00% |
| 1.5 | 4096 | 3301.62 | 3170.26 | -3.98% |

The paired per-repeat average-TTFT changes are -4.52%, -8.17%, and -6.42% at 0.9 requests/s, and
-0.05%, -7.84%, and -0.68% at 1.5 requests/s. Thus all six paired traces improve, although the
high-pressure effect is small in two of three repeats. Median per-run average TTFT moves from
1467.76 to 1347.80 ms at 0.9 requests/s and from 3058.71 to 3057.23 ms at 1.5 requests/s.

Raw treatment report: `learning/experiments/015_static2304_raw.md`.

Machine-readable treatment data: `learning/results/015_static2304.json`. The control data remains
`learning/results/012_static2048.json`.

## Interpretation

The hypothesis is confirmed for the tested workload. The service-time model selected a budget that
uses more of the latency allowance without crossing the hard SLO. The 2304 budget does not produce
a measurable throughput change, but it reduces average TTFT and improves TTFT SLO attainment in
both load regimes. The largest gains are on 4096-token prompts, which benefit most from admitting
more prefill tokens per turn.

The experiment also bounds the tradeoff. Relative to 2048, tail TPOT rises by about 13%, so 2304 is
not universally better for deployments whose objective is to minimize TPOT below the stated
one-second limit. It is preferable for the declared objective: maximize prefill responsiveness
subject to a hard one-second maximum-TPOT constraint. The 171.61 ms observed margin is adequate for
the tested trace but must be retested across the broader load matrix in the main experiment.

## Decision

Accept 2304 as the final candidate decode-active prefill budget for the main experiment:

```bash
--max-prefill-length 8192 \
--max-prefill-streak 1 \
--decode-active-prefill-length 2304
```

Keep 2048 as the conservative deployment profile when minimizing decode jitter is more important
than TTFT. Do not increase to 2560: experiment 014 predicts 1068.34 ms after the measured reserve,
which violates the SLO before validation.

## Validation

- Six treatment traces completed: two arrival rates and three repeats.
- Every trace contains 24 requests and the balanced 256/1024/4096-token mix.
- Trace seeds, input-length schedules, and arrival offsets exactly match the 2048 control.
- All 144 treatment requests satisfy the one-second maximum-TPOT SLO.
- No treatment token gap exceeds one second.
- Server shutdown completed cleanly and all eight GPUs returned to zero allocated memory.
- The failed CUDA 11.5 startup produced no benchmark samples and is excluded from results.

## Next step

Experiment 016 is the main experiment. Compare the original/default scheduling profile, the
fairness-only policy, and the final 2304-token optimized profile on Qwen3-32B TP=4 across a broader
fixed load matrix. Report at least three repetitions, tail TTFT/TPOT, throughput, power, and SLO
attainment. Experiment 017 will then perform ablation and robustness checks.
