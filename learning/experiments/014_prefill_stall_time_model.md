# Experiment 014: prefill execution-time calibration

## Status

Complete.

## Hypothesis

Pending tokens predict how much work is available, but not how long the next prefill batch will
stall decode. Direct per-batch CUDA execution times should expose a stable relationship between
admitted prefill tokens and stall duration on Qwen3-32B TP=4, allowing the scheduler budget to be
chosen from a latency objective rather than an indirect queue threshold.

## Principle

CUDA work is asynchronous, so host wall-clock timing around `forward_batch` is not a valid GPU
duration measurement. The instrumentation records CUDA events on the engine stream immediately
before and after the complete prefill forward path. The ending event is read only when that batch
has completed, preserving overlap scheduling and avoiding a synchronization in the submission
path.

Timing is opt-in with existing prefill telemetry, runs only on TP rank 0, and records the variables
needed to interpret execution cost: selected budget, admitted and pending tokens, prefill request
count, chunk count, cached tokens, maximum sequence length, active decode requests, and whether the
budget bound the batch. Samples are drained after each idle telemetry flush so long-running servers
do not retain an unbounded history.

## Experiment matrix

- Model: Qwen3-32B BF16, TP=4 on GPUs 0-3
- Workload: experiment 012 mixed 256/1024/4096-token prompts
- Output length: 256 tokens
- Arrival rates: 0.9 and 1.5 requests/s
- Repeats: 3
- Seed: 800042
- Prefill policy: static decode-active budget 4096
- Decode-idle prefill budget: 8192
- Maximum prefill streak: 1
- Comparison data: observed decode gaps from experiment 012 static 4096

Static 4096 intentionally supplies a broad range of batch sizes including the known unsafe region.
The experiment is diagnostic: its purpose is to fit and validate a service-time model, not to
re-establish the best deployment profile.

## Results

Six measured traces produced 105 prefill timing samples. Six are decode-idle first batches; the
stall-time model uses the remaining 99 decode-active batches.

Simple least-squares fit:

```text
prefill_execution_ms = 11.1947 + 0.315291 * admitted_tokens
```

Model quality:

| Metric | Value |
| --- | ---: |
| Active prefill samples | 99 |
| Pearson correlation | 0.999867 |
| R-squared | 0.999734 |
| Training RMSE | 8.85 ms |
| Maximum absolute residual | 26.63 ms |
| Leave-one-trace-out mean RMSE | 9.00 ms |
| Leave-one-trace-out maximum absolute error | 26.77 ms |

Adding prefill request count as a second feature reduces training RMSE to 6.14 ms, but admitted
tokens alone explain 99.97% of execution-time variance and are sufficient for an initial controller.
Active decode request count has only 0.089 correlation with the token-model residual.

Observed execution-time buckets:

| Admitted tokens | Samples | Average ms | P90 ms | Maximum ms |
| ---: | ---: | ---: | ---: | ---: |
| 1-512 | 26 | 80.78 | 93.49 | 94.30 |
| 513-1024 | 1 | 173.80 | 173.80 | 173.80 |
| 1025-2048 | 11 | 330.29 | 331.63 | 331.84 |
| 2049-3072 | 9 | 712.64 | 749.72 | 749.72 |
| 3073-4095 | 4 | 1160.28 | 1260.28 | 1260.28 |
| 4096 | 48 | 1303.67 | 1310.84 | 1311.59 |

The workload contains no samples between 2334 and 3368 admitted tokens, so the fitted line rather
than direct observations covers that interval. All 52 batches above one second admit at least 3368
tokens. No observed batch at or below 2334 tokens exceeds 750 ms.

CUDA execution time is not the entire externally visible TPOT gap. Across the six traces, maximum
GPU prefill time is 1308.72-1311.59 ms while maximum TPOT is 1334.79-1550.36 ms. The difference
ranges from 25.92 to 241.64 ms due to scheduling, adjacent work, and streaming overhead. Reserving
250 ms for this measured worst case leaves a 750 ms GPU execution target for a one-second TPOT SLO.

Model-derived candidates:

| Budget | Predicted GPU ms | Plus 250 ms reserve | Assessment |
| ---: | ---: | ---: | --- |
| 2048 | 656.91 | 906.91 | Known safe, conservative |
| 2304 | 737.62 | 987.62 | Largest practical candidate below SLO |
| 2560 | 818.34 | 1068.34 | Predicted SLO violation |
| 3072 | 979.77 | 1229.77 | Unsafe despite GPU time below one second |
| 4096 | 1302.63 | 1552.63 | Matches measured violations |

End-to-end metrics with timing enabled differ from experiment 012 static 4096 by at most 0.41% at
0.9 requests/s and 0.14% at 1.5 requests/s. Throughput changes by +0.014% and -0.016%, supporting
that asynchronous event timing has no measurable perturbation in these traces.

Raw benchmark report: `learning/experiments/014_static4096_timing_raw.md`.

Machine-readable output is `learning/results/014_static4096_timing.json`; all batch timing samples
are in `learning/results/014_static4096_timing_telemetry.jsonl`.

## Interpretation

The hypothesis is confirmed for this hardware and model profile. Admitted prefill tokens are a
direct and highly stable predictor of GPU stall time, unlike pending tokens, which only describe
available queue work. The measured slope corresponds to about 3172 admitted prefill tokens/s for
Qwen3-32B BF16 at TP=4 on L20.

The execution model also explains the earlier experiments mechanistically. A 4096-token prefill
takes about 1.30 seconds before host and scheduling overhead, so it cannot satisfy a one-second
maximum TPOT target. A 2048 budget is safe but leaves roughly 90 ms of the end-to-end latency budget
unused under the conservative 250 ms reserve. A 2304 budget uses that headroom and is the next
model-driven candidate.

The formula is not portable across models, tensor-parallel sizes, attention backends, or GPU types.
It should be treated as a calibrated L20/Qwen3-32B profile. The opt-in raw samples make recalibration
possible without embedding this specific slope into generic scheduler code.

## Decision

Keep the execution-time instrumentation as an opt-in diagnostic extension of prefill telemetry. It
adds no events or metadata scans when telemetry is disabled, records only on rank 0, and drains raw
samples after each idle flush.

Do not implement a runtime latency controller yet. First validate the model's recommended
2304-token static budget against 2048 using identical traces. If 2304 retains 100% TPOT SLO while
improving TTFT, the fitted service-time model can then become the basis for a target-driven budget
selector.

## Validation

- All 106 prefill batches, including warmup, produced exactly one execution sample.
- Samples are split across seven flushes: one warmup and six measured traces.
- Six trace schedules, seeds, and input-length orders match experiment 012 exactly.
- Opt-in timing reproduces the static-4096 throughput and tail metrics within run-to-run noise.
- Telemetry tests cover execution sample recording and draining.
- Eighteen focused scheduler and benchmark tests pass.
- The server exited cleanly and all eight GPUs returned to zero allocated memory.

## Next step

Experiment 015 should compare a static decode-active budget of 2304 against the existing 2048
baseline on the identical mixed-length traces. This directly tests the model's predicted 987.62 ms
worst-case envelope before any automatic controller is introduced.
