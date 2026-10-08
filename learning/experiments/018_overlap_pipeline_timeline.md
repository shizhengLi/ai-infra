# Experiment 018: overlap-pipeline result visibility

## Status

Complete. The mechanism was confirmed and the opt-in treatment passed the replay and boundary
matrix. It remains disabled by default until a fresh held-out validation is complete.

## Objective

Identify why two decode requests experience one 1.34-second client-visible token gap even though
the scheduler alternates decode with 2304-token prefill batches whose individual CUDA execution is
approximately 0.74 seconds.

## Hypothesis

The overlap loop schedules and calls `_forward` for the next batch before processing the previous
batch's result. If the next prefill `_forward` call does not return promptly, a completed decode
result can remain unsent while the following prefill executes. In that case, batch-order fairness
does not imply response-visible fairness.

## Instrumentation principle

When existing prefill telemetry is enabled on TP rank 0, record these events with
`time.perf_counter_ns()`:

- request received;
- batch selected;
- `_forward` start and return;
- previous batch result-processing start;
- output copy ready;
- tokenizer result sent.

Each batch receives a monotonically increasing diagnostic ID and includes its phase, size, and
request IDs. The benchmark stores its `perf_counter` trace origin, placing server events and client
token timestamps in the same Linux monotonic-clock domain. Events are buffered only in the opt-in
diagnostic path and drained at each idle telemetry flush; the default serving path performs none of
this work.

## Fixed replay

- Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- FlashInfer attention, memory ratio 0.8, CUDA Graph max batch 64
- Normal/active prefill budgets: 8192/2304
- Maximum prefill streak: 1
- 24 balanced 256/1024/4096-token inputs, 256 output tokens
- Arrival rate: 0.5 requests/s
- Trace seed: 1600042

## Procedure

1. Run focused unit tests and verify instrumentation is absent when telemetry is disabled.
2. Replay the exact failing trace with pipeline and CUDA timing enabled.
3. Align the two client gaps with server batch events and measure where the missing time accrues.
4. Select a corrective mechanism only from the observed timeline.
5. Replay the failing trace with that mechanism, then measure three original light-load and three
   overload traces if the targeted replay succeeds.

## Decision rule

A mechanism can proceed to the boundary matrix only if the exact failing replay has no token gap
above one second. It is accepted from this experiment only if all 72 light-load requests satisfy
the TPOT SLO and mean 2.5 requests/s throughput is no more than 1% below experiment 016's
active-only result.

## Pre-treatment diagnosis

The unmodified replay reproduced the 1338.63 ms maximum TPOT. The unsafe client interval is
34.427-35.765 seconds relative to the benchmark trace origin. The decisive server sequence is:

| Relative time | Event |
| ---: | --- |
| 34.248 s | Start 2304-token prefill batch 990 |
| 34.424 s | `_forward` returns; previous decode result is sent |
| 34.426 s | Start decode batch 991 for the two affected requests |
| 34.505 s | Decode `_forward` returns |
| 34.997 s | Start the next 2304-token prefill batch 992 |
| 35.763 s | Prefill `_forward` returns; only now process decode batch 991 |
| 35.763 s | Send decode results for both affected requests |

CUDA timing reports 721.92 ms for prefill batch 990 and 740.43 ms for batch 992. The critical
delay is not decode GPU execution: decode batch 991 is submitted between them. Its result remains
unprocessed because `overlap_loop` calls `_forward` for batch 992 before `_process_last_data` for
batch 991, and the prefill `_forward` call blocks the host for approximately 765 ms.

This confirms the hypothesis. Batch-order fairness did schedule decode between the two prefill
batches, but host result processing made that decode invisible until after the second prefill.

## Targeted treatment

Add opt-in `--decode-result-before-prefill`. When the previous submitted batch is decode and prefill
work is pending, the overlap loop processes and sends that decode result before submitting the next
prefill batch. Other decode-only and prefill-only iterations retain the existing overlap order.

The treatment will first replay seed 1600042. It proceeds to the six-trace boundary matrix only if
that exact replay has no gap above one second.

## Targeted replay result

With `--decode-result-before-prefill`, the exact seed-1600042 replay changed as follows:

| Metric | Untreated | Treatment |
| --- | ---: | ---: |
| Output throughput | 86.2498 token/s | 86.2517 token/s |
| Maximum TPOT | 1338.63 ms | 768.32 ms |
| Requests satisfying TPOT SLO | 22/24 | 24/24 |
| Token gaps above one second | 2 | 0 |

The treatment timeline also verifies the intended order rather than only the final metric. Results
for the affected decode requests were sent at 35.021 seconds relative to the trace origin; prefill
batch 992 was selected immediately afterward and entered `_forward` at 35.022 seconds. That prefill
returned at 35.762 seconds. The decode response therefore became client-visible before the blocking
prefill call.

## Boundary matrix

The boundary matrix reused experiment 016's exact prompts and arrival schedules so the policy is
paired against its active-only control. Telemetry was disabled to exclude diagnostic overhead.

| Load | Seeds | Requests | Output throughput | Max TPOT | TPOT SLO | Gaps > 1 s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 req/s | 1600042, 1610042, 1620042 | 72 | 99.9199 token/s | 777.32 ms | 72/72 | 0 |
| 2.5 req/s | 1900042, 1910042, 1920042 | 72 | 258.8160 token/s | 814.53 ms | 72/72 | 0 |

At 0.5 requests/s, experiment 016's control reached 70/72 TPOT passes with a 1339.39 ms maximum.
The treatment reached 72/72 and reduced that maximum by 41.97%. Mean throughput changed by
+0.0012%, while average TTFT increased by 1.05%.

At 2.5 requests/s, mean throughput changed from 258.9709 to 258.8160 token/s, a -0.060% change that
is inside the pre-registered 1% budget. Average TTFT increased by 0.21%. The per-trace throughput
changes were -0.094%, +0.010%, and -0.097%, so the aggregate is not hiding a large single-trace
regression.

## Interpretation

Maximum-prefill-streak fairness constrains batch selection but not when an already-submitted decode
result becomes visible to the client. In the overlap pipeline, result processing for batch N normally
occurs after `_forward` submits batch N+1. This is beneficial overlap when N+1 returns quickly, but a
long prefill can hold a completed decode response behind a second prefill-sized host wait.

The treatment introduces a narrow barrier only at the risky transition: previous batch is decode and
pending work includes prefill. It does not synchronize every batch and does not change prefill token
budgets. The six paired traces show that this removes the response-visibility gap without a measurable
throughput cost at light load and with only 0.060% aggregate cost at overload.

## Decision

Accept `--decode-result-before-prefill` as the experiment 018 candidate. Keep it opt-in and disabled
by default. The exact replay and all 144 boundary requests pass the TPOT criterion, and overload
throughput regression is below the 1% limit.

This is not yet the final deployment claim because the treatment was derived from and evaluated on
the experiment 016 trace family. Experiment 019 must test fresh, pre-registered seeds across the full
0.5/0.9/1.5/2.5 requests/s range.

## Artifacts

- Diagnostic replay: `learning/experiments/018_timeline_raw.md`
- Targeted replay: `learning/experiments/018_targeted_replay_raw.md`
- Light-load boundary: `learning/experiments/018_treatment_0p5_raw.md`
- Overload boundary: `learning/experiments/018_treatment_2p5_raw.md`
- Machine-readable results: `learning/results/018_*.json`
- Pipeline timelines: `learning/results/018_*timeline.jsonl`

## Validation

- Final scheduler-core and Poisson-trace regression suite: 28 passed.
- The treatment reports `server_decode_result_before_prefill=true` in every benchmark artifact.
- The six boundary traces have the same seeds, prompts, and scheduled arrival offsets as their
  experiment 016 controls.

## Next experiment

Experiment 019 is a held-out confirmation. Compare active-only 2304 with and without the response
barrier on three fresh traces at each of 0.5, 0.9, 1.5, and 2.5 requests/s. Require 100% TPOT SLO,
no token gap above one second, and no more than 1% mean throughput regression at every rate before
promoting the option into the recommended L20 deployment profile.
