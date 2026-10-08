# Experiment 017: light-load failure analysis and global prefill bound

## Status

Complete. The diagnostic reproduced the failure and rejected the global-budget hypothesis. The
global 2304-token bound is not accepted as a corrective change.

## Objective

Explain the two one-second TPOT violations left by experiment 016, then test whether applying the
model-derived 2304-token limit to every prefill batch closes that coverage gap without worsening
the measured overload-throughput tradeoff.

## Hypothesis

The active-only policy can select the normal 8192-token budget during a scheduler transition even
while requests are externally observed to be decoding. A globally bounded 2304-token policy should
remove that state-dependent gap. At 2.5 requests/s, decode is nearly always runnable, so the global
and active-only policies should behave similarly and have comparable throughput.

## Principle

Experiment 016 limits prefill to 2304 tokens only when the internal decode queue is runnable; the
decode-idle budget remains 8192. Its two failures share one 1339 ms token gap in the first light-load
trace. Rank-0 prefill telemetry records the selected budget, admitted tokens, active decode count,
chunk state, and CUDA execution time, allowing that gap to be attributed to an actual scheduler
transition rather than inferred from request timestamps.

The treatment sets the normal budget to 2304 and lets the active budget reuse it. Thus every
prefill selection has the same bound while `max_prefill_streak=1` remains unchanged.

## Fixed configuration

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3
- Attention backend: FlashInfer selected by `auto`
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Requests: 24 per trace, balanced 256/1024/4096-token prompts
- Output length: 256 tokens
- Maximum prefill streak: 1
- SLO: per-request maximum TPOT <= 1000 ms
- Light-load traces: 0.5 requests/s, seeds 1600042/1610042/1620042
- Overload traces: 2.5 requests/s, seeds 1900042/1910042/1920042

## Profiles

| Profile | Normal budget | Decode-active budget | Source |
| --- | ---: | ---: | --- |
| Active-only control | 8192 | 2304 | Experiment 016 |
| Global bound | 2304 | 0 (reuse normal) | Experiment 017 |

## Procedure

1. Replay light-load seed 1600042 with the active-only profile and rank-0 timing telemetry.
2. Correlate unsafe prefill batches with the external token-gap interval.
3. Run the global-bound profile for all three original 0.5 requests/s traces.
4. Run it separately for the three original 2.5 requests/s traces. The separate command uses seed
   1900042 so the benchmark's rate-index offset does not change experiment 016's trace seeds.
5. Compare paired SLO, TTFT, throughput, and prefill execution behavior.

## Pre-registered decision criteria

The global bound is a corrective candidate if:

1. all 72 light-load requests satisfy the one-second TPOT SLO;
2. no measured global-bound token gap exceeds one second;
3. its 2.5 requests/s mean output throughput is no more than 1% below the experiment 016
   active-only profile; and
4. all six boundary traces complete without correctness or server failures.

Passing these boundary checks is not a new final deployment claim. A final claim requires a fresh
held-out workload after the mechanism is selected.

## Commands

The diagnostic server used the experiment 016 active-only profile plus rank-0 telemetry:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp-size 4 --memory-ratio 0.8 --cuda-graph-max-bs 64 \
  --max-running-requests 128 --max-prefill-length 8192 \
  --max-prefill-streak 1 --decode-active-prefill-length 2304 \
  --prefill-telemetry-path learning/results/017_replay_telemetry.jsonl
```

The treatment server changed only the effective budget policy:

```bash
--max-prefill-length 2304 \
--max-prefill-streak 1 \
--decode-active-prefill-length 0
```

The client used `seed=1600042` for the 0.5 requests/s traces and a separate invocation with
`seed=1900042` for the 2.5 requests/s traces. This exactly reproduces the seeds generated at those
rate positions in experiment 016.

## Diagnostic replay

The first 0.5 requests/s trace reproduced the failure:

| Metric | Experiment 016 | Telemetry replay |
| --- | ---: | ---: |
| Output throughput | 86.26 token/s | 86.25 token/s |
| Maximum TPOT | 1339.39 ms | 1339.05 ms |
| Requests meeting TPOT SLO | 22/24 | 22/24 |
| Token gaps above one second | 2 | 2 |

The two unsafe gaps belong to 256-token-input requests scheduled at 27.8548 and 32.7494 seconds.
Both stop receiving tokens at approximately 34.415 seconds and resume at 35.754 seconds. A
4096-token request arrives at 34.1719 seconds and a 1024-token request at 34.3379 seconds.

The ordered telemetry segment for that arrival cluster contains:

| Selected budget | Pending tokens | Prefill requests | Admitted tokens | Active decode | CUDA execution |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2304 | 4104 | 1 | 2304 | 2 | 721.84 ms |
| 2304 | 2829 | 2 | 2304 | 2 | 740.49 ms |
| 2304 | 522 | 1 | 522 | 3 | 175.24 ms |

The first two rows cover the 4096-token request's chunks and part of the following 1024-token
request. Crucially, the scheduler already selected the active 2304-token budget while decode was
runnable. The unsafe interval is therefore not caused by falling back to the 8192-token default.

The trace does contain 8192-budget batches with zero active decode requests, including a
4101-token/1328.38 ms batch before the failing arrival cluster and a 4101-token/1326.23 ms batch
near the end. Neither aligns with the reproduced unsafe token interval.

## Boundary results

Aggregate results compare the global treatment with the matching active-only experiment 016
traces:

| Rate | Profile | Output tok/s | Average TTFT ms | Max TPOT ms | TPOT SLO | Gaps >1 s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | Active-only | 99.92 | 745.43 | 1339.39 | 97.2% | 2 total |
| 0.5 | Global 2304 | 99.93 | 751.84 | 1340.15 | 97.2% | 2 total |
| 2.5 | Active-only | 258.97 | 4865.86 | 789.03 | 100.0% | 0 |
| 2.5 | Global 2304 | 258.87 | 4870.97 | 803.59 | 100.0% | 0 |

Paired changes from active-only to global 2304:

| Rate | Mean output throughput | Average TTFT |
| ---: | ---: | ---: |
| 0.5 | +0.007% | +0.86% |
| 2.5 | -0.038% | +0.11% |

The three paired overload throughput changes are -0.125%, +0.058%, and -0.049%. Global bounding
therefore adds no meaningful overload cost, but also provides no measurable benefit. At light load,
the exact failing trace remains unsafe while the other two traces remain safe, matching the
active-only control request for request.

Raw reports:

- `learning/experiments/017_replay_raw.md`
- `learning/experiments/017_global_0p5_raw.md`
- `learning/experiments/017_global_2p5_raw.md`

Machine-readable results and timing telemetry use the corresponding `017_*` names under
`learning/results/`.

## Interpretation

The hypothesis is rejected. The active-only policy did not miss the failing cluster, and applying
2304 globally cannot fix a gap that already occurs under the 2304 active budget. The external
1339 ms gap spans an arrival cluster whose ordered telemetry has multiple bounded prefill chunks.
Although `max_prefill_streak=1` alternates the scheduler's selected batch types when decode is
runnable, this experiment does not yet expose when each decode result becomes visible to the
client relative to the overlapping prefill submissions.

This distinction matters: bounding one prefill CUDA execution to roughly 740 ms is insufficient if
one client-visible token interval can cover work from more than one bounded prefill submission.
The experiment 014 model remains accurate for individual batch execution time; its assumption that
one batch plus a 250 ms reserve bounds one external TPOT interval is what fails on this trace.

## Decision

Reject the global 2304 profile as a corrective optimization. It fails criteria 1 and 2, while
passing criteria 3 and 4:

- light-load SLO: 70/72 requests, not 72/72;
- maximum global-bound gap: 1340.15 ms, above one second;
- overload throughput change: -0.038%, within the 1% limit;
- all six boundary traces completed correctly.

Keep the active-only 2304 policy as the previously documented balanced profile, but retain the
strict-SLO rejection from experiment 016. Do not reduce the global normal budget: it changes
decode-idle behavior without addressing the observed mechanism.

## Validation

- The diagnostic replay matches experiment 016 within 0.01 token/s and 0.35 ms maximum TPOT.
- Telemetry records every prefill batch in the replay and is flushed in four idle snapshots.
- All six global-bound traces completed with the exact experiment 016 trace seeds and input orders.
- All 72 overload requests satisfy the TPOT SLO; 70 of 72 light-load requests satisfy it.
- All server processes exited cleanly and all GPUs returned to zero allocated memory.

## Next step

Experiment 018 should add a short-lived overlap-pipeline timeline diagnostic for batch selection,
GPU completion, decode-result processing, and token-result sending. Replay seed 1600042 to identify
why two scheduled prefill chunks cover one externally visible token interval. Only after that
timeline is known should the experiment compare a response-aware fairness rule or a smaller active
budget against the existing 2304 profile.
