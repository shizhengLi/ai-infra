# L20 optimization progress

## Current status

| Phase | Status | Result |
| --- | --- | --- |
| Environment | Complete | 8 x L20 detected; PyTorch 2.9.1+cu128; CUDA 12.8 JIT verified |
| Reproducible harness | Complete | Deterministic workload; Markdown/JSON output; 3-run statistics |
| CUDA Graph sizing | Complete | 23 -> 19 graphs; initialization 8.3615 -> 7.3553 s |
| CUDA Graph value | Complete | Enabled graph improves median throughput by 30.97% vs eager |
| Attention backend sweep | Complete | FA3 compatibility restored; `fa,fi` -0.17%, pure `fa` -15.00% vs `fi` median |
| KV/radix experiments | Concurrent adaptive reserve accepted | Batched calls 41.67 -> 8.33; reclamation -1.73%; throughput -0.036%; paired matches preserved |
| Online serving | Complete | TP=4 peaks at 503.35 token/s; latency knee between C=8 and C=32 |
| Prefill budget | Complete | 32768 cuts P99.9 TPOT 46.82% but regresses average TTFT 23.99% |
| TP 2/4/8 | Complete | TP=4 is efficiency optimum; TP=8 is 22% faster at 57% more total power |
| Scheduling fairness | Complete | Prefill streak 1 cuts worst decode stall 62% at 0.13% throughput cost |
| Adaptive prefill | Complete | Active budget 4096 halves worst TPOT with no measured throughput loss |
| Poisson budget sweep | Complete | 2048 reaches 100% TPOT SLO at 2.5 req/s with 0.12% throughput cost |
| Queue-adaptive prefill | Complete | 3 overload interventions restore 100% TPOT SLO with <0.01% throughput change |
| Mixed-length pressure | Complete | Static 2048 alone reaches 100% TPOT SLO; adaptive threshold 4096 triggers too late |
| Overload threshold 2048 | Complete | Restores 100% TPOT SLO but is behaviorally equivalent to static 2048 |
| Prefill stall-time model | Complete | 99 samples fit `11.19 + 0.31529 * tokens` ms with R-squared 0.99973 |
| Model-derived budget | Complete | 2304 keeps 100% TPOT SLO and improves average TTFT 6.10%/1.85% vs 2048 |
| Main scheduling experiment | Complete, criteria failed | 286/288 SLO passes; max TPOT 13.32s -> 1.34s, but light-load SLO and 2.5 req/s throughput criteria fail |
| Global prefill bound | Complete, rejected | Reproduces 1.34s gap under 2304 active budget; global 2304 does not improve 70/72 light-load SLO |
| Decode response visibility | Complete, accepted | Fresh held-out matrix passes 288/288 requests; max TPOT 1.12s -> 0.79s with at most 0.125% per-rate throughput cost |
| PyNCCL environment | Complete | Wheel `libnccl.so.2` linked with rpath; TP=4 API reached ready |
| Graph-tail prefill guard | Complete, rejected | Ordering changed as intended, but C=32 P90 TPOT regressed 28.69% |
| Graph-tail non-blocking priority | Complete, rejected | Long-output validation: C=32 P90 TPOT +0.07%, P99.9 -9.68%, throughput -0.32% |

## Decisions

- Start with Qwen3-0.6B because it makes iteration cheap; do not generalize its absolute throughput
  results to production-size models.
- Use offline fixed-shape decode first to isolate engine behavior.
- Keep generated benchmark facts separate from interpretation notes.
- Do not accept parameter sweeps as source optimizations unless they lead to a robust automatic
  policy or a documented deployment profile.
- Keep SM89 automatic attention on `fi`: the repaired `fa,fi` path is effectively tied and pure
  `fa` is 15.00% slower on the fixed 256-input/128-output workload.
- Recommend `--max-prefill-streak 1 --decode-active-prefill-length 2304
  --decode-result-before-prefill` for the calibrated L20/Qwen3-32B TP=4 profile. Experiment 019's
  fresh matrix passed 288/288 requests with at most 0.125% per-rate throughput regression. Keep the
  generic default disabled because this policy is model, hardware, and workload dependent.
- Accept opt-in rank-0 Radix Cache telemetry. Shared-system reuse matches 776/1,036 tokens after the
  cold request and lowers post-first TTFT by 69.21% versus unique prompts. The baseline is far below
  cache capacity, so no eviction-policy claim is justified yet.
- Keep LRU ordering: under a controlled 4096-token pool it retained 21/21 expected useful prefixes
  across three seeds and evicted older C/D first. Target node granularity instead: pressure eviction
  reclaimed 3.061x the requested tokens because whole 532-token leaves were removed.
- Retain page-aligned partial-leaf eviction behind its default-disabled flag. It passed all primary
  experiment 022 rules: 21/21 hot survivors, 1.000x amplification, 32.35% less pressure reclamation,
  and pooled D/C matches increasing from 18 to 1,048 tokens. Do not recommend it for deployment yet:
  exact trimming caused 32 pressure eviction calls per repetition and regressed hot TTFT by 16.23%.
- For the fixed Qwen3-32B TP=4, 4096-page, 16-output-token profile, pair partial eviction with a
  16-page reserve. Experiment 023 reduced pressure calls from 32 to 2, kept target overshoot at
  1.000x, retained 21/21 hot probes and 348.3/519 mean D tokens, and limited hot TTFT change to
  +3.47% versus whole-leaf control. Keep both defaults disabled until variable-length validation.
- Accept output-aware adaptive reserve for production-shaped follow-up. Experiment 024 passed every
  pre-registered mixed-length rule across three seeds: complete-sequence calls fell 82.35%, actual
  reclamation fell 8.34%, all 21 hot probes remained 519/519 hits, throughput changed -0.078%, and
  hot TTFT improved 4.23%. Keep the generic default disabled because concurrency and the 128-page
  cap have not yet been validated under realistic arrivals.
- Experiment 025 validated the same adaptive cap under concurrent phase batches across three seeds.
  Mean complete-sequence eviction calls fell 80.0% (41.67 -> 8.33), actual reclamation fell 1.73%,
  throughput changed -0.036%, survivor TTFT improved 1.64%, and all 21 paired survivor matches were
  preserved. The adaptive sum did not over-reserve in this workload; keep it opt-in pending open-loop
  arrival and cap-sensitivity testing.
- Experiment 026 rejected the raw adaptive sum for open-loop arrivals. At seed 3100042, caps 64/128/256
  reduced eviction calls 70.0%/81.7%/85.0% but each lost at least one paired survivor prefix; cap256
  also raised survivor TTFT 78.8%. No cap entered Stage B. Keep adaptive reserve restricted to the
  phase-batched profile and investigate age/priority-aware reserve accounting.
- Experiment 027 added an opt-in age-aware reserve that discounts requests by remaining-output
  fraction. On the fixed open-loop trace, age-aware caps 64 and 128 reduced eviction calls 65.5%
  and 77.6%, with throughput and TTFT within limits, but cap64 lost A's survivor match and cap128
  lost B's. Both candidates are rejected; age alone is insufficient to identify safe eviction.
- Experiment 028 added opt-in recent-match protection around raw adaptive-64 eviction. Protection
  windows of 4 and 8 reduced eviction calls 69.0%, but hot4 lost G's survivor match and hot8 lost
  A's. A bounded recent-node list is therefore rejected as a safe open-loop hotness signal.
- Experiment 029 added decayed request-level hotness ranking. Decay 0.90 and 0.99 reduced eviction
  calls 70.2% and 66.7%, with throughput and TTFT within limits, but decay0.90 lost A and decay0.99
  lost G in the survivor probes. Decayed reuse frequency is rejected as a generic open-loop signal.
- Experiment 030 closed the adaptive-reserve policy branch with a deployment freeze gate. The
  phase-batched adaptive-max-128 profile from experiment 025 remains accepted only as an explicit
  opt-in scope; raw, age-aware, recent-match, and decayed-hotness open-loop variants are rejected
  because each loses a paired survivor. No reliable workload-level identity exists in the current
  request/cache API, so no new generic heuristic is enabled.

## Experiment 031 result

Experiment 031 audited the PyNCCL sizing premise and rejected the optimization: the active
  `SchedulerConfig.max_forward_len` already equals `max_extend_tokens=8192`, selecting 80 MiB/rank
  for Qwen3-32B BF16 rather than the 400 MiB context-window bound. A regression test should preserve
  this invariant; no GPU matrix or runtime change is justified.

## Experiment 032 result

Experiment 032 added NVTX attribution for TP collectives and measured the existing direct PyNCCL
path (`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`). Nsight showed TP=4 all-reduce kernels at about 41.9% of
cumulative GPU kernel time, clearing the communication bottleneck gate. In a paired three-repeat
Qwen3-32B TP=4 online matrix, direct communication improved throughput 9.54%/11.82%/11.34% at
concurrency 1/8/32 and reduced average TPOT 3.81%/7.00%/8.00%; P90 TPOT improved at all three
loads. Accept it as an explicit L20 TP=4 deployment profile, but keep the generic default unchanged
pending TP=8 online and CUDA Graph validation.

## Experiment 033 result

Experiment 033 validated the direct path at TP8 and with CUDA Graphs. With Graphs disabled, TP8
throughput improved 10.20%/5.52%/5.26% at concurrency 1/8/32 and P90 TPOT improved at all loads.
With TP4 Graphs enabled through batch 64, throughput improved at C=8/32 but P90 TPOT regressed
4.49%/9.69%, so the strict Graph-mode gate failed. Accept direct PyNCCL as an explicit graph-disabled
L20 TP4/TP8 profile; keep generic and Graph-mode defaults unchanged.

## Experiment 034 result

Experiment 034 added per-concurrency steady-state warmups and repeated the TP4 Graph comparison five
times. The direct path remained stable and improved throughput 1.34%/7.68%/13.45% at C=1/8/32,
but P90 TPOT changed -0.06%/+2.64%/+11.91%. The Graph tail regression therefore persists after
removing first-request noise. Keep direct PyNCCL explicit and graph-disabled; close the communication
branch until a new Graph scheduler hypothesis is available.

## Experiment 035 result

Experiment 035 added Graph telemetry for logical and padded decode batch sizes. Symmetric and direct
traces had the same 530 Graph batches; P90-tail activity concentrated in logical 20/24/28/31/32
transitions that replay padded batch 24/24/32/32/32. Scheduler completion intervals above 30 ms were
130 padded-32 samples in each treatment, with direct mode about 0.4 ms slower on average. Target
padded-32 scheduler transitions next; do not alter Graph padding or collective defaults yet.

## Experiment 036 result

Experiment 036 added a default-disabled, padded-batch-specific result-before-prefill guard and ran a
paired three-repeat TP4 Graph matrix. Telemetry confirmed all five target transitions moved pending
prefill from 147.13 ms before decode result delivery to 0.52 ms after it, and decode
`forward_return -> result_sent` fell from 320.15 ms to 206.14 ms. The synchronization boundary was
counterproductive: C=32 throughput changed -0.22%, P99.9 TPOT improved 9.54%, but P90 TPOT regressed
28.69% (210.55 -> 270.96 ms). Reject guard 32 and keep it disabled by default.

## Experiment 037 result

Experiment 037 replaced the synchronous guard with a one-shot, non-blocking decode priority at
padded-32/prefill boundaries. The corrected treatment added exactly five decode batches (596 -> 601)
without creating a decode-only loop. At C=32 it changed throughput -0.63%, P90 TPOT -0.41%, P99.9
TPOT -7.06%, and average TTFT +0.02%. Keep the option as an explicit calibrated profile, but do not
enable it by default until a second seed and longer-output validation confirm the small tail gain.

## Experiment 038 result

Experiment 038 repeated the priority profile with independent seed `3800042` and output length 128.
The treatment added exactly five decode batches (2,036 -> 2,041) and kept the same 610 padded-32
batches. At C=32, throughput changed -0.32%, P90 TPOT changed +0.07%, P99.9 TPOT improved 9.68%,
and P90 TTFT rose 1.13%. The P90 gate failed; reject the profile as a deployment optimization and
keep it only as a default-disabled research flag.

## Next experiment

Experiment 041 completed kernel/NVTX attribution for padded-32 Graph replay. In the filtered replay
interval, NCCL all-reduce consumed 52.0% of summed GPU kernel time and three BF16 GEMM classes another
46.0%; sampler/copy and capture-shape branches remain deprioritized. Experiment 042 should re-measure
direct PyNCCL versus symmetric PyNCCL under this Graph workload with a strict C=32 P90 gate. The
adaptive-reserve profile remains frozen and generic cache heuristics remain disabled.

## Experiment 039 result

Experiment 039 instrumented `Engine.forward_batch` and scheduler `copy_ready` events with opt-in CUDA
events. On Qwen3-32B TP=4 L20, the control matrix was 38.040/166.885/267.630 tok/s at C=1/8/32;
C=32 P90 TPOT was 211.305 ms. Across 596 timed batches, padded-32 Graph replay averaged 34.075 ms,
versus 0.020 ms for sampling and 0.007 ms for the sampled-token copy. The remaining tail is therefore
below the sampler and is most plausibly in Graph replay or batch-shape scheduling. No default behavior
was changed.

## Experiment 040 result

Experiment 040 added opt-in explicit Graph shapes 20 and 28. The paired Qwen3-32B TP4 matrix changed
C=32 throughput +0.06% and P90 TPOT -0.003%, while C=1/C=8 throughput changed -0.34%/-0.23%.
Telemetry confirmed the shape distribution moved from 130 padded-32 decode batches to 120, but the
remaining padded-32 model time was unchanged (34.073 -> 34.106 ms). Reject the shape list as a
deployment optimization; retain the CLI for future profiling.

## Experiment 041 result

Experiment 041 added an opt-in `MiniSGL.GraphReplay.bs=*` NVTX range and captured a padded-32 Graph
replay with Nsight Systems. The filtered interval contained 29.629 ms of summed GPU kernel time:
52.0% NCCL all-reduce, 25.1%/16.2%/4.7% three BF16 GEMM classes, and 1.4% norms/attention/activation
and copies. Keep defaults unchanged; use this measured communication residual to define experiment 042.
