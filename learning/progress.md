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
| TP 2/4/8 | Complete | TP=4 is efficiency optimum; TP=8 reaches 1.66-1.72x TP2 throughput at 41.5-43.0% efficiency |
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
| FP8 W8A8 | Complete, accepted candidate | TP4 throughput +47-55%; memory/card -15.48%; MMLU -0.70 point |
| INT4 AWQ-Marlin | Complete, quality gate failed | Qwen3-32B fits TP1; TP2 scales 1.60-1.75x; MMLU -1.23/-1.75 points |

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

## Sequence continuation

Experiments 039-043 moved from timing instrumentation to Graph kernel attribution, collective
validation, and a BF16 GEMM control. Experiment 043 rejected the reduction-mode change; the next
experiment is shape-level Graph replay timing and scheduler/kernel attribution. The adaptive-reserve
profile remains frozen and generic cache heuristics remain disabled.

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

## Experiment 042 result

Experiment 042 re-measured direct PyNCCL versus symmetric PyNCCL with CUDA Graphs enabled. Direct
communication improved throughput 1.17%/8.00%/13.39% at C=1/8/32, but C=32 P90 TPOT regressed
14.96% (210.150 -> 241.596 ms). Padded-32 model timing also rose 1.509 ms. Reject direct PyNCCL for
Graph mode; retain it only as an explicit graph-disabled profile.

## Experiment 043 result

Experiment 043 isolated PyTorch's BF16 reduced-precision reduction mode with the opt-in
`--disable-bf16-reduced-precision-reduction` flag. On Qwen3-32B TP4 with CUDA Graphs, treatment
throughput changed -0.37%/-0.44%/-0.21% at C=1/8/32 and P90 TPOT changed +0.17%/+0.19%/+6.73%.
It failed the strict C=32 tail gate and is rejected as an optimization. The flag remains available
only as a default-disabled diagnostic control; the next experiment is shape-level Graph replay
timing and scheduler/kernel attribution.

## Experiment 044 result

Experiment 044 grouped the fresh experiment-043 CUDA-event traces by padded Graph shape. Padded 24
and padded 32 were the only shapes above 30 ms: 20/20 and 130/130 control events. Padded 32 averaged
34.082 ms and represented logical batches 28/31/32; padded 24 averaged 32.541 ms and represented
logical batches 20/24. This confirms a shape-specific GPU cost, not sampler or host-copy overhead.
Do not change generic Graph padding or collective semantics; experiment 045 must target the padded
24/32 replay path and pass the C=32 P90 gate.

## Experiment 045 result

Experiment 045 tested a default-disabled eager fallback for padded-24/32 Graph batches. Treatment
changed C=32 throughput -0.11% and P90 TPOT +3.72% (221.709 -> 229.953 ms), so it failed the strict
tail gate. Keep `--cuda-graph-disable-bs` as a diagnostic switch only; Graph execution remains the
default. The next experiment should add Nsight markers around the padded-24/32 replay and scheduler
interval to identify a concrete kernel or synchronization target.

## Experiment 046 result

Experiment 046 added an opt-in scheduler NVTX range and captured a short Nsight Systems probe.
Padded-24/32 SchedulerForward ranges were about 59.6/59.4 ms, with nested GraphReplay about
56.4/56.2 ms. Kernel composition stayed NCCL 51.5% plus three BF16 GEMM classes about 46.7%; no
new shape-specific synchronization kernel appeared. Do not add another scheduler barrier. The next
experiment should isolate one NCCL or BF16 GEMM implementation change under the strict C=32 gate.

## Interview-oriented next steps

After reviewing `learning/grok.md`, the roadmap is narrowed to a defensible interview narrative:
baseline and Nsight attribution, accepted direct PyNCCL in graph-disabled mode, explicit rejection
of Graph-tail heuristics, then held-out multi-card scaling and communication tuning. Experiment 047
has now completed the TP2/TP4/TP8 Qwen3-32B scaling matrix with three repeats and scaling-efficiency
reporting. Experiment 049 has completed the open-loop arrival study; experiment 050 is the fresh
held-out final evaluation. Experiment 048 completed the PyNCCL buffer/protocol sweep. Defer a GEMM rewrite until `ncu`/CUTLASS tooling and a concrete kernel
candidate are available.

## Experiment 047 result

Experiment 047 completed the held-out TP scaling matrix for Qwen3-32B BF16 on L20. With CUDA Graphs
disabled and direct PyNCCL enabled, TP4 delivered 1.41-1.54x TP2 throughput at 70.7-76.8% efficiency
for C=1/8/32. TP8 delivered 1.66-1.72x at only 41.5-43.0% efficiency. The result makes TP4 the
best measured efficiency point and shows that TP8's extra aggregate throughput is increasingly paid
for by communication. TP2 used memory ratio 0.9 after a 0.7 KV-cache allocation failure; TP4 and TP8
used 0.7, so the result is directional and this limitation is recorded in
`experiments/047_tp_scaling.md`. The benchmark and telemetry artifacts are under `learning/results/`.

The online benchmark did not include kernel-level NCCL attribution. Experiment 046's Nsight result
(about 51.5% NCCL kernel time in the measured padded Graph replay) remains context, not a measurement
from this run. Experiment 048 then ran the one-variable direct-PyNCCL buffer/algorithm sweep with the
strict C=32 P90 TPOT and correctness gates; its result is recorded below.

## Experiment 048 result

Experiment 048 swept PyNCCL buffer thresholds and NCCL protocol/algorithm choices. The direct path
(`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`) had the best broad microbenchmark behavior; non-zero internal
buffers add device-to-device copies and were slower for most decode-sized messages. `NCCL_ALGO=Tree`
and `NCCL_PROTO=Simple` were also poor microbenchmark choices. `NCCL_PROTO=LL128` improved the
80-320 KiB microbenchmark points, but the online Qwen3-32B TP4 matrix changed throughput
`+0.76%/-3.56%/-0.87%` and P90 TPOT `-0.80%/+1.50%/+2.18%` at C=1/8/32; it failed the C=8/C=32
gate and increased C=32 P99.9 TPOT by 2.49%. Keep direct buffer=0 as the explicit graph-disabled
profile and leave generic NCCL defaults unchanged. Experiment 049 then ran open-loop arrival rates
with the accepted TP4 profile; its result is recorded below.

## Experiment 049 result

Experiment 049 used seeded Poisson/open-loop arrivals with a balanced 256/1024/4096-token prompt
mix and 128 generated tokens. The accepted TP4 Direct PyNCCL graph-disabled profile kept TPOT-SLO
attainment at 100% for 0.5/0.9/1.5/2.5 requests/s, but TTFT-SLO attainment fell from 94.4% and
91.7% at 0.5/0.9 req/s to 59.7% and 44.4% at 1.5/2.5 req/s. The rate increase exposes a prefill
queueing knee rather than a decode token-progress failure. Keep the profile unchanged and use
roughly 0.9 requests/s as the measured operating point when a 2-second TTFT SLO is required. The
next experiment is 050, a fresh-seed held-out confirmation with longer output and both unique and
shared-prefix requests.

## Experiment 050 result

Experiment 050 completed the fresh-seed final validation with 128-token outputs. Unique-prefix runs
achieved 35.87 tok/s, 288.18 ms average TTFT, and 25.83 ms average TPOT. Shared-system-prefix runs
achieved 37.23 tok/s, 120.56 ms average TTFT, and 26.12 ms average TPOT. The first shared request
had 407 ms TTFT; the following eleven were 94-97 ms, confirming Radix Cache reuse. All 24 requests
returned the full output with no server or integrity errors. Accept the explicit TP4 L20 profile for
the scoped Qwen3-32B workload: Direct PyNCCL, CUDA Graph disabled, prefill streak 1, active prefill
budget 1024, and maximum prefill 2048. Keep generic defaults unchanged and close the planned
optimization loop; a future experiment needs a new bottleneck hypothesis.

## Experiment 051 result

Experiment 051 opened the low-precision phase with a cold-cache GEMM screen over all four
Qwen3-32B TP4 local projection shapes at M=1/8/32. FP8 W8A8, INT8 W8A8, and GPTQ-Marlin INT4 all
executed on L20 SM89; maximum kernel relative-L2 error was 0.501% and maximum timing coefficient of
variation was 0.46%. FP8 including dynamic activation quantization improved all 12 points by
1.47-2.07x, INT8 by 1.09-1.68x, and INT4 W4A16 by 2.16-3.99x. FP8/INT8 halve linear weight storage;
INT4 quarters it. Select FP8 for experiment 052 because it produced a broad measured gain without
requiring a calibrated GPTQ/AWQ checkpoint. Do not claim INT4 model quality from this screen: naive
channelwise RTN had 14.49-15.78% synthetic output relative-L2 error. Generic defaults remain
unchanged.

## Experiment 052 result

Experiment 052 added default-disabled FP8 configuration, per-output weight scales, dynamic per-token
activation quantization, a shared linear dispatch, and TP-aware scale loading. It supports converting
BF16 checkpoint shards during load or directly consuming FP8 `weight` plus `weight_scale`; row/O
projection wrappers retain their all-reduce after the FP8 GEMM. Embedding, LM head, KV cache, and
collectives remain BF16, and MoE is explicitly unsupported. The focused suite passed 10/10 and the
core/misc/benchmark selection passed 78/78. A Qwen3-32B TP4 meta build had the expected 256 FP8
weight/scale pairs. Real Qwen3-0.6B BF16 and FP8 smoke runs both returned `[3, 4, 3, 4]` for the fixed
request. Keep FP8 disabled by default and proceed to experiment 053 for paired quality, memory,
CUDA-Graph, TTFT, TPOT, and throughput evaluation.

## Experiment 053 result

Experiment 053 completed the paired FP8 end-to-end gate. The final Qwen3-32B TP4 Graph matrix
improved throughput by 55.00%/49.48%/47.34% at C=1/8/32 and average TPOT by
37.05%/35.65%/34.69%. Sampled peak memory fell from 18,229 to 15,407 MiB/card. The final
570-question MMLU comparison scored 80.35% BF16 versus 79.65% FP8 with 100% parse rates, a
0.70-point drop inside the one-point gate. Stage A showed that FP8 eager regresses small-model
throughput by 19-22%, whereas CUDA Graph converts it into a 10-21% gain; the accepted result is
therefore Graph-scoped. FP8 remains default-disabled pending experiment 055's fresh-seed mixed-length
and open-loop validation. Experiment 054 subsequently addressed calibrated INT4 integration.

## Experiment 054 result

Experiment 054 added default-disabled calibrated AWQ-Marlin support: metadata validation, packed
INT4 TP sharding and fused-projection merging, AWQ-to-Marlin weight/scale/zero-point conversion,
persistent workspaces, and shared linear dispatch. A dtype mismatch found by the first Qwen3-4B
Graph run was fixed by converting floating scales to the selected runtime dtype while preserving
packed INT32 weights and zero points. The final core/misc/benchmark selection passed 84/84 tests.

Qwen3-32B-AWQ now runs on one L20 at 34.69/169.66/197.74 tok/s and on two L20s at
60.64/278.69/316.09 tok/s for C=1/8/32. TP2 scaling efficiency is 87.41%/82.13%/79.92%; sampled
total memory is roughly 21-23 GiB, enabling a large reduction in GPU count relative to TP4 FP8.
The one-point MMLU gate failed: official checkpoint-default FP16 AWQ dropped 1.23 points and the
BF16-controlled run dropped 1.75 points from BF16. Accept the implementation and capacity result,
but keep AWQ default-disabled and treat it as a conditional low-memory candidate in experiment 055.

## Experiment 055 result

Experiment 055 completed the held-out quantization comparison with a fresh seed (`5500042`), three
repeats at 0.5/1.0/1.5 requests/s, and balanced 256/1024/4096-token prompts. FP8 TP4 beat BF16
TP4 on P90 TTFT at every rate (1099/1553/1352 ms versus 1534/3355/2792 ms), average TPOT
(19.71/27.16/25.92 ms versus 34.00/42.22/41.26 ms), and sampled per-card memory (16453 versus
19109 MiB). TTFT-SLO attainment was 100/97.2/97.2% for FP8 and 97.2/66.7/72.2% for BF16; both
achieved 100% TPOT-SLO at all rates.

AWQ TP1 used about 24.3 GiB on one card but achieved only 61.1/27.8/22.2% TTFT-SLO attainment;
4096-token P90 TTFT reached 4.81/10.37/10.56 seconds. It remains a capacity-first option and is
not a default serving choice. Fresh MMLU was BF16 467/570 (81.93%), FP8 468/570 (82.11%), and AWQ
469/570 (82.28%), all with 100% parse rate. These results show no new held-out regression but do
not erase experiment 054's AWQ quality risk from a different seed. Accept FP8 as the current
Graph/TP4 deployment candidate, keep BF16 as reference, and leave global defaults unchanged.

## Experiment 056 result

Experiment 056 swept static FP8 TP4 prefill/decode parameters under the same held-out mixed-length
open-loop workload. Five configurations produced 45 traces and 540 complete requests with no
errors; TPOT-SLO attainment was 100% throughout. The accepted 4096/1/1024 baseline reached
153.81 +/- 8.59 tok/s and 2489.72 ms P90 TTFT at 1.5 req/s, with 80.6% TTFT-SLO attainment.
Reducing `max_prefill_length` to 2048 made no material change. Reducing decode-active budget to 512
increased prefill batches from 232 to 372--380 and reduced throughput to about 148.6--148.7 tok/s,
without improving TTFT-SLO. `max_prefill_streak=2` raised throughput slightly to 154.68 tok/s but
increased P99.9 TPOT from 244.49 to 447.27 ms.

Telemetry reported the same maximum pending prefill pressure, 8472 tokens, for every configuration.
Reject all static alternatives and keep the experiment055 baseline. The next scheduler change
should be a pending-pressure adaptive budget, not another broad static sweep.
