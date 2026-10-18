# Mini-SGLang L20 optimization plan

## Goal

Build measurable, reviewable Mini-SGLang improvements for NVIDIA L20 (Ada, SM89), progressing from
single-GPU engine overhead to realistic online serving and finally PCIe tensor parallelism.

## Measurement rules

1. Change one primary variable per comparison.
2. Pin the model, input/output distribution, random seed, dtype, backend, and request concurrency.
3. Separate initialization/JIT/CUDA Graph capture time from steady-state generation time.
4. Run at least three steady-state repetitions. Report mean, median, and dispersion.
5. Use unique random prompts for throughput runs; use explicit shared prefixes only in radix tests.
6. Record failed and neutral experiments. A plausible explanation is not a measured improvement.
7. Keep CUDA 12.8 selected because the installed PyTorch build is `cu128`.

## L20-specific reasoning

L20 is SM89 with 48 GB GDDR6 and a large L2 cache, but it has no NVLink. Single-GPU decode is
usually memory-bandwidth and launch-overhead sensitive. CUDA Graph coverage, decode batch padding,
attention backend selection, and KV-cache sizing are therefore the first targets. At TP > 1, PCIe
collectives are expected to become a first-order cost, so scaling efficiency matters more than raw
multi-GPU throughput.

## Roadmap

### Phase 0: reproducibility and baseline

- Validate CUDA/PyTorch/FlashInfer toolchain.
- Add a deterministic offline harness that emits Markdown and JSON.
- Establish Qwen3-0.6B single-L20 baseline for fast iteration.

### Phase 1: CUDA Graph and attention backend

- Ensure requested/automatic graph sizes never exceed runnable concurrency.
- Include the exact requested max graph batch size instead of silently rounding down.
- Compare `fi`, `fa`, and graph-disabled execution across decode batch sizes.
- Measure both startup cost and output throughput.

### Phase 2: KV cache and prefill

- Sweep `memory_ratio`, `page_size`, and `max_extend_tokens` on a larger model.
- Add explicit radix hit-rate/eviction telemetry before changing eviction policy.
- Benchmark no-shared-prefix, shared-system-prompt, and multi-turn workloads separately.

### Phase 3: online scheduling

- Capture TTFT, TPOT, P90/P99, request throughput, and output throughput.
- Sweep request arrival rate and concurrency.
- Profile overlap scheduling before changing batch assembly or priorities.

### Phase 4: 8-L20 tensor parallelism

- Compare TP 1/2/4/8 using a model that requires or benefits from multiple GPUs.
- Report scaling efficiency and NCCL/PyNCCL time on the PCIe topology.
- Optimize communication only after an Nsight Systems trace attributes the bottleneck.

### Phase 5: low-precision inference

- Screen FP8 W8A8, INT8 W8A8, and INT4 weight-only kernels on Qwen3-32B TP4 layer shapes.
- Separate quantization error from kernel correctness error and activation-quantization overhead.
- Add quantized weight metadata, scale loading, TP sharding, and `QuantLinear` dispatch only after
  the kernel screen identifies a viable path.
- Validate end-to-end accuracy, memory, TTFT, TPOT, throughput, and GPUs per replica before making a
  deployment claim. Weight-only quantization does not directly reduce BF16 collective message size.

## Current execution point

本节及其后早期实验段落保留为历史决策记录；截至实验 083 的有效状态、默认策略和 084-088
后续计划以文档末尾的“2026-10 当前执行总览”为准。

Experiments 000-015 completed environment validation, baseline measurement, focused optimization,
parameter exploration, and candidate selection. Experiment 016 completed the main held-out
evaluation. The active-only 2304-token candidate improved the global maximum TPOT from 13.32 to
1.34 seconds and satisfied 286 of 288 requests, but failed the strict acceptance criteria at light
load and regressed overload throughput by 1.14%.

- Follow-up experiment 002a removed the FA3/CUTLASS compatibility blocker and completed the L20
  backend sweep. `fi` remains the default based on measurement rather than backend availability.
- Experiment 016 is the completed main experiment; its failed criteria are retained as results.
- Experiment 017 reproduced the remaining 1.34-second gap with timing telemetry and rejected the
  global 2304-token bound: the failing cluster already used multiple 2304-token active batches.
- Experiment 018 proved the remaining gap was a response-visibility issue in the overlap pipeline:
  a completed decode result waited behind the following blocking prefill `_forward`. The opt-in
  result-before-prefill barrier reduced maximum TPOT from 1.34 seconds to 0.78 seconds, passed all
  144 boundary requests, and changed overload throughput by -0.060%.
- Experiment 019 completed the fresh-seed held-out matrix across 0.5/0.9/1.5/2.5 requests/s. The
  response barrier passed all 288 treatment requests, reduced held-out maximum TPOT from 1.12 to
  0.79 seconds, and limited per-rate throughput regression to 0.125%. It is now part of the scoped
  L20/Qwen3-32B TP=4 recommendation while remaining disabled by default.
- Experiment 020 added opt-in rank-0 Radix Cache telemetry and completed the larger-model baseline.
  Unique prompts hit 0.291% of matchable tokens, shared-system prompts hit 68.670%, and real
  multi-turn conversations hit 60.619%. Shared-system post-first TTFT fell 69.21% versus unique
  prompts. No eviction occurred because the largest resident set was only 12,799 of 305,066 tokens.
- Experiment 021 forced eviction with a 4096-token pool across three fresh seeds. LRU ordering
  retained all 21/21 pooled useful probes and consistently evicted older C/D first, but whole-leaf
  eviction reclaimed 3.061x the requested pressure tokens. LRU replacement is rejected; compressed
  node granularity is the measured target.
- Experiment 022 implemented default-disabled, page-aligned partial-leaf tail eviction. Across three
  fresh seeds it retained 21/21 hot probes, reduced pressure amplification from 3.061x to 1.000x,
  lowered reclaimed tokens by 32.35%, and raised pooled D/C matched tokens from 18 to 1,048. Exact
  trimming also caused 32 pressure eviction calls per run and increased hot TTFT by 16.23%, so the
  primitive is accepted but not added to the deployment profile.
- Experiment 023 screened 16/32/64-page eviction reserves and selected 16 pages by the pre-registered
  rule. Across three seeds it reduced pressure calls from 32 to 2, kept reclamation within 0.13% of
  exact trimming, retained 21/21 hot probes, and limited hot TTFT to +3.47% versus whole-leaf
  control. Reserve 16 is accepted only for the fixed 16-output-token experiment profile.
- Experiment 024 implemented a bounded output-aware reserve and validated it at uniform 16/64/128
  output lengths plus a three-seed mixed-length workload. In the mixed confirmation it retained all
  21/21 complete hot-prefix matches, reduced mean complete-sequence eviction calls from 34 to 6 and
  reclamation by 8.34%, changed throughput by -0.078%, and improved hot TTFT by 4.23%. The policy is
  accepted for production-shaped concurrency testing but remains disabled by default.
- Experiment 025 extended the validation to concurrent mixed-length groups. Across three seeds,
  adaptive-max-128 reduced mean eviction calls from 41.67 to 8.33 and reclamation by 1.73%, with
  -0.036% concurrent throughput change, -1.64% survivor TTFT change, and no paired survivor-match
  regression. It passes the concurrent acceptance rules but remains opt-in until open-loop arrival
  and cap-sensitivity tests are complete.
- Experiment 026 tested open-loop arrivals with adaptive caps 64/128/256. Every adaptive cap lost
  at least one paired survivor prefix despite exact target reclamation; cap256 also regressed
  survivor TTFT by 78.8%. The raw admitted-demand sum is therefore rejected for open-loop use, and
  the adaptive recommendation remains limited to phase-batched workloads.
- Experiment 027 tested age-aware reserve caps 64/128 on the same open-loop trace. Calls fell 65.5%
  and 77.6% with no throughput/TTFT regression, but cap64 lost A's survivor match and cap128 lost
  B's. Age alone is rejected as a safe eviction signal; the next candidate must account for prefix
  hotness or recent match history.
- Experiment 028 tested recent-match protection windows of 4 and 8 nodes around raw adaptive-64
  eviction. Both reduced calls by 69.0%, but hot4 lost G and hot8 lost A in the survivor probes.
  A bounded recent-node window is rejected; the next candidate needs decayed request-level reuse
  evidence across the whole open-loop trace.
- Experiment 029 tested decayed request-level hotness with decay 0.90 and 0.99. Calls fell 70.2%
  and 66.7%, but decay0.90 lost A and decay0.99 lost G. Decayed reuse frequency is rejected as a
  generic open-loop eviction signal.
- Experiment 030 closed the adaptive-reserve policy branch with a deployment freeze gate. The
  phase-batched adaptive-max-128 profile from experiment 025 is accepted only as an explicit
  opt-in scope. Raw, age-aware, recent-match, and decayed-hotness open-loop variants each lost a
  paired survivor, and the current request/cache API has no reliable workload-level identity to
  replace them. Keep all generic defaults disabled; reopen only with a new identity/admission API
  and a fresh three-seed open-loop validation.
- Experiment 031 audited the PyNCCL sizing premise and rejected the proposed change. The active
  `SchedulerConfig.max_forward_len` already follows `max_extend_tokens=8192`, selecting 80 MiB/rank
  for Qwen3-32B BF16; the 400 MiB context-window calculation belongs only to the unused base
  property. Preserve this source invariant with a regression test and do not run a redundant GPU
  matrix.
- Experiment 032 completed TP collective attribution. TP=4 all-reduce kernels were about 41.9% of
  cumulative GPU kernel time, and the existing direct PyNCCL path (`MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`)
  improved paired online throughput by 9.54%/11.82%/11.34% at concurrency 1/8/32 while reducing
  average TPOT 3.81%/7.00%/8.00%. Accept it as an explicit L20 TP=4 profile; keep the generic
  default unchanged until TP=8 online and CUDA Graph validation are complete.
- Experiment 033 validated the direct path at TP8 and with CUDA Graphs. Graph-disabled TP8
  throughput improved 10.20%/5.52%/5.26% at concurrency 1/8/32 with P90 TPOT improving at every
  load. TP4 Graph mode improved throughput at C=8/32 but regressed P90 TPOT 4.49%/9.69%, so the
  strict Graph gate failed. Accept direct PyNCCL only as an explicit graph-disabled L20 TP4/TP8
  profile; keep generic and Graph-mode defaults unchanged.
- Experiment 034 added two per-concurrency warmups and five steady-state repeats. Direct Graph mode
  remained stable and improved throughput 1.34%/7.68%/13.45% at C=1/8/32, but P90 TPOT changed
  -0.06%/+2.64%/+11.91%. The tail regression is real, not first-request noise. Close the collective
  branch: keep direct PyNCCL explicit and graph-disabled until a new Graph scheduler hypothesis is
  tested.
- Experiment 035 added logical/padded batch telemetry. Symmetric and direct traces had identical
  Graph-shape distributions; P90-tail activity concentrated in 20/24/28/31/32 logical batches that
  replay padded 24/24/32/32/32 shapes. More than 30 ms scheduler completion intervals were dominated
  by padded-32 samples, with direct mode about 0.4 ms slower. Investigate scheduler transitions
  around padded 32 before changing Graph padding or collectives.
- Experiment 036 tested a padded-32 result-before-prefill guard. Telemetry proved the five target
  transitions moved prefill selection from 147.13 ms before decode result delivery to 0.52 ms after
  it, but the new synchronization boundary raised C=32 P90 TPOT 28.69% while throughput changed
  -0.22%. Reject the guard for deployment and keep its generic option disabled by default.
- Experiment 037 tested a corrected one-shot, non-blocking decode priority. It added exactly five
  decode turns at the five target boundaries, changed C=32 throughput -0.63%, P90 TPOT -0.41%, and
  P99.9 TPOT -7.06%, with average TTFT +0.02%. Keep it as an explicit calibrated profile; require
  an independent seed and longer-output confirmation before any default change.
- Experiment 038 performed that independent long-output confirmation. At output length 128, priority
  added exactly five decode turns, changed C=32 throughput -0.32%, P90 TPOT +0.07%, P99.9 TPOT
  -9.68%, and P90 TTFT +1.13%. Reject it as a deployment optimization; retain only the
  default-disabled research flag.

## Experiment 043 result

Experiment 043 added an explicit `--disable-bf16-reduced-precision-reduction` control and paired it
against the default BF16 GEMM reduction mode. On the Qwen3-32B TP4 Graph matrix, treatment changed
throughput -0.37%/-0.44%/-0.21% at C=1/8/32 and changed P90 TPOT +0.17%/+0.19%/+6.73%. Reject it as
an optimization; retain the switch only as a default-disabled diagnostic control.

## Experiment 044 result

Experiment 044 grouped fresh experiment-043 CUDA-event telemetry by padded Graph shape. Padded 24
and padded 32 were the only shapes with model intervals above 30 ms: 20/20 and 130/130 control
events, respectively. Padded 32 averaged 34.082 ms and represented logical batches 28/31/32;
padded 24 averaged 32.541 ms and represented logical batches 20/24. Do not change sampler, copy,
collective, or generic Graph padding behavior based on this attribution.

## Experiment 045 result

Experiment 045 added a default-disabled `--cuda-graph-disable-bs 24 32` eager fallback. It changed
C=32 throughput -0.11% and P90 TPOT +3.72% (221.709 -> 229.953 ms), failing the strict tail gate.
Reject the eager fallback; keep the diagnostic switch but preserve Graph execution by default.

## Experiment 046 result

Experiment 046 added opt-in `MiniSGL.SchedulerForward.decode.bs=*` NVTX ranges and captured padded
24/32 Graph replays. SchedulerForward was about 59.6/59.4 ms while nested GraphReplay was about
56.4/56.2 ms; kernel composition remained NCCL 51.5% plus three BF16 GEMM classes about 46.7%.
No new shape-specific synchronization kernel was found. Do not add another scheduler barrier or
priority policy.

## Experiment 047 result

Experiment 047 measured Qwen3-32B BF16 TP2/TP4/TP8 on L20 with CUDA Graphs disabled
and direct PyNCCL enabled. TP4 reached 1.41-1.54x TP2 throughput with 70.7-76.8%
parallel efficiency across C=1/8/32. TP8 reached 1.66-1.72x with only 41.5-43.0%
efficiency, confirming diminishing returns from per-step TP communication. TP4 is
the best measured efficiency point; TP8 remains a throughput option but is not a
cost-neutral scale-out. TP2 had to use memory ratio 0.9 after a 0.7 KV-cache
allocation failure, while TP4/TP8 used 0.7, so this is a directional scaling
result rather than a perfectly memory-matched sweep. The online run did not have
kernel-level NCCL attribution; retain experiment 046's 51.5% Nsight context and
collect a fresh profile in the communication sweep.

## Interview-oriented roadmap

The recommendations in `learning/grok.md` are useful, but the measured results now narrow the
credible interview story. Do not present a parameter sweep or an unvalidated kernel guess as an
optimization. Present the work as a bottleneck-driven loop:

1. Establish a reproducible Qwen3-32B BF16 TP4 L20 baseline with throughput, TTFT, TPOT, P90/P99.9,
   GPU timing, and Nsight evidence.
2. Show the accepted explicit graph-disabled direct-PyNCCL profile: it improved graph-disabled TP4
   throughput by about 9--12% in experiments 032--034, while Graph mode was rejected on the strict
   C=32 tail gate.
3. Show the rejected alternatives and why they were rejected: Graph shape changes, result barriers,
   decode priority, BF16 reduction mode, and eager fallback all failed the tail/throughput gates.
4. Complete one held-out multi-card scaling study and one communication tuning study before making
   a deployment recommendation. Experiments 047 and 048 now satisfy these two study slots; 049-050
   are the serving-load and held-out confirmation phases.

### Planned experiments

| Experiment | Scope | Why it is interview-useful | Acceptance gate |
| --- | --- | --- | --- |
| 047 | TP2/TP4/TP8 scaling on Qwen3-32B, graph-disabled direct PyNCCL profile | **Complete:** quantifies TP scaling efficiency and exposes the TP8 communication wall | Three repeats; throughput and scaling reported; NCCL share remains a follow-up profile |
| 048 | PyNCCL buffer/algorithm sweep in graph-disabled mode | **Complete:** direct buffer=0 remains best; protocol candidates fail online gate | One variable at a time; no C=32 P90 or correctness regression |
| 049 | Open-loop arrival matrix with the accepted TP4 profile | **Complete:** TTFT knee appears around 1.5 req/s for mixed long prompts | Report arrival-rate curves, not only fixed concurrency |
| 050 | Held-out final evaluation | **Complete:** fresh-seed long-output unique/shared-prefix validation passes | Fresh seed, longer output, shared-prefix and unique-prefix cases |
| 051 | L20 low-precision GEMM capability and shape sweep | **Complete:** FP8 dynamic W8A8 wins all 12 cold-cache points; INT4 is fastest but needs calibrated quality validation | All paths execute on SM89; kernel error is separated from quantization error; five timing samples |
| 052 | Quantization configuration, loader metadata, and `QuantLinear` skeleton | **Complete:** default-disabled FP8 loads BF16/pre-quantized weights and runs a real Qwen3-0.6B smoke test | BF16 remains unchanged; focused loader/dispatch tests pass |
| 053 | FP8 W8A8 end-to-end integration | **Complete:** Qwen3-32B TP4 gains 47-55% throughput with a 0.70-point MMLU drop | Quality, Graph, memory, TTFT, TPOT, throughput, and repeatability gates pass |
| 054 | INT4 AWQ-Marlin end-to-end integration | **Complete:** Qwen3-32B fits TP1/TP2 and saves cards, but misses the one-point MMLU gate | TP1/TP2 and Graph pass; default-FP16/BF16 quality drops 1.23/1.75 points |
| 055 | Held-out quantization comparison | **Complete:** fresh open-loop mixed-length runs select FP8 for the current TP4 service profile; AWQ remains capacity-only | Three repeats at three rates; tokens/s, tails, memory, and fresh quality reported |
| 056 | FP8 prefill/decode scheduler sweep | **Complete:** static chunk/budget changes do not improve TTFT SLO; baseline retained and dynamic pressure control identified | Five configurations; 45 runs; no throughput/SLO regression; telemetry explains queue behavior |
| 057 | Pending-pressure adaptive prefill budget | **Complete:** existing threshold/overload branch triggers, but fixed smaller budgets do not reduce queue peak or improve TTFT SLO | Three dynamic policies; 27 runs; trigger telemetry and end-to-end SLO comparison |
| 058 | Joint pending-token and active-decode pressure gate | **Complete:** joint gating recovers queue-only throughput loss, but does not improve TTFT SLO | Two joint policies; 18 runs; default compatibility, telemetry, throughput, TTFT and TPOT gates |
| 059 | Page-aware admission and safe decode retraction | **In progress:** estimator plus short L20 A/B is complete with no regression; retraction and full matrix remain | P90 TTFT -15%, pending peak -20%, throughput >=98% baseline, no stale/retracted result |
| 060 | Preallocated `Req` token buffer | **Complete, conditional:** L20 A/B removes 100% append allocations and 99.95% copied tokens with no end-to-end gain; CPU wall-time -50% gate failed | Long-output append CPU time -50%, token-identical output, no decode regression |
| 061 | L20 FP8 KV cache v2 | **Stage 2 telemetry complete:** opt-in FP8 pool, dtype-aware page accounting, first-eager-prefill per-layer calibration, and rank-scoped real-model summary; fused store and quality/perf gates remain | Capacity +70% target exceeded (2.00x tokens); real short-request weighted saturation 0.00226%; long-context quality/ITL/TTFT gates remain |
| 062 | FP8 KV long-context quality/performance gate | **Complete, performance rejected:** 4096/8192/16384 eager A/B, long-context saturation and deterministic output smoke pass; no end-to-end speedup | 2.00x KV capacity; 0.000843% weighted saturation; TPOT +36.73%--+69.34%, throughput -10.26%--29.22%; proceed to fused cast/store |
| 063 | FP8 KV fused cast/store | **Complete, conditional:** opt-in BF16/FP16-to-FP8 fused conversion and indexed page store; correctness passes and end-to-end TPOT improves, but pure write kernel remains slower | TPOT -14.6%--29.0% vs same-code legacy; byte-identical output; microbenchmark +13.6%--60.7%, keep default disabled |
| 064 | FP8 KV native pair conversion | **Complete, conditional:** SM89 native BF16/FP16 pair conversion lowers selected write-kernel timings, but non-representable scales are not byte-exact | At least 3/4 row sizes improve 20%; byte-exact output for calibrated scales; otherwise keep opt-in |
| 065 | FlashInfer page size >1 | **Stage 1 complete, conditional:** metadata and BF16/FP8/Graph smoke pass; page 32 is a candidate, page 16 regresses in the first TP4 profile; default remains 1 | Complete C=1/8/32, 8K/16K, eviction and Graph-shape matrix; require no P99 regression before changing default |
| 066 | L20 memory-aware CUDA Graph shapes | **Stage 2 complete, conditional:** with 3 repeats and 128-token output, l20 cuts padding 15.65% -> 1.48%, adds 0.33 GiB/card, and improves throughput 0.67%--1.00%; low-load P99 TPOT still regresses, so default remains coarse | Broader arrival/output matrix and no low-load P99 regression before changing default |
| 067 | Batched tokenization | **Stage 3 complete, conditional:** CPU tokenizer median improves 2.26x; 1 ms wait changes low-rate P90 TTFT -4.32%/+0.08%, still misses the -5% gate; multi-tokenizer streaming smoke passes | Test higher arrival rates and explicit abort cancellation before promoting any wait window |

Experiment 047 is complete. Use Qwen3-32B BF16 for TP2/4/8 because its dense checkpoint does not fit
TP1 on a 48 GiB L20 at the production configuration. Experiment 048 is also complete: direct buffer=0 remains the
best communication setting, while LL128's microbenchmark gains did not survive the online C=8/C=32
gate. Experiment 049 found 100% TPOT-SLO attainment but only 59.7%/44.4% TTFT-SLO attainment at
1.5/2.5 requests/s. Experiment 050 is complete: fresh-seed 128-token unique/shared-prefix runs
passed with stable decode timing. The selected profile is now validated for the scoped L20/Qwen3-32B
configuration. If a TP1 reference is needed for a scaling plot, use a separately labeled Qwen3-14B
run rather than mixing models.

Do not start a GEMM kernel rewrite yet. Nsight shows GEMM is material, but the current environment
lacks `ncu` and the available CUTLASS/FA dependency versions have already caused compatibility
blockers. First finish scaling and communication tuning; only then choose a kernel change with a
measurable baseline and a correctness test.

## Experiment 051 result

Experiment 051 established the new low-precision baseline on Qwen3-32B TP4 local layer shapes. All
FP8, INT8, and GPTQ-Marlin INT4 paths executed on L20 SM89, with maximum kernel relative-L2 error
0.501% and maximum timing coefficient of variation 0.46%. With a 128 MiB L2 flush outside each timed
call, FP8 including dynamic activation quantization improved all 12 points by 1.47-2.07x. INT8
improved them by 1.09-1.68x, while INT4 W4A16 kernel speedup was 2.16-3.99x. Select FP8 for the first
end-to-end implementation. INT4 remains a later candidate because naive channelwise RTN produced
14.49-15.78% synthetic output relative-L2 error and requires calibrated GPTQ/AWQ quality evidence.

## Experiment 052 result

Experiment 052 implemented the default-disabled `--quantization fp8` boundary. Dense transformer
linears now support BF16-to-FP8 conversion at load time and pre-quantized FP8 weight/scale state;
column-parallel scales shard and merge with outputs, while row-parallel scales replicate. Embedding,
LM head, KV cache, and collectives stay BF16. Ten focused tests and 78 selected regressions passed.
A Qwen3-32B TP4 meta build produced the expected 256 FP8 weight/scale pairs, and paired Qwen3-0.6B
BF16/FP8 smoke runs both generated `[3, 4, 3, 4]`. Accept the scaffold but keep it off by default;
experiment 053 must establish end-to-end quality, memory, Graph compatibility, and serving speed.

## Experiment 053 result

Experiment 053 validated FP8 W8A8 end to end on L20. Qwen3-32B TP4 Graph throughput improved
55.00%/49.48%/47.34% at C=1/8/32, average TPOT fell 37.05%/35.65%/34.69%, and sampled peak memory
fell 2,822 MiB/card (15.48%). On 570 seeded MMLU questions, BF16 scored 80.35% and FP8 scored
79.65%, a 0.70-point drop that passes the one-point gate; both parse rates were 100%. Small-model
eager FP8 regressed throughput by 19-22%, while Graph FP8 improved it by 10-21%, so the accepted
claim is explicitly Graph-scoped. Keep FP8 default-disabled until experiment 055 confirms mixed
long prompts and open-loop tails with a fresh seed. Experiment 054 subsequently tested INT4.

## Experiment 054 result

Experiment 054 implemented calibrated INT4 AWQ loading through the shared Marlin W4A16 kernel. The
official Qwen3-32B-AWQ checkpoint now loads all dense projections, captures Graph shapes through
batch 32, and runs on one or two L20s. Controlled BF16 TP1 reached 34.69/169.66/197.74 tok/s at
C=1/8/32; TP2 reached 60.64/278.69/316.09 tok/s, a 1.60-1.75x speedup with 79.9-87.4% scaling
efficiency. TP1 used 21.0-21.8 GiB total and TP2 used 22.6-23.2 GiB total, versus 61.6 GiB sampled
across the FP8 TP4 baseline. This is a strong capacity and tokens/s/GPU result, not a peak-throughput
win over FP8 TP4.

The strict quality gate failed. On the same 570 MMLU questions, BF16/FP8 scored 80.35%/79.65%,
checkpoint-default FP16 AWQ scored 79.12% (-1.23 points), and dtype-controlled BF16 AWQ scored
78.60% (-1.75 points). Accept the implementation and TP feasibility evidence, keep AWQ
default-disabled, and carry it into experiment 055 only as a conditional low-memory profile.

## Experiment 055 result

Experiment 055 used a new seed (`5500042`), balanced 256/1024/4096-token prompts, 128-token
outputs, three repeats, and open-loop rates of 0.5/1.0/1.5 requests/s. FP8 TP4 improved P90 TTFT
over BF16 TP4 at all rates (1099/1553/1352 ms versus 1534/3355/2792 ms), reduced average TPOT
from 34.00/42.22/41.26 ms to 19.71/27.16/25.92 ms, and reduced sampled memory from 19109 to
16453 MiB/card. TTFT-SLO attainment was 100/97.2/97.2% for FP8 versus 97.2/66.7/72.2% for BF16;
TPOT-SLO attainment stayed at 100% for both.

AWQ TP1 used only about 24.3 GiB on one card, but its TTFT-SLO attainment was 61.1/27.8/22.2%
and its 4096-token P90 TTFT reached 4807/10371/10555 ms. Keep AWQ default-disabled and describe it
as a capacity-first option, not a latency-sensitive serving choice. The fresh MMLU seed scored
81.93%/82.11%/82.28% for BF16/FP8/AWQ with 100% parsing; this finds no new regression, but does
not overturn experiment 054's AWQ quality risk because a single 570-question seed is noisy.
Accept FP8 as the current Graph/TP4 deployment candidate, retain BF16 as the reference, and do
not change global defaults.

## Experiment 056 result

Experiment 056 retuned the accepted FP8 TP4 profile with four static scheduler alternatives against
the 4096/1/1024 baseline. All five configurations ran 45 open-loop traces (540 requests) with no
request errors and 100% TPOT-SLO attainment. Reducing normal prefill length to 2048 left the 1.5
req/s TTFT SLO unchanged at 80.6% and changed throughput by less than 0.1%. Reducing the
decode-active budget to 512 increased prefill batches from 232 to 372--380, lowered 1.5 req/s
throughput from 153.81 to 148.62--148.70 tok/s, and did not improve TTFT SLO. Allowing two
consecutive prefill batches raised throughput only to 154.68 tok/s but increased P99.9 TPOT from
244.49 to 447.27 ms.

All configurations reached the same maximum pending prefill pressure of 8472 tokens. Reject the
static alternatives and retain 4096/1/1024. The next scheduler hypothesis is a pending-pressure
adaptive budget, with a small budget only when the prefill queue crosses a threshold and the normal
budget when pressure is low.

## Experiment 057 result

Experiment 057 tested the existing pressure-aware overload branch with overload budgets 512/256 and
thresholds 4096/2048. All 27 open-loop runs (324 requests) completed with 100% TPOT-SLO attainment,
and telemetry confirmed the dynamic branch was selected. However, every configuration still reached
only 80.6% TTFT-SLO at 1.5 req/s. The 4096/512 policy reduced throughput from 153.81 to 151.68
tok/s and increased P90 TTFT from 2489.72 to 2608.29 ms; earlier or more aggressive policies were
worse. Maximum pending prefill pressure remained 8472 tokens in every case.

Reject the fixed threshold/smaller-budget policy as a default. The next scheduler hypothesis must
combine pending tokens with active decode count and remaining request prefill length, and should
restore the large budget after pressure falls instead of repeatedly splitting the same queue.

## Experiment 058 result

Experiment 058 added a default-compatible active-decode threshold to the overload policy. Two joint
configurations completed 18 open-loop runs and 216 requests with 100% TPOT-SLO attainment. At 1.5
req/s, the 4096 pending / 512 budget / 6 active-decode candidate reached 153.47 tok/s, 2496.96 ms
P90 TTFT, and 235.16 ms P99.9 TPOT. This recovered nearly all throughput lost by experiment057's
queue-only 512 budget, but did not improve the baseline's 80.6% TTFT-SLO attainment.

Telemetry explains both outcomes: joint gating reduced prefill batches from the queue-only policy's
255 to 237, while maximum pending pressure stayed at 8472 tokens. Retain the mechanism as an opt-in
diagnostic/tuning control, but do not change the L20 deployment default. Further fixed-threshold
sweeps are unlikely to address the queueing cause; the next scheduler experiment should evaluate
admission control or waiting-time/remaining-prefill-aware ordering.

## Success criteria

- Performance claims use at least three comparable measurements and include variance.
- Accepted changes have focused tests and do not regress correctness.
- The result is meaningful on a model/load representative of the optimized path.
- Each experiment has a numbered Markdown note under `learning/experiments/`.

## 实验 068 当前状态

实验 068 已完成 safe decode retraction 的离线和一次 L20 限页压力验证：候选按 page demand
选择，当前 in-flight batch 受保护，已有 KV 的请求暂停后原地恢复，默认路径保持关闭。
`tests/core` 当前 109/109 通过。在线结果显示吞吐变化 -0.18%/-0.93%，P90 TTFT 没有改善，
但输出长度、page integrity 正确；高达 143/706 次 retraction 暴露 reservation thrashing。
因此性能 gate 拒绝，下一步若继续必须先加入 hysteresis/收益阈值，不直接扩大流量或改变默认值。

## Experiment 067 result

Experiment 067 Stage 3 added an opt-in tokenizer micro-batch wait. With a 1 ms wait, P90 TTFT
changed -4.32%/+0.08% at 1.5/2.5 req/s versus the batch=1 control, while throughput changed
+0.07%/-0.03%; the -5% gate still failed. Both profiles retained 100% SLO attainment, and
num_tokenizer=2/4 streaming smoke passed. Keep batch=32 as a high-concurrency CPU candidate and
keep the wait default at 0 ms; do not claim a low-rate TTFT win. See
`learning/experiments/067_batch_tokenization.md`。

## Experiment 069 result

实验 069 完成了 safe decode retraction 的 hysteresis 版本。冷却窗口和 pending-token 压力门槛
将一次 L20 在线压力矩阵中的 retraction 次数减少 83.0%，retracted token 减少 83.6%，吞吐
不降，2.5 req/s 的 P90 TTFT 相比实验 068 control 改善 2.34%。该结果有调度振荡治理和资源
状态机设计的面试价值，但只有单次重复，不能证明跨 workload 稳定收益。因此将其标记为
**条件性接受、默认关闭**；后续若继续，应做至少三次重复和不同 output length/page size 的
敏感性验证，而不是直接扩大默认流量。

## Experiment 070 result

实验 070 的多次重复没有通过 hysteresis 性能 gate：关闭 overlap 的 2 次重复中，cooldown=4
和 pending gate=2048 将 retraction 降低 77.1%，但 1.5 req/s P90 TTFT 回退 5.57%，高负载
基本持平。因此保留为 opt-in thrashing 控制，不改变默认。开启 overlap 的 12 请求限页压力
在 4096/8192 pages 均触发 cache allocator 的 `0 evictable` 断言，说明提交但尚未完成的
pipeline batch 的未来 KV reservation 没有被 admission 完整建模。下一实验优先修复这个安全
不变量，再重新评估 overlap 下的性能。

## 实验 071 结果

实验 070 暴露的 overlap pipeline allocator assertion 已通过 decode admission guard 修复：
`CacheManager.can_allocate_paged` 读取当前可见 free/evictable 容量，scheduler 在准备 batch 前
裁剪未来 page demand 超限的 decode 请求，并让 telemetry 记录实际执行的 batch。focused tests
24/24 和完整 `tests/core` 通过；Qwen3-4B-AWQ 单卡 L20 功能冒烟 6/6 请求完成、SLO 100%，无
allocator assertion。该修复的价值是保持 cache allocator 安全不变量，尚无端到端性能收益证据，
默认仍不启用额外激进策略。下一步是 TP4 限页 overlap 压力和 starvation/fairness 评估。

## 实验 072 结果

TP4 Qwen3-32B-AWQ 在 96 page KV cache 下完成了两轮 overlap pressure trace（12 请求短 burst、
32 请求长 burst），没有 allocator assertion、请求丢失或进程异常退出。长 burst P90 TTFT 为
19.71 秒，证明小 cache 下 tail queueing 已成为主要问题。新增容量裁剪 telemetry 在两轮均为
0 次，最大 decode batch 为 3，因此实验没有证明 `_fit_decode_batch` 在真实线上被触发；不能把
稳定性结果解释为性能提升。下一步转为确定性 cache occupancy/fairness 边界测试。

## 实验 073 结果

实验 073 将容量裁剪从固定前缀改为 round-robin 起点。确定性单 slot 9 轮选择从 UID 0 独占
（9/0/0）变为 0/1/2 各 3 次；真实 cache occupancy 边界仍严格遵守可见 page 容量。完整核心
测试 117/117 通过。该改动解决容量受限时的公平性风险，默认参数和完整 batch 快路径不变，
没有端到端吞吐收益证据；后续需要在线裁剪 trace 验证等待轮数和 tail latency。

## 实验 075 结果

实验 075 将容量 cursor 从列表下标改为下一个 UID，修复请求完成/取消后索引漂移造成的跳过：
`[0,1,2,3] -> [0,1]` 后移除 0，新集合 `[1,2,3]` 正确选择 `[2,3]`。核心测试 119/119 通过，
完整 batch 快路径和容量参数不变。下一步需要动态在线 trace 验证新 UID、取消请求和尾延迟。

## 实验 076 结果

实验 076 增加 `_decode_capacity_wait_rounds` 与裁剪事件字段，能够记录 deferred UID 的累计
等待轮数和 selected UID 的服务前等待轮数。确定性单 slot trace 正确呈现 1/2/3 轮等待；核心
测试 120/120 通过。该改动只增强观测，不改变 admission 或 batch 快路径，下一步用真实线上
裁剪事件验证最大等待轮数与 P99 TTFT。

## 实验 077 结果

在 L20 GPU 2 上将 Qwen3-4B-AWQ 的 KV cache 压到 `page_size=32`、`num_pages=4`（128 token），
用真实 HTTP 请求做短请求和高并发 burst。两轮共 11 个请求全部完成，TTFT/TPOT SLO 均为 100%，
没有 allocator assertion；P90 TTFT 分别为 123.29 ms 和 657.95 ms。4 个 telemetry trace
没有产生 `decode_capacity_trimmed/deferred`，因为 admission 已先将活跃 decode batch 控制在
可容纳范围内。该实验确认在线边界稳定性，并提供了“未命中裁剪分支”的证据；不能把它解释为
等待轮数或吞吐收益。详见 `learning/experiments/077_online_wait_round_validation.md`。

## 实验 078 结果

`token` 与 `page-aware` reservation 在 L20 4-page 边界 trace 上进行了三次重复 A/B。token
吞吐 338.39±3.23 tok/s、P90 TTFT 268.51 ms，但每轮出现 16 次 decode 容量裁剪，P99 TPOT
42.89 ms；page-aware 无裁剪、P99 TPOT 5.69 ms，但吞吐下降 12.84%、P90 TTFT 上升 33.62%。
因此 page-aware 保留为 tail/safety 优先场景的 opt-in，不修改默认路径；后续推广前需在更大
page 数、不同输出长度和 TP4 上重复验证。详见 `learning/experiments/078_page_aware_reservation_ab.md`。

## 实验 079 结果

TP4 PyNCCL 默认路径在 L20 GPU 2/3/6/7 上成功启动：NCCL wheel 的版本化
`libnccl.so.2` 通过绝对链接和 rpath 加载，Gloo 只承担 CPU bootstrap。Qwen3-32B-AWQ、
8 个 16-token page 的三次 A/B 中，token 吞吐 136.63±3.90 tok/s，page-aware
135.50±1.90 tok/s，P90 TTFT 511.49/519.38 ms，均无 capacity telemetry 事件。结论是
PyNCCL 链接修复接受，但 TP4 admission 已提前消化页压力，不能据此宣称 page-aware 收益或
修改默认策略。详见 `learning/experiments/079_pynccl_tp4_reservation_validation.md`。

## 实验 074 结果

在 3 page 容量、请求 demand 为 2/1/1 的确定性模型中，round-robin 选择循环为
`[0,1] -> [2,0] -> [1,2]`，避免重 page 请求和轻 page 请求互相饥饿。测试通过，默认完整
batch 快路径不变；该实验只接受为调度公平性证据，下一步需要动态请求集合的在线等待轮数和
tail latency 验证。

## 实验 080 结果

实验 080 将总 KV cache 容量固定为 128 token，在 TP4 Qwen3-32B-AWQ 上比较 page size
8/16/32。三组均完成 24/24 请求且没有 allocator assertion；page size=8/16/32 的吞吐
分别为 132.04、136.63、131.74 tok/s，P99 TPOT 分别为 36.58、35.01、80.04 ms。遥测中
没有容量裁剪或回收事件，说明 admission 先于 decode fit guard 控制了活跃集合。当前证据
支持保留 16 作为基线，不支持把更小或更大的 page size 改成默认值；后续应在长输出、高到达率
和更大 page 数下验证内部碎片及 P99/P99.9 尾延迟。详见
`learning/experiments/080_tp4_page_size_sensitivity.md`。

## 实验 081 结果

实验 081 在固定 1024-token KV cache 下把输出长度提高到 128 token，重新比较 page size。
page size=8/16/32 的吞吐为 305.82/294.03/297.83 tok/s，P99 TPOT 为 38.81/44.97/33.98 ms；
没有容量裁剪、回收或请求失败。长 decode 的排序不同于实验 080 的 16-token workload，说明
page 粒度与 decode batch shape、page 对齐和持续 KV 增长耦合，不能只用短请求结果改默认值。
后续应扩大输出长度、到达率和 page pressure，并记录 allocator 碎片和 P99.9 延迟。详见
`learning/experiments/081_tp4_long_decode_page_size.md`。

## 实验 082 结果

实验 082 用 16 个 128-token 输出请求制造 KV pressure。1024-token cache 在 20/80 req/s
下仍保持约 381/393 tok/s，但 TTFT P90 约 3.26/3.47 秒；256-token cache 下吞吐降为
89.96 tok/s、TTFT P90 为 19.77 秒、decode batch 全部为 1。两组都没有 capacity trim 或
allocator assertion，说明当前实现主要在 admission 阶段排队，而不是在 decode 阶段裁剪。
后续需要在 256-token cache 上比较 token/page-aware reservation 的等待公平性，再决定是否
扩展容量裁剪 telemetry 或策略。详见 `learning/experiments/082_tp4_kv_pressure_admission.md`。

## 实验 083 结果

在 256-token cache 和 16 个 128-token 输出请求的压力下，token/page-aware reservation 的
吞吐为 88.99/90.14 tok/s，P90 TTFT 为 19.96/19.72 秒；差异只有约 1.30%。两组都没有
capacity trim/deferred，decode batch 全为 1，说明 admission 先行串行化，A/B 没有命中公平
性 cursor。保留 token 为默认、page-aware 为 opt-in；下一步构造可命中容量裁剪分支的确定性
多候选 decode trace。详见 `learning/experiments/083_tp4_reservation_fairness_under_pressure.md`。

## 2026-10 当前执行总览（实验 069-083）

本节是当前计划的权威状态，早期章节保留为历史决策记录。实验 074 在
`learning/progress.md` 中按用户要求位于 075 之前，本文件不再调整该顺序。

### 已接受或保留为默认候选

| 能力 | 实验/代码证据 | 当前决策 |
| --- | --- | --- |
| PyNCCL wheel runtime linker | 实验 079；TP4 Qwen3-32B-AWQ 在 L20 GPU 2/3/6/7 ready | 接受，默认 PyNCCL 可用；Gloo 仅做 CPU bootstrap |
| decode admission safety guard | 实验 071/072；限页 overlap 无 allocator assertion | 接受 correctness 修复，不宣称吞吐收益 |
| capacity round-robin + UID cursor | 实验 073/074/075/076；确定性选择 0/1/2 公平循环 | 接受为公平性基础设施，等待线上 trim-path 验证 |
| FP8 W8A8 weight | 实验 053/055；Graph/TP4 性能与质量 gate 通过 | 作为 L20 Graph/TP4 显式候选，BF16 仍为 reference |
| AWQ/INT4-Marlin | 实验 054/055；TP1/TP2 容量优势，质量 gate 风险 | capacity-first opt-in，不改默认 |

### 条件性或 default-disabled 能力

| 能力 | 证据 | 不进入默认的原因 |
| --- | --- | --- |
| page-aware reservation | 实验 078/079/083 | 特定 trim trace 安全收益，但 TP4/256 pressure 已被 admission 串行化；吞吐收益不足 |
| page size 8/16/32 | 实验 080/081 | 短输出与长输出排序不同，未形成全局最优结论 |
| L20 Graph shape policy | 实验 066 | padding 15.65% -> 1.48%，但低负载 P99 TPOT 有回退 |
| tokenizer micro-batch wait | 实验 067 | CPU median 2.26x，但 1 ms wait 未通过低速率 TTFT gate |
| FP8 KV cache fused/native path | 实验 061-064 | capacity 约 2x，但长上下文 TPOT/写 kernel/scale quality gate 未完成 |
| safe retraction hysteresis | 实验 068-070 | 减少 thrashing，但多次重复未稳定改善 TTFT；保持 opt-in |

### 实验 080-083 的统一解释

这四个实验共同说明，page 和 reservation 不能脱离 admission 行为评价：

- 实验 080 固定 128-token cache、输出 16，page size=16 吞吐最高；
- 实验 081 固定 1024-token cache、输出 128，page size=8 吞吐最高，page size=32 TPOT P99
  最低，排序发生变化；
- 实验 082 将请求数提高到 16，256-token cache 下吞吐降到 89.96 tok/s、P90 TTFT 19.77 s，
  但 capacity trim 仍为 0，说明 admission 先排队；
- 实验 083 在同一压力下比较 token/page-aware，吞吐仅 88.99/90.14 tok/s，6160 个 decode
  batch 全为 1，两个策略都没有命中公平性 cursor。

因此当前系统的首要未知量不是“page-aware 是否平均更快”，而是怎样构造一个既有多个 decode
候选、又只有部分候选的未来 page demand 能容纳的 trace，从而真正验证 073-076 的公平性实现。

### 实验 084 已完成：capacity trim/fairness 集成 trace

实验 084 使用 CPU-only deterministic trace 直接走 `_fit_decode_batch` 和 pipeline telemetry。
固定容量为 3 page，UID 0/1/2 的 demand 为 2/1/1，连续 9 轮选择模式为
`[0,1] -> [2,0] -> [1,2]`，每个 UID 服务 6 次。动态阶段覆盖 UID 0 完成、UID 1 取消、
UID 3/4/5 加入，并用容量 0 制造全量 deferred round。最终 13 个事件中有 12 个
`decode_capacity_trimmed` 和 1 个 `decode_capacity_deferred`；恢复容量后 cursor 没有回退，
新 UID 没有继承旧等待计数。focused 测试 15/15、完整 `tests/core` 121/121 通过。

该结果接受公平性、动态 membership 和 telemetry 集成正确性 gate，但只证明 scheduler 的
容量选择路径，不等价于 L20 在线吞吐收益。实验 085 仍需在真实 L20 pressure matrix 中寻找
会自然命中该分支的 workload；在此之前不扩大 page-aware 默认范围。

## 实验 084-088 计划

### 实验 084：确定性 capacity trim/fairness 集成 trace（已完成）

目标是绕过 admission 直接串行化的问题，构造多个 running decode 请求的异构剩余 demand：
例如容量剩余 3 page，候选 demand 为 2/1/1 page，且每一轮只允许 1-2 个请求进入 decode。
需要同时覆盖请求完成、取消和新 UID 插入。

验收：

- 至少 9 个 round，UID 0/1/2 服务次数差异不超过 1；
- deferred UID 的 wait rounds 单调累计，服务后清零；
- capacity trim/deferred telemetry 实际大于 0；
- 无 stale result、page table 不变量破坏或 allocator assertion；
- token/page-aware 的选择结果可解释，测试固定 seed 可重复。

### 实验 085：page-aware 在真实 pressure 下的边界矩阵

固定 page size=16，组合 256/512/1024 token cache、128/512 output、20/80 req/s 和 8/16 请求。
先用 token policy 建立 baseline，再只切换 page-aware。报告吞吐、P90/P99 TTFT、P99 TPOT、
最大 pending、trim/deferred 次数、wait-round P99 和完成顺序。

接受规则：只有在至少一个真实 trim-path workload 中 page-aware 同时满足容量安全、wait-round
公平性改善、吞吐不低于 98% 和 P99 TTFT 不回退 5% 才考虑扩大 opt-in 范围；否则保留为
安全优先策略。

### 实验 086：Req 预分配 buffer 上游化

把实验 060 的实现整理成独立变更：`Req` 初始化一次 `torch.empty(max_device_len)`，append
只写入下一个位置并返回有效长度 view。补充 aborted request、capacity view、tokenizer 输出
顺序和 4096/8192 output 的 CPU benchmark。

验收是 CPU append wall-time、allocation 次数和 token identity；如果端到端 tok/s 不变，必须
如实写成 CPU overhead/可扩展性 PR，不把 microbenchmark 结果包装成 serving 吞吐提升。

### 实验 087：FP8 KV cache 最终 quality/performance gate

在 Qwen3-32B TP4 上比较 BF16 KV 与 FP8 KV 的 4096/8192/16384 context、128/512 output。
固定 scale calibration，分别测 capacity、P99 TPOT、写 kernel 时间、长上下文 token agreement
和显存。只有 fused cast/store 真实降低端到端 TPOT 且质量 gate 通过，才继续做默认候选。

### 实验 088：可审查的 upstream PR 拆分

不要提交一个包含所有实验的“大优化 PR”。拆成：

1. PyNCCL wheel linker + focused build/load test；
2. telemetry schema 和 capacity wait-round tests；
3. Req preallocated buffer；
4. page-aware admission/fairness（依赖 084/085 结果）；
5. Graph shape policy 或 FP8 KV（只有各自 gate 通过才提交）。

每个 PR 都附最小复现、默认行为说明、L20 benchmark Markdown/JSON、回归测试和失败实验链接。

## 当前停止条件

在实验 084 未命中真实 capacity trim 分支前，不继续扩大 page-aware 默认范围；在实验 087
质量/端到端 gate 未通过前，不把 FP8 KV 宣称为生产收益；在没有 Nsight 或可归因 profile 前，
不重写 GEMM/attention CUDA kernel。这样可以保证项目叙事是 bottleneck-driven，而不是无止境
调参。
