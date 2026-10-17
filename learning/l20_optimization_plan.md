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
| 061 | L20 FP8 KV cache v2 | **Stage 2 in progress:** opt-in FP8 pool, dtype-aware page accounting, and first-eager-prefill per-layer calibration; fused store and quality/perf gates remain | Capacity +70% target exceeded (2.00x tokens); long-context ITL/TTFT and quality gates remain |
| 062 | FlashInfer page size >1 | **Planned:** remove page-size-1 limitation and validate 16/32-token pages, based on upstream #110 | Correctness under chunk/eviction/Graph, metadata overhead reduction, no P99 regression |
| 063 | L20 memory-aware CUDA Graph shapes | **Planned:** fine-grained shapes plus padding/graph-hit telemetry, based on upstream #56/#160 | Padding waste -30%, capture memory <=1 GiB/card, throughput or TTFT improvement |
| 064 | Batched tokenization | **Planned:** batch plain/chat tokenization and restore uid order, based on upstream #117/#55 | Tokenizer wall time 2x better at batch 32, TTFT P90 -5%, abort/order correctness |

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
