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

## Next experiment

Experiment 047: target the measured NCCL/BF16 GEMM implementation with one isolated kernel/collective
change, preserving symmetric Graph defaults and the strict C=32 P90 gate.

## Success criteria

- Performance claims use at least three comparable measurements and include variance.
- Accepted changes have focused tests and do not regress correctness.
- The result is meaningful on a model/load representative of the optimized path.
- Each experiment has a numbered Markdown note under `learning/experiments/`.
