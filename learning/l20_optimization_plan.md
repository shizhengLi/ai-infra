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
- The online-scheduling sequence is complete for this workload. Experiment 022 remains in Phase 2:
  compare default-disabled, page-aligned partial-leaf tail eviction against experiment 021.

## Success criteria

- Performance claims use at least three comparable measurements and include variance.
- Accepted changes have focused tests and do not regress correctness.
- The result is meaningful on a model/load representative of the optimized path.
- Each experiment has a numbered Markdown note under `learning/experiments/`.
