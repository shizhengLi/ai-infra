# Experiment 031: PyNCCL communication-buffer auto-sizing

## Status

Planned. This is the next implementation experiment after the adaptive-reserve branch was closed
in experiment 030.

## Why this experiment

The current PyNCCL communicator sizes its symmetric internal buffer from the model's maximum
sequence length. Qwen3-32B advertises a 40,960-token context, so the TP=4 server can reserve a
buffer close to:

```text
40960 * hidden_size(5120) * BF16(2 bytes) = 400 MiB per rank
```

The scheduler normally bounds one forward batch by `max_extend_tokens=8192` tokens. The current
upper bound is therefore tied to the model's context window instead of the engine's actual batch
budget. This is a concrete memory-accounting inefficiency and a better next target than another
generic cache-hotness heuristic.

## Interview value

This experiment demonstrates a complete systems optimization loop:

1. trace a reservation to its source (`EngineConfig.max_forward_len` -> `init_pynccl`);
2. derive a safe upper bound from scheduler batch semantics;
3. preserve a direct NCCL fallback when a tensor exceeds the internal buffer;
4. measure memory capacity, initialization, communication, and serving latency together;
5. accept the change only if correctness and throughput remain stable.

The result is easy to explain as “reduce over-reserved communication memory without changing the
collective contract,” and it directly uses the L20 PCIe/TP setup described in `grok.md`.

## Hypothesis

Sizing the PyNCCL internal buffer from the maximum scheduler forward batch, with a small safety
margin, will free memory on every TP rank without changing model outputs or steady-state serving
performance. The freed memory should increase available KV-cache capacity or reduce initialization
pressure.

## Proposed implementation

Replace the context-window-derived bound with a scheduler-aware bound:

```text
forward_tokens = max(max_extend_tokens, max_running_requests)
required_bytes = forward_tokens * hidden_size * dtype.itemsize
buffer_bytes = ceil_to_alignment(required_bytes * safety_margin)
```

The implementation must keep an explicit environment cap for debugging and retain the existing
direct `ncclAllReduce` path when a runtime tensor is larger than the symmetric buffer. It must log
the requested bound, selected buffer size, and whether the cap was applied. The default safety
margin and alignment are implementation details to be fixed before Stage A, not tuned after seeing
the results.

## Stage A matrix

Use Qwen3-32B BF16 on L20 with TP=4, GPUs 0-3, `fi`, CUDA Graph max batch size 64, and the
calibrated scheduling profile (`max-prefill-streak=1`, decode-active prefill 2304, result-before-
prefill). Keep model, prompts, output lengths, concurrency, and seeds fixed.

| Cell | Buffer policy | Purpose |
| --- | --- | --- |
| Control | current context-window bound | Existing behavior |
| Candidate 1 | fixed 96 MiB cap | Covers the 8192-token BF16 forward bound with margin |
| Candidate 2 | fixed 128 MiB cap | Conservative manual cap |
| Treatment | scheduler-derived bound | Proposed automatic policy |

Run three fresh-server repetitions per cell. A short fixed-shape decode pass should establish
startup and throughput behavior before any longer online run.

## Metrics

- PyNCCL requested and selected buffer bytes per rank;
- free memory before/after model and communicator initialization;
- KV-cache pages/tokens allocated;
- server initialization time;
- output throughput, average/P90 TTFT, average/P90 TPOT, and maximum TPOT;
- NCCL/PyNCCL errors and output correctness;
- peak memory and GPU power when available.

## Acceptance rules

Accept the automatic policy only if all three repetitions satisfy every rule:

- zero output or collective errors;
- identical deterministic outputs to the control;
- no more than 1% steady-state throughput regression;
- no more than 3% P90 TTFT or TPOT regression;
- selected buffer is no larger than the current context-window allocation;
- at least 128 MiB additional free memory per rank, or a measured increase in KV capacity;
- no new synchronization or allocator failure under the calibrated online workload.

If a smaller fixed cap passes but the automatic policy does not, keep the result as a documented
manual deployment profile and investigate the missing forward-shape bound before changing the
default.

## Stage B

Only the best Stage A candidate enters a three-seed online confirmation. TP=8 is a secondary
robustness check because it crosses the two L20 PCIe/NUMA groups; it is not the primary acceptance
target. Communication overlap or asynchronous collectives are explicitly deferred until buffer
sizing is measured, so later code changes have an attributed baseline.

## Artifacts

- planned implementation: `python/minisgl/engine/config.py`, `python/minisgl/engine/engine.py`,
  and `python/minisgl/kernel/pynccl.py`;
- benchmark: `benchmark/online/bench_l20.py` or a focused TP memory/throughput harness;
- result note: this file will be updated with the fixed commands, raw reports, JSON, and decision;
- all measurements go under `learning/results/031_*`.
