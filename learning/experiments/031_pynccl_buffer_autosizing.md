# Experiment 031: PyNCCL communication-buffer auto-sizing

## Status

Complete, hypothesis rejected. The server already sizes the PyNCCL buffer from the scheduler's
forward budget, so no buffer-sizing code change is justified by this experiment.

## Why this experiment

The base `EngineConfig` exposes a context-window-derived `max_forward_len`, but the actual server
configuration is `ServerArgs -> SchedulerConfig`. `SchedulerConfig` overrides that property and
returns `max_extend_tokens`. Qwen3-32B advertises a 40,960-token context, but the calibrated
server uses an 8,192-token forward budget, so the selected buffer is:

```text
8192 * hidden_size(5120) * BF16(2 bytes) = 80 MiB per rank
```

The context-window calculation would be 400 MiB, but it is not used by the serving path. The
initial optimization hypothesis was therefore based on the wrong configuration layer. This is a
useful negative result: source-level accounting must be checked before launching a costly GPU
matrix.

## Interview value

This audit demonstrates a complete systems optimization loop:

1. trace a reservation to its source (`EngineConfig.max_forward_len` -> `init_pynccl`);
2. trace the subclass override that changes the bound;
3. calculate the real selected allocation before changing code;
4. reject an optimization whose premise is not present;
5. turn the source invariant into a regression test.

The negative result is easy to explain in an interview: validate the configuration inheritance
before changing a distributed runtime, then preserve the already-correct bound. It directly uses
the L20 PCIe/TP setup described in `grok.md`.

## Hypothesis

The hypothesis that the serving path over-allocates a context-window-sized buffer is rejected. The
current 80 MiB allocation already follows the scheduler budget and is capped by the existing
`MINISGL_PYNCCL_MAX_BUFFER_SIZE` environment setting.

## Source audit

The active call chain is:

```text
ServerArgs.max_extend_tokens (8192)
  -> SchedulerConfig.max_forward_len (8192)
  -> Engine._init_communication()
  -> init_pynccl(max_size_bytes=8192 * hidden_size * dtype.itemsize)
```

`init_pynccl` still applies the explicit environment cap and the C++ wrapper still falls back to
direct `ncclAllReduce` for tensors larger than the internal buffer. Neither path needs to change for
this experiment.

## Validation

The source-level audit used the local Qwen3-32B ModelScope snapshot and TP=4 configuration:

```text
max_extend_tokens = 8192
max_forward_len = 8192
max_seq_len = 40960
hidden_size = 5120
dtype = BF16 (2 bytes)
selected buffer = 80.0 MiB/rank
context-window bound = 400.0 MiB/rank (not selected)
```

No GPU Stage A matrix was run because the proposed treatment would not exercise a different code
path. Running 96/128 MiB caps would increase the reservation relative to the actual control and
would not test the stated optimization.

## Regression guard

Add a unit test asserting that `SchedulerConfig.max_forward_len` follows
`max_extend_tokens`, so a future refactor cannot silently restore context-window sizing to the
server path.

## Decision

Reject the buffer-sizing optimization as unnecessary for the current serving path. Keep the
existing sizing and environment cap. Move the next optimization to TP collective attribution and
communication/computation overlap, where the current code still performs synchronous dependency
points inside row-parallel and output-projection layers.

## Artifacts

- source audit: `python/minisgl/scheduler/config.py`, `python/minisgl/engine/engine.py`, and
  `python/minisgl/kernel/pynccl.py`;
- regression test: `tests/core/test_engine_config.py`;
- no `learning/results/031_*` GPU matrix is recorded because the treatment was not distinct from
  control.
