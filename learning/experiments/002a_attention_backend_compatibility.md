# Experiment 002a: restore FlashAttention 3 on L20

## Status

Complete on 2026-10-08. The attention backend sweep is no longer blocked.

## Objective

Make the installed `sgl-kernel 0.3.21` FlashAttention 3 kernels usable on L20 (SM89) without
downgrading CUTLASS DSL or weakening the FlashAttention 4 path, then finish the experiment 002
backend comparison.

## Root cause

`sgl_kernel.flash_attn` imports its optional `_fa4_interface`. That module was written for
`nvidia-cutlass-dsl==4.2.1`, but the environment contains CUTLASS DSL 4.7.1 because the installed
FlashInfer requires a newer CUTLASS DSL. Importing the optional FA4 module therefore raises:

```text
AttributeError: module 'cutlass._mlir.dialects.nvvm' has no attribute 'RoundingModeKind'
```

The outer `sgl_kernel.flash_attn` module only catches `ImportError`, so this optional FA4 failure
also prevents loading the independent FA3 functions. L20 uses FA3, not FA4.

Downgrading CUTLASS DSL would trade one known failure for a FlashInfer dependency conflict. Editing
the virtual environment would also be non-reproducible. The fix therefore belongs in Mini-SGLang's
backend adapter.

## Implementation principle

`python/minisgl/attention/fa.py` now loads the FlashAttention function through a cached helper. It
uses a compatibility retry only when both of these conditions hold:

1. Mini-SGLang selected FA3 (`version == 3`).
2. The exception exactly identifies the known missing CUTLASS `nvvm.RoundingModeKind` API.

For that retry, the loader temporarily supplies a stub optional `_fa4_interface`, imports the FA3
entry points, and immediately restores `sys.modules`. FA4 on SM100 and every unrelated import error
still fail normally. This keeps the workaround narrow and makes a future dependency upgrade expose
new incompatibilities instead of hiding them.

## Validation

Focused unit tests cover the known FA3 recovery, unrelated `AttributeError` propagation, and the
rule that FA4 must not use the workaround:

```bash
.venv/bin/python -m pytest -q tests/core/test_fa_compat.py
```

Result: **4 passed**.

A direct BF16 FA3 kernel check on L20 used tensors of shape `(16, 16, 128)`. All output values were
finite; against PyTorch SDPA, maximum absolute error was 0.00390625 and mean absolute error was
1.192e-07.

Small end-to-end runs then verified both `fa,fi` and pure `fa`, including model loading, CUDA Graph
capture, prefill, decode, and token generation. The full controlled sweep used the same cached
Qwen3-0.6B snapshot for all backends:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmark/offline/bench_l20.py \
  --model /data2/lszlsz/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --num-prompts 128 --input-len 256 --output-len 128 --repeats 3 \
  --attention-backend BACKEND --cuda-graph-max-bs 128 \
  --max-running-requests 128 --max-seq-len 2048 --memory-ratio 0.8
```

## Performance results

| Backend | Initialization (s) | Mean (token/s) | Median (token/s) | Population SD | vs `fi` median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fi` | 4.6099 | 10142.94 | 10148.73 | 13.66 | baseline |
| `fa,fi` | 4.4744 | 10130.52 | 10131.53 | 3.13 | -0.17% |
| `fa` | 4.6128 | 8624.67 | 8626.22 | 5.27 | -15.00% |

Raw reports:

- `002a_fi_raw.md`
- `002a_fa_fi_raw.md`
- `002a_fa_raw.md`

## Interpretation

The repair is functionally successful: FA3 now runs on SM89 under the existing dependency set.
For this decode-heavy 256/128 workload, replacing only prefill with FA changes end-to-end
throughput by -0.17%, which is too small to justify a default-policy change. Replacing both phases
with FA makes decode slower and regresses median throughput by 15.00%.

This result does not establish that FA prefill is never useful. Longer-prompt, shorter-output
workloads should be measured separately if prefill throughput becomes a bottleneck.

## Decision

- Close the attention backend compatibility blocker.
- Keep automatic SM89 selection on `fi`.
- Retain explicit `fa` and `fa,fi` as usable experimental/deployment choices.
- Continue the main roadmap at experiment 017; this follow-up does not renumber it.
