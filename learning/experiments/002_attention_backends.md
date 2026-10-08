# Experiment 002: attention backend selection on SM89

## Hypothesis

The current SM89 automatic choice (`fi` for both prefill and decode) may not be optimal. A hybrid
`fa,fi` backend may improve prefill while retaining FlashInfer decode performance; `fa` establishes
whether FlashAttention decode is competitive for this shape.

## Principle

Prefill processes many query tokens and benefits from kernels optimized for full attention. Decode
processes one new token per sequence and uses a paged KV cache, where FlashInfer is often stronger.
Mini-SGLang permits separate prefill/decode backends, so the correct comparison is end-to-end under
fixed prompt and output shapes, followed by isolated prefill-heavy and decode-heavy workloads.

## Controlled variables

Same as experiment 001 post-change: one L20, Qwen3-0.6B BF16, 128 requests, 256 input tokens, 128
output tokens, three repeats, concurrency 128, automatic graph maximum capped to 128, radix cache,
page size 1, and memory ratio 0.8. Only `--attention-backend` changes.

## Commands

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 proxychains .venv/bin/python benchmark/offline/bench_l20.py \
  --model Qwen/Qwen3-0.6B \
  --num-prompts 128 --input-len 256 --output-len 128 --repeats 3 \
  --attention-backend BACKEND --max-running-requests 128 \
  --max-seq-len 2048 --memory-ratio 0.8 \
  --markdown-out learning/experiments/002_BACKEND_raw.md \
  --json-out learning/results/002_BACKEND.json
```

`BACKEND` is `fi`, `fa,fi`, or `fa`. The experiment 001 optimized run supplies the `fi` result.

## Results

`fi` succeeded in experiment 001 at a stable median 10145.02 token/s. `fa,fi` failed on its first
warm-up prefill before measurements. The import path raised:

```text
AttributeError: module 'cutlass._mlir.dialects.nvvm' has no attribute 'RoundingModeKind'
```

Installed relevant packages:

- `sgl-kernel 0.3.21`
- `nvidia-cutlass-dsl 4.7.1`
- `flashinfer-python 0.6.18.post1`
- standalone `flash-attn` is not installed

Direct inspection confirmed CUTLASS DSL 4.7.1 does not expose `nvvm.RoundingModeKind`. A pure `fa`
run would use the same failing prefill import path, so it was not repeated.

## Interpretation

The experiment cannot compare performance until the FlashAttention dependency matrix is repaired.
This is an environment compatibility result, not evidence that FlashInfer is faster. Changing
dependencies during a backend performance experiment would add an uncontrolled variable, so that
work is deferred to a dedicated compatibility experiment.

## Decision

The compatibility issue was resolved in follow-up experiment 002a. The completed sweep showed that
`fa,fi` is effectively tied with `fi` for this workload while pure `fa` is 15.00% slower. Keep SM89
`auto` on `fi`; this is now a measured performance decision rather than a dependency workaround.

## Next step

See `002a_attention_backend_compatibility.md` for the scoped compatibility implementation, tests,
and completed backend sweep. The main roadmap continues with experiment 017.
