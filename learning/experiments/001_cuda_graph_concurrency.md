# Experiment 001: concurrency-aware CUDA Graph sizing

## Hypothesis

When `max_running_req` is lower than the automatic CUDA Graph limit, graph sizes above maximum
runnable concurrency are unreachable. Capping capture sizes should reduce initialization work and
graph memory without reducing steady-state throughput.

## Principle

The scheduler owns only `max_running_req` table slots, so a decode batch cannot exceed that value.
The current automatic graph policy uses only free GPU memory: on this L20 it selects 160 even when
maximum concurrency is 128. Graphs for 136, 144, 152, and 160 can never be replayed in this setup.

The same sizing helper also returns `[1, 2, 4]` for an explicit maximum of 1, and rounds a maximum
such as 10 down to 8. The intended invariant is: capture sizes are positive, unique, no larger than
runnable concurrency, and include the requested effective maximum.

## Controlled variables

- GPU: one NVIDIA L20 (GPU 0), SM89
- Model/dtype: Qwen3-0.6B, BF16
- Attention: FlashInfer (`fi`)
- Requests: 128, each 256 input and 128 output tokens
- Repeats/seeds: 3; base seed 42, incremented per repeat
- Max running requests: 128
- KV cache: radix, page size 1, memory ratio 0.8
- CUDA/PyTorch: CUDA toolkit 12.8, PyTorch 2.9.1+cu128

## Baseline command

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 proxychains .venv/bin/python benchmark/offline/bench_l20.py \
  --model Qwen/Qwen3-0.6B \
  --num-prompts 128 --input-len 256 --output-len 128 --repeats 3 \
  --attention-backend fi --max-running-requests 128 \
  --max-seq-len 2048 --memory-ratio 0.8 \
  --markdown-out learning/experiments/001_baseline_raw.md \
  --json-out learning/results/001_baseline.json
```

## Failed attempt

The first invocation added `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Graph capture completed, but
Transformers 4.57.3 called Hugging Face `model_info` while patching tokenizer behavior and raised
`OfflineModeIsEnabled`. No throughput measurement was produced. Subsequent commands use the existing
proxy rather than forced offline mode.

Observed baseline graph list before the tokenizer failure:

```text
[1, 2, 4, 8, 16, 24, ..., 128, 136, 144, 152, 160]
```

The four sizes above 128 are unreachable under the configured concurrency.

## Results

| Metric | Baseline | Concurrency-aware | Change |
| --- | ---: | ---: | ---: |
| Captured graph sizes | 23 | 19 | -4 unreachable graphs |
| Maximum graph batch | 160 | 128 | matches scheduler limit |
| Engine initialization | 8.3615 s | 7.3553 s | -12.03% |
| Free memory after graph capture | about 8.40 GiB | 8.44 GiB | about +0.04 GiB |
| Median throughput | 10103.48 token/s | 10145.02 token/s | +0.41% |

Baseline throughput runs: 8078.15, 10152.98, 10103.48 token/s. The first run contained an
additional warm-up effect; the latter two were stable.

Post-change throughput runs: 10127.02, 10145.02, 10154.64 token/s, population standard deviation
11.44 token/s. This change is not expected to improve a batch size already covered by both graph
sets, so the small throughput difference is treated as noise rather than a claimed speedup.

Raw reports:

- `learning/experiments/001_baseline_raw.md`
- `learning/experiments/001_optimized_raw.md`
- `learning/results/001_baseline.json`
- `learning/results/001_optimized.json`

Focused standard-library tests cover disabled graphs, maxima 1/3/10, concurrency capping, automatic
policy, duplicate sizes, invalid sizes, and explicit sizes above concurrency.

## Decision

Accept. The change removes provably unreachable work, improves initialization in this measurement,
preserves steady-state throughput, and fixes small/non-aligned maximum semantics.

## Next step

Compare FlashInfer, FlashAttention, and hybrid prefill/decode backends on the same L20 workload.
