# Experiment 019: held-out response-barrier validation

## Status

Complete. The response-barrier treatment passed every pre-registered criterion and is accepted for
the calibrated L20/Qwen3-32B deployment profile.

## Objective

Determine whether experiment 018's decode-result-before-prefill barrier generalizes to fresh
Poisson traces across the complete light-to-overload range, without materially reducing throughput.

## Hypothesis

The active-only 2304-token prefill policy already bounds one prefill interruption, but the overlap
loop can delay a completed decode response behind the following prefill submission. Processing the
previous decode result before pending prefill work should remove this second response-visibility
delay. Because the barrier applies only at a decode-to-prefill transition, its throughput cost should
remain below 1%.

## Profiles

Both profiles use `max_prefill_streak=1`, a normal prefill budget of 8192 tokens, and a 2304-token
decode-active budget.

| Profile | Decode result before prefill |
| --- | --- |
| Active-only control | Disabled |
| Response-barrier treatment | Enabled |

No parameter will be changed in response to this experiment's intermediate results.

## Experiment matrix

- Model: Qwen3-32B BF16
- Hardware: 4 x NVIDIA L20, GPUs 0-3
- Tensor parallelism: 4 with PyNCCL
- Attention backend: FlashInfer selected by `auto`
- Memory ratio: 0.8
- CUDA Graph maximum batch size: 64
- Maximum running requests: 128
- Requests per trace: 24
- Input lengths: balanced 256/1024/4096 tokens, shuffled per trace
- Output length: 256 tokens
- Arrival rates: 0.5, 0.9, 1.5, and 2.5 requests/s
- Repeats: 3 per profile and rate
- Fresh base seed: 2400042
- TTFT SLO: 2000 ms
- Per-request maximum-TPOT SLO: 1000 ms
- Total: 24 traces and 576 requests

The benchmark derives independent seeds by rate and repeat. The two profiles reuse the exact same
seeds, prompt/input orders, and scheduled arrival offsets. These seeds were not used in experiments
008-018.

## Pre-registered acceptance criteria

Promote the response barrier into the recommended L20 profile only if:

1. Treatment TPOT SLO attainment is 100% at every arrival rate.
2. No treatment token gap exceeds one second.
3. Treatment mean output throughput is no more than 1% below the paired active-only control at each
   arrival rate.
4. All three repetitions at every rate complete without request or server failure.
5. Seeds, input orders, and scheduled arrival offsets match exactly between profiles.

Average/P90/P99 TTFT, P99/P99.9 TPOT, maximum TPOT, GPU utilization, power, and paired per-trace
throughput changes are secondary metrics. The policy remains opt-in if any primary criterion fails.

## Commands

Common server command:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m minisgl \
  --model /data2/lszlsz/.cache/modelscope/models/Qwen--Qwen3-32B/snapshots/master \
  --tp-size 4 --memory-ratio 0.8 --cuda-graph-max-bs 64 \
  --max-running-requests 128 --max-prefill-length 8192 \
  --max-prefill-streak 1 --decode-active-prefill-length 2304
```

The treatment adds `--decode-result-before-prefill`. The benchmark command is:

```bash
.venv/bin/python benchmark/online/bench_poisson_l20.py \
  --input-lens 256 1024 4096 --output-len 256 --request-count 24 \
  --arrival-rates 0.5 0.9 1.5 2.5 --repeats 3 --seed 2400042 \
  --gpu-indices 0,1,2,3 --ttft-slo-ms 2000 --tpot-slo-ms 1000 \
  --server-tp 4 --server-memory-ratio 0.8 --server-graph-max-bs 64 \
  --server-max-extend-tokens 8192 --server-max-prefill-streak 1 \
  --server-decode-active-prefill-tokens 2304
```

The treatment benchmark additionally records `--server-decode-result-before-prefill` so result
metadata captures the changed server policy.

## Planned artifacts

- `learning/experiments/019_control_raw.md`
- `learning/experiments/019_treatment_raw.md`
- `learning/results/019_control.json`
- `learning/results/019_treatment.json`

## Execution log

- The first control startup selected `/usr/bin/nvcc` (CUDA 11.5) from the non-login shell and
  failed before model loading because that compiler rejects `-std=c++20`. No benchmark request or
  measurement ran. The failed parent and all workers were stopped, GPU memory returned to zero,
  and subsequent commands explicitly pin `/usr/local/cuda-12.8` as required by experiment 000.

## Results

All 24 traces and 576 requests completed. The treatment results below contain 288 requests on 12
fresh traces.

| Rate | Profile | Output tok/s | Stddev | Avg TTFT ms | P99.9 TPOT ms | Max TPOT ms | TPOT SLO |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | Control | 114.9100 | 20.1750 | 956.44 | 819.65 | 1119.69 | 95.83% |
| 0.5 | Treatment | 114.9104 | 20.1388 | 964.34 | 764.69 | 773.70 | 100% |
| 0.9 | Control | 183.4767 | 8.0836 | 1017.95 | 825.74 | 830.86 | 100% |
| 0.9 | Treatment | 183.3462 | 8.0820 | 1032.77 | 777.48 | 781.89 | 100% |
| 1.5 | Control | 219.6906 | 19.0978 | 1637.76 | 815.22 | 835.34 | 100% |
| 1.5 | Treatment | 219.4357 | 19.1118 | 1657.68 | 781.59 | 786.56 | 100% |
| 2.5 | Control | 253.7302 | 6.2346 | 3772.16 | 791.63 | 797.46 | 100% |
| 2.5 | Treatment | 253.4128 | 6.2494 | 3787.90 | 779.72 | 781.54 | 100% |

The fresh light-load control reproduced the response-visibility failure on seed 2400042: three
requests exceeded the TPOT SLO and the trace maximum was 1119.69 ms. On the identical trace, the
treatment reduced the maximum to 773.70 ms and all 24 requests passed.

## Paired changes

| Rate | Throughput change | Average TTFT change | Max TPOT change |
| ---: | ---: | ---: | ---: |
| 0.5 | +0.0003% | +0.83% | -30.90% |
| 0.9 | -0.0711% | +1.46% | -5.89% |
| 1.5 | -0.1160% | +1.22% | -5.84% |
| 2.5 | -0.1251% | +0.42% | -2.00% |

Across the 12 individual paired traces, throughput changes range from -0.139% to +0.036%. The
largest aggregate regression is 0.125%, one eighth of the pre-registered 1% limit. The modest
0.4-1.5% average-TTFT increase is the expected cost of making the preceding decode result visible
before submitting pending prefill work.

## Acceptance evaluation

| Criterion | Result | Status |
| --- | --- | --- |
| 100% treatment TPOT SLO at every rate | 288/288 requests pass | Pass |
| No treatment token gap above one second | 0 gaps | Pass |
| Per-rate mean throughput regression <= 1% | Worst is -0.125% | Pass |
| All repetitions complete | 12/12 treatment and 12/12 control | Pass |
| Exact paired workload | Seeds, input orders, and arrival offsets match | Pass |

## Interpretation

The held-out light trace independently reproduces the failure mode that motivated experiment 018,
while the treatment removes it. This is stronger evidence than replaying only the discovery trace.
At moderate and overload rates, where the active-only control already passes, the barrier preserves
100% SLO and slightly tightens the maximum TPOT with negligible throughput cost.

The barrier should remain a scoped policy rather than a universal default. The result validates
Qwen3-32B BF16 at TP=4 on L20 for the tested mixed-length workload and arrival-rate range. Different
models, accelerators, prefill kernels, or token budgets can change the decode-to-prefill tradeoff and
should be recalibrated.

## Decision

Accept the response barrier into the recommended L20 profile:

```bash
--max-prefill-streak 1 \
--decode-active-prefill-length 2304 \
--decode-result-before-prefill
```

Keep the generic server default disabled. The recommended profile now has both mechanism evidence
from experiment 018 and fresh held-out validation from experiment 019.

## Validation

- Control and treatment each contain 12 completed traces and 288 requests.
- All paired seeds, input-length orders, and scheduled arrival offsets match exactly.
- Result metadata records the barrier as false for control and true for treatment.
- All servers exited cleanly; ports 1919/1920 and GPU allocations returned to idle.
- Raw reports: `learning/experiments/019_control_raw.md` and
  `learning/experiments/019_treatment_raw.md`.
- Machine-readable data: `learning/results/019_control.json` and
  `learning/results/019_treatment.json`.

## Next experiment

Experiment 020 begins the deferred radix-cache phase. Add opt-in hit/eviction telemetry, then
measure no-shared-prefix, shared-system-prefix, and multi-turn workloads before changing cache
policy.
