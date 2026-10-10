# Experiment 034: CUDA Graph tail variance after direct PyNCCL

## Status

Complete. The Graph-mode tail regression from experiment 033 persists after per-concurrency warmup
and five steady-state repetitions. Direct PyNCCL improves throughput and average TPOT, but its P90
TPOT is still worse at C=8/32. Do not enable it automatically in CUDA Graph mode.

## Objective and method

Experiment 033 found direct communication compatible with CUDA Graph replay but observed P90 TPOT
regressions. The earlier benchmark had only one global warmup. This experiment added the opt-in
`--steady-warmup-repeats` benchmark parameter, which runs unrecorded batches before each concurrency
group. The test then used two unrecorded warmups plus five measured repetitions for each of C=1, 8,
and 32.

Configuration: Qwen3-32B BF16, TP4 on L20 GPUs 0-3, `fi`, CUDA Graph sizes through batch 64,
memory ratio 0.7, maximum prefill 2048, streak 1, active decode-prefill budget 1024, input 256,
output 32, seed `3400042`. The symmetric-buffer control and direct treatment used identical server
and client settings.

## Results

| C | Symmetric tok/s | Direct tok/s | Throughput change | Avg TPOT change | P90 TPOT change | P99.9 TPOT change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.08 | 38.59 | +1.34% | -0.22% | -0.06% | +0.32% |
| 8 | 166.83 | 179.65 | +7.68% | -2.06% | +2.64% | -20.60% |
| 32 | 267.67 | 303.66 | +13.45% | -8.09% | +11.91% | -20.63% |

The control throughput standard deviation was only 0.04, 0.32, and 0.05 tok/s for C=1/8/32;
the direct treatment was similarly stable. This rules out the experiment 033 conclusion being only
a first-request artifact. Direct mode reduces the extreme P99.9 tail, but shifts enough samples in
the P90 region to regress the pre-registered P90 criterion at C=8/32. Mean GPU memory was 32,523 MiB
for control and 32,493 MiB for treatment.

Artifacts:

- `learning/results/034_tp4_graph_symmetric_steady_seed3400042.json`
- `learning/results/034_tp4_graph_direct_steady_seed3400042.json`
- `learning/experiments/034_tp4_graph_symmetric_steady_seed3400042_raw.md`
- `learning/experiments/034_tp4_graph_direct_steady_seed3400042_raw.md`

## Decision

- Reject automatic direct-buffer selection for CUDA Graph mode under the current P90 TPOT SLO.
- Retain direct PyNCCL as an explicit graph-disabled L20 TP4/TP8 profile, where both throughput and
  P90 TPOT improved in experiments 032/033.
- Keep the generic default unchanged. The communication optimization is understood well enough for
  an interview-quality deployment profile; the remaining Graph-mode issue is a scheduling/tail
  tradeoff, not a reason to introduce asynchronous collectives prematurely.

The communication branch is now closed pending a new Graph scheduler hypothesis. The next useful
optimization should measure which decode batches are pushed into the P90 tail before changing the
collective implementation.
