# Experiment 050: Held-Out Final Validation

## Purpose

Validate the selected L20 serving profile on a fresh seed and longer output, including both unique
prefixes and shared system prefixes. This is the final check before presenting the optimization as a
deployment recommendation.

## Fixed profile and workload

- Model: Qwen3-32B BF16, TP=4 on L20 GPUs 0-3.
- Communication: `MINISGL_PYNCCL_MAX_BUFFER_SIZE=0`.
- CUDA Graphs: disabled (`--cuda-graph-max-bs 0`).
- Scheduler: maximum prefill 2048, prefill streak 1, decode-active prefill budget 1024.
- Output: 128 tokens per request, with EOS ignored.
- Requests: 12 sequential requests in each scenario.
- Fresh seed: `5000042`.
- Unique scenario: 1024-token independent user prompts.
- Shared scenario: a 768-token shared system prompt plus 256-token unique user prompts.

Each scenario used a fresh server, so its cache state was isolated. The harness verified that every
request returned the requested 128 streamed tokens.

## Results

| Scenario | Requests | Avg input tokens | Output tok/s | Avg TTFT ms | P50 TTFT ms | P90 TTFT ms | Avg TPOT ms | Avg E2E s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unique prefix | 12 | 1032 | 35.87 | 288.18 | 286.51 | 289.11 | 25.83 | 3.569 |
| Shared system | 12 | 1037 | 37.23 | 120.56 | 94.25 | 96.69 | 26.12 | 3.438 |

The shared-system first request had 407.03 ms TTFT. The remaining eleven requests were between
93.85 and 96.69 ms, demonstrating that the shared system prefix was reused after the first request.
Compared with unique prefixes, shared-system throughput improved 3.79%, average TTFT improved
58.16%, and average TPOT changed +1.13%.

## Interpretation

The selected profile is stable for long decode output and both cache regimes. Unique-prefix requests
show a tight TTFT distribution around 286-289 ms, so the longer output does not create a new tail
problem. Shared-prefix reuse materially reduces prefill TTFT after the first request without changing
decode TPOT. This supports separating prefill/cache behavior from decode/collective behavior in the
interview explanation.

This is a validation result, not proof of universal production capacity. The workload is sequential,
TP4 only, and uses 12 requests per scenario; experiment 049 remains the evidence for the open-loop
TTFT capacity knee.

## Final decision

- Accept the explicit TP4 L20 profile: Direct PyNCCL, graph-disabled, prefill streak 1, active prefill
  budget 1024, and maximum prefill 2048.
- Keep generic Mini-SGLang defaults unchanged; the selected profile is scoped to this L20/Qwen3-32B
  serving configuration.
- Do not enable LL128, Tree, non-zero PyNCCL buffers, Graph-tail barriers, or eager Graph fallbacks.
- Close the planned optimization loop. Further work should use a new hypothesis and a new baseline,
  not continue parameter tuning without a measured bottleneck.

## Artifacts

- `results/050_unique_seed5000042.json` and `.md`
- `results/050_shared_seed5000042.json` and `.md`
- `results/050_unique_telemetry.jsonl`
- `results/050_shared_telemetry.jsonl`

Both servers exited cleanly; all eight GPUs reported zero utilization and zero used memory afterward.
