# Experiment 009: decode-active prefill budget

## Status

Running.

## Hypothesis

Experiment 008 bounded starvation to one 8192-token prefill batch, approximately 2.9 seconds.
Reducing the prefill budget to 4096 only while decode is runnable should roughly halve the maximum
anchor token gap. Retaining 8192 while decode is idle avoids globally reducing prefill batch size.

## Principle

Prefill batch duration determines the minimum interruption bound when prefill and decode execute on
the same stream. The scheduler can use workload state to select its token budget: full-size prefill
when no decode latency is at risk, and smaller chunks when decode requests are already running.

This experiment retains `max_prefill_streak=1`, changes only the decode-active budget from the
default 8192 to 4096, and reuses the mixed-arrival and closed-batch workloads from experiment 008.

## Controlled variables

- Model: Qwen3-32B BF16
- Hardware: GPUs 0-3, four NVIDIA L20 cards
- Tensor parallelism: 4, PyNCCL
- Normal prefill budget: 8192 tokens
- Maximum prefill streak: 1
- Baseline decode-active budget: 8192 tokens
- Treatment decode-active budget: 4096 tokens
- Mixed workload: 8 anchors, then 24 long-prefill requests after 2 seconds
- Closed workload: 32 requests with 1024 input and 256 output tokens
- Repeats: 3

## Results

Pending.

## Interpretation

Pending.

## Decision

Pending.
