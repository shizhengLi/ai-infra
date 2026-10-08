# Experiment 020: Radix Cache observability baseline

## Status

Complete. All acceptance checks passed. The metrics, workloads, and interpretation rules below were
fixed before collecting the formal data.

## Objective

Add opt-in Radix Cache telemetry and establish how Mini-SGLang reuses KV prefixes for independent,
shared-system-prefix, and multi-turn requests on Qwen3-32B TP=4.

## Principle

Radix Cache performance must be explained in tokens rather than only request counts. A request can
match none, part, or almost all of its prefill prefix. The scheduler intentionally matches only the
first `input_len - 1` tokens so at least one token is extended. The primary reuse metric is therefore:

```text
token_hit_rate = sum(matched_tokens) / sum(matchable_tokens)
```

Instrumentation is placed at the ownership boundaries:

1. `CacheManager.match_req`: matchable and matched tokens for each request.
2. `CacheManager.cache_req`: existing and newly inserted prefix tokens.
3. `CacheManager._allocate`: requested and actual eviction tokens.

The telemetry also records miss/partial/full-hit request counts, current evictable/protected cache
size, maximum resident tokens, and per-operation samples. It is enabled only on TP rank 0 when an
output path is supplied; the default path performs no counter or sample updates.

## Workloads

Each scenario uses a fresh server so cache state cannot leak across scenarios. A distinct warmup
request is excluded from measured metrics.

| Scenario | Requests | Structure | Expected behavior |
| --- | ---: | --- | --- |
| Unique | 12 | Independent 1024-token user prompts | Only template-level reuse; near-zero token hit rate |
| Shared system | 12 | Shared 768-token system text plus unique 256-token user text | First request cold, later requests reuse most system tokens |
| Multi-turn | 4 conversations x 3 turns | Per-conversation 256-token system text, 128-token new user text, 32 generated tokens per turn | Turn 1 cold; turns 2-3 reuse growing history |

Requests are sequential. This intentionally removes overlap races: a later request is submitted only
after the prior request has finished and its prefix is evictable in the tree.

## Fixed environment

- Model: Qwen3-32B BF16
- Hardware: 4 x NVIDIA L20, GPUs 0-3
- Tensor parallelism: 4 with PyNCCL
- CUDA 12.8 and FlashInfer attention
- Memory ratio: 0.8, page size 1
- CUDA Graph maximum batch size: 64
- Normal/decode-active prefill budgets: 8192/2304
- Maximum prefill streak: 1
- Decode result before prefill: enabled
- Output length: 32 tokens
- Base seed: 3000042

The normal KV pool is retained in experiment 020. Eviction counters are validated by unit tests, but
an absence of runtime eviction is a valid capacity result. A later pressure experiment may reduce
the page pool only after the baseline is measured.

## Decision rules

Instrumentation is accepted if:

1. Default-disabled telemetry produces no samples or counter updates.
2. Unit tests cover miss, partial hit, full hit, insertion, eviction, snapshot, and sample draining.
3. Rank-0 JSONL output is valid and contains all measured requests.
4. Unique prompts show materially lower token hit rate than shared-system and later multi-turn turns.
5. Reported matched tokens never exceed matchable tokens, and cache size accounting remains valid.

This is an observability experiment, not a cache-policy performance claim. No eviction or matching
policy will be changed based on intermediate scenario results.

## Planned artifacts

- `benchmark/online/bench_radix_l20.py`
- `learning/experiments/020_unique_raw.md`
- `learning/experiments/020_shared_system_raw.md`
- `learning/experiments/020_multi_turn_raw.md`
- `learning/results/020_*.json`
- `learning/results/020_*_telemetry.jsonl`

## Implementation

The change adds an opt-in `--cache-telemetry-path` JSONL stream. Rank 0 records cumulative counters
and drains detailed operation samples at idle, shutdown, or every 1024 operations under sustained
load. With no output path, the recording methods return before changing counters or allocating
sample dictionaries.

The paired benchmark generates token-count-controlled prompts, performs one excluded warmup, sends
requests sequentially, and records the actual chat-template input length. Multi-turn requests feed
the generated assistant text back into the next turn, rather than approximating conversation
history.

## Protocol correction

The first shared-system and multi-turn attempts accidentally used server defaults
`memory_ratio=0.9` and CUDA Graph maximum batch size 160. This differed from the preregistered
`0.8/64` environment, so those attempts were excluded before comparing scenarios. Their raw notes
are retained as `020_*_pilot_config_mismatch.md`; formal shared-system and multi-turn runs were
repeated from fresh servers with a 305,066-token KV pool and graph sizes ending at 64. The unique
run already used the registered configuration.

## Results

Telemetry below excludes warmup UID 0 and includes measured UIDs 1-12.

| Scenario | Matchable tokens | Matched tokens | Token hit rate | Average TTFT | P50 TTFT | Post-first TTFT | New resident tokens* |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unique | 12,372 | 36 | 0.291% | 370.13 ms | 349.56 ms | 349.52 ms | 12,799 |
| Shared system | 12,432 | 8,537 | 68.670% | 144.95 ms | 108.14 ms | 107.62 ms | 4,358 |
| Multi-turn | 6,780 | 4,110 | 60.619% | 117.41 ms | 70.79 ms | n/a | 3,133 |

`*` The resident-token counters include the distinct warmup because they describe final server
state, while hit-rate counters in this table exclude it.

All requests are classified as partial hits because the chat template itself shares 1-3 leading
tokens. The token metric correctly distinguishes the effectively cold unique workload from useful
prefix reuse. After the first cache-filling request, the shared-system workload matches exactly 776
of 1,036 matchable tokens on every request. Its post-first TTFT is 69.21% lower than the comparable
unique workload, while TPOT stays flat at about 23.94 ms.

### Multi-turn growth

| Turn | Matchable tokens/request | Matched tokens/request | Token hit rate | Average TTFT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 396 | 1-3 | 0.631% average | 145.29 ms after the first shape-cold request |
| 2 | 565 | 428 | 75.752% | 68.75 ms |
| 3 | 734 | 597 | 81.335% | 70.89 ms |

The matched prefix grows by 169 tokens between turns 2 and 3, tracking the newly cached prior-turn
segment: user message, generated assistant response, and chat-template delimiters. This verifies that
the online API path reuses real conversation history, not only synthetic system prefixes.

The first measured request in each fresh server is slower than subsequent equal-shape requests even
after the short warmup, so average TTFT includes shape-cold overhead. Cache conclusions therefore use
both token telemetry and post-first latency, not the aggregate latency alone.

## Capacity result

No runtime eviction occurred. Final/high-water resident sizes were 12,799 tokens for unique, 4,358
for shared-system, and 3,133 for multi-turn, versus 305,066 available pages. The largest workload
used only 4.20% of capacity. This is a measured capacity result, not evidence that eviction policy is
optimal; policy comparison requires a deliberately constrained or much longer workload.

## Decision

Accept the telemetry and benchmark infrastructure. Every preregistered check passed:

1. Disabled telemetry is a tested no-op.
2. Unit tests cover classification, insertion, eviction, snapshots, draining, and actual cache
   manager integration.
3. All three rank-0 JSONL streams are valid and contain exactly the 12 measured match samples.
4. Matched tokens never exceed matchable tokens; shared and later multi-turn requests show material
   reuse.
5. Cache integrity checks passed and all formal runs shut down cleanly.

Do not change radix matching or eviction policy from this baseline. Experiment 021 should create a
controlled-capacity reuse workload that forces eviction, revisits previously cached prefixes, and
measures useful-hit retention before selecting a policy change.

## Verification

- Focused cache telemetry/cache allocation tests: 11 passed
- Full `tests/core`: 28 passed
- Three formal telemetry JSONL files parse successfully; each contains measured UIDs 1-12 exactly
  once and zero `matched_tokens > matchable_tokens` violations
- `compileall` and `git diff --check`: passed; Black check passed for the new benchmark, telemetry
  implementation, CLI/config, and tests (the scheduler retains its pre-existing local formatting)
- Post-run cleanup: no GPU compute processes and no listener on port 1919
