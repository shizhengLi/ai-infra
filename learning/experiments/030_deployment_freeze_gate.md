# Experiment 030: Adaptive reserve deployment freeze gate

## Status

Complete. The adaptive reserve is frozen as an opt-in phase-batched profile. No generic
open-loop reserve policy is accepted, and no new eviction heuristic is enabled by default.

## Objective

Experiments 026-029 tested four ways to make the output-aware adaptive reserve safe for open-loop
arrivals. Every candidate reduced allocator re-entry, but every candidate lost at least one paired
survivor prefix. Experiment 030 is the pre-registered decision gate: determine whether the server
currently has a reliable workload-level identity that can be used as a hard protection signal. If
not, freeze the strongest measured profile and close this policy branch.

## Decision rule

The phase-batched profile can be frozen only when experiment 025 remains within all of its original
acceptance limits: no paired survivor loss, at least 25% fewer complete-sequence eviction calls,
no more than 5% reclamation increase, exact target reclamation, no more than 3% throughput
regression, and no more than 5% survivor-TTFT regression. A generic policy is rejected when it
fails the survivor rule, even if its performance metrics pass.

## Identity feasibility

The current cache API receives token IDs, request output state, and cache match history. The
letters used in the benchmark (`A` through `I`) are harness labels, not request metadata passed to
the server or part of the cache key. Adding those labels as a new cache identity would change API
and cache-key semantics, so it is not a safe optimization that can be evaluated in this branch.
The existing age, recent-match, and decayed-hotness signals are therefore only heuristics, not
explicit workload identities.

## Evidence matrix

| Experiment | Policy | Survivor rule | Performance result | Decision |
| ---: | --- | --- | --- | --- |
| 025 | Adaptive max 128, phase batches | 21/21 paired matches preserved | Calls -80.0%, throughput -0.036%, survivor TTFT -1.64% | Freeze as scoped profile |
| 026 | Raw adaptive caps 64/128/256, open loop | Every cap lost at least one pair | Calls -70.0% to -85.0%; cap256 TTFT +78.8% | Reject generic use |
| 027 | Age-aware caps 64/128, open loop | cap64 lost A; cap128 lost B | Calls -65.5%/-77.6%; latency limits passed | Reject generic use |
| 028 | Recent-match windows 4/8, open loop | hot4 lost G; hot8 lost A | Calls -69.0%; latency limits passed | Reject generic use |
| 029 | Decayed hotness 0.90/0.99, open loop | decay0.90 lost A; decay0.99 lost G | Calls -70.2%/-66.7%; latency limits passed | Reject generic use |

The four open-loop experiments all fail the same correctness-oriented gate. This is stronger
evidence than their call-reduction numbers: a policy that reclaims a prefix needed by the next
probe is not safe for a generic serving workload.

## Frozen deployment profile

The accepted scope is deliberately narrow:

- partial-leaf eviction enabled;
- adaptive reserve cap 128 pages;
- concurrent phase-batched admission matching experiment 025;
- explicit opt-in configuration only;
- generic defaults remain disabled;
- age-aware, recent-match, and decayed-hotness options remain experimental and disabled.

This is a deployment profile, not a new global default. The scheduler cannot infer phase barriers or
future workload identity from the current request API, so enabling the profile for arbitrary
open-loop traffic would violate the experiment boundary.

## Reproducibility and verification

The decision is backed by the existing fresh-server artifacts:

- `learning/experiments/025_concurrent_*_raw.md`
- `learning/results/025_concurrent_*.json`
- `learning/results/025_concurrent_*_telemetry.jsonl`
- `learning/experiments/026_*` through `learning/experiments/029_*`

The machine-readable decision is stored in
`learning/results/030_deployment_gate.json`. The focused regression set passed with `54 passed, 5
subtests passed`:

```text
PYTHONPATH=python .venv/bin/python -m pytest -q -o addopts='' \
  tests/core/test_partial_leaf_eviction.py tests/core/test_cache_allocate.py \
  tests/core/test_cache_telemetry.py tests/core/test_schedule_policy.py \
  tests/core/test_scheduler.py tests/engine/test_graph.py tests/kernel/test_pynccl_link.py
```

The full suite collected 63 passing tests but could not be clean because three existing CUDA JIT
tests hit the environment's `/usr/bin/nvcc` rejection of `-std=c++20`, and `test_tensor` is exposed
as a pytest fixture-shaped function without fixtures. JSON parsing, bytecode compilation, and
`git diff --check` passed.

## Reopen conditions

Reopen this branch only when all of the following are available:

1. A new API supplies an explicit workload-level prefix identity or admission class.
2. That identity is propagated without changing tokenization or cache-key semantics.
3. A fresh three-seed open-loop matrix passes the existing survivor, reclamation, throughput, and
   TTFT rules.

Until then, phase-batched adaptive reserve is the final recommendation for this optimization
branch, and no additional generic hotness heuristic is planned.
