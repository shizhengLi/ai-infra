from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible Mini-SGLang offline benchmark for L20"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", "--attn", default="fi")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-prefill-length", type=int, default=8192)
    parser.add_argument("--memory-ratio", type=float, default=0.8)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--cache-type", choices=["radix", "naive"], default="radix")
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def make_workload(
    args: argparse.Namespace, repeat: int
) -> tuple[list[list[int]], list[SamplingParams]]:
    rng = random.Random(args.seed + repeat)
    prompts = [
        [rng.randint(0, 10_000) for _ in range(args.input_len)]
        for _ in range(args.num_prompts)
    ]
    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
        for _ in prompts
    ]
    return prompts, sampling_params


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def git_worktree_state() -> tuple[bool, str]:
    status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, check=False, text=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], capture_output=True, check=False
    ).stdout
    return bool(status.strip()), hashlib.sha256(diff).hexdigest()[:12]


def markdown_report(result: dict[str, Any]) -> str:
    config = result["config"]
    environment = result["environment"]
    measurements = result["measurements"]
    run_rows = "\n".join(
        f"| {run['repeat']} | {run['duration_s']:.4f} | {run['throughput_tok_s']:.2f} |"
        for run in measurements["runs"]
    )
    graph_max = config["cuda_graph_max_bs"]
    return f"""# L20 offline benchmark

## Purpose

Measure end-to-end offline decode throughput with deterministic, non-shared-prefix inputs. Each
repeat uses a different seed so the radix cache cannot turn later repeats into prefix-cache tests.

## Principle

The workload keeps input and requested output lengths fixed. `ignore_eos=True` guarantees the same
number of generated tokens in every run. Initialization time is reported separately because CUDA
Graph capture and kernel JIT affect startup but not steady-state throughput.

## Environment

- Timestamp (UTC): `{result['timestamp_utc']}`
- Git revision: `{environment['git_revision']}`
- Git worktree dirty: `{environment['git_dirty']}`
- Tracked diff SHA-256: `{environment['git_diff_sha256']}`
- GPU: `{environment['gpu_name']}` (SM `{environment['compute_capability']}`)
- GPU memory: `{environment['gpu_memory_gib']:.2f} GiB`
- Python: `{environment['python']}`
- PyTorch: `{environment['torch']}`
- PyTorch CUDA: `{environment['torch_cuda']}`

## Configuration

- Model: `{config['model']}`
- Attention backend: `{config['attention_backend']}`
- Requests: `{config['num_prompts']}`
- Input/output tokens per request: `{config['input_len']}/{config['output_len']}`
- Repeats: `{config['repeats']}`
- Max running requests: `{config['max_running_requests']}`
- CUDA Graph max batch size: `{graph_max if graph_max is not None else 'auto'}`
- Resolved CUDA Graph sizes: `{config['resolved_cuda_graph_bs']}`
- Memory ratio: `{config['memory_ratio']}`
- Page size/cache: `{config['page_size']}` / `{config['cache_type']}`

## Results

- Engine initialization: **{measurements['initialization_s']:.4f} s**
- Mean throughput: **{measurements['throughput_mean_tok_s']:.2f} token/s**
- Median throughput: **{measurements['throughput_median_tok_s']:.2f} token/s**
- Population standard deviation: **{measurements['throughput_pstdev_tok_s']:.2f} token/s**

| Repeat | Duration (s) | Throughput (token/s) |
| ---: | ---: | ---: |
{run_rows}

## Conclusion

This file records raw benchmark facts. Interpret the comparison and next action in
`learning/progress.md` and the corresponding numbered experiment note.
"""


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if min(args.num_prompts, args.input_len, args.output_len) < 1:
        raise ValueError("request count and token lengths must be positive")

    config = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "repeats": args.repeats,
        "seed": args.seed,
        "attention_backend": args.attention_backend,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "max_running_requests": args.max_running_requests,
        "max_seq_len": args.max_seq_len,
        "max_prefill_length": args.max_prefill_length,
        "memory_ratio": args.memory_ratio,
        "page_size": args.page_size,
        "cache_type": args.cache_type,
    }

    init_start = time.perf_counter()
    llm = LLM(
        args.model,
        attention_backend=args.attention_backend,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_running_req=args.max_running_requests,
        max_seq_len_override=args.max_seq_len,
        max_extend_tokens=args.max_prefill_length,
        memory_ratio=args.memory_ratio,
        page_size=args.page_size,
        cache_type=args.cache_type,
    )
    initialization_s = time.perf_counter() - init_start
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    config["resolved_cuda_graph_bs"] = llm.engine.graph_runner.graph_bs_list
    git_dirty, git_diff_sha256 = git_worktree_state()

    # Warm up sampling and the short-prefill path without sharing a prefix with measured inputs.
    llm.generate([[10_001, 10_002, 10_003]], SamplingParams(temperature=0.0, max_tokens=4))

    total_output_tokens = args.num_prompts * args.output_len
    runs = []
    for repeat in range(args.repeats):
        prompts, sampling_params = make_workload(args, repeat)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        duration_s = time.perf_counter() - start
        actual_tokens = sum(len(output["token_ids"]) for output in outputs)
        if actual_tokens != total_output_tokens:
            raise RuntimeError(f"expected {total_output_tokens} output tokens, got {actual_tokens}")
        throughput = actual_tokens / duration_s
        runs.append(
            {
                "repeat": repeat + 1,
                "duration_s": duration_s,
                "output_tokens": actual_tokens,
                "throughput_tok_s": throughput,
            }
        )
        print(
            f"repeat={repeat + 1} duration={duration_s:.4f}s "
            f"throughput={throughput:.2f} token/s",
            flush=True,
        )

    throughputs = [run["throughput_tok_s"] for run in runs]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "git_revision": git_revision(),
            "git_dirty": git_dirty,
            "git_diff_sha256": git_diff_sha256,
            "gpu_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_gib": properties.total_memory / (1024**3),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "config": config,
        "measurements": {
            "initialization_s": initialization_s,
            "runs": runs,
            "throughput_mean_tok_s": statistics.mean(throughputs),
            "throughput_median_tok_s": statistics.median(throughputs),
            "throughput_pstdev_tok_s": statistics.pstdev(throughputs),
        },
    }

    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown_report(result))


if __name__ == "__main__":
    main()
