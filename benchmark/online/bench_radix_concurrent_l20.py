from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from minisgl.benchmark.client import generate_prompt
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


GROUPS = (
    ("fill", ("A", "B", "C", "D", "E", "F")),
    ("refresh", ("A", "B")),
    ("pressure", ("G", "H", "I")),
    ("probe-survivor", ("I", "H", "G", "B", "A", "F", "E")),
    ("probe-evicted", ("D", "C")),
)
OUTPUT_LENS = {"A": 16, "B": 64, "C": 128, "D": 16, "E": 64, "F": 128, "G": 16, "H": 64, "I": 128}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent Radix Cache benchmark for L20")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--seed", type=int, default=3100042)
    parser.add_argument("--server-tp", type=int, default=4)
    parser.add_argument("--server-num-pages", type=int, default=4576)
    parser.add_argument("--server-radix-partial-eviction", action="store_true")
    parser.add_argument("--server-radix-partial-eviction-reserve-pages", type=int, default=0)
    parser.add_argument(
        "--server-radix-partial-eviction-adaptive-reserve-max-pages", type=int, default=0
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, check=False, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], capture_output=True, check=False, text=True
    ).stdout
    return {"git_revision": revision or "unknown", "git_dirty": bool(status.strip())}


async def run_request(
    client: OpenAI,
    model: str,
    prompt: str,
    output_len: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_len,
        temperature=0.0,
        extra_body={"ignore_eos": True, "top_k": 1},
    )
    timestamps = [started]
    server_uid: int | None = None
    async for chunk in response:
        if server_uid is None and chunk.id and chunk.id.startswith("cmpl-"):
            server_uid = int(chunk.id.removeprefix("cmpl-"))
        timestamps.append(time.perf_counter())
    if len(timestamps) < output_len + 1:
        raise RuntimeError(
            f"request returned {len(timestamps) - 1} stream events, expected {output_len} tokens"
        )
    token_timestamps = timestamps[: output_len + 1]
    return {
        "server_uid": server_uid,
        "output_tokens": output_len,
        "ttft_ms": (token_timestamps[1] - token_timestamps[0]) * 1000,
        "tpot_avg_ms": statistics.mean(
            b - a for a, b in zip(token_timestamps[1:], token_timestamps[2:])
        )
        * 1000,
        "e2e_s": token_timestamps[-1] - token_timestamps[0],
        "started_s": started,
        "finished_s": token_timestamps[-1],
    }


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    survivor = [run for run in runs if run["phase"] == "probe-survivor"]
    starts = [run["started_s"] for run in runs]
    finishes = [run["finished_s"] for run in runs]
    output_tokens = sum(run["output_tokens"] for run in runs)
    duration = max(finishes) - min(starts)
    return {
        "requests": len(runs),
        "output_tokens": output_tokens,
        "duration_s": duration,
        "concurrent_output_throughput_tok_s": output_tokens / duration,
        "ttft_avg_ms": statistics.mean(run["ttft_ms"] for run in runs),
        "ttft_p90_ms": percentile([run["ttft_ms"] for run in runs], 0.90),
        "survivor_group_ttft_avg_ms": statistics.mean(run["ttft_ms"] for run in survivor),
        "survivor_group_tpot_avg_ms": statistics.mean(run["tpot_avg_ms"] for run in survivor),
        "e2e_p90_s": percentile([run["e2e_s"] for run in runs], 0.90),
    }


def markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    summary = report["summary"]
    rows = "\n".join(
        f"| {run['request']} | {run['phase']} | {run['prefix']} | {run['output_len']} | "
        f"{run['server_uid']} | {run['ttft_ms']:.2f} | {run['tpot_avg_ms']:.2f} | "
        f"{run['e2e_s']:.3f} |"
        for run in report["runs"]
    )
    return f"""# L20 concurrent Radix Cache workload result

## Principle

Each phase submits its requests concurrently and waits for the whole group before starting the next
phase. This exercises adaptive reserve summation over admitted batches while preserving the
experiment 024 prefix-pressure order.

## Configuration

- Model: `{report['model']}`
- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- Tensor parallelism: `{config['server_tp']}`
- KV pages / page size: `{config['server_num_pages']} / 1`
- Partial-leaf eviction: `{config['server_radix_partial_eviction']}`
- Fixed reserve: `{config['server_radix_partial_eviction_reserve_pages']}` pages
- Adaptive reserve maximum: `{config['server_radix_partial_eviction_adaptive_reserve_max_pages']}` pages
- Concurrent groups: `fill6, refresh2, pressure3, survivor7, evicted2`
- Seed: `{config['seed']}`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | {summary['requests']} |
| Output tokens | {summary['output_tokens']} |
| Concurrent output throughput | {summary['concurrent_output_throughput_tok_s']:.2f} token/s |
| Average TTFT | {summary['ttft_avg_ms']:.2f} ms |
| P90 TTFT | {summary['ttft_p90_ms']:.2f} ms |
| Survivor-group average TTFT | {summary['survivor_group_ttft_avg_ms']:.2f} ms |
| Survivor-group average TPOT | {summary['survivor_group_tpot_avg_ms']:.2f} ms |
| P90 E2E | {summary['e2e_p90_s']:.3f} s |

## Requests

| Request | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
{rows}

Cache eviction and match details are stored in the paired rank-0 telemetry JSONL file.
"""


async def main() -> None:
    args = parse_args()
    if args.prompt_tokens < 2 or args.server_num_pages < 1:
        raise ValueError("prompt tokens and server pages must be positive")
    if (
        args.server_radix_partial_eviction_reserve_pages > 0
        and args.server_radix_partial_eviction_adaptive_reserve_max_pages > 0
    ):
        raise ValueError("fixed and adaptive reserve are mutually exclusive")

    random.seed(args.seed)
    async with OpenAI(base_url=args.base_url, api_key="dummy") as client:
        models = await client.models.list()
        if not models.data:
            raise RuntimeError("server returned no models")
        model = models.data[0].id
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        warmup_prompt = generate_prompt(tokenizer, 64)
        await run_request(client, model, warmup_prompt, 8)
        prompts = {label: generate_prompt(tokenizer, args.prompt_tokens) for label in OUTPUT_LENS}
        runs: list[dict[str, Any]] = []
        request_index = 0
        for phase, labels in GROUPS:
            started_group = time.perf_counter()
            results = await asyncio.gather(
                *(
                    run_request(client, model, prompts[label], OUTPUT_LENS[label])
                    for label in labels
                )
            )
            finished_group = time.perf_counter()
            for label, result in zip(labels, results, strict=True):
                request_index += 1
                result.update(
                    request=request_index,
                    phase=phase,
                    prefix=label,
                    output_len=OUTPUT_LENS[label],
                    group_size=len(labels),
                    group_duration_s=finished_group - started_group,
                )
                runs.append(result)

    measured_uids = [run["server_uid"] for run in runs]
    if any(uid is None for uid in measured_uids) or len(set(measured_uids)) != len(measured_uids):
        raise RuntimeError(f"missing or duplicate server UIDs: {measured_uids}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            **git_state(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "model": model,
        "config": {
            **vars(args),
            "markdown_out": str(args.markdown_out),
            "json_out": str(args.json_out),
            "groups": [[phase, list(labels)] for phase, labels in GROUPS],
        },
        "summary": build_summary(runs),
        "runs": runs,
    }
    print(json.dumps({"config": report["config"], "summary": report["summary"]}, indent=2))
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown_report(report))
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
