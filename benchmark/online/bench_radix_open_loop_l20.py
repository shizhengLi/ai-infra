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


REQUESTS = (
    (0.00, "fill", "A", 16),
    (0.00, "fill", "B", 64),
    (0.00, "fill", "C", 128),
    (0.00, "fill", "D", 16),
    (0.00, "fill", "E", 64),
    (0.00, "fill", "F", 128),
    (0.75, "refresh", "A", 16),
    (0.75, "refresh", "B", 64),
    (1.50, "pressure", "G", 16),
    (1.50, "pressure", "H", 64),
    (1.50, "pressure", "I", 128),
    (4.50, "probe-survivor", "I", 128),
    (4.50, "probe-survivor", "H", 64),
    (4.50, "probe-survivor", "G", 16),
    (4.50, "probe-survivor", "B", 64),
    (4.50, "probe-survivor", "A", 16),
    (4.50, "probe-survivor", "F", 128),
    (4.50, "probe-survivor", "E", 64),
    (6.00, "probe-evicted", "D", 16),
    (6.00, "probe-evicted", "C", 128),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop Radix Cache benchmark for L20")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--seed", type=int, default=3100042)
    parser.add_argument("--server-tp", type=int, default=4)
    parser.add_argument("--server-num-pages", type=int, default=4576)
    parser.add_argument("--server-radix-partial-eviction", action="store_true")
    parser.add_argument("--server-radix-partial-eviction-reserve-pages", type=int, default=0)
    parser.add_argument(
        "--server-radix-partial-eviction-adaptive-reserve-max-pages", type=int, default=0
    )
    parser.add_argument(
        "--server-radix-partial-eviction-adaptive-reserve-mode",
        choices=["raw", "age-aware"],
        default="raw",
    )
    parser.add_argument("--server-radix-partial-eviction-hotness-decay", type=float, default=0.0)
    parser.add_argument(
        "--server-radix-partial-eviction-protect-recent-matches", type=int, default=0
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
    client: OpenAI, model: str, prompt: str, output_len: int, offset_s: float, origin: float
) -> dict[str, Any]:
    await asyncio.sleep(max(0.0, origin + offset_s - time.perf_counter()))
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
    tokens = timestamps[: output_len + 1]
    return {
        "server_uid": server_uid,
        "output_tokens": output_len,
        "scheduled_offset_s": offset_s,
        "started_s": started,
        "finished_s": tokens[-1],
        "ttft_ms": (tokens[1] - tokens[0]) * 1000,
        "tpot_avg_ms": statistics.mean(b - a for a, b in zip(tokens[1:], tokens[2:])) * 1000,
        "e2e_s": tokens[-1] - tokens[0],
    }


def build_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    survivors = [run for run in runs if run["phase"] == "probe-survivor"]
    duration = max(run["finished_s"] for run in runs) - min(run["started_s"] for run in runs)
    return {
        "requests": len(runs),
        "output_tokens": sum(run["output_tokens"] for run in runs),
        "trace_duration_s": duration,
        "open_loop_output_throughput_tok_s": sum(run["output_tokens"] for run in runs) / duration,
        "ttft_avg_ms": statistics.mean(run["ttft_ms"] for run in runs),
        "ttft_p90_ms": percentile([run["ttft_ms"] for run in runs], 0.90),
        "survivor_ttft_avg_ms": statistics.mean(run["ttft_ms"] for run in survivors),
        "survivor_tpot_avg_ms": statistics.mean(run["tpot_avg_ms"] for run in survivors),
        "e2e_p90_s": percentile([run["e2e_s"] for run in runs], 0.90),
    }


def markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    summary = report["summary"]
    rows = "\n".join(
        f"| {run['request']} | {run['scheduled_offset_s']:.2f} | {run['phase']} | "
        f"{run['prefix']} | {run['output_len']} | {run['server_uid']} | "
        f"{run['ttft_ms']:.2f} | {run['tpot_avg_ms']:.2f} | {run['e2e_s']:.3f} |"
        for run in report["runs"]
    )
    return f"""# L20 open-loop Radix Cache workload result

## Principle

All requests are launched against one fixed open-loop trace. Clients do not wait for earlier
responses, so short and long requests can be admitted while older decodes are still running.

## Configuration

- Model: `{report['model']}`
- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- Tensor parallelism: `{config['server_tp']}`
- KV pages / page size: `{config['server_num_pages']} / 1`
- Partial-leaf eviction: `{config['server_radix_partial_eviction']}`
- Fixed reserve: `{config['server_radix_partial_eviction_reserve_pages']}` pages
- Adaptive reserve maximum: `{config['server_radix_partial_eviction_adaptive_reserve_max_pages']}` pages
- Adaptive reserve mode: `{config['server_radix_partial_eviction_adaptive_reserve_mode']}`
- Protected recent matches: `{config['server_radix_partial_eviction_protect_recent_matches']}`
- Hotness decay: `{config['server_radix_partial_eviction_hotness_decay']}`
- Trace offsets: `0.00 / 0.75 / 1.50 / 4.50 / 6.00 s`
- Seed: `{config['seed']}`

## Summary

| Metric | Value |
| --- | ---: |
| Requests | {summary['requests']} |
| Output tokens | {summary['output_tokens']} |
| Open-loop output throughput | {summary['open_loop_output_throughput_tok_s']:.2f} token/s |
| Average TTFT | {summary['ttft_avg_ms']:.2f} ms |
| P90 TTFT | {summary['ttft_p90_ms']:.2f} ms |
| Survivor average TTFT | {summary['survivor_ttft_avg_ms']:.2f} ms |
| Survivor average TPOT | {summary['survivor_tpot_avg_ms']:.2f} ms |
| P90 E2E | {summary['e2e_p90_s']:.3f} s |

## Requests

| Request | Offset s | Phase | Prefix | Output | Server UID | TTFT ms | TPOT ms | E2E s |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
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
        await run_request(client, model, warmup_prompt, 8, 0.0, time.perf_counter())
        prompts = {
            label: generate_prompt(tokenizer, args.prompt_tokens)
            for label in sorted({item[2] for item in REQUESTS})
        }
        origin = time.perf_counter() + 0.25
        results = await asyncio.gather(
            *(
                run_request(client, model, prompts[prefix], output_len, offset, origin)
                for offset, _phase, prefix, output_len in REQUESTS
            )
        )

    runs = []
    for index, ((offset, phase, prefix, output_len), result) in enumerate(
        zip(REQUESTS, results, strict=True), start=1
    ):
        result.update(
            request=index,
            phase=phase,
            prefix=prefix,
            output_len=output_len,
        )
        runs.append(result)
    uids = [run["server_uid"] for run in runs]
    if any(uid is None for uid in uids) or len(set(uids)) != len(uids):
        raise RuntimeError(f"missing or duplicate server UIDs: {uids}")

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
            "trace": [list(item) for item in REQUESTS],
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
