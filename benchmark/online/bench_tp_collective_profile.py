"""Microbenchmark PyNCCL collectives at several message sizes.

Run with torchrun, for example:

    PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \
      benchmark/online/bench_tp_collective_profile.py --tp 4 --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from minisgl.distributed import set_tp_info
from minisgl.kernel import init_pynccl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-SGLang TP collective profile")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def _init_process_group() -> None:
    dist.init_process_group(backend="gloo", init_method="env://")


def _measure(comm, elements: int, dtype: torch.dtype, warmup: int, repetitions: int) -> dict[str, float]:
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(elements, dtype=dtype, device=device)
    for _ in range(warmup):
        comm.all_reduce(tensor, "sum")
    torch.cuda.synchronize(device)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        comm.all_reduce(tensor, "sum")
    end.record()
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    message_bytes = elements * tensor.element_size()
    avg_ms = elapsed_ms / repetitions
    return {
        "rank": rank,
        "message_bytes": message_bytes,
        "avg_ms": avg_ms,
        "algorithmic_gib_per_s": message_bytes / (avg_ms / 1000) / 2**30,
    }


def main() -> None:
    args = parse_args()
    if args.tp < 2 or args.warmup < 0 or args.repetitions < 1:
        raise ValueError("tp must be >=2, warmup must be non-negative, repetitions must be positive")
    if min(args.tokens) < 1 or args.hidden_size < 1:
        raise ValueError("tokens and hidden-size must be positive")
    if int(os.environ.get("LOCAL_WORLD_SIZE", args.tp)) != args.tp:
        raise ValueError("torchrun LOCAL_WORLD_SIZE must match --tp")

    _init_process_group()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    set_tp_info(rank, args.tp)
    dtype = _dtype(args.dtype)
    tp_cpu_group = dist.group.WORLD
    assert tp_cpu_group is not None
    comm = init_pynccl(
        tp_rank=rank,
        tp_size=args.tp,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max(args.tokens) * args.hidden_size * torch.tensor([], dtype=dtype).element_size(),
    )

    measurements: list[dict[str, float | int]] = []
    for token_count in args.tokens:
        local = _measure(
            comm,
            token_count * args.hidden_size,
            dtype,
            args.warmup,
            args.repetitions,
        )
        gathered: list[dict[str, float | int] | None] | None = [None] * args.tp if rank == 0 else None
        dist.gather_object(local, gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            measurements.append(
                {
                    "tokens": token_count,
                    "message_bytes": local["message_bytes"],
                    "avg_ms_mean": sum(float(item["avg_ms"]) for item in gathered) / args.tp,
                    "avg_ms_max": max(float(item["avg_ms"]) for item in gathered),
                    "algorithmic_gib_per_s_mean": sum(
                        float(item["algorithmic_gib_per_s"]) for item in gathered
                    )
                    / args.tp,
                }
            )

    if rank == 0:
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_shape": {"hidden_size": args.hidden_size, "dtype": args.dtype},
            "config": {
                "tp": args.tp,
                "tokens": args.tokens,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
            },
            "measurements": measurements,
        }
        print(json.dumps(report, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
