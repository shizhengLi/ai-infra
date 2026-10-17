from __future__ import annotations

import argparse
import json
import time

import torch

from minisgl.core import Req, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Req host-token append allocation")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def make_req(input_len: int, output_len: int) -> Req:
    return Req(
        input_ids=torch.arange(input_len, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=0,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,  # type: ignore[arg-type]
    )


def measure_preallocated(input_len: int, output_len: int) -> float:
    req = make_req(input_len, output_len)
    next_token = torch.tensor([1], dtype=torch.int32)
    start = time.perf_counter()
    for _ in range(output_len):
        req.complete_one()
        req.append_host(next_token)
    return time.perf_counter() - start


def measure_cat(input_len: int, output_len: int) -> float:
    ids = torch.arange(input_len, dtype=torch.int32)
    next_token = torch.tensor([1], dtype=torch.int32)
    start = time.perf_counter()
    for _ in range(output_len):
        ids = torch.cat([ids, next_token])
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if min(args.input_len, args.output_len, args.repeats) <= 0:
        raise ValueError("input length, output length, and repeats must be positive")
    preallocated = [measure_preallocated(args.input_len, args.output_len) for _ in range(args.repeats)]
    cat = [measure_cat(args.input_len, args.output_len) for _ in range(args.repeats)]
    result = {
        "input_len": args.input_len,
        "output_len": args.output_len,
        "repeats": args.repeats,
        "preallocated_s": preallocated,
        "cat_s": cat,
        "preallocated_mean_s": sum(preallocated) / len(preallocated),
        "cat_mean_s": sum(cat) / len(cat),
        "preallocated_allocations": 1,
        "cat_allocations": args.output_len,
        "preallocated_copy_bytes": (args.input_len + args.output_len)
        * torch.tensor([], dtype=torch.int32).element_size(),
        "cat_copy_bytes": sum(args.input_len + index for index in range(args.output_len))
        * torch.tensor([], dtype=torch.int32).element_size(),
    }
    result["speedup"] = result["cat_mean_s"] / result["preallocated_mean_s"]
    result["copy_reduction"] = result["cat_copy_bytes"] / result["preallocated_copy_bytes"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
