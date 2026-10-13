"""Benchmark low-precision GEMMs on Qwen3-32B tensor-parallel layer shapes.

This is a kernel screening benchmark, not an end-to-end serving benchmark. FP8 and
INT8 results report both GEMM-only time and dynamic activation quantization plus
GEMM time. Weight quantization and Marlin repacking are offline operations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from sgl_kernel import (
    fp8_scaled_mm,
    gptq_marlin_gemm,
    gptq_marlin_repack,
    int8_scaled_mm,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
)
from sgl_kernel.scalar_type import scalar_types


QWEN3_32B_TP4_SHAPES = {
    "qkv_proj": (5120, 2560),
    "o_proj": (2048, 5120),
    "gate_up_proj": (5120, 12800),
    "down_proj": (6400, 5120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L20 quantized GEMM screening benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5100042)
    parser.add_argument("--cache-flush-mib", type=int, default=128)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def _measure(
    fn: Callable[[], torch.Tensor | None],
    warmup: int,
    repetitions: int,
    samples: int,
    cache_flush: torch.Tensor,
) -> dict[str, Any]:
    for _ in range(warmup):
        cache_flush.add_(1.0)
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        elapsed_ms = 0.0
        for _ in range(repetitions):
            # The flush is ordered before the start event and is not part of the
            # measured interval. This avoids benchmarking L2-resident weights.
            cache_flush.add_(1.0)
            start.record()
            fn()
            end.record()
            end.synchronize()
            elapsed_ms += start.elapsed_time(end)
        values.append(elapsed_ms / repetitions)
    return {
        "samples_ms": values,
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "pstdev_ms": statistics.pstdev(values),
    }


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    ref_norm = torch.linalg.vector_norm(reference_f).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(1, -1), reference_f.reshape(1, -1)
    )
    return {
        "relative_l2": (torch.linalg.vector_norm(delta) / ref_norm).item(),
        "cosine_similarity": cosine.item(),
        "max_abs": delta.abs().max().item(),
    }


def _quantize_rows(
    value: torch.Tensor, qmax: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = (value.float().abs().amax(dim=1) / qmax).clamp_min(1e-12)
    scaled = (value.float() / scales[:, None]).clamp(-qmax, qmax)
    quantized = scaled.to(dtype) if dtype.is_floating_point else scaled.round().to(dtype)
    return quantized, scales


def _pack_gptq_int4(values: torch.Tensor) -> torch.Tensor:
    size_k, size_n = values.shape
    if size_k % 8:
        raise ValueError("INT4 GPTQ packing requires K divisible by 8")
    packed = torch.zeros((size_k // 8, size_n), device=values.device, dtype=torch.int32)
    for index in range(8):
        packed |= values[index::8].to(torch.int32) << (4 * index)
    return packed


def _permute_marlin_scales(scales: torch.Tensor, size_n: int) -> torch.Tensor:
    permutation = []
    for index in range(4):
        permutation.extend(2 * index + offset for offset in (0, 1, 8, 9, 16, 17, 24, 25))
    return scales.reshape(-1, 32)[:, permutation].reshape(-1, size_n).contiguous()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() or "unknown"


def _run_shape(
    label: str,
    size_m: int,
    size_k: int,
    size_n: int,
    warmup: int,
    repetitions: int,
    samples: int,
    seed: int,
    cache_flush_mib: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    activation = torch.randn((size_m, size_k), device="cuda", dtype=torch.bfloat16) * 0.1
    # Store [N, K] so .T has the column-major stride required by scaled_mm.
    weight_nk = torch.randn((size_n, size_k), device="cuda", dtype=torch.bfloat16) * 0.1
    weight_kn = weight_nk.T
    bf16_reference = activation @ weight_kn
    cache_flush = torch.empty(
        cache_flush_mib * 2**18, device="cuda", dtype=torch.float32
    ).zero_()

    bf16_timing = _measure(
        lambda: torch.mm(activation, weight_kn), warmup, repetitions, samples, cache_flush
    )

    fp8_weight, fp8_weight_scale = _quantize_rows(weight_nk, 448.0, torch.float8_e4m3fn)
    fp8_activation = torch.empty_like(activation, dtype=torch.float8_e4m3fn)
    fp8_activation_scale = torch.empty(size_m, device="cuda", dtype=torch.float32)

    def fp8_quantize() -> None:
        sgl_per_token_quant_fp8(activation, fp8_activation, fp8_activation_scale)

    def fp8_gemm() -> torch.Tensor:
        return fp8_scaled_mm(
            fp8_activation,
            fp8_weight.T,
            fp8_activation_scale,
            fp8_weight_scale,
            torch.bfloat16,
        )

    def fp8_linear() -> torch.Tensor:
        fp8_quantize()
        return fp8_gemm()

    fp8_quantize()
    fp8_output = fp8_gemm()
    fp8_dequant_a = (
        fp8_activation.float() * fp8_activation_scale[:, None]
    ).to(torch.bfloat16)
    fp8_dequant_w = (fp8_weight.float() * fp8_weight_scale[:, None]).to(torch.bfloat16)
    fp8_quant_reference = fp8_dequant_a @ fp8_dequant_w.T
    fp8_gemm_timing = _measure(fp8_gemm, warmup, repetitions, samples, cache_flush)
    fp8_linear_timing = _measure(fp8_linear, warmup, repetitions, samples, cache_flush)

    int8_weight, int8_weight_scale = _quantize_rows(weight_nk, 127.0, torch.int8)
    int8_activation = torch.empty_like(activation, dtype=torch.int8)
    int8_activation_scale = torch.empty((size_m, 1), device="cuda", dtype=torch.float32)

    def int8_quantize() -> None:
        sgl_per_token_group_quant_int8(
            activation,
            int8_activation,
            int8_activation_scale,
            size_k,
            1e-10,
            -128.0,
            127.0,
            False,
            False,
            None,
            False,
        )

    def int8_gemm() -> torch.Tensor:
        return int8_scaled_mm(
            int8_activation,
            int8_weight.T,
            int8_activation_scale,
            int8_weight_scale,
            torch.bfloat16,
        )

    def int8_linear() -> torch.Tensor:
        int8_quantize()
        return int8_gemm()

    int8_quantize()
    int8_output = int8_gemm()
    int8_dequant_a = (
        int8_activation.float() * int8_activation_scale
    ).to(torch.bfloat16)
    int8_dequant_w = (int8_weight.float() * int8_weight_scale[:, None]).to(torch.bfloat16)
    int8_quant_reference = int8_dequant_a @ int8_dequant_w.T
    int8_gemm_timing = _measure(int8_gemm, warmup, repetitions, samples, cache_flush)
    int8_linear_timing = _measure(int8_linear, warmup, repetitions, samples, cache_flush)

    weight_max = weight_kn.float().amax(dim=0, keepdim=True)
    weight_min = weight_kn.float().amin(dim=0, keepdim=True)
    int4_scale = torch.maximum((weight_max / 7.0).abs(), (weight_min / -8.0).abs())
    int4_scale = int4_scale.clamp_min(1e-12).to(torch.bfloat16)
    int4_signed = (weight_kn.float() / int4_scale.float()).round().clamp(-8, 7).to(torch.int32)
    int4_dequant_weight = int4_signed.to(torch.bfloat16) * int4_scale
    int4_gptq = _pack_gptq_int4(int4_signed + 8)
    empty_perm = torch.empty(0, device="cuda", dtype=torch.int32)
    int4_marlin = gptq_marlin_repack(int4_gptq, empty_perm, size_k, size_n, 4)
    int4_marlin_scale = _permute_marlin_scales(int4_scale, size_n)
    workspace = torch.zeros(
        max(256, (size_n // 64) * 16), device="cuda", dtype=torch.int32
    )

    def int4_gemm() -> torch.Tensor:
        return gptq_marlin_gemm(
            activation,
            None,
            int4_marlin,
            int4_marlin_scale,
            None,
            None,
            None,
            None,
            workspace,
            scalar_types.uint4b8,
            size_m,
            size_n,
            size_k,
        )

    int4_output = int4_gemm()
    int4_quant_reference = activation @ int4_dequant_weight
    int4_timing = _measure(int4_gemm, warmup, repetitions, samples, cache_flush)
    torch.cuda.synchronize()

    bf16_ms = bf16_timing["median_ms"]
    fp8_ms = fp8_gemm_timing["median_ms"]
    fp8_linear_ms = fp8_linear_timing["median_ms"]
    int8_ms = int8_gemm_timing["median_ms"]
    int8_linear_ms = int8_linear_timing["median_ms"]
    int4_ms = int4_timing["median_ms"]
    weight_elements = size_k * size_n
    result = {
        "label": label,
        "m": size_m,
        "k": size_k,
        "n": size_n,
        "bf16": {
            "timing": bf16_timing,
            "weight_mib": weight_elements * 2 / 2**20,
        },
        "fp8_w8a8": {
            "gemm_timing": fp8_gemm_timing,
            "dynamic_linear_timing": fp8_linear_timing,
            "gemm_speedup": bf16_ms / fp8_ms,
            "dynamic_linear_speedup": bf16_ms / fp8_linear_ms,
            "weight_mib": (weight_elements + size_n * 4) / 2**20,
            "quantization_error": _error(fp8_output, bf16_reference),
            "kernel_error": _error(fp8_output, fp8_quant_reference),
        },
        "int8_w8a8": {
            "gemm_timing": int8_gemm_timing,
            "dynamic_linear_timing": int8_linear_timing,
            "gemm_speedup": bf16_ms / int8_ms,
            "dynamic_linear_speedup": bf16_ms / int8_linear_ms,
            "weight_mib": (weight_elements + size_n * 4) / 2**20,
            "quantization_error": _error(int8_output, bf16_reference),
            "kernel_error": _error(int8_output, int8_quant_reference),
        },
        "int4_weight_only": {
            "gemm_timing": int4_timing,
            "gemm_speedup": bf16_ms / int4_ms,
            "weight_mib": (weight_elements / 2 + size_n * 2) / 2**20,
            "quantization_error": _error(int4_output, bf16_reference),
            "kernel_error": _error(int4_output, int4_quant_reference),
        },
    }

    del (
        activation,
        weight_nk,
        weight_kn,
        bf16_reference,
        fp8_weight,
        fp8_dequant_a,
        fp8_dequant_w,
        fp8_quant_reference,
        int8_weight,
        int8_dequant_a,
        int8_dequant_w,
        int8_quant_reference,
        int4_signed,
        int4_dequant_weight,
        int4_gptq,
        int4_marlin,
        int4_quant_reference,
        cache_flush,
    )
    torch.cuda.empty_cache()
    return result


def _markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report["measurements"]:
        bf16_ms = item["bf16"]["timing"]["median_ms"]
        for name, key in (
            ("BF16", "bf16"),
            ("FP8 W8A8", "fp8_w8a8"),
            ("INT8 W8A8", "int8_w8a8"),
            ("INT4 W4A16", "int4_weight_only"),
        ):
            data = item[key]
            timing = data.get("gemm_timing", data.get("timing"))
            speedup = bf16_ms / timing["median_ms"]
            quant_error = data.get("quantization_error", {}).get("relative_l2")
            kernel_error = data.get("kernel_error", {}).get("relative_l2")
            rows.append(
                f"| {item['label']} | {item['m']} | {name} | {timing['median_ms']:.4f} | "
                f"{speedup:.2f}x | {data['weight_mib']:.2f} | "
                f"{'-' if quant_error is None else f'{quant_error:.5f}'} | "
                f"{'-' if kernel_error is None else f'{kernel_error:.5f}'} |"
            )
    warmup = report["config"]["warmup"]
    repetitions = report["config"]["repetitions"]
    samples = report["config"]["samples"]
    cache_flush_mib = report["config"]["cache_flush_mib"]
    return f"""# Experiment 051 raw quantized GEMM results

- Timestamp: `{report['timestamp_utc']}`
- Git revision: `{report['environment']['git_revision']}`
- GPU: `{report['environment']['gpu_name']}` (SM `{report['environment']['compute_capability']}`)
- PyTorch/CUDA: `{report['environment']['torch']}` / `{report['environment']['torch_cuda']}`
- sgl-kernel: `{report['environment']['sgl_kernel']}`
- Warmup/repetitions/samples: `{warmup}` / `{repetitions}` / `{samples}`
- L2 cache flush before each timed call: `{cache_flush_mib}` MiB (flush excluded from timing)

Times are kernel-only medians. FP8/INT8 dynamic activation-quantization timings are retained in the
JSON artifact. Weight quantization and Marlin repacking are offline and excluded.

| Layer (TP4 local) | M | Path | Median ms | vs BF16 | Weight MiB | Quant rel-L2 | Kernel rel-L2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}
"""


def main() -> None:
    args = parse_args()
    if min(args.batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if args.warmup < 0 or min(args.repetitions, args.samples) < 1:
        raise ValueError("warmup must be non-negative; repetitions and samples must be positive")
    if args.cache_flush_mib < 1:
        raise ValueError("cache flush size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise RuntimeError(f"this L20 experiment requires SM89, found SM{major}{minor}")

    import sgl_kernel

    measurements = []
    for shape_index, (label, (size_k, size_n)) in enumerate(QWEN3_32B_TP4_SHAPES.items()):
        for size_m in args.batch_sizes:
            print(f"benchmarking {label}: M={size_m}, K={size_k}, N={size_n}", flush=True)
            measurements.append(
                _run_shape(
                    label,
                    size_m,
                    size_k,
                    size_n,
                    args.warmup,
                    args.repetitions,
                    args.samples,
                    args.seed + shape_index * 100 + size_m,
                    args.cache_flush_mib,
                )
            )

    props = torch.cuda.get_device_properties(0)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "git_revision": _git_revision(),
            "gpu_name": props.name,
            "compute_capability": f"{major}.{minor}",
            "gpu_memory_gib": props.total_memory / 2**30,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "sgl_kernel": getattr(sgl_kernel, "__version__", "unknown"),
        },
        "config": {
            "model": "Qwen3-32B",
            "tensor_parallel_size": 4,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "samples": args.samples,
            "seed": args.seed,
            "cache_flush_mib": args.cache_flush_mib,
        },
        "measurements": measurements,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(report))


if __name__ == "__main__":
    main()
