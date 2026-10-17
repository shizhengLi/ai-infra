from __future__ import annotations

import argparse
import json
import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper


FP8_MAX = 448.0


def _parse_scales(value: str) -> list[float]:
    scales = [float(item) for item in value.split(",") if item]
    if not scales or any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("scales must be a comma-separated list of positives")
    return scales


def _build_wrapper(
    workspace: torch.Tensor,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dtype: torch.dtype,
    seq_len: int,
) -> BatchDecodeWithPagedKVCacheWrapper:
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend="fa2",
        use_tensor_cores=True,
    )
    indptr = torch.tensor([0, seq_len], dtype=torch.int32)
    indices = torch.arange(seq_len, dtype=torch.int32, device=workspace.device)
    last_page_len = torch.tensor([1], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=1,
        seq_lens=seq_lens,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        non_blocking=False,
    )
    return wrapper


def _measure_ms(fn, repeats: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    q = torch.randn((1, args.num_qo_heads, args.head_dim), device=device, dtype=torch.bfloat16)
    k = torch.randn(
        (args.seq_len, args.num_kv_heads, args.head_dim), device=device, dtype=torch.bfloat16
    ) * args.activation_std
    v = torch.randn_like(k) * args.activation_std
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    bf16_wrapper = _build_wrapper(
        workspace,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        kv_dtype=torch.bfloat16,
        seq_len=args.seq_len,
    )
    fp8_wrapper = _build_wrapper(
        workspace,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        kv_dtype=torch.float8_e4m3fn,
        seq_len=args.seq_len,
    )
    q_for_run = q
    paged_bf16 = (k.view(args.seq_len, 1, args.num_kv_heads, args.head_dim), v.view(args.seq_len, 1, args.num_kv_heads, args.head_dim))
    ref = bf16_wrapper.run(q_for_run, paged_bf16)
    bf16_ms = _measure_ms(lambda: bf16_wrapper.run(q_for_run, paged_bf16), args.repeats)
    calibrated_scale = max(float(k.float().abs().amax().item()), float(v.float().abs().amax().item())) / (FP8_MAX * 0.95)
    calibrated_scale = max(calibrated_scale, 1e-6)
    rows = []
    for scale in args.scales:
        k_fp8 = (k.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        v_fp8 = (v.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        paged_fp8 = (k_fp8.view(args.seq_len, 1, args.num_kv_heads, args.head_dim), v_fp8.view(args.seq_len, 1, args.num_kv_heads, args.head_dim))
        out = fp8_wrapper.run(q_for_run, paged_fp8, k_scale=scale, v_scale=scale)
        error = out.float() - ref.float()
        rows.append(
            {
                "scale": scale,
                "max_abs_error": float(error.abs().max().item()),
                "relative_l2": float(error.norm().item() / ref.float().norm().item()),
                "cosine": float(torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()),
                "saturation_fraction": float(((k.float().abs() >= scale * FP8_MAX).sum() + (v.float().abs() >= scale * FP8_MAX).sum()).item() / (2 * k.numel())),
                "fp8_ms": _measure_ms(lambda: fp8_wrapper.run(q_for_run, paged_fp8, k_scale=scale, v_scale=scale), args.repeats),
            }
        )
    return {
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "seed": args.seed,
        "seq_len": args.seq_len,
        "num_qo_heads": args.num_qo_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "activation_std": args.activation_std,
        "calibrated_scale": calibrated_scale,
        "bf16_ms": bf16_ms,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BF16 and FP8 KV attention on FlashInfer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--activation-std", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=6100042)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--scales", type=_parse_scales, default=_parse_scales("0.03125,0.0625,0.125,0.25,0.5,1.0"))
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
