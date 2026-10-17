import pytest
import torch

from minisgl.distributed import DistributedInfo, set_tp_info
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _resolve_kv_cache_dtype
from minisgl.kvcache.mha_pool import FP8_KV_DTYPE, MHAKVCache, calibrate_fp8_kv_scale
from minisgl.server.args import parse_args


def test_kv_cache_dtype_cli_is_default_compatible() -> None:
    args, _ = parse_args(["--model", "unused", "--dtype", "bfloat16"])
    assert args.kv_cache_dtype == "auto"
    assert args.kv_cache_scale == 1.0
    assert args.kv_cache_calibration == "fixed"
    assert args.kv_cache_calibration_summary is None


def test_kv_cache_fp8_cli_and_scale() -> None:
    args, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "float8_e4m3fn",
            "--kv-cache-scale",
            "0.25",
        ]
    )
    assert args.kv_cache_dtype == "float8_e4m3fn"
    assert args.kv_cache_scale == 0.25
    assert args.kv_cache_calibration == "fixed"


def test_kv_cache_first_batch_calibration_is_finite() -> None:
    k = torch.tensor([[-2.0, 0.5]], dtype=torch.bfloat16)
    v = torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)
    scale = calibrate_fp8_kv_scale(k, v)
    assert scale == pytest.approx(3.0 / (448.0 * 0.95), rel=1e-5)


def test_kv_cache_dtype_resolution() -> None:
    base = dict(model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    assert _resolve_kv_cache_dtype(EngineConfig(**base)) is torch.bfloat16
    assert _resolve_kv_cache_dtype(
        EngineConfig(**base, kv_cache_dtype="float8_e4m3fn")
    ) is torch.float8_e4m3fn


def test_fp8_cache_allocates_half_width_storage() -> None:
    set_tp_info(rank=0, size=1)
    cache = MHAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=8,
        num_pages=4,
        page_size=1,
        dtype=FP8_KV_DTYPE,
        device=torch.device("cpu"),
        scale=0.125,
    )
    assert cache.dtype is FP8_KV_DTYPE
    assert cache.scale == 0.125
    assert cache.get_scale(0) == 0.125


def test_fp8_calibration_summary_is_rank_scoped(tmp_path) -> None:
    cache = MHAKVCache(
        num_kv_heads=1,
        num_layers=1,
        head_dim=8,
        num_pages=1,
        page_size=1,
        dtype=FP8_KV_DTYPE,
        device=torch.device("cpu"),
        calibration="first_batch",
        calibration_summary=str(tmp_path / "kv.json"),
    )
    cache.write_calibration_summary(str(tmp_path / "kv.json"), rank=2)
    summary = (tmp_path / "kv.rank2.json").read_text(encoding="utf-8")
    assert '"rank": 2' in summary
    assert '"saturation_fraction": 0.0' in summary
    assert cache.k_cache(0).element_size() == 1


def test_kv_cache_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MHAKVCache(1, 1, 8, 1, 1, torch.bfloat16, torch.device("cpu"), scale=0.0)
