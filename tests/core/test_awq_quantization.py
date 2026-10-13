from types import SimpleNamespace

import pytest
import torch

from minisgl.layers import LinearReplicated
from minisgl.layers import linear as linear_module
from minisgl.models.weight import _get_merge_dim, _shard_tensor
from minisgl.quantization import QuantizationConfig
from minisgl.quantization.awq import _pack_cols, _unpack_cols
from minisgl.server.args import parse_args


AWQ_METADATA = {
    "bits": 4,
    "group_size": 128,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
}
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_awq(values: torch.Tensor) -> torch.Tensor:
    size_k, size_n = values.shape
    ordered = values.reshape(size_k, size_n // 8, 8)[:, :, AWQ_ORDER].reshape(size_k, size_n)
    return _pack_cols(ordered, size_k, size_n)


def test_awq_config_requires_calibrated_checkpoint_metadata() -> None:
    parsed, _ = parse_args(
        ["--model", "unused", "--dtype", "bfloat16", "--quantization", "awq"]
    )

    assert parsed.quantization == "awq"
    assert QuantizationConfig.from_name("awq", AWQ_METADATA) == QuantizationConfig(
        method="awq", bits=4, group_size=128, zero_point=True
    )
    with pytest.raises(ValueError, match="requires quantization_config"):
        QuantizationConfig.from_name("awq")
    with pytest.raises(ValueError, match="requires 4-bit GEMM"):
        QuantizationConfig.from_name("awq", {**AWQ_METADATA, "bits": 8})


def test_awq_pack_helpers_round_trip() -> None:
    values = torch.arange(2 * 64, dtype=torch.int32).reshape(2, 64) % 16

    packed = _pack_cols(values, 2, 64)

    assert torch.equal(_unpack_cols(packed, 2, 64), values)


def test_tp_sharding_and_merge_dimensions_handle_awq_metadata() -> None:
    qweight = torch.arange(128 * 16, dtype=torch.int32).reshape(128, 16)
    scales = torch.arange(2 * 128, dtype=torch.bfloat16).reshape(2, 128)

    assert torch.equal(
        _shard_tensor("model.layers.0.self_attn.q_proj.qweight", qweight, 1, 2, 2),
        qweight[:, 8:],
    )
    assert torch.equal(
        _shard_tensor("model.layers.0.self_attn.o_proj.qweight", qweight, 1, 2, 2),
        qweight[64:],
    )
    assert torch.equal(
        _shard_tensor("model.layers.0.self_attn.q_proj.scales", scales, 1, 2, 2),
        scales[:, 64:],
    )
    assert torch.equal(
        _shard_tensor("model.layers.0.self_attn.o_proj.scales", scales, 1, 2, 2),
        scales[1:],
    )
    assert _get_merge_dim("model.layers.0.self_attn.q_proj.qweight") == 1
    assert _get_merge_dim("model.layers.0.self_attn.q_proj.scales") == 1
    assert _get_merge_dim("model.layers.0.self_attn.q_proj.weight") == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Marlin")
def test_awq_marlin_linear_matches_dequantized_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("Marlin requires SM80 or newer")
    monkeypatch.setattr(
        linear_module, "get_tp_info", lambda: SimpleNamespace(rank=0, size=1)
    )
    torch.manual_seed(5400042)
    size_k = size_n = 128
    quantized = torch.randint(0, 16, (size_k, size_n), device="cuda", dtype=torch.int32)
    zero_points = torch.randint(4, 12, (1, size_n), device="cuda", dtype=torch.int32)
    scales = (torch.rand(1, size_n, device="cuda", dtype=torch.bfloat16) + 0.25) / 16
    layer = LinearReplicated(
        size_k,
        size_n,
        has_bias=False,
        quantization_config=QuantizationConfig(
            method="awq", bits=4, group_size=128, zero_point=True
        ),
    )
    layer.load_state_dict(
        {
            "qweight": _pack_awq(quantized),
            "scales": scales,
            "qzeros": _pack_awq(zero_points),
        }
    )
    inputs = torch.randn(8, size_k, device="cuda", dtype=torch.bfloat16) * 0.1

    output = layer.forward(inputs)
    reference_weight = (quantized - zero_points).to(torch.bfloat16) * scales
    reference = inputs @ reference_weight
    relative_error = torch.linalg.vector_norm(
        output.float() - reference.float()
    ) / torch.linalg.vector_norm(reference.float())

    assert output.shape == (8, size_n)
    assert output.dtype == torch.bfloat16
    assert relative_error.item() < 0.01
