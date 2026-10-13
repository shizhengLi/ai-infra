from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from minisgl.layers import LinearOProj, LinearReplicated, LinearRowParallel
from minisgl.layers import linear as linear_module
from minisgl.models.weight import _get_merge_info, _shard_tensor
from minisgl.quantization import FP8_DTYPE, QuantizationConfig, quantize_fp8_weight
from minisgl.server.args import parse_args


def test_quantization_cli_is_default_disabled() -> None:
    default, _ = parse_args(["--model", "unused", "--dtype", "bfloat16"])
    fp8, _ = parse_args(
        ["--model", "unused", "--dtype", "bfloat16", "--quantization", "fp8"]
    )

    assert default.quantization is None
    assert default.quantization_config is None
    assert fp8.quantization == "fp8"
    assert fp8.quantization_config == QuantizationConfig(method="fp8")


def test_unknown_quantization_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported quantization method"):
        QuantizationConfig.from_name("int4")


def test_fp8_weight_quantization_is_per_output_channel() -> None:
    weight = torch.linspace(-2.0, 2.0, 4 * 128, dtype=torch.float32).reshape(4, 128)

    quantized, scale = quantize_fp8_weight(weight)
    reconstructed = quantized.float() * scale[:, None]
    relative_error = torch.linalg.vector_norm(reconstructed - weight) / torch.linalg.vector_norm(
        weight
    )

    assert quantized.dtype == FP8_DTYPE
    assert quantized.shape == weight.shape
    assert scale.dtype == torch.float32
    assert scale.shape == (4,)
    assert relative_error.item() < 0.04


def test_fp8_linear_loads_float_weight_and_prequantized_state() -> None:
    config = QuantizationConfig(method="fp8")
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    layer = LinearReplicated(128, 128, has_bias=False, quantization_config=config)

    layer.load_state_dict({"weight": weight})
    saved = layer.state_dict()
    restored = LinearReplicated(128, 128, has_bias=False, quantization_config=config)
    restored.load_state_dict(dict(saved))

    assert layer.weight.dtype == FP8_DTYPE
    assert layer.weight_scale.shape == (128,)
    assert torch.equal(restored.weight.float(), layer.weight.float())
    assert torch.equal(restored.weight_scale, layer.weight_scale)


def test_prequantized_fp8_weight_requires_scale() -> None:
    layer = LinearReplicated(
        128,
        128,
        has_bias=False,
        quantization_config=QuantizationConfig(method="fp8"),
    )

    with pytest.raises(ValueError, match="Missing scale"):
        layer.load_state_dict({"weight": torch.empty(128, 128, dtype=FP8_DTYPE)})


def test_unquantized_linear_preserves_f_linear_behavior() -> None:
    weight = torch.randn(16, 32)
    inputs = torch.randn(4, 32)
    layer = LinearReplicated(32, 16, has_bias=False)
    layer.load_state_dict({"weight": weight})

    assert torch.equal(layer.forward(inputs), F.linear(inputs, weight))


def test_tp_sharding_handles_fp8_scale_metadata() -> None:
    scale = torch.arange(8, dtype=torch.float32)

    col_rank_1 = _shard_tensor(
        "model.layers.0.self_attn.q_proj.weight_scale", scale, 1, 2, num_kv_heads=2
    )
    row_rank_1 = _shard_tensor(
        "model.layers.0.self_attn.o_proj.weight_scale", scale, 1, 2, num_kv_heads=2
    )
    kv_rank_1 = _shard_tensor(
        "model.layers.0.self_attn.k_proj.weight_scale", scale, 1, 4, num_kv_heads=2
    )

    assert torch.equal(col_rank_1, scale[4:])
    assert torch.equal(row_rank_1, scale)
    assert torch.equal(kv_rank_1, scale[:4])
    assert _get_merge_info("model.layers.0.self_attn.q_proj.weight_scale") == (
        "model.layers.0.self_attn.qkv_proj.weight_scale",
        "q",
        ("q", "k", "v"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP8 kernel")
def test_fp8_linear_kernel_matches_dequantized_reference() -> None:
    if torch.cuda.get_device_capability()[0:2] < (8, 9):
        pytest.skip("FP8 scaled GEMM requires SM89 or newer")

    device = torch.device("cuda")
    config = QuantizationConfig(method="fp8")
    weight = torch.randn(128, 128, device=device, dtype=torch.bfloat16) * 0.1
    inputs = torch.randn(8, 128, device=device, dtype=torch.bfloat16) * 0.1
    layer = LinearReplicated(128, 128, has_bias=False, quantization_config=config)
    layer.load_state_dict({"weight": weight})

    output = layer.forward(inputs)
    from sgl_kernel import sgl_per_token_quant_fp8

    quantized_inputs = torch.empty_like(inputs, dtype=FP8_DTYPE)
    input_scale = torch.empty(inputs.shape[0], device=device, dtype=torch.float32)
    sgl_per_token_quant_fp8(inputs, quantized_inputs, input_scale)
    dequantized_inputs = (quantized_inputs.float() * input_scale[:, None]).to(torch.bfloat16)
    dequantized_weight = (layer.weight.float() * layer.weight_scale[:, None]).to(
        torch.bfloat16
    )
    reference = F.linear(dequantized_inputs, dequantized_weight)
    relative_error = torch.linalg.vector_norm(
        output.float() - reference.float()
    ) / torch.linalg.vector_norm(reference.float())

    assert output.dtype == torch.bfloat16
    assert output.shape == (8, 128)
    assert relative_error.item() < 0.01


@pytest.mark.parametrize("layer_type", [LinearOProj, LinearRowParallel])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP8 kernel")
def test_tp_linear_wrappers_use_fp8_dispatch(
    monkeypatch: pytest.MonkeyPatch, layer_type: type[LinearOProj] | type[LinearRowParallel]
) -> None:
    if torch.cuda.get_device_capability()[0:2] < (8, 9):
        pytest.skip("FP8 scaled GEMM requires SM89 or newer")
    monkeypatch.setattr(
        linear_module, "get_tp_info", lambda: SimpleNamespace(rank=0, size=1)
    )
    layer = layer_type(
        128,
        128,
        has_bias=False,
        quantization_config=QuantizationConfig(method="fp8"),
    )
    layer.load_state_dict(
        {"weight": torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)}
    )

    output = layer.forward(torch.randn(4, 128, device="cuda", dtype=torch.bfloat16))

    assert output.dtype == torch.bfloat16
    assert output.shape == (4, 128)
