from types import SimpleNamespace

import pytest

from minisgl.attention import fa


@pytest.fixture(autouse=True)
def clear_flash_attn_loader_cache():
    fa._load_flash_attn_with_kvcache.cache_clear()
    yield
    fa._load_flash_attn_with_kvcache.cache_clear()


def test_fa3_skips_incompatible_optional_fa4(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def flash_attn_with_kvcache():
        return None

    def import_module(name: str):
        nonlocal calls
        calls += 1
        assert name == "sgl_kernel.flash_attn"
        if calls == 1:
            raise AttributeError(
                "module 'cutlass._mlir.dialects.nvvm' has no attribute 'RoundingModeKind'"
            )
        fa4_stub = fa.sys.modules["sgl_kernel._fa4_interface"]
        assert fa4_stub.flash_attn_varlen_func is None
        return SimpleNamespace(flash_attn_with_kvcache=flash_attn_with_kvcache)

    monkeypatch.delitem(fa.sys.modules, "sgl_kernel._fa4_interface", raising=False)
    monkeypatch.setattr(fa.importlib, "import_module", import_module)

    loaded = fa._load_flash_attn_with_kvcache(version=3)

    assert loaded is flash_attn_with_kvcache
    assert calls == 2
    assert "sgl_kernel._fa4_interface" not in fa.sys.modules


@pytest.mark.parametrize("version", [3, 4])
def test_flash_attn_loader_preserves_unrelated_attribute_errors(
    monkeypatch: pytest.MonkeyPatch, version: int
):
    def import_module(name: str):
        raise AttributeError("unrelated import failure")

    monkeypatch.setattr(fa.importlib, "import_module", import_module)

    with pytest.raises(AttributeError, match="unrelated import failure"):
        fa._load_flash_attn_with_kvcache(version=version)


def test_fa4_does_not_use_fa3_compatibility_path(monkeypatch: pytest.MonkeyPatch):
    def import_module(name: str):
        raise AttributeError(
            "module 'cutlass._mlir.dialects.nvvm' has no attribute 'RoundingModeKind'"
        )

    monkeypatch.setattr(fa.importlib, "import_module", import_module)

    with pytest.raises(AttributeError, match="RoundingModeKind"):
        fa._load_flash_attn_with_kvcache(version=4)
