"""Tests for pluggable scoring backends."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import dualdocking.scoring.backends as mod
from dualdocking.scoring.backends import list_builtin_scoring_engines, resolve_scoring_backend


def _fake_boltz_module() -> SimpleNamespace:
    """Create a fake Boltz module exposing required functions."""

    def _score_batch(
        _target_id: str, smiles_list: list[str], *, use_cache: bool = True, device: str | None = None
    ) -> list[float]:
        assert isinstance(use_cache, bool)
        return [float(device or "0")] * len(smiles_list)

    def _context_info() -> dict[str, str]:
        return {"backend": "fake"}

    def _prune_cache() -> int:
        return 7

    return SimpleNamespace(
        score_batch=_score_batch,
        get_scoring_context_info=_context_info,
        prune_stale_cache=_prune_cache,
        _BOLTZ_BACKEND="cli",
        _CONTEXT_HASH=None,
        _NIM_ACTIVE_URLS=None,
        _NIM_ACTIVE_URLS_KEY=None,
    )


def test_builtin_boltz_alias_from_legacy_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy scorer.name alias should resolve to built-in boltz2 engine."""
    fake_module = _fake_boltz_module()

    def _fake_import(module_name: str):
        if module_name == "dualdocking.scoring.boltz2":
            return fake_module
        return __import__(module_name)

    monkeypatch.setattr(mod.importlib, "import_module", _fake_import)
    monkeypatch.delenv("BOLTZ_BACKEND", raising=False)

    engine_name, backend = resolve_scoring_backend({"name": "Boltz-2", "backend": "cli"})
    assert engine_name == "boltz2"
    assert backend.prune_stale_cache() == 7
    assert backend.score_batch("CDK2", ["CCO", "CCC"], device="1") == [1.0, 1.0]
    assert "boltz2" in list_builtin_scoring_engines()


def test_builtin_boltz_nim_sets_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boltz NIM config should be projected into environment variables."""
    fake_module = _fake_boltz_module()

    def _fake_import(module_name: str):
        if module_name == "dualdocking.scoring.boltz2":
            return fake_module
        return __import__(module_name)

    monkeypatch.setattr(mod.importlib, "import_module", _fake_import)
    monkeypatch.delenv("BOLTZ_NIM_URLS", raising=False)
    monkeypatch.delenv("BOLTZ_NIM_URL", raising=False)

    engine_name, backend = resolve_scoring_backend(
        {
            "engine": "boltz2",
            "backend": "nim",
            "nim": {"urls": ["http://localhost:8000", "http://localhost:8001"]},
        }
    )
    assert engine_name == "boltz2"
    assert backend.get_scoring_context_info()["backend"] == "fake"
    assert mod.os.environ["BOLTZ_BACKEND"] == "nim"
    assert mod.os.environ["BOLTZ_NIM_URLS"] == "http://localhost:8000,http://localhost:8001"
    assert mod.os.environ["BOLTZ_NIM_URL"] == "http://localhost:8000"


def test_external_scoring_backend_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """External scorer spec `module` should use module-level callables."""
    module = ModuleType("test_external_score_backend")

    def _score_batch(_target_id: str, smiles_list: list[str], **_kwargs) -> list[float]:
        return [-3.0] * len(smiles_list)

    def _context_info() -> dict[str, str]:
        return {"backend": "external"}

    def _prune_cache() -> int:
        return 0

    module.score_batch = _score_batch
    module.get_scoring_context_info = _context_info
    module.prune_stale_cache = _prune_cache
    monkeypatch.setitem(sys.modules, "test_external_score_backend", module)

    engine_name, backend = resolve_scoring_backend({"engine": "test_external_score_backend"})
    assert engine_name == "test_external_score_backend"
    assert backend.score_batch("CDK2", ["CCO"]) == [-3.0]


def test_external_scoring_backend_factory_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    """External scorer spec `module:factory` should instantiate backend object."""
    module = ModuleType("test_external_score_backend_factory")

    def _score_batch(_target_id: str, smiles_list: list[str], **_kwargs) -> list[float]:
        return [-4.0] * len(smiles_list)

    def _context_info() -> dict[str, str]:
        return {"backend": "factory"}

    def _prune_cache() -> int:
        return 1

    def _make_backend() -> SimpleNamespace:
        return SimpleNamespace(
            score_batch=_score_batch,
            get_scoring_context_info=_context_info,
            prune_stale_cache=_prune_cache,
        )

    module.make_backend = _make_backend
    monkeypatch.setitem(sys.modules, "test_external_score_backend_factory", module)

    engine_name, backend = resolve_scoring_backend({"engine": "test_external_score_backend_factory:make_backend"})
    assert engine_name == "test_external_score_backend_factory:make_backend"
    assert backend.prune_stale_cache() == 1
    assert backend.score_batch("CDK2", ["CCO", "CCC"]) == [-4.0, -4.0]


def test_external_scoring_backend_missing_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """External scorer backend missing required methods should raise TypeError."""
    module = ModuleType("test_external_score_backend_invalid")

    def _score_batch(_target_id: str, smiles_list: list[str], **_kwargs) -> list[float]:
        return [-1.0] * len(smiles_list)

    module.score_batch = _score_batch
    monkeypatch.setitem(sys.modules, "test_external_score_backend_invalid", module)

    with pytest.raises(TypeError, match="must provide callables"):
        resolve_scoring_backend({"engine": "test_external_score_backend_invalid"})
