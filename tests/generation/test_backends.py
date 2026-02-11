"""Tests for pluggable generation backends."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from dualdocking.generation.backends import list_generation_backends, resolve_generation_backend


def test_builtin_freedpp_alias_from_legacy_name() -> None:
    """Legacy generator.name alias should resolve to built-in FREED++ backend."""
    backend_name, backend = resolve_generation_backend({"name": "FREED++"})
    assert backend_name == "freedpp"
    assert callable(backend)
    assert "freedpp" in list_generation_backends()


def test_external_generation_backend_module_default_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    """External backend spec `module` should use callable `generate`."""
    module = ModuleType("test_external_gen_backend")

    def _generate(_request):
        return ["CCO"]

    module.generate = _generate
    monkeypatch.setitem(sys.modules, "test_external_gen_backend", module)

    backend_name, backend = resolve_generation_backend({"backend": "test_external_gen_backend"})
    assert backend_name == "test_external_gen_backend"
    assert backend(SimpleNamespace()) == ["CCO"]


def test_external_generation_backend_module_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    """External backend spec `module:callable` should resolve that callable."""
    module = ModuleType("test_external_gen_backend_attr")

    def _custom_generate(_request):
        return ["CCC"]

    module.custom_generate = _custom_generate
    monkeypatch.setitem(sys.modules, "test_external_gen_backend_attr", module)

    backend_name, backend = resolve_generation_backend({"backend": "test_external_gen_backend_attr:custom_generate"})
    assert backend_name == "test_external_gen_backend_attr:custom_generate"
    assert backend(SimpleNamespace()) == ["CCC"]


def test_external_generation_backend_missing_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid external backend module should fail with explicit error."""
    module = ModuleType("test_external_gen_backend_missing")
    monkeypatch.setitem(sys.modules, "test_external_gen_backend_missing", module)

    with pytest.raises(ValueError, match="must expose callable"):
        resolve_generation_backend({"backend": "test_external_gen_backend_missing"})
