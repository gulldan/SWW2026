"""Tests for pluggable generation backends."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dualdocking.generation.backends import GenerationRequest, list_generation_backends, resolve_generation_backend


def test_builtin_freedpp_alias_from_legacy_name() -> None:
    """Legacy generator.name alias should resolve to built-in FREED++ backend."""
    backend_name, backend = resolve_generation_backend({"name": "FREED++"})
    assert backend_name == "freedpp"
    assert callable(backend)
    assert "freedpp" in list_generation_backends()


def test_builtin_reinvent4_alias() -> None:
    """REINVENT4 aliases should resolve to built-in reinvent4 backend."""
    for name in ("REINVENT4", "reinvent4", "reinvent"):
        backend_name, backend = resolve_generation_backend({"name": name})
        assert backend_name == "reinvent4", f"Failed for name={name!r}"
        assert callable(backend)
    assert "reinvent4" in list_generation_backends()


def test_reinvent4_backend_passes_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in REINVENT4 backend should pass config parameters correctly."""
    import dualdocking.generation.reinvent4 as reinvent4_mod

    captured: dict[str, object] = {}

    def _fake_generate(config) -> list[str]:
        captured["config"] = config
        return ["CCO"]

    monkeypatch.setattr(reinvent4_mod, "generate", _fake_generate)

    _backend_name, backend = resolve_generation_backend({"backend": "reinvent4"})
    request = GenerationRequest(
        pair_id="CDK2_ERK2",
        pair_config={"target_a": {"id": "CDK2"}, "target_b": {"id": "ERK2"}},
        mode="single_a",
        seed=42,
        output_dir=Path("output"),
        n_molecules=10,
        epochs_override=None,
        steps_per_epoch_override=None,
        generator_config={
            "reinvent4": {
                "algorithm": "dap",
                "batch_size": 256,
                "n_steps": 300,
                "vina_program": "unidock",
            }
        },
        quality_gate={},
    )
    result = backend(request)
    assert result == ["CCO"]

    cfg = captured["config"]
    assert getattr(cfg, "algorithm") == "dap"
    assert getattr(cfg, "batch_size") == 256
    assert getattr(cfg, "n_steps") == 300
    assert getattr(cfg, "vina_program") == "unidock"
    assert getattr(cfg, "target_id") == "CDK2"
    assert getattr(cfg, "reward_mode") == "single_a"


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


def test_freedpp_backend_passes_reward_modes_passthrough_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built-in FREED++ backend should pass reward-modes passthrough config."""
    import dualdocking.generation.freedpp as freedpp_mod

    captured: dict[str, object] = {}

    def _fake_generate(config) -> list[str]:
        captured["config"] = config
        return ["CCO"]

    monkeypatch.setattr(freedpp_mod, "generate", _fake_generate)

    _backend_name, backend = resolve_generation_backend({"backend": "freedpp"})
    request = GenerationRequest(
        pair_id="CDK2_ERK2",
        pair_config={"target_a": {"id": "CDK2"}, "target_b": {"id": "ERK2"}},
        mode="single_a",
        seed=42,
        output_dir=Path("output"),
        n_molecules=10,
        epochs_override=None,
        steps_per_epoch_override=None,
        generator_config={
            "freedpp": {
                "pass_reward_modes_to_container": False,
                "reward_modes_path_in_container": "/workspace/custom_reward_modes.yaml",
            }
        },
        quality_gate={},
    )
    result = backend(request)
    assert result == ["CCO"]

    cfg = captured["config"]
    assert getattr(cfg, "pass_reward_modes_to_container") is False
    assert getattr(cfg, "reward_modes_path_in_container") == "/workspace/custom_reward_modes.yaml"
