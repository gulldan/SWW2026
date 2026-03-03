"""Tests for shared reward selectivity registry and evaluators."""

from __future__ import annotations

import sys
import types
from uuid import uuid4

import pytest

from dualdocking.reward.selectivity import (
    evaluate_reward_mode,
    get_reward_modes_config_source,
    list_reward_modes,
    load_reward_plugins,
    load_reward_modes_from_yaml,
    register_reward_mode,
    resolve_reward_mode,
    reward_mode_requires_dual_docking,
    reward_mode_requires_secondary_target,
)


def test_builtin_modes_present():
    modes = set(list_reward_modes())
    assert {
        "single_a",
        "single_b",
        "double",
        "linear_equal",
        "biased_a",
        "selective_a",
        "selective_weighted",
    }.issubset(modes)
    assert get_reward_modes_config_source() is not None


def test_mode_requirements_flags():
    assert reward_mode_requires_secondary_target("single_a") is False
    assert reward_mode_requires_dual_docking("single_a") is False

    assert reward_mode_requires_secondary_target("single_b") is True
    assert reward_mode_requires_dual_docking("single_b") is False

    assert reward_mode_requires_secondary_target("double") is True
    assert reward_mode_requires_dual_docking("double") is True


def test_builtin_reward_values():
    assert evaluate_reward_mode("single_a", z_a=-1.5) == pytest.approx(-1.5)
    assert evaluate_reward_mode("single_b", z_b=2.0) == pytest.approx(2.0)
    assert evaluate_reward_mode("double", z_a=-1.0, z_b=2.0) == pytest.approx(2.0)
    assert evaluate_reward_mode("linear_equal", z_a=-1.0, z_b=2.0) == pytest.approx(1.0)
    assert evaluate_reward_mode("biased_a", z_a=-1.0, z_b=2.0) == pytest.approx(-0.4)
    assert evaluate_reward_mode("selective_a", z_a=-1.0, z_b=2.0) == pytest.approx(-3.0)


def test_selective_weighted_lambda_override():
    mode = resolve_reward_mode("selective_weighted", {"lambda_off": 2.5})
    assert mode.evaluate(z_a=-1.0, z_b=2.0) == pytest.approx(-6.0)


def test_selective_weighted_rejects_non_positive_lambda():
    with pytest.raises(ValueError, match="lambda_off must be > 0"):
        evaluate_reward_mode("selective_weighted", z_a=-1.0, z_b=2.0, params={"lambda_off": 0.0})


def test_unknown_mode_fails():
    with pytest.raises(KeyError, match="Unknown reward mode"):
        evaluate_reward_mode("nonexistent_mode", z_a=-1.0)


def test_custom_mode_registration():
    def _custom_eval(z_a, z_b, params):
        za = float(z_a)
        zb = float(z_b)
        bonus = float(params.get("bonus", 0.0))
        return za - 0.5 * zb + bonus

    register_reward_mode(
        name="custom_half_off",
        evaluator=_custom_eval,
        requires_secondary_target=True,
        requires_dual_docking=True,
        default_params={"bonus": 0.0},
        allow_override=True,
    )
    value = evaluate_reward_mode("custom_half_off", z_a=-2.0, z_b=4.0, params={"bonus": 1.5})
    assert value == pytest.approx(-2.5)


def test_load_reward_plugins_registers_mode(monkeypatch):
    mode_name = f"plugin_mode_{uuid4().hex[:8]}"
    module_name = f"tmp_reward_plugin_{uuid4().hex[:8]}"
    plugin_module = types.ModuleType(module_name)

    def _plugin_eval(z_a, z_b, params):
        za = float(z_a)
        zb = float(z_b)
        scale = float(params.get("scale", 1.0))
        return za - scale * zb

    def _register():
        register_reward_mode(
            name=mode_name,
            evaluator=_plugin_eval,
            requires_secondary_target=True,
            requires_dual_docking=True,
            default_params={"scale": 0.25},
        )

    plugin_module.register_reward_modes = _register
    monkeypatch.setitem(sys.modules, module_name, plugin_module)

    assert load_reward_plugins(module_name) == [module_name]
    assert load_reward_plugins(module_name) == []
    assert evaluate_reward_mode(mode_name, z_a=-2.0, z_b=4.0) == pytest.approx(-3.0)


def test_load_reward_modes_from_yaml_registers_mode(tmp_path):
    mode_name = f"yaml_mode_{uuid4().hex[:8]}"
    cfg_path = tmp_path / "reward_modes.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "version: 1",
                "modes:",
                f"  {mode_name}:",
                "    kind: selective_weighted",
                "    requires_secondary_target: true",
                "    requires_dual_docking: true",
                "    default_params:",
                "      lambda_off: 3.0",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_reward_modes_from_yaml(cfg_path)
    assert loaded == str(cfg_path)
    assert evaluate_reward_mode(mode_name, z_a=-1.0, z_b=2.0) == pytest.approx(-7.0)


def test_load_reward_modes_from_yaml_rejects_unknown_kind(tmp_path):
    cfg_path = tmp_path / "reward_modes_invalid.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "version: 1",
                "modes:",
                "  bad_mode:",
                "    kind: custom_formula",
                "    requires_secondary_target: false",
                "    requires_dual_docking: false",
                "    default_params: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown kind"):
        load_reward_modes_from_yaml(cfg_path)
