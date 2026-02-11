"""Tests for scripts/pipeline/run_full_experiment.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

import scripts.pipeline.run_full_experiment as mod
from scripts.pipeline.run_full_experiment import main


def _exp_cfg() -> dict:
    return {"modes": ["single_a"], "seeds": [42], "scorer": {"backend": "cli"}}


def _pairs_cfg_with_blocked() -> dict:
    return {
        "pairs": {
            "CORE_PAIR": {
                "set": "core",
                "target_a": {"id": "A"},
                "target_b": {"id": "B"},
            },
            "BLOCKED_PAIR": {
                "set": "blocked",
                "target_a": {"id": "X"},
                "target_b": {"id": "Y"},
            },
        }
    }


def test_dry_run_excludes_blocked_pairs(monkeypatch) -> None:
    """Blocked pairs should be excluded from task matrix."""
    monkeypatch.setattr(mod, "load_experiment", lambda _path: _exp_cfg())
    monkeypatch.setattr(mod, "load_pairs", lambda _path: _pairs_cfg_with_blocked())
    monkeypatch.setattr(mod, "_detect_gpu_count", lambda: 1)

    runner = CliRunner()
    result = runner.invoke(main, ["--dry-run"])

    assert result.exit_code == 0
    assert "--pair CORE_PAIR" in result.output
    assert "--pair BLOCKED_PAIR" not in result.output


def test_manifest_finalized_as_failed_on_exception(tmp_path, monkeypatch) -> None:
    """Manifest should be finalized even when pre-task stage fails."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "load_experiment", lambda _path: _exp_cfg())
    monkeypatch.setattr(mod, "load_pairs", lambda _path: _pairs_cfg_with_blocked())
    monkeypatch.setattr(mod, "_detect_gpu_count", lambda: 1)
    monkeypatch.setattr(
        mod, "_prepare_commands", lambda *_a, **_k: [["uv", "run", "python", "scripts/s1_prepare/prepare_all.py"]]
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("prep failed")

    monkeypatch.setattr(mod, "_run_cmd", _boom)
    monkeypatch.setattr(mod, "manifest_path", lambda *_a, **_k: tmp_path / "run_manifest_test.json")
    monkeypatch.setattr(mod, "init_manifest", lambda *_a, **_k: {})

    statuses: list[str] = []

    def _capture_finalize(_path, *, status="completed"):
        statuses.append(status)

    monkeypatch.setattr(mod, "finalize_manifest", _capture_finalize)

    runner = CliRunner()
    result = runner.invoke(main, [])

    assert result.exit_code != 0
    assert statuses == ["failed"]


def test_run_cmd_sets_boltz_env_for_builtin_engine(tmp_path, monkeypatch) -> None:
    """Boltz-specific env vars should be set only for built-in boltz2 engine."""
    captured: dict[str, dict[str, str]] = {}

    def _fake_run_project_command(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "run_project_command", _fake_run_project_command)
    monkeypatch.setattr(mod, "_SCORER_ENGINE", "boltz2")
    monkeypatch.setattr(mod, "_SCORER_BOLTZ_BACKEND", "nim")
    monkeypatch.setattr(mod, "_SCORER_NIM_URLS", ["http://localhost:8000", "http://localhost:8001"])

    mod._run_cmd(["python", "scripts/pipeline/run_stages.py", "--list-stages"], Path(tmp_path / "log.txt"), gpu_id=1)

    env = captured["env"]
    assert env["DUALDOCKING_SCORER_ENGINE"] == "boltz2"
    assert env["SCORER_CUDA_DEVICE"] == "1"
    assert env["BOLTZ_CUDA_DEVICE"] == "1"
    assert env["BOLTZ_BACKEND"] == "nim"
    assert env["BOLTZ_NIM_URLS"] == "http://localhost:8000,http://localhost:8001"
    assert env["BOLTZ_NIM_URL"] == "http://localhost:8000"


def test_run_cmd_clears_boltz_env_for_external_engine(tmp_path, monkeypatch) -> None:
    """External scorer engines must not inherit forced Boltz routing env vars."""
    captured: dict[str, dict[str, str]] = {}

    def _fake_run_project_command(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "run_project_command", _fake_run_project_command)
    monkeypatch.setattr(mod, "_SCORER_ENGINE", "my_custom_scorer")
    monkeypatch.setenv("BOLTZ_BACKEND", "nim")
    monkeypatch.setenv("BOLTZ_NIM_URLS", "http://localhost:8000")
    monkeypatch.setenv("BOLTZ_NIM_URL", "http://localhost:8000")
    monkeypatch.setenv("BOLTZ_CUDA_DEVICE", "7")

    mod._run_cmd(["python", "scripts/pipeline/run_stages.py", "--list-stages"], Path(tmp_path / "log.txt"), gpu_id=0)

    env = captured["env"]
    assert env["DUALDOCKING_SCORER_ENGINE"] == "my_custom_scorer"
    assert env["SCORER_CUDA_DEVICE"] == "0"
    assert "BOLTZ_BACKEND" not in env
    assert "BOLTZ_NIM_URLS" not in env
    assert "BOLTZ_NIM_URL" not in env
    assert "BOLTZ_CUDA_DEVICE" not in env
