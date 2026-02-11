"""Path-level tests for scripts/s2_calibrate/dock_calibration.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.s2_calibrate.dock_calibration as mod


class _FakeRunner:
    def __init__(self):
        self.commands: list[list[str]] = []

    def exec(self, args: list[str], timeout: int | None = None):  # noqa: ARG002
        self.commands.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_write_sdf_uses_correct_rdkit_embed_path_in_container(tmp_path, monkeypatch) -> None:
    """Container command must point to scripts/s1_prepare/rdkit_embed.py."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    runner = _FakeRunner()
    sdf_path = tmp_path / "output" / "data" / "calibration" / "ligands" / "ligand_00000.sdf"

    ok, err = mod._write_sdf_from_smiles(
        runner,
        "CCO",
        sdf_path,
        seed=137,
        timeout=60,
    )

    assert ok is True
    assert err == ""
    assert runner.commands
    assert runner.commands[0][1] == "/workspace/scripts/s1_prepare/rdkit_embed.py"
