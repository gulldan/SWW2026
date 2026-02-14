"""Path-level tests for dualdocking/scoring/docking.py (extracted from dock_calibration)."""

from __future__ import annotations

from types import SimpleNamespace

import dualdocking.scoring.docking as mod


class _FakeRunner:
    def __init__(self):
        self.commands: list[list[str]] = []

    def exec(self, args: list[str], timeout: int | None = None):
        self.commands.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_write_sdf_uses_correct_rdkit_embed_path_in_container(tmp_path) -> None:
    """Container command must point to cli/s1_prepare/rdkit_embed.py."""
    runner = _FakeRunner()
    sdf_path = tmp_path / "output" / "data" / "calibration" / "ligands" / "ligand_00000.sdf"

    ok, err = mod.write_sdf_from_smiles(
        runner,
        "CCO",
        sdf_path,
        seed=137,
        timeout=60,
        project_root=tmp_path,
    )

    assert ok is True
    assert err == ""
    assert runner.commands
    assert runner.commands[0][1] == "/workspace/cli/s1_prepare/rdkit_embed.py"
