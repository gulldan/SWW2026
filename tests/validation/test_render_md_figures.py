"""Tests for dualdocking.validation.render_md_figures module.

Unit tests for publication-quality figure generation.  All matplotlib
output goes to tmp_path; PyMOL and ProLIF are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dualdocking.validation.render_md_figures import (
    _add_panel_labels,
    _load_target_name,
    _resolve_gro_path,
    discover_targets,
    plot_pca_variance,
    plot_rg_hbonds,
    plot_rmsd,
    plot_rmsf,
    plot_sasa,
    render_binding_pose,
    render_summary_table,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_md_result(
    target: str = "CDK2",
    *,
    stable: bool = True,
    converged: bool = True,
    mmpbsa_dg: float = -8.5,
    sasa_mean: float = 5200.0,
    sasa_std: float = 120.0,
    pca_pc1_variance: float = 45.3,
    pca_pc2_variance: float = 18.7,
) -> dict:
    return {
        "target": target,
        "mean_rmsd": 1.5,
        "max_rmsd": 2.8,
        "stable": stable,
        "backbone_rmsd_mean": 1.2,
        "rmsf_binding_pocket": 0.8,
        "rg_mean": 14.5,
        "rg_std": 0.3,
        "mean_hbonds": 3.2,
        "converged": converged,
        "mmpbsa_dg": mmpbsa_dg,
        "sasa_mean": sasa_mean,
        "sasa_std": sasa_std,
        "pca_pc1_variance": pca_pc1_variance,
        "pca_pc2_variance": pca_pc2_variance,
    }


def _write_csvs(d: Path, *, n: int = 100) -> None:
    """Write mock CSV analysis outputs into a target directory."""
    rng = np.random.default_rng(42)
    time = np.linspace(0, 50, n)

    pd.DataFrame(
        {"time_ns": time, "rmsd_A": rng.normal(1.5, 0.2, n)}
    ).to_csv(d / "rmsd_backbone.csv", index=False)

    pd.DataFrame(
        {"time_ns": time, "rmsd_A": rng.normal(2.0, 0.3, n)}
    ).to_csv(d / "rmsd_ligand.csv", index=False)

    pd.DataFrame(
        {"residue": np.arange(1, n + 1), "rmsf_A": rng.exponential(0.5, n)}
    ).to_csv(d / "rmsf.csv", index=False)

    pd.DataFrame(
        {"time_ps": np.linspace(0, 50000, n), "rg_A": rng.normal(14.5, 0.2, n)}
    ).to_csv(d / "radius_gyration.csv", index=False)

    pd.DataFrame(
        {"time_ns": time, "n_hbonds": rng.poisson(3, n).astype(float)}
    ).to_csv(d / "hbonds.csv", index=False)

    pd.DataFrame(
        {"time_ns": time, "sasa_A2": rng.normal(5200, 120, n)}
    ).to_csv(d / "sasa.csv", index=False)

    pd.DataFrame(
        {"mode": np.arange(1, 11), "eigenvalue": np.sort(rng.exponential(1.0, 10))[::-1]}
    ).to_csv(d / "eigenvalues.csv", index=False)


def _make_target_dir(
    base: Path,
    label: str,
    target_name: str = "CDK2",
    *,
    write_gro: bool = False,
) -> Path:
    """Create a mock target directory with md_result.json and CSV outputs."""
    d = base / f"target_{label}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "md_result.json").write_text(json.dumps(_make_md_result(target_name)))
    _write_csvs(d)
    if write_gro:
        (d / "reference.gro").write_text("mock gro\n")
    return d


# ── _load_target_name ───────────────────────────────────────────────


class TestLoadTargetName:
    def test_reads_from_json(self, tmp_path: Path) -> None:
        d = tmp_path / "target_A"
        d.mkdir()
        (d / "md_result.json").write_text(json.dumps({"target": "ERK2"}))
        assert _load_target_name(d) == "ERK2"

    def test_falls_back_to_dirname(self, tmp_path: Path) -> None:
        d = tmp_path / "target_B"
        d.mkdir()
        assert _load_target_name(d) == "target_B"

    def test_falls_back_when_no_target_key(self, tmp_path: Path) -> None:
        d = tmp_path / "target_A"
        d.mkdir()
        (d / "md_result.json").write_text(json.dumps({"stable": True}))
        assert _load_target_name(d) == "target_A"


# ── _resolve_gro_path ──────────────────────────────────────────────


class TestResolveGroPath:
    def test_reference_gro_preferred(self, tmp_path: Path) -> None:
        (tmp_path / "reference.gro").write_text("gro1")
        (tmp_path / "reference_npt.gro").write_text("gro2")
        assert _resolve_gro_path(tmp_path) == tmp_path / "reference.gro"

    def test_falls_back_to_npt(self, tmp_path: Path) -> None:
        (tmp_path / "reference_npt.gro").write_text("gro2")
        assert _resolve_gro_path(tmp_path) == tmp_path / "reference_npt.gro"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert _resolve_gro_path(tmp_path) is None


# ── discover_targets ────────────────────────────────────────────────


class TestDiscoverTargets:
    def test_finds_both(self, tmp_path: Path) -> None:
        _make_target_dir(tmp_path, "A", "CDK2")
        _make_target_dir(tmp_path, "B", "ERK2")
        targets = discover_targets(tmp_path)
        assert set(targets.keys()) == {"A", "B"}

    def test_finds_one(self, tmp_path: Path) -> None:
        _make_target_dir(tmp_path, "A", "CDK2")
        targets = discover_targets(tmp_path)
        assert list(targets.keys()) == ["A"]

    def test_empty_when_no_result(self, tmp_path: Path) -> None:
        d = tmp_path / "target_A"
        d.mkdir(parents=True)
        # Directory exists but no md_result.json
        targets = discover_targets(tmp_path)
        assert targets == {}

    def test_empty_when_no_dirs(self, tmp_path: Path) -> None:
        targets = discover_targets(tmp_path)
        assert targets == {}


# ── _add_panel_labels ──────────────────────────────────────────────


class TestAddPanelLabels:
    def test_labels_placed(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        _add_panel_labels(fig, [ax1, ax2], ["A", "B"])
        # Verify text objects were added
        assert any(t.get_text() == "A" for t in ax1.texts)
        assert any(t.get_text() == "B" for t in ax2.texts)
        plt.close(fig)

    def test_single_label(self) -> None:
        fig, ax = plt.subplots()
        _add_panel_labels(fig, [ax], ["X"])
        assert any(t.get_text() == "X" for t in ax.texts)
        plt.close(fig)


# ── plot_rmsd ──────────────────────────────────────────────────────


class TestPlotRmsd:
    def test_creates_output_files(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        targets = discover_targets(base)
        plot_rmsd(targets, out)

        assert (out / "panel_BC_rmsd.png").exists()
        assert (out / "panel_BC_rmsd.svg").exists()

    def test_single_target(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")

        targets = discover_targets(base)
        plot_rmsd(targets, out)

        assert (out / "panel_BC_rmsd.png").exists()


# ── plot_rmsf ──────────────────────────────────────────────────────


class TestPlotRmsf:
    def test_creates_output_files(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        targets = discover_targets(base)
        plot_rmsf(targets, out)

        assert (out / "panel_D_rmsf.png").exists()
        assert (out / "panel_D_rmsf.svg").exists()


# ── plot_rg_hbonds ─────────────────────────────────────────────────


class TestPlotRgHbonds:
    def test_creates_output_files(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        targets = discover_targets(base)
        plot_rg_hbonds(targets, out)

        assert (out / "panel_FG_rg_hbonds.png").exists()
        assert (out / "panel_FG_rg_hbonds.svg").exists()

    def test_handles_empty_hbonds(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        d = _make_target_dir(base, "A", "CDK2")
        # Overwrite hbonds with empty CSV
        pd.DataFrame(columns=["time_ns", "n_hbonds"]).to_csv(
            d / "hbonds.csv", index=False
        )

        targets = discover_targets(base)
        plot_rg_hbonds(targets, out)

        assert (out / "panel_FG_rg_hbonds.png").exists()


# ── render_summary_table ───────────────────────────────────────────


class TestRenderSummaryTable:
    def test_creates_table(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        targets = discover_targets(base)
        render_summary_table(targets, out)

        assert (out / "summary_table.png").exists()
        assert (out / "summary_table.svg").exists()

    def test_no_results_noop(self, tmp_path: Path) -> None:
        out = tmp_path / "plots"
        out.mkdir()
        render_summary_table({}, out)
        assert not (out / "summary_table.png").exists()

    def test_nan_mmpbsa_shows_dash(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        d = _make_target_dir(base, "A", "CDK2")
        # Overwrite with NaN mmpbsa
        result = _make_md_result("CDK2", mmpbsa_dg=float("nan"))
        (d / "md_result.json").write_text(json.dumps(result))

        targets = discover_targets(base)
        # Should not raise
        render_summary_table(targets, out)
        assert (out / "summary_table.png").exists()

    def test_unstable_unconverged(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        d = _make_target_dir(base, "A", "CDK2")
        result = _make_md_result("CDK2", stable=False, converged=False)
        (d / "md_result.json").write_text(json.dumps(result))

        targets = discover_targets(base)
        render_summary_table(targets, out)
        assert (out / "summary_table.png").exists()


# ── plot_sasa ──────────────────────────────────────────────────────


class TestPlotSasa:
    def test_creates_output_files(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        targets = discover_targets(base)
        plot_sasa(targets, out)

        assert (out / "panel_H_sasa.png").exists()
        assert (out / "panel_H_sasa.svg").exists()

    def test_handles_missing_csv(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        d = _make_target_dir(base, "A", "CDK2")
        (d / "sasa.csv").unlink()

        targets = discover_targets(base)
        plot_sasa(targets, out)

        assert (out / "panel_H_sasa.png").exists()


# ── plot_pca_variance ─────────────────────────────────────────────


class TestPlotPcaVariance:
    def test_creates_output_files(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        targets = discover_targets(base)
        plot_pca_variance(targets, out)

        assert (out / "panel_I_pca.png").exists()
        assert (out / "panel_I_pca.svg").exists()

    def test_handles_missing_csv(self, tmp_path: Path) -> None:
        base = tmp_path / "md"
        out = tmp_path / "plots"
        out.mkdir()
        d = _make_target_dir(base, "A", "CDK2")
        (d / "eigenvalues.csv").unlink()

        targets = discover_targets(base)
        plot_pca_variance(targets, out)

        assert (out / "panel_I_pca.png").exists()


# ── render_binding_pose ────────────────────────────────────────────


class TestRenderBindingPose:
    @staticmethod
    def _mock_pymol() -> MagicMock:
        """Create a mock pymol package where ``from pymol import cmd`` works."""
        mock_cmd = MagicMock()
        mock_cmd.count_states.return_value = 100
        mock_pymol = MagicMock()
        mock_pymol.cmd = mock_cmd
        return mock_pymol

    def test_calls_pymol(self, tmp_path: Path) -> None:
        gro = tmp_path / "reference.gro"
        gro.write_text("mock gro\n")
        xtc = tmp_path / "trajectory.xtc"
        out_png = tmp_path / "pose.png"

        mock_pymol = self._mock_pymol()
        mock_cmd = mock_pymol.cmd

        with patch.dict("sys.modules", {"pymol": mock_pymol, "pymol.cmd": mock_cmd}):
            render_binding_pose(gro, xtc, out_png)

        mock_cmd.reinitialize.assert_called_once()
        mock_cmd.load.assert_called_once_with(str(gro), "complex")
        mock_cmd.ray.assert_called_once_with(1800, 1400)
        mock_cmd.png.assert_called_once_with(str(out_png))
        mock_cmd.delete.assert_called_once_with("all")

    def test_skips_trajectory_when_missing(self, tmp_path: Path) -> None:
        gro = tmp_path / "reference.gro"
        gro.write_text("mock gro\n")
        xtc = tmp_path / "trajectory.xtc"
        out_png = tmp_path / "pose.png"

        mock_pymol = self._mock_pymol()
        mock_cmd = mock_pymol.cmd

        with patch.dict("sys.modules", {"pymol": mock_pymol, "pymol.cmd": mock_cmd}):
            render_binding_pose(gro, xtc, out_png)

        # load_traj should NOT be called since xtc doesn't exist
        mock_cmd.load_traj.assert_not_called()

    def test_custom_dimensions(self, tmp_path: Path) -> None:
        gro = tmp_path / "reference.gro"
        gro.write_text("mock gro\n")
        xtc = tmp_path / "traj.xtc"
        out_png = tmp_path / "pose.png"

        mock_pymol = self._mock_pymol()
        mock_cmd = mock_pymol.cmd

        with patch.dict("sys.modules", {"pymol": mock_pymol, "pymol.cmd": mock_cmd}):
            render_binding_pose(gro, xtc, out_png, width=800, height=600)

        mock_cmd.ray.assert_called_once_with(800, 600)


# ── plot_prolif_interactions ───────────────────────────────────────


class TestPlotProlifInteractions:
    def test_skips_when_prolif_missing(self, tmp_path: Path) -> None:
        from dualdocking.validation.render_md_figures import plot_prolif_interactions

        gro = tmp_path / "reference.gro"
        gro.write_text("mock")
        xtc = tmp_path / "trajectory.xtc"
        out = tmp_path / "plots"
        out.mkdir()

        # Patch imports to raise ImportError
        with patch.dict("sys.modules", {"MDAnalysis": None, "prolif": None}):
            # The function catches ImportError internally
            plot_prolif_interactions(gro, xtc, out, "CDK2")

        # No output files should be created
        assert not list(out.glob("panel_E_*"))

    def test_skips_when_no_lig_atoms(self, tmp_path: Path) -> None:
        from dualdocking.validation.render_md_figures import plot_prolif_interactions

        gro = tmp_path / "reference.gro"
        gro.write_text("mock")
        xtc = tmp_path / "trajectory.xtc"
        out = tmp_path / "plots"
        out.mkdir()

        mock_mda = MagicMock()
        mock_prolif = MagicMock()

        mock_universe = MagicMock()
        mock_mda.Universe.return_value = mock_universe
        mock_lig = MagicMock()
        mock_lig.n_atoms = 0
        mock_universe.select_atoms.side_effect = lambda sel: mock_lig if "LIG" in sel else MagicMock()

        with patch.dict("sys.modules", {"MDAnalysis": mock_mda, "prolif": mock_prolif}):
            plot_prolif_interactions(gro, xtc, out, "CDK2")

        assert not list(out.glob("panel_E_*"))


# ── CLI (main) ─────────────────────────────────────────────────────


class TestCLI:
    def test_help(self) -> None:
        from click.testing import CliRunner

        from dualdocking.validation.render_md_figures import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--pair" in result.output
        assert "--mode" in result.output
        assert "--mol" in result.output
        assert "--output-dir" in result.output

    def test_no_targets_exits_gracefully(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from dualdocking.validation.render_md_figures import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "--pair", "CDK2_ERK2",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0

    def test_renders_with_mock_data(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from dualdocking.validation.render_md_figures import main

        # Build directory structure matching what main() expects
        base = tmp_path / "output" / "md" / "CDK2_ERK2" / "single_a" / "mol_0"
        _make_target_dir(base, "A", "CDK2")
        _make_target_dir(base, "B", "ERK2")

        out = tmp_path / "plots"
        runner = CliRunner()

        with patch(
            "dualdocking.validation.render_md_figures.PROJECT_ROOT", tmp_path
        ):
            result = runner.invoke(main, [
                "--pair", "CDK2_ERK2",
                "--mode", "single_a",
                "--mol", "0",
                "--output-dir", str(out),
            ])

        assert result.exit_code == 0
        assert (out / "panel_BC_rmsd.png").exists()
        assert (out / "panel_D_rmsf.png").exists()
        assert (out / "panel_FG_rg_hbonds.png").exists()
        assert (out / "panel_H_sasa.png").exists()
        assert (out / "panel_I_pca.png").exists()
        assert (out / "summary_table.png").exists()
