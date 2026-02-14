"""Tests for dualdocking.validation.md module.

Unit tests for GROMACS command generation, MDP writing, GRO merging,
topology patching, XVG parsing, and CLI wrapper. All GROMACS/subprocess
calls are mocked -- no real MD simulation is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from dualdocking.validation.md import (
    MDConfig,
    MDResult,
    _merge_gro,
    _parse_xvg,
    _patch_topology,
    _write_em_mdp,
    _write_ions_mdp,
    _write_npt_mdp,
    _write_nvt_mdp,
    _write_prod_mdp,
    build_gromacs_commands,
    select_top_molecules,
    write_summary,
)


# ── MDConfig ─────────────────────────────────────────────────────────


class TestMDConfig:
    def test_defaults(self) -> None:
        cfg = MDConfig(pair_id="CDK2_ERK2")
        assert cfg.pair_id == "CDK2_ERK2"
        assert cfg.top_n == 3
        assert cfg.simulation_time_ns == 50.0
        assert cfg.stable_window_ns == 25.0
        assert cfg.rmsd_threshold_a == 3.0
        assert cfg.mmpbsa is False
        assert cfg.modes == ["single_a", "single_b", "double"]
        assert cfg.seeds == [42, 137, 256, 512, 1024]
        assert cfg.dt == 0.002
        assert cfg.ref_t == 310
        assert cfg.ref_p == 1.0

    def test_frozen(self) -> None:
        cfg = MDConfig(pair_id="CDK2_ERK2")
        with pytest.raises(AttributeError):
            cfg.pair_id = "BACE1_GSK3B"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = MDConfig(
            pair_id="AChE_BChE",
            top_n=5,
            simulation_time_ns=100.0,
            stable_window_ns=50.0,
            rmsd_threshold_a=2.5,
            mmpbsa=True,
            modes=["double"],
            seeds=[42],
        )
        assert cfg.top_n == 5
        assert cfg.simulation_time_ns == 100.0
        assert cfg.mmpbsa is True
        assert cfg.modes == ["double"]


# ── GROMACS command generation ───────────────────────────────────────


class TestBuildGromacsCommands:
    def test_command_count(self) -> None:
        commands = build_gromacs_commands()
        # pdb2gmx, editconf, solvate, grompp_ions, genion,
        # grompp_em, mdrun_em, grompp_nvt, mdrun_nvt,
        # grompp_npt, mdrun_npt, grompp_prod, mdrun_prod
        assert len(commands) == 13

    def test_all_commands_start_with_gmx(self) -> None:
        commands = build_gromacs_commands()
        for cmd in commands:
            assert cmd[0] == "gmx"

    def test_pdb2gmx_command(self) -> None:
        commands = build_gromacs_commands()
        pdb2gmx = commands[0]
        assert pdb2gmx[1] == "pdb2gmx"
        assert "-ff" in pdb2gmx
        assert "amber99sb-ildn" in pdb2gmx
        assert "-water" in pdb2gmx
        assert "tip3p" in pdb2gmx
        assert "-ignh" in pdb2gmx

    def test_genion_command(self) -> None:
        commands = build_gromacs_commands()
        genion = commands[4]
        assert genion[1] == "genion"
        assert "-neutral" in genion
        assert "-conc" in genion
        idx = genion.index("-conc")
        assert genion[idx + 1] == "0.15"

    def test_editconf_dodecahedron(self) -> None:
        commands = build_gromacs_commands()
        editconf = commands[1]
        assert editconf[1] == "editconf"
        assert "-bt" in editconf
        idx = editconf.index("-bt")
        assert editconf[idx + 1] == "dodecahedron"

    def test_mdrun_commands_have_ntmpi(self) -> None:
        commands = build_gromacs_commands()
        mdrun_cmds = [c for c in commands if "mdrun" in c]
        assert len(mdrun_cmds) == 4  # em, nvt, npt, prod
        for cmd in mdrun_cmds:
            assert "-ntmpi" in cmd
            assert "1" in cmd

    def test_production_mdrun_uses_md_deffnm(self) -> None:
        commands = build_gromacs_commands()
        prod_mdrun = commands[-1]
        assert "mdrun" in prod_mdrun
        assert "-deffnm" in prod_mdrun
        idx = prod_mdrun.index("-deffnm")
        assert prod_mdrun[idx + 1].endswith("md")


# ── MDP writers ──────────────────────────────────────────────────────


class TestMDPWriters:
    def test_ions_mdp(self, tmp_path: Path) -> None:
        p = tmp_path / "ions.mdp"
        _write_ions_mdp(p)
        content = p.read_text()
        assert "integrator = steep" in content
        assert "nsteps = 0" in content

    def test_em_mdp(self, tmp_path: Path) -> None:
        p = tmp_path / "em.mdp"
        _write_em_mdp(p)
        content = p.read_text()
        assert "integrator = steep" in content
        assert "emtol = 1000.0" in content
        assert "pbc = xyz" in content

    def test_nvt_mdp_uses_config(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", ref_t=300)
        p = tmp_path / "nvt.mdp"
        _write_nvt_mdp(p, cfg)
        content = p.read_text()
        assert "ref_t = 300 300" in content
        assert "gen-vel = yes" in content
        assert "define = -DPOSRES" in content
        assert "pcoupl = no" in content

    def test_npt_mdp_uses_config(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", ref_p=2.0)
        p = tmp_path / "npt.mdp"
        _write_npt_mdp(p, cfg)
        content = p.read_text()
        assert "ref_p = 2.0" in content
        assert "pcoupl = Parrinello-Rahman" in content
        assert "continuation = yes" in content
        assert "define = -DPOSRES" in content

    def test_prod_mdp_nsteps(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", simulation_time_ns=50.0, dt=0.002)
        p = tmp_path / "md.mdp"
        _write_prod_mdp(p, cfg)
        content = p.read_text()
        expected_nsteps = int(50.0 * 1e3 / 0.002)  # 25_000_000
        assert f"nsteps = {expected_nsteps}" in content
        assert "define = -DPOSRES" not in content  # no restraints in production

    def test_prod_mdp_custom_time(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", simulation_time_ns=100.0, dt=0.002)
        p = tmp_path / "md.mdp"
        _write_prod_mdp(p, cfg)
        content = p.read_text()
        expected_nsteps = int(100.0 * 1e3 / 0.002)  # 50_000_000
        assert f"nsteps = {expected_nsteps}" in content


# ── GRO merging ──────────────────────────────────────────────────────


class TestMergeGro:
    def test_merges_atoms(self, tmp_path: Path) -> None:
        prot = tmp_path / "protein.gro"
        lig = tmp_path / "ligand.gro"
        out = tmp_path / "complex.gro"

        prot.write_text(
            "Protein\n"
            " 3\n"
            "    1ALA      N    1   0.100   0.200   0.300\n"
            "    1ALA     CA    2   0.400   0.500   0.600\n"
            "    1ALA      C    3   0.700   0.800   0.900\n"
            "   5.0   5.0   5.0\n"
        )
        lig.write_text(
            "Ligand\n"
            " 2\n"
            "    1LIG     C1    1   1.000   1.100   1.200\n"
            "    1LIG     C2    2   1.300   1.400   1.500\n"
            "   5.0   5.0   5.0\n"
        )

        _merge_gro(prot, lig, out)
        lines = out.read_text().splitlines()

        assert lines[0] == "Protein"
        assert lines[1].strip() == "5"
        assert len(lines) == 8  # header + count + 5 atoms + box
        assert "LIG" in lines[5]
        assert "5.0   5.0   5.0" in lines[-1]


# ── Topology patching ────────────────────────────────────────────────


class TestPatchTopology:
    def test_inserts_include_and_molecule(self, tmp_path: Path) -> None:
        topol = tmp_path / "topol.top"
        lig_itp = tmp_path / "LIG_GMX.itp"

        topol.write_text(
            '#include "amber99sb-ildn.ff/forcefield.itp"\n'
            '#include "posre.itp"\n'
            "\n"
            "[ system ]\n"
            "Protein\n"
            "\n"
            "[ molecules ]\n"
            "Protein          1\n"
        )
        lig_itp.write_text("; ligand itp\n")

        _patch_topology(topol, lig_itp)
        content = topol.read_text()

        assert '#include "LIG_GMX.itp"' in content
        assert "LIG              1" in content
        # Include should appear before [ system ]
        inc_pos = content.index('#include "LIG_GMX.itp"')
        sys_pos = content.index("[ system ]")
        assert inc_pos < sys_pos

    def test_idempotent(self, tmp_path: Path) -> None:
        topol = tmp_path / "topol.top"
        lig_itp = tmp_path / "LIG_GMX.itp"

        topol.write_text(
            "[ system ]\nProtein\n[ molecules ]\nProtein          1\n"
        )
        lig_itp.write_text("; ligand itp\n")

        _patch_topology(topol, lig_itp)
        content1 = topol.read_text()
        _patch_topology(topol, lig_itp)
        content2 = topol.read_text()

        assert content1 == content2


# ── XVG parsing ──────────────────────────────────────────────────────


class TestParseXvg:
    def test_basic_parse(self, tmp_path: Path) -> None:
        xvg = tmp_path / "test.xvg"
        xvg.write_text(
            "# comment\n"
            "@ title\n"
            "@ xaxis\n"
            "0.0 1.5\n"
            "1.0 2.3\n"
            "2.0 0.8\n"
        )
        df = _parse_xvg(xvg, columns=["time", "value"])
        assert len(df) == 3
        assert list(df.columns) == ["time", "value"]
        assert df["time"].iloc[0] == 0.0
        assert df["value"].iloc[1] == pytest.approx(2.3)

    def test_skips_comments_and_at_lines(self, tmp_path: Path) -> None:
        xvg = tmp_path / "test.xvg"
        xvg.write_text(
            "# GROMACS output\n"
            "@ s0 legend\n"
            "@ s1 legend\n"
            "1.0 2.0\n"
        )
        df = _parse_xvg(xvg, columns=["a", "b"])
        assert len(df) == 1

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        xvg = tmp_path / "test.xvg"
        xvg.write_text("0.0 1.0 2.0 3.0\n")
        df = _parse_xvg(xvg, columns=["x", "y"])
        assert len(df) == 1
        assert list(df.columns) == ["x", "y"]
        assert df["y"].iloc[0] == 1.0


# ── MDResult dataclass ───────────────────────────────────────────────


class TestMDResult:
    def test_creation(self) -> None:
        r = MDResult(
            pair="CDK2_ERK2",
            mode="double",
            mol_id=0,
            target="CDK2",
            target_label="A",
            smiles="CCO",
            mean_rmsd=1.5,
            max_rmsd=2.8,
            stable=True,
            mean_hbonds=3.2,
            mean_energy=-150.5,
        )
        assert r.stable is True
        assert r.pair == "CDK2_ERK2"

    def test_frozen(self) -> None:
        r = MDResult(
            pair="CDK2_ERK2", mode="double", mol_id=0,
            target="CDK2", target_label="A", smiles="CCO",
            mean_rmsd=1.5, max_rmsd=2.8, stable=True,
            mean_hbonds=3.2, mean_energy=-150.5,
        )
        with pytest.raises(AttributeError):
            r.stable = False  # type: ignore[misc]


# ── write_summary ────────────────────────────────────────────────────


class TestWriteSummary:
    def test_writes_new_parquet(self, tmp_path: Path) -> None:
        results = [
            MDResult(
                pair="P", mode="m", mol_id=0, target="T",
                target_label="A", smiles="C", mean_rmsd=1.0,
                max_rmsd=2.0, stable=True, mean_hbonds=1.0,
                mean_energy=-100.0,
            ),
        ]
        path = tmp_path / "summary.parquet"
        write_summary(results, path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df["pair"].iloc[0] == "P"

    def test_appends_and_deduplicates(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.parquet"
        r1 = MDResult(
            pair="P", mode="m", mol_id=0, target="T",
            target_label="A", smiles="C", mean_rmsd=1.0,
            max_rmsd=2.0, stable=True, mean_hbonds=1.0,
            mean_energy=-100.0,
        )
        write_summary([r1], path)

        # Write again with updated rmsd -- should deduplicate
        r2 = MDResult(
            pair="P", mode="m", mol_id=0, target="T",
            target_label="A", smiles="C", mean_rmsd=0.5,
            max_rmsd=1.0, stable=True, mean_hbonds=2.0,
            mean_energy=-200.0,
        )
        write_summary([r2], path)
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df["mean_rmsd"].iloc[0] == pytest.approx(0.5)

    def test_no_results_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.parquet"
        write_summary([], path)
        assert not path.exists()


# ── select_top_molecules ─────────────────────────────────────────────


class TestSelectTopMolecules:
    def test_returns_empty_when_no_data(self, tmp_path: Path) -> None:
        df = select_top_molecules("P", "m", 3, [42], tmp_path)
        assert df.empty

    def test_selects_top_n(self, tmp_path: Path) -> None:
        data = pd.DataFrame({
            "smiles": ["A", "B", "C", "D"],
            "combined_min": [0.5, -1.0, 0.2, -0.5],
        })

        with patch("dualdocking.data.io_utils.stem_exists", return_value=True), \
             patch("dualdocking.data.io_utils.read_df", return_value=data):
            result = select_top_molecules("P", "m", 2, [42], tmp_path)

        assert len(result) == 2
        # Should be sorted by combined_min (ascending = best first)
        assert result["smiles"].iloc[0] == "B"  # -1.0 is best


# ── CLI test ─────────────────────────────────────────────────────────


class TestCLI:
    def test_help(self) -> None:
        from click.testing import CliRunner

        from cli.s6_md.run_md import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--pair" in result.output
        assert "--top-n" in result.output
        assert "--mmpbsa" in result.output
        assert "--dry-run" in result.output

    def test_missing_tool_exits(self) -> None:
        from click.testing import CliRunner

        from cli.s6_md.run_md import main

        runner = CliRunner()
        with patch("shutil.which", return_value=None):
            result = runner.invoke(main, ["--pair", "CDK2_ERK2"])
        assert result.exit_code != 0
