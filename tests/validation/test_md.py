"""Tests for dualdocking.validation.md_* modules.

Unit tests for GROMACS command generation, MDP writing, GRO merging,
topology patching, XVG parsing, convergence assessment, trajectory
preprocessing, GPU detection, and CLI wrapper.  All GROMACS/subprocess
calls are mocked -- no real MD simulation is needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dualdocking.validation.md_config import (
    MDConfig,
    MDResult,
    _detect_gpu_flags,
    _run,
)
from dualdocking.validation.md_mdp import (
    _write_em_mdp,
    _write_ions_mdp,
    _write_npt_mdp,
    _write_nvt_mdp,
    _write_prod_mdp,
)
from dualdocking.validation.md_setup import (
    _make_index,
    _merge_gro,
    _patch_topology,
)
from dualdocking.validation.md_analysis import (
    _assess_convergence,
    _parse_xvg,
    _preprocess_trajectory,
)
from dualdocking.validation.md_pipeline import (
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
        # New fields
        assert cfg.equil_time_ps == 200.0
        assert cfg.nstlog == 1000
        assert cfg.nstenergy == 1000
        assert cfg.gpu_auto is True
        assert cfg.convergence_ks_alpha == 0.05
        assert cfg.cmd_timeout == 600
        assert cfg.mdrun_timeout == 259200

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
            equil_time_ps=500.0,
            gpu_auto=False,
        )
        assert cfg.top_n == 5
        assert cfg.simulation_time_ns == 100.0
        assert cfg.mmpbsa is True
        assert cfg.modes == ["double"]
        assert cfg.equil_time_ps == 500.0
        assert cfg.gpu_auto is False


# ── _run ─────────────────────────────────────────────────────────────


class TestRun:
    def test_input_text_forwarded(self) -> None:
        with patch("dualdocking.validation.md_config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run(["gmx", "genion"], input_text="SOL\n", timeout=60)
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            assert kwargs["input"] == "SOL\n"

    def test_timeout_forwarded(self) -> None:
        with patch("dualdocking.validation.md_config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _run(["gmx", "mdrun"], timeout=7200)
            kwargs = mock_run.call_args[1]
            assert kwargs["timeout"] == 7200

    def test_failure_raises(self) -> None:
        with patch("dualdocking.validation.md_config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="some error")
            with pytest.raises(RuntimeError, match="Command failed"):
                _run(["gmx", "grompp"])

    def test_check_false_no_raise(self) -> None:
        with patch("dualdocking.validation.md_config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="err")
            result = _run(["gmx", "test"], check=False)
            assert result.returncode == 1


# ── _detect_gpu_flags ────────────────────────────────────────────────


class TestDetectGpu:
    def test_with_gpu(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("dualdocking.validation.md_config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA RTX 4090\n")
            flags = _detect_gpu_flags()
            assert "-nb" in flags
            assert "gpu" in flags
            assert len(flags) == 8

    def test_without_gpu(self) -> None:
        with patch("shutil.which", return_value=None):
            flags = _detect_gpu_flags()
            assert flags == []

    def test_nvidia_smi_fails(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("dualdocking.validation.md_config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            flags = _detect_gpu_flags()
            assert flags == []


# ── GROMACS command generation ───────────────────────────────────────


class TestBuildGromacsCommands:
    def test_command_count(self) -> None:
        commands = build_gromacs_commands()
        # 14 setup + 3 preprocess + 7 analysis = 24
        assert len(commands) == 24

    def test_all_commands_start_with_gmx(self) -> None:
        commands = build_gromacs_commands()
        for cmd in commands:
            assert cmd[0] == "gmx"

    def test_pdb2gmx_command(self) -> None:
        commands = build_gromacs_commands()
        pdb2gmx = commands[0]
        assert pdb2gmx[1] == "pdb2gmx"
        assert "-ff" in pdb2gmx
        assert "amber14sb" in pdb2gmx
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
        prod_mdrun = commands[13]  # 14th command (0-indexed)
        assert "mdrun" in prod_mdrun
        assert "-deffnm" in prod_mdrun
        idx = prod_mdrun.index("-deffnm")
        assert prod_mdrun[idx + 1].endswith("md")

    def test_make_ndx_present(self) -> None:
        commands = build_gromacs_commands()
        make_ndx = commands[7]
        assert make_ndx[1] == "make_ndx"

    def test_trjconv_preprocessing_present(self) -> None:
        commands = build_gromacs_commands()
        trjconv_cmds = [c for c in commands if "trjconv" in c]
        assert len(trjconv_cmds) == 3
        # Verify steps: whole, center/mol, fit
        assert "-pbc" in trjconv_cmds[0] and "whole" in trjconv_cmds[0]
        assert "-center" in trjconv_cmds[1]
        assert "-fit" in trjconv_cmds[2] and "rot+trans" in trjconv_cmds[2]

    def test_analysis_commands_use_index(self) -> None:
        commands = build_gromacs_commands()
        # Analysis commands are 17-23 (backbone rmsd, ligand rmsd, rmsf, rg, hbond-legacy, sasa, covar)
        for i in range(17, 24):
            assert "-n" in commands[i], f"Command {i} missing -n: {commands[i]}"

    def test_grompp_commands_use_index_after_make_ndx(self) -> None:
        commands = build_gromacs_commands()
        # NVT grompp (cmd 8), NPT grompp (cmd 10), prod grompp (cmd 12)
        for i in [8, 10, 12]:
            assert "-n" in commands[i], f"grompp command {i} missing -n"

    def test_maxwarn_values(self) -> None:
        commands = build_gromacs_commands()
        # ions grompp (3) and EM grompp (5) use -maxwarn 0
        for i in [3, 5]:
            idx = commands[i].index("-maxwarn")
            assert commands[i][idx + 1] == "0", f"Command {i} should use -maxwarn 0"
        # NVT (8), NPT (10), prod (12) grompp use -maxwarn 1
        for i in [8, 10, 12]:
            idx = commands[i].index("-maxwarn")
            assert commands[i][idx + 1] == "1", f"Command {i} should use -maxwarn 1"

    def test_hbond_legacy_command(self) -> None:
        commands = build_gromacs_commands()
        hbond_cmd = commands[21]
        assert hbond_cmd[1] == "hbond-legacy"

    def test_sasa_command(self) -> None:
        commands = build_gromacs_commands()
        sasa_cmd = commands[22]
        assert sasa_cmd[1] == "sasa"
        assert "-tu" in sasa_cmd
        assert "ns" in sasa_cmd

    def test_covar_command(self) -> None:
        commands = build_gromacs_commands()
        covar_cmd = commands[23]
        assert covar_cmd[1] == "covar"
        assert "-e" in covar_cmd  # eigenvalue output

    def test_no_energy_command(self) -> None:
        """Interaction energy command removed (no energygrps in prod MDP)."""
        commands = build_gromacs_commands()
        energy_cmds = [c for c in commands if "energy" in c]
        assert len(energy_cmds) == 0


# ── MDP writers ──────────────────────────────────────────────────────


class TestMDPWriters:
    def test_ions_mdp(self, tmp_path: Path) -> None:
        p = tmp_path / "ions.mdp"
        _write_ions_mdp(p)
        content = p.read_text()
        assert "integrator" in content and "steep" in content
        assert "nsteps" in content and "0" in content
        assert "cutoff-scheme" in content and "Verlet" in content
        assert "rcoulomb                 = 1.0" in content
        assert "rvdw                     = 1.0" in content

    def test_em_mdp(self, tmp_path: Path) -> None:
        p = tmp_path / "em.mdp"
        _write_em_mdp(p)
        content = p.read_text()
        assert "integrator" in content and "steep" in content
        assert "emtol" in content and "1000.0" in content
        assert "cutoff-scheme" in content and "Verlet" in content
        assert "DispCorr" in content and "EnerPres" in content
        assert "fourierspacing" in content and "0.12" in content
        assert "rcoulomb                 = 1.0" in content

    def test_nvt_mdp_uses_config(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", ref_t=300, equil_time_ps=200.0)
        p = tmp_path / "nvt.mdp"
        _write_nvt_mdp(p, cfg)
        content = p.read_text()
        assert "ref_t" in content and "300 300" in content
        assert "gen-vel" in content and "yes" in content
        assert "define" in content and "-DPOSRES -DPOSRES_LIG" in content
        assert "pcoupl" in content and "no" in content
        assert "cutoff-scheme" in content and "Verlet" in content
        assert "constraint-algorithm" in content and "LINCS" in content
        assert "lincs_order" in content and "4" in content
        assert "nstlog" in content
        assert "nstenergy" in content
        # tc-grps uses Protein_LIG / Water_and_ions
        assert "Protein_LIG Water_and_ions" in content
        # Check nsteps from equil_time_ps
        expected_nsteps = int(200.0 / 0.002)
        assert f"nsteps" in content and str(expected_nsteps) in content

    def test_npt_mdp_uses_c_rescale(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", ref_p=2.0)
        p = tmp_path / "npt.mdp"
        _write_npt_mdp(p, cfg)
        content = p.read_text()
        assert "ref_p" in content and "2.0" in content
        assert "pcoupl" in content and "C-rescale" in content
        assert "Berendsen" not in content
        assert "Parrinello-Rahman" not in content
        assert "continuation" in content and "yes" in content
        assert "define" in content and "-DPOSRES -DPOSRES_LIG" in content
        assert "tau_p" in content and "1.0" in content
        assert "Protein_LIG Water_and_ions" in content

    def test_prod_mdp_nsteps(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", simulation_time_ns=50.0, dt=0.002)
        p = tmp_path / "md.mdp"
        _write_prod_mdp(p, cfg)
        content = p.read_text()
        expected_nsteps = int(50.0 * 1e3 / 0.002)  # 25_000_000
        assert f"nsteps" in content and str(expected_nsteps) in content
        assert "define" not in content or "-DPOSRES" not in content
        # energygrps removed (GPU NB incompatible with multiple groups)
        assert "energygrps" not in content
        assert "Parrinello-Rahman" in content
        assert "tau_t" in content and "1.0 1.0" in content
        assert "constraint-algorithm" in content and "LINCS" in content
        assert "Protein_LIG Water_and_ions" in content

    def test_prod_mdp_custom_time(self, tmp_path: Path) -> None:
        cfg = MDConfig(pair_id="X", simulation_time_ns=100.0, dt=0.002)
        p = tmp_path / "md.mdp"
        _write_prod_mdp(p, cfg)
        content = p.read_text()
        expected_nsteps = int(100.0 * 1e3 / 0.002)  # 50_000_000
        assert str(expected_nsteps) in content


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

    def test_lig_inserted_before_crystal_waters(self, tmp_path: Path) -> None:
        """Ligand atoms go before crystal-water atoms in merged GRO."""
        prot_gro = tmp_path / "protein.gro"
        prot_gro.write_text(
            "Protein in water\n"
            " 5\n"
            "    1ALA      N    1   0.100   0.200   0.300\n"
            "    1ALA     CA    2   0.200   0.300   0.400\n"
            "    2HOH     OW    3   1.000   1.000   1.000\n"
            "    2HOH    HW1    4   1.100   1.000   1.000\n"
            "    2HOH    HW2    5   1.000   1.100   1.000\n"
            "   5.0   5.0   5.0\n"
        )

        lig_gro = tmp_path / "lig.gro"
        lig_gro.write_text(
            "LIG\n"
            " 2\n"
            "    1LIG      C    1   0.500   0.500   0.500\n"
            "    1LIG     C1    2   0.600   0.600   0.600\n"
            "   5.0   5.0   5.0\n"
        )

        out_gro = tmp_path / "complex.gro"
        _merge_gro(prot_gro, lig_gro, out_gro)

        lines = out_gro.read_text().splitlines()
        assert int(lines[1].strip()) == 7  # 2 prot + 2 lig + 3 water

        # Order: protein, LIG, HOH
        resnames = [line[5:10].strip() for line in lines[2:9]]
        assert resnames == ["ALA", "ALA", "LIG", "LIG", "HOH", "HOH", "HOH"]


# ── Topology patching ────────────────────────────────────────────────


class TestPatchTopology:
    def test_inserts_include_and_molecule(self, tmp_path: Path) -> None:
        topol = tmp_path / "topol.top"
        lig_itp = tmp_path / "LIG_GMX.itp"

        topol.write_text(
            '#include "amber14sb.ff/forcefield.itp"\n'
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

    def test_lig_inserted_before_sol_in_molecules(self, tmp_path: Path) -> None:
        """Crystal waters (SOL) from PDB must stay after LIG in [molecules]."""
        topol = tmp_path / "topol.top"
        lig_itp = tmp_path / "LIG_GMX.itp"

        topol.write_text(
            '#include "amber14sb.ff/forcefield.itp"\n'
            "\n"
            "[ system ]\n"
            "Protein in water\n"
            "\n"
            "[ molecules ]\n"
            "; Compound        #mols\n"
            "Protein_chain_A     1\n"
            "SOL               180\n"
        )
        lig_itp.write_text("; ligand itp\n")

        _patch_topology(topol, lig_itp)
        content = topol.read_text()

        # LIG must appear before SOL in [molecules]
        lig_pos = content.index("LIG              1")
        sol_pos = content.index("SOL               180")
        assert lig_pos < sol_pos, "LIG must come before SOL in [molecules]"

    def test_copies_posre_lig_itp(self, tmp_path: Path) -> None:
        """posre_lig.itp is copied alongside LIG_GMX.itp."""
        topol = tmp_path / "topol.top"
        topol.write_text("[ system ]\nP\n[ molecules ]\nP 1\n")

        acpype = tmp_path / "acpype"
        acpype.mkdir()
        lig_itp = acpype / "LIG_GMX.itp"
        lig_itp.write_text("; ligand\n")
        posre = acpype / "posre_lig.itp"
        posre.write_text("; posres\n")

        _patch_topology(topol, lig_itp)

        assert (tmp_path / "posre_lig.itp").exists()
        assert (tmp_path / "posre_lig.itp").read_text() == "; posres\n"


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
        # New fields have defaults
        assert np.isnan(r.backbone_rmsd_mean)
        assert np.isnan(r.rg_mean)
        assert r.converged is False
        assert np.isnan(r.mmpbsa_dg)

    def test_creation_with_new_fields(self) -> None:
        r = MDResult(
            pair="CDK2_ERK2", mode="double", mol_id=0,
            target="CDK2", target_label="A", smiles="CCO",
            mean_rmsd=1.5, max_rmsd=2.8, stable=True,
            mean_hbonds=3.2, mean_energy=-150.5,
            backbone_rmsd_mean=1.2, rg_mean=14.5, rg_std=0.3,
            converged=True, convergence_ks_pvalue=0.82,
            coulomb_sr=-50.0, lj_sr=-100.0, mmpbsa_dg=-8.5,
        )
        assert r.backbone_rmsd_mean == 1.2
        assert r.rg_mean == 14.5
        assert r.converged is True
        assert r.mmpbsa_dg == -8.5

    def test_creation_with_sasa_pca_fields(self) -> None:
        r = MDResult(
            pair="CDK2_ERK2", mode="double", mol_id=0,
            target="CDK2", target_label="A", smiles="CCO",
            mean_rmsd=1.5, max_rmsd=2.8, stable=True,
            mean_hbonds=3.2, mean_energy=-150.5,
            sasa_mean=5200.0, sasa_std=120.0,
            pca_pc1_variance=45.3, pca_pc2_variance=18.7,
        )
        assert r.sasa_mean == 5200.0
        assert r.sasa_std == 120.0
        assert r.pca_pc1_variance == 45.3
        assert r.pca_pc2_variance == 18.7

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

    def test_new_columns_present(self, tmp_path: Path) -> None:
        results = [
            MDResult(
                pair="P", mode="m", mol_id=0, target="T",
                target_label="A", smiles="C", mean_rmsd=1.0,
                max_rmsd=2.0, stable=True, mean_hbonds=1.0,
                mean_energy=-100.0, backbone_rmsd_mean=0.8,
                rg_mean=15.0, converged=True, mmpbsa_dg=-5.0,
            ),
        ]
        path = tmp_path / "summary.parquet"
        write_summary(results, path)
        df = pd.read_parquet(path)
        assert "backbone_rmsd_mean" in df.columns
        assert "rg_mean" in df.columns
        assert "converged" in df.columns
        assert "mmpbsa_dg" in df.columns
        assert "coulomb_sr" in df.columns
        assert "lj_sr" in df.columns
        assert "convergence_ks_pvalue" in df.columns

    def test_appends_and_deduplicates(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.parquet"
        r1 = MDResult(
            pair="P", mode="m", mol_id=0, target="T",
            target_label="A", smiles="C", mean_rmsd=1.0,
            max_rmsd=2.0, stable=True, mean_hbonds=1.0,
            mean_energy=-100.0,
        )
        write_summary([r1], path)

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
        assert result["smiles"].iloc[0] == "B"  # -1.0 is best


# ── _make_index ──────────────────────────────────────────────────────


class TestMakeIndex:
    def test_creates_index_with_stdin(self, tmp_path: Path) -> None:
        tpr = tmp_path / "em.tpr"
        tpr.touch()

        def _mock_make_ndx(*_args: object, **_kwargs: object) -> MagicMock:
            ndx_path = tmp_path / "index.ndx"
            ndx_path.write_text(
                "[ Protein ]\n1 2 3\n"
                "[ LIG ]\n4 5\n"
                "[ SOL ]\n6 7\n"
                "[ NA ]\n8\n",
            )
            return MagicMock(returncode=0)

        with patch("dualdocking.validation.md_setup._run") as mock_run:
            mock_run.side_effect = _mock_make_ndx
            ndx = _make_index(tpr, tmp_path, cmd_timeout=120)

        assert ndx == tmp_path / "index.ndx"
        mock_run.assert_called_once()
        kwargs = mock_run.call_args[1]
        assert kwargs["input_text"] == "q\n"
        assert "make_ndx" in mock_run.call_args[0][0]

        content = ndx.read_text()
        assert "[ Protein_LIG ]" in content
        assert "[ Water_and_ions ]" in content


# ── _preprocess_trajectory ───────────────────────────────────────────


class TestPreprocessTrajectory:
    def test_three_trjconv_calls_with_stdin(self, tmp_path: Path) -> None:
        tpr = tmp_path / "md.tpr"
        xtc = tmp_path / "md.xtc"
        tpr.touch()
        xtc.touch()

        # Create intermediate files so unlink works
        (tmp_path / "md_whole.xtc").touch()
        (tmp_path / "md_center.xtc").touch()

        with patch("dualdocking.validation.md_analysis._run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = _preprocess_trajectory(tpr, xtc, tmp_path, cmd_timeout=120)

        assert result == tmp_path / "md_fit.xtc"
        assert mock_run.call_count == 3

        # Verify stdin for each call
        calls = mock_run.call_args_list
        assert calls[0][1]["input_text"] == "System\n"
        assert calls[1][1]["input_text"] == "Protein\nSystem\n"
        assert calls[2][1]["input_text"] == "Backbone\nSystem\n"

        # Verify trjconv commands
        assert "trjconv" in calls[0][0][0]
        assert "-pbc" in calls[0][0][0] and "whole" in calls[0][0][0]
        assert "-center" in calls[1][0][0]
        assert "-fit" in calls[2][0][0] and "rot+trans" in calls[2][0][0]

    def test_intermediate_files_cleaned(self, tmp_path: Path) -> None:
        tpr = tmp_path / "md.tpr"
        xtc = tmp_path / "md.xtc"
        tpr.touch()
        xtc.touch()
        (tmp_path / "md_whole.xtc").write_text("dummy")
        (tmp_path / "md_center.xtc").write_text("dummy")

        with patch("dualdocking.validation.md_analysis._run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _preprocess_trajectory(tpr, xtc, tmp_path, cmd_timeout=120)

        assert not (tmp_path / "md_whole.xtc").exists()
        assert not (tmp_path / "md_center.xtc").exists()


# ── _analyze_trajectory ──────────────────────────────────────────────


class TestAnalyzeTrajectory:
    def test_all_analyses_called_with_stdin(self, tmp_path: Path) -> None:
        from dualdocking.validation.md_analysis import _analyze_trajectory

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        out_dir = tmp_path / "out"
        ndx = work_dir / "index.ndx"
        ndx.touch()
        xtc_fit = work_dir / "md_fit.xtc"
        xtc_fit.touch()
        (work_dir / "md.tpr").touch()

        cfg = MDConfig(pair_id="X", simulation_time_ns=50.0)

        # Mock _run and _parse_xvg to return valid data
        mock_df = pd.DataFrame({
            "time_ns": [0.0, 25.0, 50.0],
            "rmsd_nm": [0.1, 0.15, 0.12],
        })
        rmsf_df = pd.DataFrame({
            "residue": [1.0, 2.0, 3.0],
            "rmsf_nm": [0.05, 0.08, 0.06],
        })
        rg_df = pd.DataFrame({
            "time_ps": [0.0, 1000.0, 2000.0],
            "rg_nm": [1.4, 1.5, 1.45],
        })
        hbond_df = pd.DataFrame({
            "time_ns": [0.0, 25.0, 50.0],
            "n_hbonds": [3.0, 4.0, 3.0],
        })

        sasa_df = pd.DataFrame({
            "time_ns": [0.0, 25.0, 50.0],
            "sasa_nm2": [50.0, 51.0, 49.0],
        })
        eigenval_df = pd.DataFrame({
            "mode": [1.0, 2.0, 3.0],
            "eigenvalue": [5.0, 3.0, 2.0],
        })

        # 7 parse calls: bb rmsd, lig rmsd, rmsf, rg, hbond, sasa, eigenval
        parse_returns = [mock_df, mock_df, rmsf_df, rg_df, hbond_df, sasa_df, eigenval_df]
        call_idx = {"i": 0}

        def mock_parse_xvg(path, columns):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return parse_returns[idx]

        with patch("dualdocking.validation.md_analysis._run") as mock_run, \
             patch("dualdocking.validation.md_analysis._parse_xvg", side_effect=mock_parse_xvg), \
             patch("dualdocking.validation.md_analysis._assess_convergence", return_value=(True, 0.85)):
            mock_run.return_value = MagicMock(returncode=0)
            result = _analyze_trajectory(work_dir, ndx, xtc_fit, out_dir, cfg)

        # Verify 7 _run calls (bb rmsd, lig rmsd, rmsf, rg, hbond-legacy, sasa, covar)
        assert mock_run.call_count == 7

        # Verify stdin passed to each
        calls = mock_run.call_args_list
        assert calls[0][1]["input_text"] == "Backbone\nBackbone\n"  # bb rmsd
        assert calls[1][1]["input_text"] == "LIG\nLIG\n"  # lig rmsd
        assert calls[2][1]["input_text"] == "Backbone\n"  # rmsf
        assert calls[3][1]["input_text"] == "Protein\n"  # rg
        assert calls[4][1]["input_text"] == "Protein\nLIG\n"  # hbond-legacy
        assert calls[5][1]["input_text"] == "Protein\n"  # sasa
        assert calls[6][1]["input_text"] == "Backbone\nBackbone\n"  # covar

        # Verify hbond-legacy is used
        hbond_cmd = calls[4][0][0]
        assert "hbond-legacy" in hbond_cmd

        # Verify sasa and covar commands
        sasa_cmd = calls[5][0][0]
        assert "sasa" in sasa_cmd
        covar_cmd = calls[6][0][0]
        assert "covar" in covar_cmd

        # Verify result keys
        assert "mean_rmsd" in result
        assert "backbone_rmsd_mean" in result
        assert "rg_mean" in result
        assert "converged" in result
        assert "sasa_mean" in result
        assert "sasa_std" in result
        assert "pca_pc1_variance" in result
        assert "pca_pc2_variance" in result
        # Interaction energy fields are NaN (no energygrps in prod MDP)
        assert "coulomb_sr" in result
        assert "lj_sr" in result
        assert np.isnan(result["coulomb_sr"])
        assert np.isnan(result["lj_sr"])
        assert np.isnan(result["mean_energy"])


# ── _assess_convergence ──────────────────────────────────────────────


class TestConvergence:
    def test_converged_plateau(self) -> None:
        rng = np.random.default_rng(42)
        # Stable plateau: both halves from same distribution
        data = pd.Series(rng.normal(loc=2.0, scale=0.1, size=200))
        converged, pval = _assess_convergence(data)
        assert converged is True
        assert pval > 0.05

    def test_divergent_drift(self) -> None:
        # Clear drift: first half low, second half high
        first = np.full(100, 1.0)
        second = np.full(100, 3.0)
        data = pd.Series(np.concatenate([first, second]))
        converged, pval = _assess_convergence(data)
        assert converged is False

    def test_short_series_not_converged(self) -> None:
        data = pd.Series([1.0, 2.0, 3.0])
        converged, pval = _assess_convergence(data)
        assert converged is False
        assert np.isnan(pval)


# ── _parameterize_ligand ─────────────────────────────────────────────


class TestParameterizeLigand:
    def test_no_gen3d(self, tmp_path: Path) -> None:
        """Verify obabel is called without --gen3d flag."""
        from dualdocking.validation.md_setup import _parameterize_ligand

        mock_mol = MagicMock()
        mock_chem = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_allchem = MagicMock()
        mock_allchem.EmbedMolecule.return_value = 0
        mock_allchem.MMFFOptimizeMolecule.return_value = 0
        acpype_dir = tmp_path / "LIG.acpype"
        acpype_dir.mkdir()
        (acpype_dir / "LIG_GMX.itp").write_text("; test itp\n")
        (acpype_dir / "LIG_GMX.gro").write_text(
            "LIG\n 1\n    1LIG     C1    1   0.0   0.0   0.0\n   5.0   5.0   5.0\n"
        )

        with patch("dualdocking.validation.md_setup._run") as mock_run, \
             patch("rdkit.Chem.MolFromSmiles", mock_chem.MolFromSmiles), \
             patch("rdkit.Chem.AddHs", mock_chem.AddHs), \
             patch("rdkit.Chem.MolToMolFile"), \
             patch("rdkit.Chem.AllChem.EmbedMolecule", mock_allchem.EmbedMolecule), \
             patch("rdkit.Chem.AllChem.ETKDGv3", mock_allchem.ETKDGv3), \
             patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule", mock_allchem.MMFFOptimizeMolecule):
            mock_run.return_value = MagicMock(returncode=0)
            _parameterize_ligand("CCO", tmp_path, cmd_timeout=60)

        # Check obabel call doesn't have --gen3d
        for c in mock_run.call_args_list:
            cmd = c[0][0]
            if "obabel" in cmd:
                assert "--gen3d" not in cmd

    def test_genrestr_called(self, tmp_path: Path) -> None:
        """Verify gmx genrestr is called and ITP is patched."""
        from dualdocking.validation.md_setup import _parameterize_ligand

        mock_mol = MagicMock()
        mock_chem = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_allchem = MagicMock()
        mock_allchem.EmbedMolecule.return_value = 0
        mock_allchem.MMFFOptimizeMolecule.return_value = 0
        acpype_dir = tmp_path / "LIG.acpype"
        acpype_dir.mkdir()
        (acpype_dir / "LIG_GMX.itp").write_text("; test itp\n")
        (acpype_dir / "LIG_GMX.gro").write_text(
            "LIG\n 1\n    1LIG     C1    1   0.0   0.0   0.0\n   5.0   5.0   5.0\n"
        )

        with patch("dualdocking.validation.md_setup._run") as mock_run, \
             patch("rdkit.Chem.MolFromSmiles", mock_chem.MolFromSmiles), \
             patch("rdkit.Chem.AddHs", mock_chem.AddHs), \
             patch("rdkit.Chem.MolToMolFile"), \
             patch("rdkit.Chem.AllChem.EmbedMolecule", mock_allchem.EmbedMolecule), \
             patch("rdkit.Chem.AllChem.ETKDGv3", mock_allchem.ETKDGv3), \
             patch("rdkit.Chem.AllChem.MMFFOptimizeMolecule", mock_allchem.MMFFOptimizeMolecule):
            mock_run.return_value = MagicMock(returncode=0)
            _parameterize_ligand("CCO", tmp_path, cmd_timeout=60)

        # Check genrestr was called
        genrestr_calls = [
            c for c in mock_run.call_args_list
            if len(c[0]) > 0 and "genrestr" in c[0][0]
        ]
        assert len(genrestr_calls) == 1
        assert genrestr_calls[0][1]["input_text"] == "0\n"

        # Check ITP was patched with POSRES_LIG block
        itp_content = (acpype_dir / "LIG_GMX.itp").read_text()
        assert "#ifdef POSRES_LIG" in itp_content
        assert '#include "posre_lig.itp"' in itp_content

    def test_embed_failure(self, tmp_path: Path) -> None:
        """EmbedMolecule returning -1 should raise."""
        from dualdocking.validation.md_setup import _parameterize_ligand

        mock_mol = MagicMock()
        mock_chem = MagicMock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_allchem = MagicMock()
        mock_allchem.EmbedMolecule.return_value = -1  # failure

        with patch("rdkit.Chem.MolFromSmiles", mock_chem.MolFromSmiles), \
             patch("rdkit.Chem.AddHs", mock_chem.AddHs), \
             patch("rdkit.Chem.AllChem.EmbedMolecule", mock_allchem.EmbedMolecule), \
             patch("rdkit.Chem.AllChem.ETKDGv3", MagicMock()):
            with pytest.raises(RuntimeError, match="3D embedding failed"):
                _parameterize_ligand("CCO", tmp_path, cmd_timeout=60)


# ── _setup_system ────────────────────────────────────────────────────


class TestSetupSystem:
    def test_genion_stdin(self, tmp_path: Path) -> None:
        """Verify genion receives SOL stdin."""
        from dualdocking.validation.md_setup import _setup_system

        cfg = MDConfig(pair_id="X", gpu_auto=False)

        with patch("dualdocking.validation.md_setup._run") as mock_run, \
             patch("dualdocking.validation.md_setup._merge_gro"), \
             patch("dualdocking.validation.md_setup._patch_topology"), \
             patch("dualdocking.validation.md_setup._write_ions_mdp"), \
             patch("dualdocking.validation.md_setup._write_em_mdp"), \
             patch("dualdocking.validation.md_setup._write_nvt_mdp"), \
             patch("dualdocking.validation.md_setup._write_npt_mdp"), \
             patch("dualdocking.validation.md_setup._write_prod_mdp"), \
             patch("dualdocking.validation.md_setup._make_index") as mock_ndx:
            mock_run.return_value = MagicMock(returncode=0)
            mock_ndx.return_value = tmp_path / "index.ndx"

            protein_pdb = tmp_path / "protein.pdb"
            protein_pdb.touch()
            acpype_dir = tmp_path / "LIG.acpype"
            acpype_dir.mkdir()
            (acpype_dir / "LIG_GMX.gro").touch()
            (acpype_dir / "LIG_GMX.itp").touch()

            _setup_system(protein_pdb, acpype_dir, tmp_path, cfg)

        # Find the genion call
        genion_calls = [
            c for c in mock_run.call_args_list
            if len(c[0]) > 0 and "genion" in c[0][0]
        ]
        assert len(genion_calls) == 1
        assert genion_calls[0][1]["input_text"] == "SOL\n"

    def test_uses_amber14sb(self, tmp_path: Path) -> None:
        """Verify pdb2gmx uses amber14sb force field."""
        from dualdocking.validation.md_setup import _setup_system

        cfg = MDConfig(pair_id="X", gpu_auto=False)

        with patch("dualdocking.validation.md_setup._run") as mock_run, \
             patch("dualdocking.validation.md_setup._merge_gro"), \
             patch("dualdocking.validation.md_setup._patch_topology"), \
             patch("dualdocking.validation.md_setup._write_ions_mdp"), \
             patch("dualdocking.validation.md_setup._write_em_mdp"), \
             patch("dualdocking.validation.md_setup._write_nvt_mdp"), \
             patch("dualdocking.validation.md_setup._write_npt_mdp"), \
             patch("dualdocking.validation.md_setup._write_prod_mdp"), \
             patch("dualdocking.validation.md_setup._make_index") as mock_ndx:
            mock_run.return_value = MagicMock(returncode=0)
            mock_ndx.return_value = tmp_path / "index.ndx"

            protein_pdb = tmp_path / "protein.pdb"
            protein_pdb.touch()
            acpype_dir = tmp_path / "LIG.acpype"
            acpype_dir.mkdir()
            (acpype_dir / "LIG_GMX.gro").touch()
            (acpype_dir / "LIG_GMX.itp").touch()

            _setup_system(protein_pdb, acpype_dir, tmp_path, cfg)

        # Find the pdb2gmx call
        pdb2gmx_calls = [
            c for c in mock_run.call_args_list
            if len(c[0]) > 0 and "pdb2gmx" in c[0][0]
        ]
        assert len(pdb2gmx_calls) == 1
        cmd = pdb2gmx_calls[0][0][0]
        assert "amber14sb" in cmd
        assert "amber99sb-ildn" not in cmd


# ── CLI test ─────────────────────────────────────────────────────────


class TestCLI:
    def test_help(self) -> None:
        from click.testing import CliRunner

        from cli.s9_md.run_md import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--pair" in result.output
        assert "--top-n" in result.output
        assert "--mmpbsa" in result.output
        assert "--dry-run" in result.output
        assert "--simulation-time" in result.output
        assert "--no-gpu" in result.output

    def test_missing_tool_exits(self) -> None:
        from click.testing import CliRunner

        from cli.s9_md.run_md import main

        runner = CliRunner()
        with patch("shutil.which", return_value=None):
            result = runner.invoke(main, ["--pair", "CDK2_ERK2"])
        assert result.exit_code != 0
