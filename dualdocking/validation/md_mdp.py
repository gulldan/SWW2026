"""MDP file writers for GROMACS MD simulations.

Modern protocol: amber14sb force field (Maier et al. 2015), 1.0 nm cutoffs
(AMBER standard), C-rescale barostat (Bernetti & Bussi 2020) for NPT
equilibration, Protein_LIG / Water_and_ions thermostat coupling groups,
and ligand position restraints during equilibration.
"""

from __future__ import annotations

from pathlib import Path

from dualdocking.validation.md_config import _MDP_VERLET, MDConfig


def _write_ions_mdp(path: Path) -> None:
    """Write a minimal MDP for ion placement (grompp only)."""
    path.write_text(
        "integrator               = steep\n"
        "nsteps                   = 0\n"
        "ensemble-temperature-setting = not available\n" + _MDP_VERLET
    )


def _write_em_mdp(path: Path) -> None:
    """Write energy minimization MDP."""
    path.write_text(
        "integrator               = steep\n"
        "emtol                    = 1000.0\n"
        "emstep                   = 0.01\n"
        "nsteps                   = 50000\n"
        "nstlist                  = 10\n"
        "ensemble-temperature-setting = not available\n" + _MDP_VERLET
    )


def _write_equil_mdp(
    path: Path,
    config: MDConfig,
    *,
    pressure_block: str,
    velocity_block: str,
) -> None:
    """Write shared equilibration MDP content for NVT/NPT phases.

    Thermostat groups couple protein+ligand together, separate from
    solvent+ions.  Both protein and ligand are position-restrained.
    """
    nsteps = int(config.equil_time_ps / config.dt)
    path.write_text(
        f"integrator               = md\n"
        f"dt                       = {config.dt}\n"
        f"nsteps                   = {nsteps}\n"
        f"nstxout-compressed       = {config.nstxout_compressed}\n"
        f"nstlog                   = {config.nstlog}\n"
        f"nstenergy                = {config.nstenergy}\n" + _MDP_VERLET + "tcoupl                   = V-rescale\n"
        f"tc-grps                  = Protein_LIG Water_and_ions\n"
        f"tau_t                    = 0.1 0.1\n"
        f"ref_t                    = {config.ref_t} {config.ref_t}\n"
        + pressure_block
        + velocity_block
        + "constraints              = h-bonds\n"
        "constraint-algorithm     = LINCS\n"
        "lincs_order              = 4\n"
        "define                   = -DPOSRES -DPOSRES_LIG\n"
    )


def _write_nvt_mdp(path: Path, config: MDConfig) -> None:
    """Write NVT equilibration MDP (position restraints, no pressure coupling)."""
    _write_equil_mdp(
        path,
        config,
        pressure_block="pcoupl                   = no\n",
        velocity_block=f"gen-vel                  = yes\ngen-temp                 = {config.ref_t}\n",
    )


def _write_npt_mdp(path: Path, config: MDConfig) -> None:
    """Write NPT equilibration MDP (C-rescale barostat; Bernetti & Bussi 2020)."""
    _write_equil_mdp(
        path,
        config,
        pressure_block=(
            "pcoupl                   = C-rescale\n"
            "pcoupltype               = isotropic\n"
            "tau_p                    = 1.0\n"
            f"ref_p                    = {config.ref_p}\n"
            "compressibility          = 4.5e-5\n"
        ),
        velocity_block="gen-vel                  = no\ncontinuation             = yes\n",
    )


def _write_prod_mdp(path: Path, config: MDConfig) -> None:
    """Write production MD MDP (Parrinello-Rahman barostat)."""
    nsteps = int(config.simulation_time_ns * 1e3 / config.dt)
    path.write_text(
        f"integrator               = md\n"
        f"dt                       = {config.dt}\n"
        f"nsteps                   = {nsteps}\n"
        f"nstxout-compressed       = {config.nstxout_compressed}\n"
        f"nstlog                   = {config.nstlog}\n"
        f"nstenergy                = {config.nstenergy}\n" + _MDP_VERLET + "tcoupl                   = V-rescale\n"
        f"tc-grps                  = Protein_LIG Water_and_ions\n"
        f"tau_t                    = 1.0 1.0\n"
        f"ref_t                    = {config.ref_t} {config.ref_t}\n"
        f"pcoupl                   = Parrinello-Rahman\n"
        "pcoupltype               = isotropic\n"
        "tau_p                    = 2.0\n"
        f"ref_p                    = {config.ref_p}\n"
        "compressibility          = 4.5e-5\n"
        "gen-vel                  = no\n"
        "continuation             = yes\n"
        "constraints              = h-bonds\n"
        "constraint-algorithm     = LINCS\n"
        "lincs_order              = 4\n"
    )
