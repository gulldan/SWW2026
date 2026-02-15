"""MD validation configuration: dataclasses, constants, and utility helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# ── Defaults ─────────────────────────────────────────────────────────
_DEFAULT_RMSD_THRESHOLD_A = 3.0  # Angstrom
_DEFAULT_STABLE_WINDOW_NS = 25.0
_DEFAULT_SIM_NS = 50.0
_DEFAULT_DT = 0.002  # ps
_DEFAULT_NSTXOUT = 5000  # write every 10 ps
_DEFAULT_REF_T = 310  # K (physiological)
_DEFAULT_REF_P = 1.0  # bar

# ── Common MDP parameters (AMBER standard: 1.0 nm cutoff) ───────────
_MDP_VERLET = (
    "cutoff-scheme            = Verlet\n"
    "coulombtype              = PME\n"
    "rcoulomb                 = 1.0\n"
    "rvdw                     = 1.0\n"
    "vdw-modifier             = Potential-shift\n"
    "DispCorr                 = EnerPres\n"
    "fourierspacing           = 0.12\n"
    "pme_order                = 4\n"
    "pbc                      = xyz\n"
)


@dataclass(frozen=True)
class MDConfig:
    """Configuration for an MD validation run."""

    pair_id: str
    top_n: int = 3
    modes: list[str] = field(default_factory=lambda: ["single_a", "single_b", "double"])
    seeds: list[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])
    simulation_time_ns: float = _DEFAULT_SIM_NS
    stable_window_ns: float = _DEFAULT_STABLE_WINDOW_NS
    rmsd_threshold_a: float = _DEFAULT_RMSD_THRESHOLD_A
    mmpbsa: bool = False
    output_dir: Path = Path("output/md")
    results_dir: Path = Path("output/results")
    targets_dir: Path = Path("output/data/targets")
    dt: float = _DEFAULT_DT
    ref_t: int = _DEFAULT_REF_T
    ref_p: float = _DEFAULT_REF_P
    nstxout_compressed: int = _DEFAULT_NSTXOUT
    equil_time_ps: float = 200.0
    nstlog: int = 1000
    nstenergy: int = 1000
    gpu_auto: bool = True
    convergence_ks_alpha: float = 0.05
    cmd_timeout: int = 600
    mdrun_timeout: int = 259200  # 72 h


@dataclass(frozen=True)
class MDResult:
    """Result from a single molecule x target MD run."""

    pair: str
    mode: str
    mol_id: int
    target: str
    target_label: str
    smiles: str
    mean_rmsd: float
    max_rmsd: float
    stable: bool
    mean_hbonds: float
    mean_energy: float
    backbone_rmsd_mean: float = float("nan")
    backbone_rmsd_final: float = float("nan")
    rmsf_binding_pocket: float = float("nan")
    rg_mean: float = float("nan")
    rg_std: float = float("nan")
    converged: bool = False
    convergence_ks_pvalue: float = float("nan")
    coulomb_sr: float = float("nan")
    lj_sr: float = float("nan")
    mmpbsa_dg: float = float("nan")


# ── Utility: external command runner ─────────────────────────────────
def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    timeout: int = 600,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command with logging, stdin support, and timeout."""
    logger.debug("$ {}", " ".join(cmd[:8]))
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        input=input_text,
    )
    if check and result.returncode != 0:
        tail = (result.stderr or "")[-1000:]
        raise RuntimeError(
            f"Command failed (rc={result.returncode}): {' '.join(cmd[:6])}\n{tail}"
        )
    return result


# ── GPU detection ────────────────────────────────────────────────────
def _detect_gpu_flags() -> list[str]:
    """Return GROMACS mdrun GPU flags if an NVIDIA GPU is available."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        r = subprocess.run(  # noqa: S603
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "gpu"]
    except Exception:
        logger.debug("nvidia-smi detection failed")
    return []
