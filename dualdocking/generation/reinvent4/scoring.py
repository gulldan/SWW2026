"""Dual-target docking scoring for REINVENT4.

Provides ``score_batch()`` — the pure scoring function that docks SMILES
against one or two targets via UniDock, normalizes to z-scores, and
evaluates the reward mode.

Also provides REINVENT4 plugin classes (``Parameters``, ``DualDockingComponent``)
that follow the ``reinvent_plugins`` component protocol.  When a thin wrapper
file ``comp_dual_docking.py`` re-exports these classes into REINVENT4's plugin
directory, the component is discovered automatically.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# REINVENT4 plugin protocol (graceful fallback when reinvent4 not installed)
# ---------------------------------------------------------------------------

try:
    from reinvent_plugins.components.add_tag import add_tag
    from reinvent_plugins.components.component_results import ComponentResults
except ImportError:

    def add_tag(label: str, text: str = "True") -> Callable:  # type: ignore[misc]
        """Fallback decorator matching REINVENT4's ``add_tag`` API."""

        def _wrapper(cls: type) -> type:
            setattr(cls, label, text)
            return cls

        return _wrapper

    @dataclass
    class ComponentResults:  # type: ignore[no-redef]
        """Fallback when reinvent4 not installed."""

        scores: list[np.ndarray]
        metadata: dict | None = None


# ---------------------------------------------------------------------------
# Constants (consistent with FREED++ dual_reward_patch)
# ---------------------------------------------------------------------------

SCORE_MIN: float = -20.0
SCORE_MAX: float = 10.0
MAX_Z_ABS: float = 10.0
INVALID_Z_PENALTY: float = 4.0
_GPU_SUB_BATCH: int = 16
_TIMEOUT_PER_LIGAND: float = 5.0  # seconds per ligand for timeout calculation
_TIMEOUT_MIN: float = 60.0  # minimum timeout (seconds)


# ---------------------------------------------------------------------------
# Reward statistics tracking
# ---------------------------------------------------------------------------


@dataclass
class RewardStats:
    """Accumulates per-molecule scoring outcomes during an RL run.

    Lifecycle: ``reset()`` before training, ``record()`` per molecule during
    training, ``write_metadata()`` after training.
    """

    total: int = 0
    invalid_a: int = 0
    invalid_b: int = 0

    def reset(self) -> None:
        """Zero all counters."""
        self.total = 0
        self.invalid_a = 0
        self.invalid_b = 0

    def record(self, *, valid_a: bool, valid_b: bool | None = None) -> None:
        """Record outcome for one molecule.

        Args:
            valid_a: Whether target-A docking succeeded.
            valid_b: Whether target-B docking succeeded (``None`` when
                the reward mode does not use target B).
        """
        self.total += 1
        if not valid_a:
            self.invalid_a += 1
        if valid_b is not None and not valid_b:
            self.invalid_b += 1

    def to_metadata(
        self,
        *,
        reward_mode: str,
        primary_target: str,
        reward_params: dict | None = None,
    ) -> dict:
        """Build the metadata dict consumed by ``validate_reward_quality()``.

        Args:
            reward_mode: Reward strategy name.
            primary_target: ``"both"`` / target-A id / target-B id.
            reward_params: Optional extra reward parameters.

        Returns:
            Metadata dict with the schema expected by the quality gate.
        """
        total = max(self.total, 1)  # avoid division by zero
        frac_a = self.invalid_a / total
        frac_b = self.invalid_b / total
        return {
            "total_calls": self.total,
            "invalid_count_a": self.invalid_a,
            "invalid_count_b": self.invalid_b,
            "invalid_reward_fraction_a": round(frac_a, 6),
            "invalid_reward_fraction_b": round(frac_b, 6),
            "timeout_count_a": 0,
            "timeout_count_b": 0,
            "timeout_fraction_a": 0.0,
            "timeout_fraction_b": 0.0,
            "invalid_z_penalty": INVALID_Z_PENALTY,
            "max_z_abs": MAX_Z_ABS,
            "primary_target": primary_target,
            "reward_mode": reward_mode,
            "reward_params": reward_params or {},
            "run_valid": frac_a <= 0.10 and frac_b <= 0.10,
        }

    def write_metadata(
        self,
        path: Path,
        *,
        reward_mode: str,
        primary_target: str,
        reward_params: dict | None = None,
    ) -> None:
        """Write ``reward_metadata.json`` to *path*.

        Args:
            path: Output file path.
            reward_mode: Reward strategy name.
            primary_target: ``"both"`` / target-A id / target-B id.
            reward_params: Optional extra reward parameters.
        """
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.to_metadata(
            reward_mode=reward_mode,
            primary_target=primary_target,
            reward_params=reward_params,
        )
        path.write_text(json.dumps(meta, indent=2))
        logger.info(
            "Wrote reward metadata: {} (total={}, invalid_a={}, invalid_b={})", path, self.total, self.invalid_a, self.invalid_b
        )


_reward_stats = RewardStats()


# ---------------------------------------------------------------------------
# Docking helpers
# ---------------------------------------------------------------------------


def smiles_to_pdbqt(smiles: str, out_path: Path) -> bool:
    """Convert SMILES to PDBQT via RDKit 3D embedding + meeko.

    Args:
        smiles: Input SMILES string.
        out_path: Output PDBQT file path.

    Returns:
        ``True`` if conversion succeeded.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, _error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                out_path.write_text(pdbqt_string)
                return True
    except Exception:
        logger.debug("SMILES->PDBQT failed: {}", smiles)
    return False


def dock_single(
    ligand_pdbqt: Path,
    receptor_pdbqt: Path,
    center: str,
    size: str,
    vina_program: str = "unidock",
    exhaustiveness: int = 8,
) -> float | None:
    """Dock one ligand against one receptor and return the best score.

    Args:
        ligand_pdbqt: Ligand PDBQT file.
        receptor_pdbqt: Receptor PDBQT file.
        center: Comma-separated ``"cx,cy,cz"`` center coordinates.
        size: Comma-separated ``"sx,sy,sz"`` box dimensions.
        vina_program: Docking binary name.
        exhaustiveness: Docking exhaustiveness.

    Returns:
        Best docking score (kcal/mol) or ``None`` on failure.
    """
    out_pdbqt = ligand_pdbqt.with_suffix(".out.pdbqt")
    cx, cy, cz = center.split(",")
    sx, sy, sz = size.split(",")

    cmd = [
        vina_program,
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        cx.strip(),
        "--center_y",
        cy.strip(),
        "--center_z",
        cz.strip(),
        "--size_x",
        sx.strip(),
        "--size_y",
        sy.strip(),
        "--size_z",
        sz.strip(),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "1",
        "--out",
        str(out_pdbqt),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
        if result.returncode != 0:
            logger.debug("Docking failed (rc={}): {}", result.returncode, ligand_pdbqt.name)
            return None
        return parse_vina_score(out_pdbqt)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("Docking error: {}", exc)
        return None


def _parse_sdf_energy(text: str) -> list[float]:
    """Extract ENERGY scores from a UniDock SDF output."""
    scores: list[float] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith("> <Uni-Dock RESULT>"):
            continue
        if idx + 1 >= len(lines):
            continue
        payload = lines[idx + 1].strip()
        for token in payload.replace(";", " ").split():
            if token.startswith("ENERGY="):
                try:
                    scores.append(float(token.split("=", 1)[1]))
                except ValueError:
                    continue
    return scores


def _run_gpu_sub_batch(  # noqa: PLR0913
    chunk: list[Path],
    receptor_pdbqt: Path,
    cx: str,
    cy: str,
    cz: str,
    sx: str,
    sy: str,
    sz: str,
    output_dir: Path,
    vina_program: str,
    exhaustiveness: int,
) -> None:
    """Run one GPU sub-batch of UniDock, with CPU fallback on failure.

    On timeout, partial results already written to *output_dir* are kept.
    """
    timeout = max(_TIMEOUT_MIN, len(chunk) * _TIMEOUT_PER_LIGAND)
    cmd = [
        vina_program,
        "--receptor",
        str(receptor_pdbqt),
    ]
    for lig in chunk:
        cmd += ["--gpu_batch", str(lig)]
    cmd += [
        "--center_x",
        cx,
        "--center_y",
        cy,
        "--center_z",
        cz,
        "--size_x",
        sx,
        "--size_y",
        sy,
        "--size_z",
        sz,
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "1",
        "--dir",
        str(output_dir),
        "--verbosity",
        "0",
    ]

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "GPU sub-batch failed (rc={}, n={}), falling back to CPU",
                result.returncode,
                len(chunk),
            )
            _run_cpu_fallback(
                chunk,
                receptor_pdbqt,
                cx,
                cy,
                cz,
                sx,
                sy,
                sz,
                output_dir,
                vina_program,
                exhaustiveness,
            )
    except subprocess.TimeoutExpired:
        logger.warning("GPU sub-batch timed out ({:.0f}s, n={}), CPU fallback for missing", timeout, len(chunk))
        _run_cpu_fallback(
            chunk,
            receptor_pdbqt,
            cx,
            cy,
            cz,
            sx,
            sy,
            sz,
            output_dir,
            vina_program,
            exhaustiveness,
        )
    except OSError as exc:
        logger.warning("GPU sub-batch OS error: {}, CPU fallback", exc)
        _run_cpu_fallback(
            chunk,
            receptor_pdbqt,
            cx,
            cy,
            cz,
            sx,
            sy,
            sz,
            output_dir,
            vina_program,
            exhaustiveness,
        )


def _run_cpu_fallback(  # noqa: PLR0913
    chunk: list[Path],
    receptor_pdbqt: Path,
    cx: str,
    cy: str,
    cz: str,
    sx: str,
    sy: str,
    sz: str,
    output_dir: Path,
    vina_program: str,
    exhaustiveness: int,
) -> None:
    """Dock ligands one-by-one via CPU (``--ligand``) as last resort."""
    per_lig_timeout = max(30.0, _TIMEOUT_PER_LIGAND * 3)
    for lig in chunk:
        out_sdf = output_dir / f"{lig.stem}_out.sdf"
        out_pdbqt = output_dir / f"{lig.stem}_out.pdbqt"
        if out_sdf.exists() or out_pdbqt.exists():
            continue  # already have result from partial GPU run
        cmd = [
            vina_program,
            "--receptor",
            str(receptor_pdbqt),
            "--ligand",
            str(lig),
            "--center_x",
            cx,
            "--center_y",
            cy,
            "--center_z",
            cz,
            "--size_x",
            sx,
            "--size_y",
            sy,
            "--size_z",
            sz,
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            "1",
            "--out",
            str(out_pdbqt),
        ]
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=per_lig_timeout,
                check=False,
            )


def dock_gpu_batch(
    ligand_pdbqts: list[Path],
    receptor_pdbqt: Path,
    center: str,
    size: str,
    output_dir: Path,
    vina_program: str = "unidock",
    exhaustiveness: int = 8,
) -> dict[Path, float | None]:
    """Dock a batch of ligands via UniDock ``--gpu_batch`` on GPU.

    Splits into sub-batches of ``_GPU_SUB_BATCH`` to avoid timeouts
    and VRAM exhaustion.  Falls back to single-ligand CPU docking
    for failed sub-batches.

    Args:
        ligand_pdbqts: Ligand PDBQT files.
        receptor_pdbqt: Receptor PDBQT file.
        center: Comma-separated ``"cx,cy,cz"`` center coordinates.
        size: Comma-separated ``"sx,sy,sz"`` box dimensions.
        output_dir: Directory for output files.
        vina_program: Docking binary name/path.
        exhaustiveness: Docking exhaustiveness.

    Returns:
        Mapping from ligand path to best score (or ``None``).
    """
    cx, cy, cz = center.split(",")
    sx, sy, sz = size.split(",")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CPU-only mode: dock each ligand individually (--ligand).
    # GPU batch (--gpu_batch) hangs on complex REINVENT4 molecules,
    # making CPU-only faster overall (no timeout waits).
    logger.debug("Docking {} ligands via CPU (--ligand each)", len(ligand_pdbqts))
    _run_cpu_fallback(
        ligand_pdbqts,
        receptor_pdbqt,
        cx.strip(),
        cy.strip(),
        cz.strip(),
        sx.strip(),
        sy.strip(),
        sz.strip(),
        output_dir,
        vina_program,
        exhaustiveness,
    )

    # Parse results
    scores: dict[Path, float | None] = {}
    for lig in ligand_pdbqts:
        sdf_path = output_dir / f"{lig.stem}_out.sdf"
        pdbqt_out = output_dir / f"{lig.stem}_out.pdbqt"
        if sdf_path.exists():
            energy_list = _parse_sdf_energy(sdf_path.read_text())
            if energy_list:
                best = min(energy_list)
                scores[lig] = best if SCORE_MIN <= best <= SCORE_MAX else None
            else:
                scores[lig] = None
            sdf_path.unlink(missing_ok=True)
        elif pdbqt_out.exists():
            scores[lig] = parse_vina_score(pdbqt_out)
            pdbqt_out.unlink(missing_ok=True)
        else:
            scores[lig] = None
    return scores


def parse_vina_score(pdbqt_path: Path) -> float | None:
    """Parse best docking score from Vina/UniDock output PDBQT.

    Returns:
        Score (kcal/mol) if valid within ``[SCORE_MIN, SCORE_MAX]``, else ``None``.
    """
    try:
        for line in pdbqt_path.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                score = float(parts[3])
                if SCORE_MIN <= score <= SCORE_MAX:
                    return score
                logger.debug("Score out of range: {} (file={})", score, pdbqt_path.name)
                return None
    except (OSError, IndexError, ValueError):
        pass
    return None


def capped_zscore(raw: float, mean: float, std: float) -> float:
    """Compute z-score clamped to ``[-MAX_Z_ABS, MAX_Z_ABS]``.

    Args:
        raw: Raw docking score.
        mean: Calibration mean.
        std: Calibration standard deviation.

    Returns:
        Clamped z-score.
    """
    if std < 1e-12:
        return 0.0
    z = (raw - mean) / std
    return max(-MAX_Z_ABS, min(MAX_Z_ABS, z))


# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockingTarget:
    """Configuration for one docking target."""

    pdbqt_path: Path
    center: str
    size: str
    norm_mean: float
    norm_std: float


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def score_batch(
    smiles_list: list[str],
    target_a: DockingTarget,
    target_b: DockingTarget | None,
    reward_mode: str,
    vina_program: str = "unidock",
) -> np.ndarray:
    """Score a batch of SMILES via GPU batch docking + reward-mode scalarization.

    Uses UniDock ``--gpu_batch`` to dock all ligands at once on GPU,
    falling back to ``--ligand_index`` if GPU batch fails.

    Args:
        smiles_list: SMILES strings to score.
        target_a: Primary docking target.
        target_b: Secondary target (required for dual-docking modes).
        reward_mode: Reward mode name (see ``dualdocking.reward``).
        vina_program: Docking binary.

    Returns:
        1-D float64 array of rewards, shape ``(len(smiles_list),)``.
    """
    from dualdocking.reward import resolve_reward_mode, reward_mode_requires_secondary_target

    mode = resolve_reward_mode(reward_mode)
    needs_target_b = reward_mode_requires_secondary_target(reward_mode)
    rewards = np.full(len(smiles_list), -INVALID_Z_PENALTY, dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="reinvent4_dock_") as tmpdir:
        tmp = Path(tmpdir)

        # --- Phase 1: Prepare all ligand PDBQTs ---
        valid_indices: list[int] = []
        lig_paths: list[Path] = []
        for i, smi in enumerate(smiles_list):
            lig_pdbqt = tmp / f"lig_{i}.pdbqt"
            if smiles_to_pdbqt(smi, lig_pdbqt):
                valid_indices.append(i)
                lig_paths.append(lig_pdbqt)

        if not lig_paths:
            for _ in smiles_list:
                _reward_stats.record(valid_a=False, valid_b=False if needs_target_b else None)
            return rewards

        # --- Phase 2: GPU batch dock against target A ---
        out_a = tmp / "out_a"
        scores_a_map = dock_gpu_batch(
            lig_paths,
            target_a.pdbqt_path,
            target_a.center,
            target_a.size,
            out_a,
            vina_program,
        )

        # --- Phase 3: GPU batch dock against target B (if needed) ---
        scores_b_map: dict[Path, float | None] = {}
        if needs_target_b and target_b is not None:
            out_b = tmp / "out_b"
            scores_b_map = dock_gpu_batch(
                lig_paths,
                target_b.pdbqt_path,
                target_b.center,
                target_b.size,
                out_b,
                vina_program,
            )

        # --- Phase 4: Compute rewards ---
        valid_set = set(valid_indices)
        for i in range(len(smiles_list)):
            if i not in valid_set:
                _reward_stats.record(valid_a=False, valid_b=False if needs_target_b else None)
                continue

            lig_pdbqt = tmp / f"lig_{i}.pdbqt"
            score_a = scores_a_map.get(lig_pdbqt)
            if score_a is None:
                _reward_stats.record(valid_a=False, valid_b=False if needs_target_b else None)
                continue
            z_a = capped_zscore(score_a, target_a.norm_mean, target_a.norm_std)

            z_b: float | None = None
            if needs_target_b and target_b is not None:
                score_b = scores_b_map.get(lig_pdbqt)
                if score_b is None:
                    _reward_stats.record(valid_a=True, valid_b=False)
                    continue
                z_b = capped_zscore(score_b, target_b.norm_mean, target_b.norm_std)

            _reward_stats.record(valid_a=True, valid_b=True if needs_target_b else None)
            rewards[i] = -mode.evaluate(z_a=z_a, z_b=z_b)

    return rewards


# ---------------------------------------------------------------------------
# REINVENT4 plugin classes
# ---------------------------------------------------------------------------


@add_tag("__parameters")
@dataclass
class Parameters:
    """REINVENT4 component parameters from TOML ``[*.specific]`` section.

    All fields are lists — REINVENT4 multi-endpoint convention.
    Only the first element of each list is used.
    """

    receptor_a: list[str] = field(default_factory=list)
    center_a: list[str] = field(default_factory=list)
    size_a: list[str] = field(default_factory=list)
    norm_mean_a: list[float] = field(default_factory=list)
    norm_std_a: list[float] = field(default_factory=list)
    receptor_b: list[str] = field(default_factory=list)
    center_b: list[str] = field(default_factory=list)
    size_b: list[str] = field(default_factory=list)
    norm_mean_b: list[float] = field(default_factory=list)
    norm_std_b: list[float] = field(default_factory=list)
    reward_mode: list[str] = field(default_factory=lambda: ["single_a"])
    vina_program: list[str] = field(default_factory=lambda: ["unidock"])


@add_tag("__component")
class DualDockingComponent:
    """REINVENT4 scoring component wrapping UniDock + reward modes.

    Instantiated by REINVENT4's scoring pipeline from TOML parameters.
    Called once per RL batch with a list of SMILES.
    """

    def __init__(self, params: Parameters) -> None:
        self.reward_mode = params.reward_mode[0] if params.reward_mode else "single_a"
        self.vina_program = params.vina_program[0] if params.vina_program else "unidock"

        if not params.receptor_a:
            msg = "DualDockingComponent requires receptor_a"
            raise ValueError(msg)

        self.target_a = DockingTarget(
            pdbqt_path=Path(params.receptor_a[0]),
            center=params.center_a[0],
            size=params.size_a[0],
            norm_mean=params.norm_mean_a[0] if params.norm_mean_a else 0.0,
            norm_std=params.norm_std_a[0] if params.norm_std_a else 1.0,
        )

        self.target_b: DockingTarget | None = None
        if params.receptor_b and params.receptor_b[0]:
            self.target_b = DockingTarget(
                pdbqt_path=Path(params.receptor_b[0]),
                center=params.center_b[0],
                size=params.size_b[0],
                norm_mean=params.norm_mean_b[0] if params.norm_mean_b else 0.0,
                norm_std=params.norm_std_b[0] if params.norm_std_b else 1.0,
            )

        logger.info(
            "DualDockingComponent: mode={}, target_a={}, target_b={}",
            self.reward_mode,
            self.target_a.pdbqt_path.name,
            self.target_b.pdbqt_path.name if self.target_b else "none",
        )

    def __call__(self, smilies: list[str]) -> ComponentResults:
        """Score a batch of SMILES (called by REINVENT4's RL loop).

        Args:
            smilies: SMILES strings generated by the agent.

        Returns:
            ComponentResults with one endpoint score array.
        """
        scores = score_batch(
            smilies,
            self.target_a,
            self.target_b,
            self.reward_mode,
            self.vina_program,
        )
        return ComponentResults(scores=[scores])
