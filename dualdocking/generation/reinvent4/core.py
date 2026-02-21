"""REINVENT4 SMILES-based RL generation backend.

REINVENT4 runs natively in Python 3.12 (no Docker needed for training).
Receptor PDBQT preparation reuses the FREED++ Docker image as a one-time
step; the trained agent and SMILES sampling are fully native.

References:
    - REINVENT4: https://github.com/MolecularAI/REINVENT4
    - Loeffler et al., J Cheminformatics 2024, PMID 38383444
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

from dualdocking.generation.freedpp import io as _io_helpers
from dualdocking.generation.reinvent4.config import Reinvent4TOMLConfig, write_toml
from dualdocking.generation.reinvent4.scoring import DockingTarget, _reward_stats
from dualdocking.reward import reward_mode_requires_dual_docking, reward_mode_requires_secondary_target
from dualdocking.scoring.binding_box import extract_binding_box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "output" / "data" / "targets"
CACHE_DIR: Path = PROJECT_ROOT / "output" / "cache" / "reinvent4_models"
CACHE_KEY_VERSION: str = "v1_reinvent4_20260219"

# Docker image used only for receptor PDB -> PDBQT conversion.
_DEFAULT_DOCKER_IMAGE: str = "dualdocking/freedpp:latest"
_DOCKER_BIN: str = shutil.which("docker") or "docker"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Reinvent4GenerationConfig:
    """Parameters for a single REINVENT4 generation run.

    Attributes:
        target_id: Target identifier for receptor A.
        n_molecules: Number of molecules to extract from scaffold memory.
        seed: Random seed.
        output_dir: Run output directory.
        reward_mode: Reward strategy (same modes as FREED++).
        second_target_id: Target identifier for receptor B (dual modes).
        algorithm: RL learning strategy (``"dap"`` recommended).
        generator_type: REINVENT4 model architecture label (``"reinvent"``,
            ``"libinvent"``, ``"linkinvent"``, ``"mol2mol"``).  Currently
            used only in the cache key; the actual architecture is determined
            by the prior model checkpoint, not by this field.
        batch_size: Molecules sampled per RL step.
        n_steps: Total RL steps.
        use_cache: Whether to cache / reuse trained agent.
        use_diversity_filter: Enable scaffold diversity filter.
        prior_model: Path to prior model (empty = ChEMBL prior).
        box_padding: Binding-box padding in Angstroms.
        vina_program: Docking binary (``"unidock"``).
        training_timeout: Max wall-clock seconds for training.
    """

    target_id: str
    n_molecules: int = 1000
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("output/results/smoke"))
    reward_mode: str = "single_a"
    second_target_id: str | None = None

    # REINVENT4-specific
    algorithm: str = "dap"
    generator_type: str = "reinvent"
    batch_size: int = 128
    n_steps: int = 500
    use_cache: bool = True
    use_diversity_filter: bool = True
    prior_model: str = ""
    box_padding: float = 10.0
    vina_program: str = "unidock"
    training_timeout: int = 86400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(config: Reinvent4GenerationConfig) -> str:
    """Compute a SHA-256 cache key for the trained agent model."""
    raw = "|".join(
        [
            CACHE_KEY_VERSION,
            config.target_id,
            config.second_target_id or "",
            config.reward_mode,
            config.algorithm,
            config.generator_type,
            str(config.batch_size),
            str(config.n_steps),
            str(config.seed),
            config.vina_program,
            config.prior_model,
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _find_cached_model(key: str) -> Path | None:
    """Look up a cached trained agent by cache key."""
    candidate = CACHE_DIR / key
    if candidate.is_dir() and any(candidate.iterdir()):
        logger.info("Found cached REINVENT4 agent: {}", key)
        return candidate
    return None


def _resolve_pdb_path(target_id: str) -> Path:
    """Find PDB file for a target in the prepared-targets directory."""
    return _io_helpers.resolve_pdb_path(target_id, data_dir=DATA_DIR)


def _prepare_receptor(
    pdb_path: Path,
    out_dir: Path,
    padding: float = 10.0,
) -> tuple[Path, str, str]:
    """Convert receptor PDB to PDBQT and extract binding box.

    Uses Docker-based conversion via the FREED++ image (one-time step).
    The resulting PDBQT is cached in *out_dir*.
    """
    return _io_helpers.prepare_receptor(
        pdb_path,
        out_dir,
        image=_DEFAULT_DOCKER_IMAGE,
        docker_bin=_DOCKER_BIN,
        padding=padding,
        run_subprocess=subprocess.run,
        extract_binding_box=lambda path, pad: extract_binding_box(path, padding=pad),
        log=logger,
    )


def _load_normalization(target_id: str) -> tuple[float, float]:
    """Load calibration mean/std for *target_id* from normalization.yaml.

    Returns:
        ``(mean, std)`` tuple.

    Raises:
        FileNotFoundError: Config file missing.
        KeyError: Target not in calibration data.
    """
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization.yaml not found: {norm_path}")

    with norm_path.open() as fh:
        data = yaml.safe_load(fh)

    targets = data.get("targets", {})
    if target_id not in targets:
        msg = f"Target '{target_id}' not in normalization.yaml"
        raise KeyError(msg)

    entry = targets[target_id]
    return float(entry["mean"]), float(entry["std"])


def _install_scoring_plugin() -> None:
    """Verify that the DualDocking plugin is discoverable by REINVENT4.

    The plugin file ``comp_dual_docking.py`` ships as part of the
    ``reinvent_plugins`` namespace package (installed via pyproject.toml).
    This function verifies it can be imported.
    """
    try:
        import reinvent_plugins.components.comp_dual_docking  # noqa: F401
    except ImportError:
        raise ImportError(
            "DualDocking plugin not found in reinvent_plugins namespace. Ensure the dualdocking package is installed (uv sync)."
        ) from None


def _parse_output_smiles(workspace: Path, n_molecules: int) -> list[str]:
    """Parse generated SMILES from REINVENT4 output files.

    Looks for ``scaffold_memory*.csv`` in the workspace, then falls back
    to any CSV with a ``SMILES`` or ``smiles`` column.

    Args:
        workspace: REINVENT4 workspace directory.
        n_molecules: Maximum number of molecules to return.

    Returns:
        Deduplicated SMILES list (up to *n_molecules*).
    """
    import pandas as pd

    smiles: list[str] = []
    seen: set[str] = set()

    # Primary: scaffold memory CSV (REINVENT4 default output)
    for csv_path in sorted(workspace.rglob("scaffold_memory*.csv")):
        try:
            df = pd.read_csv(csv_path)
            col = next((c for c in df.columns if c.lower() == "smiles"), None)
            if col:
                for smi in df[col].dropna():
                    smi_str = str(smi).strip()
                    if smi_str and smi_str not in seen:
                        seen.add(smi_str)
                        smiles.append(smi_str)
        except Exception as exc:
            logger.warning("Error reading {}: {}", csv_path, exc)

    # Fallback: any results CSV
    if not smiles:
        for csv_path in sorted(workspace.rglob("*.csv")):
            if "scaffold_memory" in csv_path.name:
                continue
            try:
                df = pd.read_csv(csv_path)
                col = next((c for c in df.columns if c.lower() == "smiles"), None)
                if col:
                    for smi in df[col].dropna():
                        smi_str = str(smi).strip()
                        if smi_str and smi_str not in seen:
                            seen.add(smi_str)
                            smiles.append(smi_str)
            except Exception:  # noqa: S110
                pass

    if not smiles:
        logger.warning("No SMILES found in REINVENT4 output: {}", workspace)

    # Truncate to budget
    if len(smiles) > n_molecules:
        smiles = smiles[:n_molecules]

    logger.info("Parsed {} unique SMILES from REINVENT4 output", len(smiles))
    return smiles


def _resolve_n_steps(config: Reinvent4GenerationConfig) -> int:
    """Resolve RL step count from config, with env override."""
    env_steps = os.environ.get("REINVENT4_N_STEPS")
    if env_steps is not None:
        return int(env_steps)
    return config.n_steps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate(config: Reinvent4GenerationConfig) -> list[str]:
    """Run the full REINVENT4 generation pipeline and return SMILES.

    Pipeline steps:
        1. Check trained-agent cache
        2. Prepare workspace
        3. Load normalization stats
        4. Prepare receptor A (PDB -> PDBQT + binding box)
        5. Prepare receptor B if needed
        6. Install scoring plugin into REINVENT4 plugin dir
        7. Generate TOML config
        8. Run REINVENT4 via CLI
        9. Write reward metadata
        10. Parse output SMILES
        11. Cache trained agent
        12. Write output (Parquet + CSV)

    Args:
        config: REINVENT4 generation configuration.

    Returns:
        List of generated SMILES strings.

    Raises:
        ValueError: If dual mode requested but ``second_target_id`` is unset.
        FileNotFoundError: If normalization.yaml is missing.
    """
    needs_second = reward_mode_requires_secondary_target(config.reward_mode)
    if needs_second and not config.second_target_id:
        raise ValueError(f"{config.reward_mode} mode requires second_target_id")

    n_steps_total = 10 if not needs_second else 11
    pbar = tqdm(
        total=n_steps_total,
        desc=f"REINVENT4 [{config.target_id}]",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    # 1. Cache check
    pbar.set_postfix_str("Cache check")
    key = _cache_key(config)
    cached_agent = _find_cached_model(key) if config.use_cache else None
    pbar.update(1)

    # 2. Workspace
    pbar.set_postfix_str("Preparing workspace")
    workspace = config.output_dir / "reinvent4_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    pbar.update(1)

    # 3. Normalization stats
    pbar.set_postfix_str("Loading normalization")
    mean_a, std_a = _load_normalization(config.target_id)
    mean_b, std_b = (0.0, 1.0)
    if needs_second and config.second_target_id:
        mean_b, std_b = _load_normalization(config.second_target_id)
    pbar.update(1)

    # 4. Receptor A
    pbar.set_postfix_str(f"Receptor A ({config.target_id})")
    pdb_a = _resolve_pdb_path(config.target_id)
    pdbqt_a, center_a, size_a = _prepare_receptor(pdb_a, workspace, config.box_padding)
    target_a = DockingTarget(pdbqt_path=pdbqt_a, center=center_a, size=size_a, norm_mean=mean_a, norm_std=std_a)
    pbar.update(1)

    # 5. Receptor B (if dual mode)
    target_b: DockingTarget | None = None
    if needs_second and config.second_target_id:
        pbar.set_postfix_str(f"Receptor B ({config.second_target_id})")
        pdb_b = _resolve_pdb_path(config.second_target_id)
        pdbqt_b, center_b, size_b = _prepare_receptor(pdb_b, workspace, config.box_padding)
        target_b = DockingTarget(pdbqt_path=pdbqt_b, center=center_b, size=size_b, norm_mean=mean_b, norm_std=std_b)
        pbar.update(1)

    # 6. Install scoring plugin
    pbar.set_postfix_str("Installing plugin")
    _install_scoring_plugin()
    pbar.update(1)

    # 7. TOML config
    pbar.set_postfix_str("Generating config")
    n_steps = _resolve_n_steps(config)

    agent_model = ""
    if cached_agent is not None:
        ckpt_files = sorted(cached_agent.glob("*.ckpt"))
        if ckpt_files:
            agent_model = str(ckpt_files[-1])
            logger.info("Using cached agent: {}", agent_model)

    toml_cfg = Reinvent4TOMLConfig(
        output_dir=workspace,
        prior_model=config.prior_model,
        agent_model=agent_model,
        algorithm=config.algorithm,
        batch_size=config.batch_size,
        n_steps=n_steps,
        seed=config.seed,
        use_diversity_filter=config.use_diversity_filter,
        target_a=target_a,
        target_b=target_b,
        reward_mode=config.reward_mode,
        vina_program=config.vina_program,
    )
    toml_path = write_toml(toml_cfg, workspace / "reinvent4_config.toml")
    pbar.update(1)

    # 8. Run REINVENT4
    pbar.set_postfix_str("REINVENT4 training")
    _reward_stats.reset()
    reinvent_bin = shutil.which("reinvent") or "reinvent"
    try:
        subprocess.run(  # noqa: S603
            [reinvent_bin, str(toml_path)],
            check=True,
            timeout=config.training_timeout,
        )
    except FileNotFoundError:
        msg = "REINVENT4 CLI not found. Install with: uv add 'reinvent>=4.7' or check that 'reinvent' is on PATH."
        raise FileNotFoundError(msg) from None
    pbar.update(1)

    # 9. Write reward metadata
    pbar.set_postfix_str("Writing reward metadata")
    if reward_mode_requires_dual_docking(config.reward_mode):
        primary_target = "both"
    elif needs_second:
        primary_target = config.second_target_id or config.target_id
    else:
        primary_target = config.target_id
    _reward_stats.write_metadata(
        workspace / "reward_metadata.json",
        reward_mode=config.reward_mode,
        primary_target=primary_target,
    )
    pbar.update(1)

    # 10. Parse output
    pbar.set_postfix_str("Parsing SMILES")
    smiles = _parse_output_smiles(workspace, config.n_molecules)
    pbar.update(1)

    # 11. Cache agent
    pbar.set_postfix_str("Caching agent")
    if config.use_cache and cached_agent is None:
        ckpt_files = sorted(workspace.rglob("*.ckpt"))
        if ckpt_files:
            cache_dest = CACHE_DIR / key
            cache_dest.mkdir(parents=True, exist_ok=True)
            for ckpt in ckpt_files:
                shutil.copy2(ckpt, cache_dest / ckpt.name)
            logger.info("Agent cached: {}", cache_dest)
    pbar.update(1)

    # 12. Write output
    pbar.set_postfix_str("Writing results")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    from dualdocking.data.io_utils import write_df

    generated_df = pd.DataFrame({"smiles": smiles})
    write_df(generated_df, config.output_dir / "generated", csv_too=True)
    logger.info("Generated {} molecules -> {}", len(smiles), config.output_dir / "generated.parquet")
    pbar.close()

    return smiles
