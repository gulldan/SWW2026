"""Docker wrapper for the FREED++ molecular generator.

FREED++ requires Python 3.7 / PyTorch 1.12 / DGL 0.9, which is incompatible
with this project (Python >=3.11), so it runs in an isolated Docker container.

Training reward uses UniDock GPU (``unidock_gpu``, GPU-accelerated docking).
Final scoring uses Boltz-2 (``scorer_boltz2.py``, ~58 sec/mol, GPU).

References:
    - FREED++ paper: https://arxiv.org/abs/2401.09840
    - Original FREED: https://arxiv.org/abs/2110.01219
    - Repository: https://github.com/AIRI-Institute/FFREED
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dualdocking.command_runner import run_project_command
from dualdocking.generation import freedpp_docker as _docker_helpers
from dualdocking.generation import freedpp_io as _io_helpers
from dualdocking.generation import freedpp_model as _model_helpers
from dualdocking.reward import reward_mode_requires_secondary_target
from dualdocking.scoring.binding_box import extract_binding_box
from dualdocking.utils.hardware import detect_primary_gpu_vram_gb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "output" / "data" / "targets"
CACHE_DIR: Path = PROJECT_ROOT / "output" / "cache" / "freedpp_models"

DEFAULT_DOCKER_IMAGE: str = "dualdocking/freedpp:latest"
DOCKER_BIN: str = shutil.which("docker") or "docker"
CACHE_KEY_VERSION: str = "v3_clean_single_b_20260207"
FREED_DATA_DIR: Path = PROJECT_ROOT / "docker" / "freedpp" / "data"
FREED_DATA_FILES: tuple[Path, ...] = (
    FREED_DATA_DIR / "zinc_crem.json",
    FREED_DATA_DIR / "alert_collections.csv",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Parameters for a single FREED++ generation run.

    Attributes:
        target_id: ChEMBL or internal target identifier for receptor A.
        n_molecules: Number of molecules to sample after training.
        seed: Random seed for reproducibility.
        output_dir: Directory for output files and workspace.
        reward_mode: Reward strategy -- ``"single_a"`` (dock against A only),
            ``"single_b"`` (dock against B only), ``"double"``
            (dock against both, worst-case), ``"linear_equal"`` (z_a + z_b),
            ``"biased_a"`` (z_a + 0.3*z_b), ``"selective_a"`` (z_a - z_b),
            or ``"selective_weighted"`` (z_a - lambda_off * z_b).
        second_target_id: Target identifier for receptor B. Required when
            *reward_mode* is ``"single_b"`` or ``"double"``.
        epochs: Number of FREED++ training epochs.
        num_sub_proc: Number of concurrent docking sub-processes. 0 means
            auto-detect based on GPU VRAM.
        starting_smile: SMARTS-like seed scaffold for fragment-based growth.
        use_cache: Whether to use / populate the trained-model cache.
        docker_image: Docker image tag for the FREED++ container.
        training_timeout: Maximum wall-clock seconds for Docker training.
        box_padding: Padding in Angstroms around the binding box.
        vina_program: Docking program name (``"qvina02"``, ``"unidock"``,
            or ``"unidock_gpu"``).
        steps_per_epoch: RL steps per training epoch.
        qg_max_invalid_fraction: Quality-gate threshold for invalid SMILES
            fraction per epoch.
        qg_max_timeout_fraction: Quality-gate threshold for docking timeout
            fraction per epoch.
        pass_reward_modes_to_container: Whether to export
            ``DUALDOCKING_REWARD_MODES_PATH`` into the container environment.
            Enable for models that should reuse host/project reward-mode
            declarations; disable for models with self-contained reward logic.
        reward_modes_path_in_container: Container path for
            ``reward_modes.yaml`` when passthrough is enabled.
    """

    target_id: str
    n_molecules: int = 1000
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("output/results/smoke"))
    reward_mode: str = "single_a"  # See dualdocking.reward.selectivity for available modes.
    second_target_id: str | None = None

    # FREED++-specific
    epochs: int = 100
    num_sub_proc: int = 0  # 0 = auto-detect based on GPU VRAM
    starting_smile: str = "c1([*:1])c([*:2])ccc([*:3])c1"
    use_cache: bool = True
    docker_image: str = DEFAULT_DOCKER_IMAGE
    training_timeout: int = 86400  # 24 h
    box_padding: float = 10.0  # Angstroms
    vina_program: str = "unidock_gpu"  # "qvina02" | "unidock" | "unidock_gpu"
    steps_per_epoch: int = 256
    qg_max_invalid_fraction: float = 0.10
    qg_max_timeout_fraction: float = 0.05
    pass_reward_modes_to_container: bool = True
    reward_modes_path_in_container: str = "/opt/reward_modes.yaml"


# ---------------------------------------------------------------------------
# Docking program resolution (vina_program from FREED++ API)
# ---------------------------------------------------------------------------

_VINA_PATHS: dict[str, str] = {
    "qvina02": "/opt/freed/ffreed/env/qvina02",
    "unidock": "/usr/local/bin/unidock",
    # In the container Uni-Dock GPU is installed as `unidock`.
    # We keep the `unidock_gpu` key for metadata consistency.
    "unidock_gpu": "/usr/local/bin/unidock",
}


# ---------------------------------------------------------------------------
# Wrappers around extracted helper modules (backward-compatible API)
# ---------------------------------------------------------------------------


def _auto_num_sub_proc() -> int:
    """Pick ``num_sub_proc`` based on GPU VRAM."""
    return _model_helpers.auto_num_sub_proc(
        env_num_sub_proc=os.getenv("FREEDPP_NUM_SUB_PROC"),
        detect_vram_gb=detect_primary_gpu_vram_gb,
        log=logger,
    )


def _ensure_docker_image(image: str = DEFAULT_DOCKER_IMAGE) -> None:
    """Ensure the FREED++ Docker image exists, building it if necessary."""
    _docker_helpers.ensure_docker_image(
        image,
        docker_bin=DOCKER_BIN,
        project_root=PROJECT_ROOT,
        freed_data_files=FREED_DATA_FILES,
        run_subprocess=subprocess.run,
        run_project_command=run_project_command,
        log=logger,
    )


def _cache_key(config: GenerationConfig) -> str:
    """Compute a SHA256-based cache key from generation parameters."""
    return _model_helpers.cache_key(config, key_version=CACHE_KEY_VERSION)


def _find_cached_model(key: str) -> Path | None:
    """Look up a cached trained model by its cache key."""
    return _model_helpers.find_cached_model(key, cache_dir=CACHE_DIR, log=logger)


def _resolve_pdb_path(target_id: str) -> Path:
    """Find the PDB file for a target in the prepared targets directory."""
    return _io_helpers.resolve_pdb_path(target_id, data_dir=DATA_DIR)


def _prepare_receptor(
    pdb_path: Path,
    out_dir: Path,
    image: str = DEFAULT_DOCKER_IMAGE,
    padding: float = 10.0,
) -> tuple[Path, str, str]:
    """Convert a PDB file to PDBQT and compute the binding box."""
    return _io_helpers.prepare_receptor(
        pdb_path,
        out_dir,
        image=image,
        docker_bin=DOCKER_BIN,
        padding=padding,
        run_subprocess=subprocess.run,
        extract_binding_box=lambda path, pad: extract_binding_box(path, padding=pad),
        log=logger,
    )


def _resolve_vina_path(program: str) -> str:
    """Map a ``vina_program`` name to its container-internal binary path."""
    return _model_helpers.resolve_vina_path(program, vina_paths=_VINA_PATHS)


def _build_freedpp_args(
    config: GenerationConfig,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    commands: str = "train,sample",
) -> list[str]:
    """Build the CLI argument list for FREED++ ``main.py``."""
    return _model_helpers.build_freedpp_args(
        config,
        pdbqt_a,
        center_a,
        size_a,
        commands=commands,
        auto_num_sub_proc=_auto_num_sub_proc,
        resolve_vina_path=_resolve_vina_path,
        log=logger,
    )


_RUN_WITH_PATCH: Path = PROJECT_ROOT / "docker" / "freedpp" / "run_with_patch.py"


def _run_freedpp(
    config: GenerationConfig,
    workspace: Path,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    pdbqt_b: Path | None = None,
    center_b: str | None = None,
    size_b: str | None = None,
    commands: str = "train,sample",
    checkpoint: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run FREED++ inside a Docker container.

    Args:
        checkpoint: Container path to ``model_*.pth`` for resuming training.
    """

    def _build_args(
        cfg: GenerationConfig,
        pdbqt: Path,
        center: str,
        size: str,
        cmds: str = "train,sample",
    ) -> list[str]:
        return _model_helpers.build_freedpp_args(
            cfg,
            pdbqt,
            center,
            size,
            commands=cmds,
            auto_num_sub_proc=_auto_num_sub_proc,
            resolve_vina_path=_resolve_vina_path,
            log=logger,
            checkpoint=checkpoint,
        )

    # Mount the local run_with_patch.py so fixes take effect without
    # rebuilding the Docker image.
    extra: list[str] = []
    if _RUN_WITH_PATCH.exists():
        extra = ["-v", f"{_RUN_WITH_PATCH}:/opt/run_with_patch.py:ro"]

    return _docker_helpers.run_freedpp(
        config,
        workspace,
        pdbqt_a,
        center_a,
        size_a,
        pdbqt_b=pdbqt_b,
        center_b=center_b,
        size_b=size_b,
        commands=commands,
        docker_bin=DOCKER_BIN,
        build_freedpp_args=_build_args,
        popen=subprocess.Popen,
        pipe=subprocess.PIPE,
        getenv=os.getenv,
        tqdm_factory=tqdm,
        log=logger,
        extra_docker_args=extra,
    )


def parse_training_log(log_path: Path) -> list[dict[str, int | float]]:
    """Parse a FREED++ training log and extract per-epoch metrics."""
    return _docker_helpers.parse_training_log(log_path)


def _parse_output_smiles(workspace: Path) -> list[str]:
    """Read generated SMILES from FREED++ output files."""
    return _io_helpers.parse_output_smiles(workspace, log=logger)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate(config: GenerationConfig) -> list[str]:
    """Run the full FREED++ generation pipeline and return SMILES.

    Pipeline steps:
        1. Check / build Docker image
        2. Check trained-model cache
        3. Prepare workspace and copy normalization config
        4. Prepare receptor A (PDB -> PDBQT + binding box)
        5. Prepare receptor B if needed (single_b / double mode)
        6. Determine commands (train+sample or sample-only from cache)
        7. Run FREED++ in Docker
        8. Parse output SMILES
        9. Cache trained model
        10. Write output (Parquet + CSV)

    Args:
        config: Generation configuration specifying targets, reward mode,
            training hyperparameters, and output paths.

    Returns:
        List of generated SMILES strings.

    Raises:
        ValueError: If *reward_mode* requires a second target but
            *second_target_id* is not set.
        FileNotFoundError: If normalization.yaml is missing.
    """
    needs_second = reward_mode_requires_secondary_target(config.reward_mode)
    if needs_second and not config.second_target_id:
        raise ValueError(f"{config.reward_mode} mode requires second_target_id")

    n_steps = 8 if not needs_second else 9
    pbar = tqdm(
        total=n_steps,
        desc=f"Generate [{config.target_id}]",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    # 1. Docker image
    pbar.set_postfix_str("Docker image")
    _ensure_docker_image(config.docker_image)
    pbar.update(1)

    # 2. Cache check
    pbar.set_postfix_str("Cache check")
    key = _cache_key(config)
    cached_model = _find_cached_model(key) if config.use_cache else None
    pbar.update(1)

    # 3. Workspace
    pbar.set_postfix_str("Preparing workspace")
    workspace = config.output_dir / "freedpp_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    pbar.update(1)

    # 3.1. Normalization (needed for reward)
    normalization_src = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not normalization_src.exists():
        raise FileNotFoundError(f"normalization.yaml not found: {normalization_src}")
    normalization_dst = workspace / "normalization.yaml"
    if not normalization_dst.exists():
        shutil.copy2(normalization_src, normalization_dst)
        logger.info("normalization.yaml -> {}", normalization_dst)

    # 4. Prepare receptor A
    pbar.set_postfix_str(f"Receptor A -> PDBQT ({config.target_id})")
    pdb_a = _resolve_pdb_path(config.target_id)
    pdbqt_a, center_a, size_a = _prepare_receptor(
        pdb_a,
        workspace,
        config.docker_image,
        config.box_padding,
    )
    pbar.update(1)

    # 5. Prepare receptor B (single_b / double mode)
    pdbqt_b = center_b = size_b = None
    if needs_second and config.second_target_id:
        pbar.set_postfix_str(f"Receptor B -> PDBQT ({config.second_target_id})")
        pdb_b = _resolve_pdb_path(config.second_target_id)
        pdbqt_b, center_b, size_b = _prepare_receptor(
            pdb_b,
            workspace,
            config.docker_image,
            config.box_padding,
        )
        pbar.update(1)

    # 6. Determine commands + resume logic
    checkpoint: str | None = None
    skip_freedpp = False

    if cached_model is not None:
        model_target = workspace / "output" / "run"
        if not model_target.exists():
            shutil.copytree(cached_model, model_target, dirs_exist_ok=True)
        commands = "sample"
        logger.info("Using cached model ({}) -> sample only", key)
    else:
        commands = "train,sample"
        output_run = workspace / "output" / "run"

        if output_run.exists():
            # Check if previous run already produced valid samples
            mols_dir = output_run / "mols"
            has_samples = mols_dir.exists() and any(mols_dir.glob("sample_*"))

            if has_samples:
                skip_freedpp = True
                logger.info("Previous FREED++ run has valid output — reusing")
            else:
                # Look for training checkpoints to resume from
                checkpoints = sorted(output_run.rglob("model_*.pth"))
                if checkpoints:
                    latest_ckpt = checkpoints[-1]
                    # Convert host path to container path
                    rel = latest_ckpt.relative_to(workspace)
                    checkpoint = f"/workspace/{rel}"
                    logger.info("Resuming training from checkpoint: {}", checkpoint)
                else:
                    # No useful data — clean up and start fresh
                    logger.warning("Found stale output from interrupted run: {}", output_run)
                    shutil.rmtree(output_run, ignore_errors=True)

                    # Fallback: Docker (root-owned files)
                    if output_run.exists():
                        logger.info("Removing root-owned files via Docker...")
                        subprocess.run(  # noqa: S603
                            [
                                DOCKER_BIN,
                                "run",
                                "--rm",
                                "-v",
                                f"{workspace}:/ws",
                                "alpine",
                                "rm",
                                "-rf",
                                "/ws/output/run",
                            ],
                            check=False,
                            timeout=30,
                        )

                    if output_run.exists():
                        raise RuntimeError(
                            f"Failed to remove stale output directory: {output_run}. "
                            "Try removing it manually: "
                            f"sudo rm -rf '{output_run}'"
                        )
                    logger.info("Stale output removed")

    # 7. Run FREED++
    if not skip_freedpp:
        pbar.set_postfix_str(f"FREED++ ({commands})")
        _run_freedpp(
            config,
            workspace,
            pdbqt_a,
            center_a,
            size_a,
            pdbqt_b,
            center_b,
            size_b,
            commands=commands,
            checkpoint=checkpoint,
        )
    else:
        pbar.set_postfix_str("FREED++ (reusing)")
    pbar.update(1)

    # 8. Parse output
    pbar.set_postfix_str("Parsing SMILES")
    smiles = _parse_output_smiles(workspace)
    pbar.update(1)

    # 9. Cache trained model
    pbar.set_postfix_str("Caching model")
    if config.use_cache and commands == "train,sample":
        model_dir = workspace / "output" / "run"
        if model_dir.exists():
            cache_dest = CACHE_DIR / key
            cache_dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(model_dir, cache_dest, dirs_exist_ok=True)
            logger.info("Model saved to cache: {}", cache_dest)
    pbar.update(1)

    # 10. Write output
    pbar.set_postfix_str("Writing results")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    from dualdocking.data.io_utils import write_df

    generated_df = pd.DataFrame({"smiles": smiles})
    write_df(generated_df, config.output_dir / "generated", csv_too=True)
    logger.info("Generated {} molecules -> {}", len(smiles), config.output_dir / "generated.parquet")
    pbar.close()

    return smiles
