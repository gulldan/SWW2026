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

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dualdocking.command_runner import run_project_command
from dualdocking.scoring.binding_box import extract_binding_box

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
            ``"single_b"`` (dock against B only), or ``"double"``
            (dock against both A and B).
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
    """

    target_id: str
    n_molecules: int = 1000
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("output/results/smoke"))
    reward_mode: str = "single_a"  # "single_a" | "single_b" | "double"
    second_target_id: str | None = None

    # FREED++-specific
    epochs: int = 200
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


# ---------------------------------------------------------------------------
# Docker image management
# ---------------------------------------------------------------------------


def _auto_num_sub_proc() -> int:
    """Pick ``num_sub_proc`` based on GPU VRAM.

    Can be overridden via the ``FREEDPP_NUM_SUB_PROC`` environment variable.

    VRAM tiers (concurrent UniDock GPU processes)::

        >=80 GB (H100/A100-80)  -> 24
         >=40 GB (A6000 48 GB)  -> 12
         <=24 GB (RTX 4090)     ->  8

    Default (detection failure) is 8 (safe for any consumer GPU).

    Returns:
        Number of docking sub-processes to use.
    """
    env = os.getenv("FREEDPP_NUM_SUB_PROC")
    if env is not None:
        return int(env)

    vram_gb: float | None = None

    # Attempt 1: torch.cuda (fast, in-process)
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:  # noqa: S110
        pass

    # Attempt 2: nvidia-smi (works even without CUDA-enabled torch)
    if vram_gb is None:
        try:
            result = subprocess.run(
                ["/usr/bin/nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_gb = float(result.stdout.strip().split("\n")[0]) / 1024
        except Exception:  # noqa: S110
            pass

    if vram_gb is None:
        logger.debug("GPU VRAM detection failed for num_sub_proc, using default=8")
        return 8

    logger.debug("GPU VRAM detected: {:.1f} GB", vram_gb)
    if vram_gb >= 76:  # H100/H800 80 GB / A100 80 GB (nvidia-smi reports ~79.x GiB)
        return 24
    if vram_gb >= 38:  # A6000 48 GB / A100 40 GB
        return 12
    return 8  # RTX 4090 24 GB and smaller


def _ensure_docker_image(image: str = DEFAULT_DOCKER_IMAGE) -> None:
    """Ensure the FREED++ Docker image exists, building it if necessary.

    Args:
        image: Docker image tag to check/build.

    Raises:
        RuntimeError: If ``docker inspect`` fails for reasons other than
            a missing image, or if the build/fetch script fails.
        FileNotFoundError: If the Dockerfile is not found.
    """
    result = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        logger.debug("Docker-образ {} найден", image)
        return

    stderr = result.stderr or ""
    # Distinguish "image not found" from Docker daemon errors
    if "No such image" not in stderr and "not found" not in stderr.lower():
        raise RuntimeError(f"Docker inspect failed (rc={result.returncode}): {stderr.strip()}")

    logger.info("Docker-образ {} не найден, собираю...", image)
    dockerfile_dir = PROJECT_ROOT / "docker" / "freedpp"
    dockerfile_path = dockerfile_dir / "Dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile не найден: {dockerfile_path}")

    missing_files = [p for p in FREED_DATA_FILES if not p.exists()]
    if missing_files:
        logger.info("Отсутствуют FREED data files, скачиваю: {}", ", ".join(str(p) for p in missing_files))
        result = run_project_command(
            ["python", "scripts/s0_fetch/fetch_freed_fragments.py"],
            project_root=PROJECT_ROOT,
            capture_output=True,
            allow_inprocess=True,
        )
        if result.returncode != 0:
            tail = ((result.stdout or "") + (result.stderr or ""))[-2000:]
            raise RuntimeError(f"Не удалось скачать FREED data files:\n{tail}")

    subprocess.run(  # noqa: S603
        [
            DOCKER_BIN,
            "build",
            "-t",
            image,
            "-f",
            str(dockerfile_path),
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    logger.info("Docker-образ {} собран", image)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(config: GenerationConfig) -> str:
    """Compute a SHA256-based cache key from generation parameters.

    The key captures all parameters that affect the trained model, so
    that cache lookups are invalidated when any relevant setting changes.

    Args:
        config: Generation configuration.

    Returns:
        Hex-encoded cache key (first 16 characters of SHA256).
    """
    parts = [
        CACHE_KEY_VERSION,
        config.target_id,
        config.second_target_id or "",
        config.reward_mode,
        str(config.epochs),
        str(config.steps_per_epoch),
        str(config.num_sub_proc),
        str(config.seed),
        config.vina_program,
        config.starting_smile,
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _find_cached_model(key: str) -> Path | None:
    """Look up a cached trained model by its cache key.

    Args:
        key: Cache key as returned by :func:`_cache_key`.

    Returns:
        Path to the cached model directory, or ``None`` if not found.
    """
    model_dir = CACHE_DIR / key
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info("Найден кеш модели: {}", model_dir)
        return model_dir
    return None


# ---------------------------------------------------------------------------
# PDB resolution
# ---------------------------------------------------------------------------


def _resolve_pdb_path(target_id: str) -> Path:
    """Find the PDB file for a target in the prepared targets directory.

    Searches ``output/data/targets/*/A/`` and ``output/data/targets/*/B/``,
    matching *target_id* against the ``id`` field in each ``meta.yaml``.

    Args:
        target_id: Target identifier to look up.

    Returns:
        Path to the first ``.pdb`` file found for the target.

    Raises:
        FileNotFoundError: If no PDB file matches *target_id*.
    """
    import yaml

    for pair_dir in DATA_DIR.iterdir():
        if not pair_dir.is_dir():
            continue
        for role in ("A", "B"):
            role_dir = pair_dir / role
            meta_path = role_dir / "meta.yaml"
            if not meta_path.exists():
                continue
            with meta_path.open() as f:
                meta = yaml.safe_load(f)
            if meta.get("id") == target_id:
                pdb_files = list(role_dir.glob("*.pdb"))
                if pdb_files:
                    return pdb_files[0]

    raise FileNotFoundError(f"PDB файл для target '{target_id}' не найден в {DATA_DIR}")


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------


def _prepare_receptor(
    pdb_path: Path,
    out_dir: Path,
    image: str = DEFAULT_DOCKER_IMAGE,
    padding: float = 10.0,
) -> tuple[Path, str, str]:
    """Convert a PDB file to PDBQT and compute the binding box.

    Uses the FREED++ Docker container's ``prepare-receptor`` entrypoint
    to perform the conversion. Binding box is computed via
    :func:`~dualdocking.scoring.binding_box.extract_binding_box`.

    Args:
        pdb_path: Path to the input PDB file.
        out_dir: Directory for the output PDBQT file.
        image: Docker image tag with receptor preparation tools.
        padding: Binding box padding in Angstroms.

    Returns:
        Tuple of ``(pdbqt_path, center_str, size_str)`` where the center
        and size are comma-separated coordinate strings.
    """
    pdbqt_path = out_dir / (pdb_path.stem + ".pdbqt")
    if not pdbqt_path.exists():
        logger.info("Конвертация PDB → PDBQT: {} → {}", pdb_path, pdbqt_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            [
                DOCKER_BIN,
                "run",
                "--rm",
                "-v",
                f"{pdb_path.parent}:/input:ro",
                "-v",
                f"{out_dir}:/output",
                image,
                "prepare-receptor",
                f"/input/{pdb_path.name}",
                f"/output/{pdbqt_path.name}",
            ],
            check=True,
            timeout=60,
        )
    else:
        logger.debug("PDBQT уже существует: {}", pdbqt_path)

    center_str, size_str = extract_binding_box(pdb_path, padding=padding)
    logger.info("Binding box: center={}, size={}", center_str, size_str)
    return pdbqt_path, center_str, size_str


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


def _resolve_vina_path(program: str) -> str:
    """Map a ``vina_program`` name to its container-internal binary path.

    Args:
        program: Program name (e.g. ``"unidock_gpu"``) or an explicit
            absolute path.

    Returns:
        Absolute path to the docking binary inside the Docker container.
    """
    if program in _VINA_PATHS:
        return _VINA_PATHS[program]
    # Treat as an explicit path (e.g. user-provided absolute path)
    return program


# ---------------------------------------------------------------------------
# FREED++ CLI args builder
# ---------------------------------------------------------------------------


def _build_freedpp_args(
    config: GenerationConfig,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    commands: str = "train,sample",
) -> list[str]:
    """Build the CLI argument list for FREED++ ``main.py``.

    Args:
        config: Generation configuration.
        pdbqt_a: Path to the primary receptor PDBQT file.
        center_a: Comma-separated binding box center coordinates.
        size_a: Comma-separated binding box size dimensions.
        commands: Comma-separated FREED++ commands (e.g. ``"train,sample"``).

    Returns:
        List of CLI arguments for the FREED++ container entrypoint.
    """
    num_sub_proc = config.num_sub_proc if config.num_sub_proc > 0 else _auto_num_sub_proc()
    logger.debug("FREED++ num_sub_proc={} (config={}, auto={})", num_sub_proc, config.num_sub_proc, config.num_sub_proc == 0)
    return [
        "--commands",
        commands,
        "--receptor",
        f"/workspace/{pdbqt_a.name}",
        "--box_center",
        center_a,
        "--box_size",
        size_a,
        "--fragments",
        "/opt/freed/data/zinc_crem.json",
        "--alert_collections",
        "/opt/freed/data/alert_collections.csv",
        "--vina_program",
        _resolve_vina_path(config.vina_program),
        "--starting_smile",
        config.starting_smile,
        "--exp_root",
        "/workspace/output",
        "--name",
        "run",
        "--epochs",
        str(config.epochs),
        "--num_sub_proc",
        str(num_sub_proc),
        "--seed",
        str(config.seed),
        "--num_mols",
        str(config.n_molecules),
        "--steps_per_epoch",
        str(config.steps_per_epoch),
        "--reward_version",
        "soft",
        "--num_modes",
        "1",
    ]


# ---------------------------------------------------------------------------
# Docker run
# ---------------------------------------------------------------------------


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
) -> subprocess.CompletedProcess[str]:
    """Run FREED++ inside a Docker container.

    Streams Docker output in real time for progress monitoring. For
    ``"double"`` mode, mounts both receptors and writes a dual config
    JSON to the workspace.

    Args:
        config: Generation configuration.
        workspace: Host directory mounted as ``/workspace`` in the container.
        pdbqt_a: Path to receptor A PDBQT file.
        center_a: Binding box center for receptor A.
        size_a: Binding box size for receptor A.
        pdbqt_b: Path to receptor B PDBQT file (for double/single_b mode).
        center_b: Binding box center for receptor B.
        size_b: Binding box size for receptor B.
        commands: Comma-separated FREED++ commands.

    Returns:
        Completed process with stdout captured.

    Raises:
        ValueError: If ``single_b`` mode is used without receptor B.
        subprocess.CalledProcessError: If the Docker process exits non-zero.
    """
    # For single_b, B is the primary receptor for FREED++ generation + reward.
    # _build_freedpp_args param names (pdbqt_a, center_a, size_a) refer to the
    # primary receptor, not necessarily target A.
    if config.reward_mode == "single_b":
        if pdbqt_b is None:
            raise ValueError("single_b requires receptor B (second_target_id must be set)")
        if center_b is None or size_b is None:
            raise ValueError("single_b requires center_b and size_b")
        args = _build_freedpp_args(config, pdbqt_b, center_b, size_b, commands)
    else:
        args = _build_freedpp_args(config, pdbqt_a, center_a, size_a, commands)

    gpu_device = os.getenv("FREEDPP_CUDA_DEVICE")
    gpu_flag = f'"device={gpu_device}"' if gpu_device is not None else "all"

    docker_cmd: list[str] = [
        DOCKER_BIN,
        "run",
        "--rm",
        "--gpus",
        gpu_flag,
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{workspace}:/workspace",
    ]

    docker_cmd.extend(
        [
            "-e",
            f"REWARD_MODE={config.reward_mode}",
            "-e",
            f"TARGET_ID_A={config.target_id}",
            "-e",
            f"TARGET_ID_B={config.second_target_id or ''}",
            "-e",
            "NORMALIZATION_PATH=/workspace/normalization.yaml",
            "-e",
            f"QG_MAX_INVALID_FRACTION={config.qg_max_invalid_fraction}",
            "-e",
            f"QG_MAX_TIMEOUT_FRACTION={config.qg_max_timeout_fraction}",
            "-e",
            "GPU_BATCH_DOCK=1",
        ]
    )

    # Dual-target: mount second receptor and config (only needed for double mode)
    if config.reward_mode == "double" and pdbqt_b is not None:
        dual_cfg = {
            "receptor_b": f"/workspace/{pdbqt_b.name}",
            "center_b": center_b,
            "size_b": size_b,
        }
        dual_config_path = workspace / "dual_config.json"
        dual_config_path.write_text(json.dumps(dual_cfg, indent=2))

        docker_cmd.extend(
            [
                "-e",
                f"DUAL_RECEPTOR_B=/workspace/{pdbqt_b.name}",
                "-e",
                "DUAL_CONFIG=/workspace/dual_config.json",
            ]
        )

    docker_cmd.append(config.docker_image)
    docker_cmd.extend(args)

    def _ts() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
    log_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = workspace / f"freedpp_{config.reward_mode}_seed_{config.seed}_{run_id}_{log_ts}.log"

    logger.info("Запуск FREED++: {}", " ".join(docker_cmd))
    logger.info("FREED++ лог -> {}", log_path)

    epoch_bar: tqdm[None] | None = None
    log_file = None
    stderr_lines: list[str] = []
    process: subprocess.Popen[str] | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"[{_ts()}] RUN_ID={run_id}\n")
        log_file.write(f"[{_ts()}] CMD: {' '.join(docker_cmd)}\n")
        log_file.flush()

        process = subprocess.Popen(  # noqa: S603
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if "train" in commands:
            epoch_bar = tqdm(total=config.epochs, desc="FREED++ обучение", unit="epoch")

        train_start = time.monotonic()
        last_epoch_logged = -1
        line_count = 0

        # Multiple patterns to catch FREED++ epoch/iteration progress
        _epoch_re = re.compile(
            r"[Ee](?:poch|pisode)\s*[:\[#]?\s*(\d+)"  # Epoch 1, Episode 1, epoch: 1
            r"|#(\d+)"  # #1, #001
            r"|\[(\d+)/\d+\]"  # [1/200]
            r"|[Ii]ter(?:ation)?\s*[:\[#]?\s*(\d+)"  # Iter 1, Iteration: 1
        )
        _noise_patterns = ("DeprecationWarning", "SyntaxWarning", "FutureWarning", "UserWarning", "warnings.warn")

        for line in process.stdout:  # type: ignore[union-attr]
            stripped = line.rstrip()
            log_file.write(f"[{_ts()}] {stripped}\n")
            log_file.flush()
            line_count += 1

            # Skip noisy warnings
            if any(pat in stripped for pat in _noise_patterns):
                stderr_lines.append(line)
                continue

            if epoch_bar is not None:
                match = _epoch_re.search(stripped)
                if match:
                    # Take first non-None group
                    epoch_num = int(next(g for g in match.groups() if g is not None))
                    epoch_bar.n = min(epoch_num, config.epochs)
                    epoch_bar.refresh()
                    log_interval = max(1, config.epochs // 10)
                    if epoch_num > 0 and epoch_num % log_interval == 0 and epoch_num != last_epoch_logged:
                        last_epoch_logged = epoch_num
                        elapsed = time.monotonic() - train_start
                        sec_per_epoch = elapsed / epoch_num
                        remaining = (config.epochs - epoch_num) * sec_per_epoch
                        logger.info(
                            "[FREED++] Epoch {}/{} — {:.0f}s elapsed, ~{:.0f}m remaining ({:.1f}s/epoch)",
                            epoch_num,
                            config.epochs,
                            elapsed,
                            remaining / 60,
                            sec_per_epoch,
                        )
                elif line_count % 500 == 0:
                    # Fallback: show line count in postfix if no epoch pattern matched yet
                    elapsed = time.monotonic() - train_start
                    epoch_bar.set_postfix_str(f"{line_count} lines, {elapsed:.0f}s")

            low = stripped.lower()
            if any(kw in low for kw in ("error", "sample", "saving", "loss", "score", "result")):
                logger.info("[FREED++] {}", stripped)
            stderr_lines.append(line)

        process.wait(timeout=config.training_timeout)
    finally:
        if epoch_bar is not None:
            epoch_bar.close()
        if log_file is not None:
            log_file.close()

    stdout_text = "".join(stderr_lines)

    if process is None or process.returncode != 0:
        rc = process.returncode if process is not None else -1
        logger.error("FREED++ завершился с ошибкой (rc={})", rc)
        logger.error("output: {}", stdout_text[-2000:])
        logger.error("FREED++ лог -> {}", log_path)
        raise subprocess.CalledProcessError(rc, docker_cmd, output=stdout_text)

    return subprocess.CompletedProcess(docker_cmd, process.returncode, stdout=stdout_text, stderr="")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_output_smiles(workspace: Path) -> list[str]:
    """Read generated SMILES from FREED++ output files.

    FREED++ saves results in ``{exp_root}/{name}/mols/``:

    - ``sample_*.json`` -- JSON array of SMILES (primary format)
    - ``sample_*.csv`` -- CSV with a ``smiles`` column (legacy)

    Args:
        workspace: FREED++ workspace directory containing the ``output/``
            subdirectory.

    Returns:
        List of SMILES strings parsed from output files.
    """
    import pandas as pd

    mols_dir = workspace / "output" / "run" / "mols"
    all_smiles: list[str] = []

    # JSON format (primary)
    json_files = sorted(mols_dir.glob("sample_*.json"))
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
            if isinstance(data, list):
                all_smiles.extend(str(s) for s in data if s)
        except Exception as e:
            logger.warning("Error reading {}: {}", json_path, e)

    # CSV format (fallback ONLY if no JSON found)
    if not all_smiles:
        csv_files = sorted(mols_dir.glob("sample_*.csv"))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                col = "smiles" if "smiles" in df.columns else "Smiles" if "Smiles" in df.columns else None
                if col:
                    all_smiles.extend(df[col].dropna().tolist())
            except Exception as e:
                logger.warning("Error reading {}: {}", csv_path, e)

    if not all_smiles:
        logger.warning("No molecule files found in {}", mols_dir)

    logger.info("Read {} SMILES from output", len(all_smiles))
    return all_smiles


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
    needs_second = config.reward_mode in {"single_b", "double"}
    if needs_second and not config.second_target_id:
        raise ValueError(f"{config.reward_mode} mode requires second_target_id")
    n_steps = 8 if not needs_second else 9
    pbar = tqdm(
        total=n_steps,
        desc=f"Генерация [{config.target_id}]",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    # 1. Docker image
    pbar.set_postfix_str("Docker-образ")
    _ensure_docker_image(config.docker_image)
    pbar.update(1)

    # 2. Cache check
    pbar.set_postfix_str("Проверка кеша")
    key = _cache_key(config)
    cached_model = _find_cached_model(key) if config.use_cache else None
    pbar.update(1)

    # 3. Workspace
    pbar.set_postfix_str("Подготовка workspace")
    workspace = config.output_dir / "freedpp_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    pbar.update(1)

    # 3.1. Normalization (needed for reward)
    normalization_src = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not normalization_src.exists():
        raise FileNotFoundError(f"normalization.yaml не найден: {normalization_src}")
    normalization_dst = workspace / "normalization.yaml"
    if not normalization_dst.exists():
        shutil.copy2(normalization_src, normalization_dst)
        logger.info("normalization.yaml -> {}", normalization_dst)

    # 4. Prepare receptor A
    pbar.set_postfix_str(f"Receptor A → PDBQT ({config.target_id})")
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
        pbar.set_postfix_str(f"Receptor B → PDBQT ({config.second_target_id})")
        pdb_b = _resolve_pdb_path(config.second_target_id)
        pdbqt_b, center_b, size_b = _prepare_receptor(
            pdb_b,
            workspace,
            config.docker_image,
            config.box_padding,
        )
        pbar.update(1)

    # 6. Determine commands
    if cached_model is not None:
        model_target = workspace / "output" / "run"
        if not model_target.exists():
            shutil.copytree(cached_model, model_target, dirs_exist_ok=True)
        commands = "sample"
        logger.info("Используется кешированная модель ({}) → только sampling", key)
    else:
        commands = "train,sample"
        # Clear stale output from a previously interrupted run
        stale_output = workspace / "output" / "run"
        if stale_output.exists():
            logger.warning("Обнаружен stale output от прерванного запуска: {}", stale_output)
            shutil.rmtree(stale_output, ignore_errors=True)
            # Fallback: Docker (root-owned files)
            if stale_output.exists():
                logger.info("Удаление root-owned файлов через Docker...")
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
            logger.info("Stale output удалён")

    # 7. Run FREED++
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
    )
    pbar.update(1)

    # 8. Parse output
    pbar.set_postfix_str("Парсинг SMILES")
    smiles = _parse_output_smiles(workspace)
    pbar.update(1)

    # 9. Cache trained model
    pbar.set_postfix_str("Кеширование модели")
    if config.use_cache and commands == "train,sample":
        model_dir = workspace / "output" / "run"
        if model_dir.exists():
            cache_dest = CACHE_DIR / key
            cache_dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(model_dir, cache_dest, dirs_exist_ok=True)
            logger.info("Модель сохранена в кеш: {}", cache_dest)
    pbar.update(1)

    # 10. Write output
    pbar.set_postfix_str("Запись результатов")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    from dualdocking.data.io_utils import write_df

    _gen_df = pd.DataFrame({"smiles": smiles})
    write_df(_gen_df, config.output_dir / "generated", csv_too=True)
    logger.info("Сгенерировано {} молекул -> {}", len(smiles), config.output_dir / "generated.parquet")
    pbar.close()

    return smiles
