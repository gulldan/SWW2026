"""Boltz-2 affinity scoring wrapper with disk cache.

Provides ``score_batch(target, smiles_list)`` as the main entry point.
Two backends are supported:

- **cli** (default): invokes ``boltz predict`` via subprocess.
- **nim**: calls the NVIDIA NIM Boltz-2 REST API (persistent GPU server).

Backend selection: set ``BOLTZ_BACKEND=nim`` environment variable.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from dualdocking.data.mol_prep import check_ligand_size, neutralize_smiles, parse_smiles
from dualdocking.scoring import boltz2_batch as _batch_helpers
from dualdocking.scoring import boltz2_context as _context_helpers
from dualdocking.scoring import boltz2_helpers as _helpers
from dualdocking.scoring import boltz2_nim_backend as _nim_backend
from dualdocking.scoring import boltz2_scoring_api as _scoring_api
from dualdocking.scoring import boltz2_storage as _storage_helpers
from dualdocking.scoring import boltz2_subprocess as _subprocess_helpers
from dualdocking.scoring.cache_io import load_cache, save_cache
from dualdocking.utils.hardware import detect_primary_gpu_vram_gb
from dualdocking.utils.master_config import read_master_config_field
from dualdocking.utils.nim import nim_url_ready

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "cache"
SCORE_CACHE_DIR = CACHE_DIR / "boltz_scores"
DATA_DIR = PROJECT_ROOT / "output" / "data" / "targets"
MSA_CACHE_DIR = CACHE_DIR / "boltz_msa"

BOLTZ_TIMEOUT = 600  # seconds per molecule
BOLTZ_BATCH_TIMEOUT = 3600  # seconds per batch
MAX_LIGAND_ATOMS = 128
_DEFAULT_BATCH_SIZE = 64  # molecules per boltz predict invocation (internal batch=1, safe on any VRAM)


def _auto_batch_size() -> int:
    env = os.getenv("BOLTZ_BATCH_SIZE")
    if env is not None:
        return int(env)

    vram_gb = detect_primary_gpu_vram_gb()

    if vram_gb is None:
        logger.debug("GPU detection failed for batch size, using default={}", _DEFAULT_BATCH_SIZE)
        return _DEFAULT_BATCH_SIZE

    logger.debug("GPU VRAM detected for batch size: {:.1f} GB", vram_gb)
    if vram_gb >= 46:  # H100/H800 80 GB / A100 40-80 GB (nvidia-smi reports ~79.x GiB)
        return 128
    if vram_gb >= 20:  # RTX 4090 24 GB / A5000 24 GB
        return 64
    return _DEFAULT_BATCH_SIZE


BATCH_SIZE = _auto_batch_size()

# Boltz CLI parameters
_BOLTZ_CLI_ARGS_BASE: list[str] = [
    "--model",
    "boltz2",
    "--recycling_steps",
    "3",
    "--sampling_steps",
    "50",
    "--diffusion_samples",
    "1",
    "--sampling_steps_affinity",
    "50",
    "--diffusion_samples_affinity",
    "1",
    "--override",
]

_ACCELERATOR: str | None = None

# Scoring context hash — computed once per process
_CONTEXT_HASH: str | None = None

# Docker image for FREED++ (used for digest lookup)
_FREEDPP_DOCKER_IMAGE = "dualdocking/freedpp:latest"

# ---------------------------------------------------------------------------
# NIM backend configuration
# ---------------------------------------------------------------------------
_BOLTZ_BACKEND: str = os.getenv("BOLTZ_BACKEND", "cli").lower()
# Multiple NIM URLs for round-robin (comma-separated) or single URL
_BOLTZ_NIM_URL: str = os.getenv("BOLTZ_NIM_URL", "http://localhost:8000")
_BOLTZ_NIM_URLS: list[str] = [u.strip() for u in os.getenv("BOLTZ_NIM_URLS", _BOLTZ_NIM_URL).split(",") if u.strip()]
_NIM_PREDICT_ENDPOINT = "/biology/mit/boltz2/predict"
_NIM_HEALTH_ENDPOINT = "/v1/health/ready"
_NIM_TIMEOUT = 300  # seconds per molecule via NIM
# Per-instance concurrency: how many parallel affinity requests to queue per NIM instance.
# NIM processes 1-2 affinity predictions at a time; excess requests queue internally
# with a 120s GPU-acquire timeout.  With pipeline GPU-pinning (each scoring process →
# its own NIM), concurrency=2 keeps the queue short enough to avoid timeouts.
# Override with NIM_CONCURRENCY env var if needed (try 3-4 if single-process).
_NIM_CONCURRENCY = int(os.getenv("NIM_CONCURRENCY", "1"))
_NIM_HEALTH_TIMEOUT = float(os.getenv("NIM_HEALTH_TIMEOUT", "2.0"))

# Cached active NIM pool (configured URLs are probed once per process/config change).
_NIM_ACTIVE_URLS: list[str] | None = None
_NIM_ACTIVE_URLS_KEY: tuple[str, ...] | None = None
_NIM_ACTIVE_URLS_TIME: float = 0.0
_NIM_HEALTH_TTL: float = 120.0  # re-probe health every 2 minutes

# NIM prediction params — mirrors _BOLTZ_CLI_ARGS_BASE values.
# step_scale=1.638: boltz2 default used by both NIM and CLI.
_NIM_PREDICT_PARAMS: dict[str, int | float | str] = {
    "recycling_steps": 3,
    "sampling_steps": 50,
    "diffusion_samples": 1,
    "sampling_steps_affinity": 50,
    "diffusion_samples_affinity": 1,
    "step_scale": 1.638,
    "output_format": "mmcif",
}


def _configured_nim_urls() -> list[str]:
    urls = _BOLTZ_NIM_URLS or [_BOLTZ_NIM_URL]
    cleaned = [u.strip().rstrip("/") for u in urls if u and u.strip()]
    deduped = list(dict.fromkeys(cleaned))
    if not deduped:
        msg = "No NIM URLs configured. Set BOLTZ_NIM_URL or BOLTZ_NIM_URLS."
        raise RuntimeError(msg)
    return deduped


def _nim_url_ready(url: str) -> bool:
    return nim_url_ready(url, health_endpoint=_NIM_HEALTH_ENDPOINT, timeout=_NIM_HEALTH_TIMEOUT)


def _resolve_nim_urls(*, refresh: bool = False) -> list[str]:
    import time as _time

    global _NIM_ACTIVE_URLS, _NIM_ACTIVE_URLS_KEY, _NIM_ACTIVE_URLS_TIME

    configured = tuple(_configured_nim_urls())
    now = _time.monotonic()
    cache_valid = (
        not refresh
        and _NIM_ACTIVE_URLS is not None
        and configured == _NIM_ACTIVE_URLS_KEY
        and (now - _NIM_ACTIVE_URLS_TIME) < _NIM_HEALTH_TTL
    )
    if cache_valid:
        return _NIM_ACTIVE_URLS  # type: ignore[return-value]

    healthy = [url for url in configured if _nim_url_ready(url)]
    if healthy:
        dropped = [url for url in configured if url not in healthy]
        if dropped:
            logger.warning("NIM URL(s) not ready, excluded from pool: {}", ", ".join(dropped))
        _NIM_ACTIVE_URLS = healthy
        _NIM_ACTIVE_URLS_KEY = configured
        _NIM_ACTIVE_URLS_TIME = now
        return healthy

    msg = (
        f"No healthy NIM URL detected via {_NIM_HEALTH_ENDPOINT}. "
        f"Configured URLs: {', '.join(configured)}. "
        "Ensure NIM containers are running and accessible."
    )
    raise RuntimeError(msg)


def _resolve_accelerator() -> str:
    global _ACCELERATOR
    if _ACCELERATOR is not None:
        return _ACCELERATOR

    env = os.getenv("BOLTZ_ACCELERATOR", "gpu").lower()
    if env not in {"gpu", "cpu"}:
        raise ValueError(f"Unsupported BOLTZ_ACCELERATOR={env}; use 'gpu' or 'cpu'")

    if env == "gpu":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Boltz GPU requested but torch.cuda.is_available() is False. "
                    "Reinstall boltz with CUDA support and ensure NVIDIA drivers are visible."
                )
        except Exception as exc:
            raise RuntimeError(
                "Boltz GPU requested but CUDA is not available. Reinstall boltz with CUDA support and verify torch CUDA."
            ) from exc

    _ACCELERATOR = env
    return _ACCELERATOR


# ---------------------------------------------------------------------------
# Scoring context — v2 cache key
# ---------------------------------------------------------------------------


def _boltz_version() -> str:
    try:
        return importlib.metadata.version("boltz")
    except Exception:
        logger.debug("importlib.metadata.version('boltz') failed, trying master_config.yaml")
    try:
        return _master_config_field("software.boltz_version")
    except Exception:
        logger.warning("Cannot determine Boltz version; using 'unknown'")
        return "unknown"


def _master_config_field(path: str) -> str:
    return read_master_config_field(PROJECT_ROOT, path)


def _run_cmd(cmd: list[str]) -> str:
    return _context_helpers.run_cmd(cmd)


def _docker_digest() -> str:
    return _context_helpers.docker_digest(
        _FREEDPP_DOCKER_IMAGE,
        master_config_field=_master_config_field,
        log=logger,
    )


def _msa_content_hash(msa_path: Path | None) -> str:
    return _helpers.msa_content_hash(msa_path)


def _sequence_content_hash(sequence: str) -> str:
    return _helpers.sequence_content_hash(sequence)


def _scoring_context_hash() -> str:
    global _CONTEXT_HASH
    if _CONTEXT_HASH is not None:
        return _CONTEXT_HASH

    _CONTEXT_HASH = _context_helpers.context_hash(
        backend=_BOLTZ_BACKEND,
        nim_predict_params=_NIM_PREDICT_PARAMS,
        nim_urls=_BOLTZ_NIM_URLS,
        boltz_version=_boltz_version(),
        cli_args_base=_BOLTZ_CLI_ARGS_BASE,
        accelerator=os.getenv("BOLTZ_ACCELERATOR", "gpu").lower(),
        docker_digest=_docker_digest(),
        log=logger,
    )
    return _CONTEXT_HASH


def get_cache_context_hash() -> str:
    """Public accessor for the scoring-context hash.

    Returns:
        16-character hex prefix of the context digest.
    """
    return _scoring_context_hash()


def get_scoring_context_info() -> dict[str, str | int | list[str]]:
    """Return all scoring-context components for metadata persistence.

    Returns:
        Dict with keys like ``cache_context_hash``, ``backend``, and
        backend-specific fields (e.g. ``boltz_version`` for CLI,
        ``nim_urls`` for NIM).
    """
    return _context_helpers.context_info(
        backend=_BOLTZ_BACKEND,
        cache_context_hash=get_cache_context_hash(),
        nim_predict_params=_NIM_PREDICT_PARAMS,
        nim_urls=_BOLTZ_NIM_URLS,
        nim_concurrency=_NIM_CONCURRENCY,
        boltz_version=_boltz_version(),
        cli_args_base=_BOLTZ_CLI_ARGS_BASE,
        accelerator=os.getenv("BOLTZ_ACCELERATOR", "gpu").lower(),
        docker_digest=_docker_digest(),
    )


def prune_stale_cache() -> int:
    """Remove cache files whose context segment does not match the current hash.

    Returns:
        Number of stale files removed.
    """
    ctx_prefix = _scoring_context_hash()[:8]
    removed = 0
    for f in SCORE_CACHE_DIR.glob("*.json"):
        if f.suffix == ".lock":
            continue
        # Current-context files have pattern: {target_id}.{ctx[:8]}.json
        parts = f.stem.rsplit(".", 1)
        if len(parts) == 2 and parts[1] == ctx_prefix:
            continue
        # Legacy files (no context segment) or different context — remove
        f.unlink()
        logger.info("Pruned stale cache file: {}", f.name)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Sequence cache + cache key helpers
# ---------------------------------------------------------------------------

# In-memory sequence cache (target_id -> sequence)
_sequence_cache: dict[str, str] = {}


def _cache_key(
    target_id: str,
    smi: str,
    *,
    ctx_hash: str = "",
    seq_hash: str = "",
    msa_hash: str = "",
) -> str:
    return _helpers.cache_key(
        target_id,
        smi,
        ctx_hash=ctx_hash,
        seq_hash=seq_hash,
        msa_hash=msa_hash,
    )


def _cache_path(target_id: str) -> Path:
    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SCORE_CACHE_DIR / f"{target_id}.{_scoring_context_hash()[:8]}.json"


def _load_cache(target_id: str) -> dict[str, float]:
    return load_cache(_cache_path(target_id))


def _save_cache(target_id: str, cache: dict[str, float]) -> None:
    save_cache(_cache_path(target_id), cache)


def _load_sequence(target_id: str) -> str:
    return _storage_helpers.load_sequence(
        target_id,
        data_dir=DATA_DIR,
        sequence_cache=_sequence_cache,
    )


def _get_msa_path(target_id: str) -> Path | None:
    return _storage_helpers.get_msa_path(target_id, msa_cache_dir=MSA_CACHE_DIR)


def _require_msa_path(target_id: str) -> Path:
    return _helpers.require_msa_path(MSA_CACHE_DIR, target_id)


def _validate_smiles(smiles: str) -> bool:
    mol = parse_smiles(smiles)
    if mol is None:
        return False
    return check_ligand_size(mol, max_heavy=100, max_atoms=MAX_LIGAND_ATOMS, max_rotatable=None)


def _neutralize_smiles(smiles: str) -> str:
    return neutralize_smiles(smiles)


def _collect_valid_smiles(
    smiles_list: list[str],
    *,
    warning_template: str,
) -> list[tuple[int, str]]:
    return _helpers.collect_valid_smiles(
        smiles_list,
        validate_smiles=_validate_smiles,
        neutralize_smiles=_neutralize_smiles,
        max_ligand_atoms=MAX_LIGAND_ATOMS,
        warning_template=warning_template,
        log=logger,
    )


def _write_boltz_yaml(
    sequence: str,
    smiles: str,
    path: Path,
    msa_path: Path | None = None,
) -> None:
    _helpers.write_boltz_yaml(sequence, smiles, path, msa_path=msa_path)


def _boltz_env(device: str | None = None) -> dict[str, str]:
    return _subprocess_helpers.build_boltz_env(
        device,
        fallback_device=os.getenv("BOLTZ_CUDA_DEVICE"),
        log=logger,
    )


def _run_boltz_predict(yaml_path: Path, out_dir: Path, *, device: str | None = None) -> subprocess.CompletedProcess[str]:
    return _subprocess_helpers.run_boltz_predict(
        yaml_path,
        out_dir,
        project_root=PROJECT_ROOT,
        accelerator=_resolve_accelerator(),
        cli_args_base=_BOLTZ_CLI_ARGS_BASE,
        timeout=BOLTZ_TIMEOUT,
        env=_boltz_env(device=device),
        log=logger,
    )


def _parse_affinity(out_dir: Path, stem: str) -> float:
    return _helpers.parse_affinity(out_dir, stem)


def _parse_affinity_file(affinity_path: Path) -> float:
    return _helpers.parse_affinity_file(affinity_path)


def _score_single_internal(
    target_id: str,
    smiles: str,
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None = None,
) -> float:
    return _subprocess_helpers.score_single_internal(
        target_id,
        smiles,
        sequence,
        msa_path,
        device=device,
        write_boltz_yaml=_write_boltz_yaml,
        run_boltz_predict=_run_boltz_predict,
        parse_affinity=_parse_affinity,
        log=logger,
    )


def _run_boltz_predict_dir(
    input_dir: Path,
    out_dir: Path,
    timeout: int = BOLTZ_BATCH_TIMEOUT,
    *,
    device: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return _subprocess_helpers.run_boltz_predict_dir(
        input_dir,
        out_dir,
        project_root=PROJECT_ROOT,
        accelerator=_resolve_accelerator(),
        cli_args_base=_BOLTZ_CLI_ARGS_BASE,
        timeout=timeout,
        env=_boltz_env(device=device),
        log=logger,
    )


def _nim_health_check() -> bool:
    return any(_nim_url_ready(url) for url in _configured_nim_urls())


_NIM_MAX_RETRIES = 2  # total attempts = 1 + retries


def _nim_payload(sequence: str, smiles: str, msa_content: str | None = None) -> dict[str, Any]:
    return _helpers.nim_payload(sequence, smiles, predict_params=_NIM_PREDICT_PARAMS, msa_content=msa_content)


def _nim_predict_affinity(sequence: str, smiles: str) -> float:
    return _nim_backend.predict_affinity_sync(
        sequence,
        smiles,
        nim_urls=_resolve_nim_urls(),
        predict_endpoint=_NIM_PREDICT_ENDPOINT,
        request_timeout=_NIM_TIMEOUT,
        max_retries=_NIM_MAX_RETRIES,
        payload_builder=_nim_payload,
        parse_affinity=_parse_nim_affinity,
        log=logger,
    )


def _parse_nim_affinity(result: dict[str, Any], smiles: str) -> float:
    return _helpers.parse_nim_affinity(result, smiles)


async def _nim_predict_batch_async(
    sequence: str,
    smiles_indices: list[tuple[int, str]],
    *,
    nim_urls: list[str] | None = None,
    msa_content: str | None = None,
) -> list[tuple[int, float]]:
    return await _nim_backend.predict_batch_async(
        sequence,
        smiles_indices,
        nim_urls=nim_urls if nim_urls is not None else _resolve_nim_urls(),
        predict_endpoint=_NIM_PREDICT_ENDPOINT,
        request_timeout=_NIM_TIMEOUT,
        concurrency=_NIM_CONCURRENCY,
        max_retries=_NIM_MAX_RETRIES,
        payload_builder=_nim_payload,
        parse_affinity=_parse_nim_affinity,
        log=logger,
        msa_content=msa_content,
    )


def _score_batch_internal_nim(
    target_id: str,  # noqa: ARG001
    smiles_list: list[str],
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None = None,  # noqa: ARG001
) -> list[float]:
    msa_content = msa_path.read_text() if msa_path is not None else None
    return _batch_helpers.score_batch_nim(
        smiles_list,
        sequence,
        collect_valid_smiles=_collect_valid_smiles,
        resolve_nim_urls=_resolve_nim_urls,
        predict_batch_async=_nim_predict_batch_async,
        nim_concurrency=_NIM_CONCURRENCY,
        log=logger,
        msa_content=msa_content,
    )


def _score_batch_internal(
    target_id: str,
    smiles_list: list[str],
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None = None,
) -> list[float]:
    if _BOLTZ_BACKEND == "nim":
        return _score_batch_internal_nim(
            target_id,
            smiles_list,
            sequence,
            msa_path,
            device=device,
        )

    return _batch_helpers.score_batch_cli(
        target_id,
        smiles_list,
        sequence,
        msa_path,
        device=device,
        batch_timeout=BOLTZ_BATCH_TIMEOUT,
        collect_valid_smiles=_collect_valid_smiles,
        write_boltz_yaml=_write_boltz_yaml,
        run_boltz_predict_dir=_run_boltz_predict_dir,
        parse_affinity_file=_parse_affinity_file,
        score_single_internal=_score_single_internal,
        log=logger,
    )


def score_single(target_id: str, smiles: str) -> float:
    """Score a single molecule against a target via Boltz-2.

    Args:
        target_id: Target identifier matching ``configs/pairs.yaml``.
        smiles: Canonical SMILES string.

    Returns:
        Predicted affinity as log10(IC50 in uM); lower = stronger
        binding.  ``NaN`` on any error.

    Raises:
        FileNotFoundError: If the FASTA or MSA file is missing (CLI
            backend only).
    """
    return _scoring_api.score_single(
        target_id,
        smiles,
        backend=_BOLTZ_BACKEND,
        max_ligand_atoms=MAX_LIGAND_ATOMS,
        boltz_timeout=BOLTZ_TIMEOUT,
        validate_smiles=_validate_smiles,
        load_sequence=_load_sequence,
        neutralize_smiles=_neutralize_smiles,
        nim_predict_affinity=_nim_predict_affinity,
        require_msa_path=_require_msa_path,
        write_boltz_yaml=_write_boltz_yaml,
        run_boltz_predict=_run_boltz_predict,
        parse_affinity=_parse_affinity,
        log=logger,
    )


def score_batch(
    target_id: str,
    smiles_list: list[str],
    *,
    device: str | None = None,
    use_cache: bool = True,
    batch_size: int = BATCH_SIZE,
) -> NDArray[np.float64]:
    """Score a batch of SMILES against *target_id* with disk caching.

    Molecules are grouped into sub-batches and processed via a single
    ``boltz predict`` call per sub-batch (model loaded once).  Results
    are persisted to a context-segmented JSON cache after each sub-batch.

    Args:
        target_id: Target identifier matching ``configs/pairs.yaml``.
        smiles_list: Canonical SMILES strings.
        device: Optional CUDA device index for GPU pinning.
        use_cache: Skip already-scored molecules when ``True``.
        batch_size: Number of molecules per ``boltz predict`` invocation.

    Returns:
        1-D array of scores (log10 IC50 in uM), same order as
        *smiles_list*.  ``NaN`` for failed molecules.

    Raises:
        FileNotFoundError: If the FASTA or MSA file is missing (CLI
            backend only).
    """
    return _scoring_api.score_batch(
        target_id,
        smiles_list,
        device=device,
        use_cache=use_cache,
        batch_size=batch_size,
        backend=_BOLTZ_BACKEND,
        load_sequence=_load_sequence,
        require_msa_path=_require_msa_path,
        get_msa_path=_get_msa_path,
        sequence_content_hash=_sequence_content_hash,
        msa_content_hash=_msa_content_hash,
        scoring_context_hash=_scoring_context_hash,
        load_cache=_load_cache,
        save_cache=_save_cache,
        cache_key=_cache_key,
        score_batch_internal=_score_batch_internal,
        log=logger,
    )
