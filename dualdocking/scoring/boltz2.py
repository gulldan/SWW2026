"""Boltz-2 affinity scoring wrapper with disk cache.

Provides ``score_batch(target, smiles_list)`` as the main entry point.
Two backends are supported:

- **cli** (default): invokes ``boltz predict`` via subprocess.
- **nim**: calls the NVIDIA NIM Boltz-2 REST API (persistent GPU server).

Backend selection: set ``BOLTZ_BACKEND=nim`` environment variable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from dualdocking.scoring.cache_io import load_cache, save_cache

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
    """Pick batch size based on GPU VRAM.

    Override via the ``BOLTZ_BATCH_SIZE`` environment variable.  Falls
    back to ``_DEFAULT_BATCH_SIZE`` when GPU detection fails.

    Returns:
        Number of molecules per ``boltz predict`` invocation.
    """
    env = os.getenv("BOLTZ_BATCH_SIZE")
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
_NIM_CONCURRENCY = int(os.getenv("NIM_CONCURRENCY", "1"))
_NIM_HEALTH_TIMEOUT = float(os.getenv("NIM_HEALTH_TIMEOUT", "2.0"))

# Cached active NIM pool (configured URLs are probed once per process/config change).
_NIM_ACTIVE_URLS: list[str] | None = None
_NIM_ACTIVE_URLS_KEY: tuple[str, ...] | None = None
_NIM_ACTIVE_URLS_TIME: float = 0.0
_NIM_HEALTH_TTL: float = 120.0  # re-probe health every 2 minutes

# NIM prediction params — mirrors _BOLTZ_CLI_ARGS_BASE values.
# step_scale=1.5: boltz2 model default (CLI uses 1.5; NIM defaults to 1.638 if unset).
_NIM_PREDICT_PARAMS: dict[str, int | float | str] = {
    "recycling_steps": 3,
    "sampling_steps": 50,
    "diffusion_samples": 1,
    "sampling_steps_affinity": 50,
    "diffusion_samples_affinity": 1,
    "step_scale": 1.5,
    "output_format": "mmcif",
}


def _configured_nim_urls() -> list[str]:
    """Return configured NIM URLs with preserved order and deduplication."""
    urls = _BOLTZ_NIM_URLS or [_BOLTZ_NIM_URL]
    cleaned = [u.strip().rstrip("/") for u in urls if u and u.strip()]
    deduped = list(dict.fromkeys(cleaned))
    if not deduped:
        msg = "No NIM URLs configured. Set BOLTZ_NIM_URL or BOLTZ_NIM_URLS."
        raise RuntimeError(msg)
    return deduped


def _nim_url_ready(url: str) -> bool:
    """Check whether a specific NIM URL is ready."""
    health_url = f"{url}{_NIM_HEALTH_ENDPOINT}"
    try:
        req = urllib.request.Request(health_url)  # noqa: S310
        with urllib.request.urlopen(req, timeout=_NIM_HEALTH_TIMEOUT) as resp:  # noqa: S310
            body = resp.read().lower()
        if b"false" in body:
            return False
        return b"true" in body or b"ready" in body or len(body) == 0
    except Exception:
        return False


def _resolve_nim_urls(*, refresh: bool = False) -> list[str]:
    """Resolve active NIM URLs, filtering out unreachable instances.

    Probes each configured URL via the health endpoint and returns only
    those that respond as ready.  Results are cached for ``_NIM_HEALTH_TTL``
    seconds to avoid repeated probing during a scoring batch.

    Raises:
        RuntimeError: If no configured URL passes the health check.
    """
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
    """Resolve and cache the Boltz accelerator (``gpu`` or ``cpu``).

    Uses ``BOLTZ_ACCELERATOR`` env var (default ``gpu``).  Verifies CUDA
    availability when GPU is selected.

    Returns:
        ``"gpu"`` or ``"cpu"``.

    Raises:
        ValueError: If the env var is not ``gpu`` or ``cpu``.
        RuntimeError: If GPU is requested but CUDA is unavailable.
    """
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
    """Return the installed Boltz package version.

    Falls back to ``configs/master_config.yaml`` then ``"unknown"``.

    Returns:
        Version string, e.g. ``"2.2.1"``.
    """
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
    """Read a dotted key path from ``configs/master_config.yaml``.

    Args:
        path: Dot-separated key path (e.g. ``"software.boltz_version"``).

    Returns:
        The string value at the given path.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the dotted path is not found.
    """
    import yaml

    master_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    if not master_path.exists():
        msg = f"master_config.yaml not found: {master_path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(master_path.read_text())
    value = data
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break
    if not value:
        msg = f"Field '{path}' not found in master_config.yaml"
        raise KeyError(msg)
    return str(value)


def _run_cmd(cmd: list[str]) -> str:
    """Run a shell command and return stripped stdout.

    Args:
        cmd: Command and arguments.

    Returns:
        Stripped stdout on success, or ``""`` on any failure.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _docker_digest() -> str:
    """Return FREED++ Docker image digest with a multi-level fallback.

    Tries ``docker image inspect`` for repo digest, then image ID, then
    ``master_config.yaml``, and finally returns ``"unknown"``.

    Returns:
        Digest or image ID string.
    """
    out = _run_cmd(["docker", "image", "inspect", _FREEDPP_DOCKER_IMAGE, "--format", "{{index .RepoDigests 0}}"])
    if "@" in out:
        return out.split("@")[-1]
    if out:
        return out
    # Local builds often have no RepoDigests; fall back to image ID.
    image_id = _run_cmd(["docker", "image", "inspect", _FREEDPP_DOCKER_IMAGE, "--format", "{{.Id}}"])
    if image_id:
        return image_id
    try:
        return _master_config_field("software.docker_image_digest")
    except Exception:
        logger.warning("Cannot determine Docker digest; using 'unknown'")
        return "unknown"


def _msa_content_hash(msa_path: Path | None) -> str:
    """Return the first 16 hex chars of the SHA-256 of MSA file contents.

    Args:
        msa_path: Path to MSA CSV file, or ``None`` if unavailable.

    Returns:
        Hex digest prefix, or ``"no_msa"`` when *msa_path* is ``None``.
    """
    if msa_path is None:
        return "no_msa"
    return hashlib.sha256(msa_path.read_bytes()).hexdigest()[:16]


def _sequence_content_hash(sequence: str) -> str:
    """Return the first 16 hex chars of the SHA-256 of a protein sequence.

    Args:
        sequence: Amino-acid sequence string.

    Returns:
        Hex digest prefix (16 characters).
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def _scoring_context_hash() -> str:
    """Compute (and cache) the global scoring-context hash.

    The context hash encodes backend-specific parameters (NIM params or
    CLI args + Boltz version + accelerator + Docker digest) so that cache
    files from different configurations never collide.

    Returns:
        16-character hex prefix of the SHA-256 context digest.
    """
    global _CONTEXT_HASH
    if _CONTEXT_HASH is not None:
        return _CONTEXT_HASH

    if _BOLTZ_BACKEND == "nim":
        nim_params_str = json.dumps(_NIM_PREDICT_PARAMS, sort_keys=True)
        # Context hash based on params only (not URLs, since round-robin is an implementation detail)
        blob = f"nim||{nim_params_str}".encode()
        _CONTEXT_HASH = hashlib.sha256(blob).hexdigest()[:16]
        logger.info(
            "Scoring context (NIM): urls={}, params_hash={}... -> ctx={}",
            _BOLTZ_NIM_URLS,
            hashlib.sha256(nim_params_str.encode()).hexdigest()[:8],
            _CONTEXT_HASH,
        )
    else:
        boltz_ver = _boltz_version()
        cli_args = "|".join(_BOLTZ_CLI_ARGS_BASE)
        accelerator = os.getenv("BOLTZ_ACCELERATOR", "gpu").lower()
        digest = _docker_digest()

        blob = f"{boltz_ver}||{cli_args}||{accelerator}||{digest}".encode()
        _CONTEXT_HASH = hashlib.sha256(blob).hexdigest()[:16]

        logger.info(
            "Scoring context: boltz={}, cli_hash={}..., accel={}, docker={}... -> ctx={}",
            boltz_ver,
            hashlib.sha256(cli_args.encode()).hexdigest()[:8],
            accelerator,
            digest[:16],
            _CONTEXT_HASH,
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
    if _BOLTZ_BACKEND == "nim":
        nim_params_str = json.dumps(_NIM_PREDICT_PARAMS, sort_keys=True)
        return {
            "cache_context_hash": get_cache_context_hash(),
            "backend": "nim",
            "nim_urls": _BOLTZ_NIM_URLS,
            "nim_concurrency": _NIM_CONCURRENCY,
            "nim_params_hash": hashlib.sha256(nim_params_str.encode()).hexdigest()[:16],
        }
    boltz_ver = _boltz_version()
    cli_args = "|".join(_BOLTZ_CLI_ARGS_BASE)
    accelerator = os.getenv("BOLTZ_ACCELERATOR", "gpu").lower()
    digest = _docker_digest()
    return {
        "cache_context_hash": get_cache_context_hash(),
        "backend": "cli",
        "boltz_version": boltz_ver,
        "docker_digest": digest,
        "accelerator": accelerator,
        "cli_args_hash": hashlib.sha256(cli_args.encode()).hexdigest()[:16],
    }


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
    """Build a deterministic SHA-256 cache key for a (target, SMILES) pair.

    Args:
        target_id: Target identifier.
        smi: Canonical SMILES string.
        ctx_hash: Scoring-context hash prefix.
        seq_hash: Protein-sequence content hash prefix.
        msa_hash: MSA file content hash prefix.

    Returns:
        Full 64-character hex SHA-256 digest.
    """
    blob = f"{target_id}||{smi}||{ctx_hash}||{seq_hash}||{msa_hash}".encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_path(target_id: str) -> Path:
    """Return the context-segmented cache file path for a target.

    Args:
        target_id: Target identifier.

    Returns:
        Path like ``SCORE_CACHE_DIR/{target_id}.{ctx[:8]}.json``.
    """
    SCORE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SCORE_CACHE_DIR / f"{target_id}.{_scoring_context_hash()[:8]}.json"


def _load_cache(target_id: str) -> dict[str, float]:
    """Load the disk cache for *target_id* under the current context."""
    return load_cache(_cache_path(target_id))


def _save_cache(target_id: str, cache: dict[str, float]) -> None:
    """Persist *cache* entries for *target_id* to disk (atomic merge)."""
    save_cache(_cache_path(target_id), cache)


def _load_sequence(target_id: str) -> str:
    """Load and cache the protein sequence from a FASTA file.

    Reads from ``DATA_DIR/{target_id}.fasta`` and caches in
    ``_sequence_cache`` for subsequent calls.

    Args:
        target_id: Target identifier matching the FASTA filename stem.

    Returns:
        Concatenated amino-acid sequence (header lines stripped).

    Raises:
        FileNotFoundError: If the FASTA file does not exist.
    """
    if target_id in _sequence_cache:
        return _sequence_cache[target_id]

    fasta_path = DATA_DIR / f"{target_id}.fasta"
    if not fasta_path.exists():
        msg = f"FASTA not found: {fasta_path}. Run: uv run python scripts/s1_prepare/prepare_targets.py --prepare-all"
        raise FileNotFoundError(msg)

    lines = fasta_path.read_text().strip().splitlines()
    # Skip FASTA headers (lines starting with >)
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    _sequence_cache[target_id] = seq
    return seq


def _get_msa_path(target_id: str) -> Path | None:
    """Locate the pre-computed MSA file for a target.

    Args:
        target_id: Target identifier.

    Returns:
        Path to ``MSA_CACHE_DIR/{target_id}.csv`` if it exists, else ``None``.
    """
    msa_path = MSA_CACHE_DIR / f"{target_id}.csv"
    if msa_path.exists():
        return msa_path
    return None


def _validate_smiles(smiles: str) -> bool:
    """Validate a SMILES string via RDKit parsing and atom-count check.

    Args:
        smiles: SMILES string to validate.

    Returns:
        ``True`` if the SMILES parses and has at most
        ``MAX_LIGAND_ATOMS`` heavy atoms.
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit unavailable, skipping SMILES validation")
        return True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= MAX_LIGAND_ATOMS


def _neutralize_smiles(smiles: str) -> str:
    """Neutralize formal charges for Boltz-2 compatibility.

    Boltz-2 v2.2.1 crashes on charged species (``[NH2+]``, ``[O-]``).
    Strips protonation states while preserving molecular identity.

    Args:
        smiles: Input SMILES (potentially charged).

    Returns:
        Neutralized SMILES, or the original string if neutralization
        fails or RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    try:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
    except Exception:
        logger.debug("Neutralization failed for SMILES: {}", smiles)
        return smiles

    result = Chem.MolToSmiles(mol)
    return result or smiles


def _write_boltz_yaml(
    sequence: str,
    smiles: str,
    path: Path,
    msa_path: Path | None = None,
) -> None:
    """Write a Boltz-2 input YAML file for a single protein-ligand pair.

    Args:
        sequence: Protein amino-acid sequence.
        smiles: Ligand SMILES string.
        path: Destination YAML path (parent dirs created automatically).
        msa_path: Optional pre-computed MSA CSV; embedded as ``msa`` field
            in the protein entry.
    """
    import yaml

    protein_entry: dict = {
        "id": "A",
        "sequence": sequence,
    }
    if msa_path is not None:
        protein_entry["msa"] = str(msa_path.resolve())

    data = {
        "version": 1,
        "sequences": [
            {"protein": protein_entry},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def _boltz_env(device: str | None = None) -> dict[str, str]:
    """Build an environment dict for boltz subprocesses with GPU pinning.

    Args:
        device: Explicit CUDA device index (e.g. ``"0"``).  Falls back to
            the ``BOLTZ_CUDA_DEVICE`` environment variable when ``None``.

    Returns:
        Copy of ``os.environ`` with ``CUDA_VISIBLE_DEVICES`` set when a
        device is specified.
    """
    env = dict(os.environ)
    cuda_device = device if device is not None else os.getenv("BOLTZ_CUDA_DEVICE")
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.debug("Boltz GPU pinned to CUDA_VISIBLE_DEVICES={}", cuda_device)
    return env


def _run_boltz_predict(yaml_path: Path, out_dir: Path, *, device: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``boltz predict`` on a single YAML input via subprocess.

    Args:
        yaml_path: Path to the Boltz input YAML.
        out_dir: Directory for prediction outputs.
        device: Optional CUDA device index for GPU pinning.

    Returns:
        Completed subprocess result (check ``returncode``).

    Raises:
        subprocess.TimeoutExpired: If the process exceeds ``BOLTZ_TIMEOUT``.
    """
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        _resolve_accelerator(),
        *_BOLTZ_CLI_ARGS_BASE,
    ]
    logger.debug("Boltz CMD: {}", " ".join(cmd))

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=BOLTZ_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env=_boltz_env(device=device),
        check=False,
    )
    return result


def _parse_affinity(out_dir: Path, stem: str) -> float:
    """Parse ``affinity_pred_value`` from a Boltz JSON result.

    The value is log10(IC50 in micromolar); lower = stronger binding.

    Args:
        out_dir: Boltz output directory.
        stem: Input YAML stem used to locate the result subdirectory.

    Returns:
        Predicted affinity as a float.

    Raises:
        FileNotFoundError: If the affinity JSON is missing.
        KeyError: If ``affinity_pred_value`` is not in the JSON.
    """
    affinity_path = out_dir / f"boltz_results_{stem}" / "predictions" / stem / f"affinity_{stem}.json"

    if not affinity_path.exists():
        msg = f"Affinity JSON not found: {affinity_path}"
        raise FileNotFoundError(msg)

    data = json.loads(affinity_path.read_text())
    return float(data["affinity_pred_value"])


def _score_single_internal(
    target_id: str,
    smiles: str,
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None = None,
) -> float:
    """Score one molecule via an individual ``boltz predict`` call (no cache).

    Used as a retry fallback when the batch call fails for specific
    molecules.

    Args:
        target_id: Target identifier (for temp-dir naming).
        smiles: Neutralized SMILES string.
        sequence: Protein amino-acid sequence.
        msa_path: Pre-computed MSA CSV path, or ``None``.
        device: Optional CUDA device index.

    Returns:
        Predicted affinity (log10 IC50 uM), or ``NaN`` on failure.
    """
    with tempfile.TemporaryDirectory(prefix=f"boltz_retry_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        yaml_stem = "input"
        yaml_path = tmp_path / f"{yaml_stem}.yaml"
        out_dir = tmp_path / "output"

        _write_boltz_yaml(sequence, smiles, yaml_path, msa_path=msa_path)

        try:
            result = _run_boltz_predict(yaml_path, out_dir, device=device)
        except subprocess.TimeoutExpired:
            logger.debug("Boltz retry timeout for SMILES: {}", smiles)
            return float("nan")

        if result.returncode != 0:
            logger.debug("Boltz retry failed (rc={}) for SMILES: {}", result.returncode, smiles)
            return float("nan")

        try:
            return _parse_affinity(out_dir, yaml_stem)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return float("nan")


def _run_boltz_predict_dir(
    input_dir: Path,
    out_dir: Path,
    timeout: int = BOLTZ_BATCH_TIMEOUT,
    *,
    device: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``boltz predict`` on a directory of YAML input files.

    Logs key stderr lines (errors, progress) while filtering noisy
    deprecation warnings.

    Args:
        input_dir: Directory containing per-molecule YAML inputs.
        out_dir: Directory for prediction outputs.
        timeout: Maximum seconds for the batch subprocess.
        device: Optional CUDA device index for GPU pinning.

    Returns:
        Completed subprocess result.

    Raises:
        subprocess.TimeoutExpired: If the process exceeds *timeout*.
    """
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(input_dir),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        _resolve_accelerator(),
        *_BOLTZ_CLI_ARGS_BASE,
    ]
    logger.debug("Boltz batch CMD: {}", " ".join(cmd))

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=_boltz_env(device=device),
        check=False,
    )
    # Log key Boltz stderr lines (filter out warning noise)
    _boltz_noise = ("DeprecationWarning", "SyntaxWarning", "FutureWarning", "UserWarning", "warnings.warn")
    if result.stderr:
        for line in result.stderr.splitlines():
            if any(pat in line for pat in _boltz_noise):
                continue
            low = line.lower()
            if any(kw in low for kw in ("error", "running", "predict", "loaded", "sampling", "progress")):
                logger.info("[Boltz-2] {}", line.rstrip())
    return result


def _nim_health_check() -> bool:
    """Check whether the NIM Boltz-2 server is healthy and ready.

    Returns:
        ``True`` if at least one configured NIM URL responds as ready.
    """
    return any(_nim_url_ready(url) for url in _configured_nim_urls())


_NIM_MAX_RETRIES = 2  # total attempts = 1 + retries


def _nim_predict_affinity(sequence: str, smiles: str) -> float:
    """Score one molecule via the NIM REST API (synchronous).

    NIM returns two affinity values from two internal models; they are
    averaged.  Retries up to ``_NIM_MAX_RETRIES`` times on transient
    (5xx / network) errors; 4xx errors are not retried.  Each retry
    rotates to a *different* NIM URL to avoid hammering a dead instance.

    Args:
        sequence: Protein amino-acid sequence.
        smiles: Neutralized ligand SMILES.

    Returns:
        Predicted affinity as log10(IC50 in uM), or ``NaN`` on failure.
    """
    import time

    nim_urls = _resolve_nim_urls()
    n_urls = len(nim_urls)

    payload: dict = {
        "polymers": [{"id": "A", "molecule_type": "protein", "sequence": sequence}],
        "ligands": [{"id": "B", "smiles": smiles, "predict_affinity": True}],
        **_NIM_PREDICT_PARAMS,
    }

    data = json.dumps(payload).encode()

    # Build URL order: start with a random URL, then cycle through others.
    # This ensures retries always hit a different instance.
    import random
    start_idx = random.randrange(n_urls) if n_urls > 1 else 0  # noqa: S311
    url_order = [(start_idx + i) % n_urls for i in range(max(n_urls, _NIM_MAX_RETRIES + 1))]

    last_error: Exception | None = None
    for attempt in range(_NIM_MAX_RETRIES + 1):
        idx = url_order[attempt % len(url_order)]
        url = f"{nim_urls[idx]}{_NIM_PREDICT_ENDPOINT}"
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=_NIM_TIMEOUT) as resp:  # noqa: S310
                result = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500] if exc.fp else ""
            # 4xx errors are not retryable (bad input)
            if 400 <= exc.code < 500:
                logger.error("NIM HTTP {} for SMILES {}: {}", exc.code, smiles, body)
                return float("nan")
            last_error = exc
            logger.warning(
                "NIM HTTP {} for SMILES {} (attempt {}/{}, url={}): {}",
                exc.code,
                smiles,
                attempt + 1,
                _NIM_MAX_RETRIES + 1,
                nim_urls[idx],
                body,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning(
                "NIM request failed for SMILES {} (attempt {}/{}, url={}): {}",
                smiles, attempt + 1, _NIM_MAX_RETRIES + 1, nim_urls[idx], exc,
            )

        if attempt < _NIM_MAX_RETRIES:
            time.sleep(1.0 * (attempt + 1))  # linear backoff
    else:
        logger.error("NIM exhausted {} retries for SMILES {}: {}", _NIM_MAX_RETRIES + 1, smiles, last_error)
        return float("nan")

    try:
        affinities = result["affinities"]["B"]
        pred_values = affinities["affinity_pred_value"]
        # Two affinity models → average
        return float(np.mean(pred_values))
    except (KeyError, TypeError, IndexError) as exc:
        logger.error("Failed to parse NIM affinity for SMILES {}: {}", smiles, exc)
        return float("nan")


def _parse_nim_affinity(result: dict[str, Any], smiles: str) -> float:
    """Extract the averaged affinity score from a NIM response JSON.

    Args:
        result: Parsed JSON response from the NIM predict endpoint.
        smiles: SMILES string (used only for error logging).

    Returns:
        Mean of the two model affinity predictions, or ``NaN`` on parse
        failure.
    """
    try:
        affinities = result["affinities"]["B"]
        pred_values = affinities["affinity_pred_value"]
        return float(np.mean(pred_values))
    except (KeyError, TypeError, IndexError) as exc:
        logger.error("Failed to parse NIM affinity for SMILES {}: {}", smiles, exc)
        return float("nan")


async def _nim_predict_batch_async(
    sequence: str,
    smiles_indices: list[tuple[int, str]],
    *,
    nim_urls: list[str] | None = None,
) -> list[tuple[int, float]]:
    """Score molecules via NIM using async HTTP with round-robin.

    Sends concurrent requests across all available NIM URLs using
    ``aiohttp``.  Concurrency per NIM instance is bounded by
    ``_NIM_CONCURRENCY``.

    Args:
        sequence: Protein amino-acid sequence.
        smiles_indices: List of ``(original_index, neutralized_smiles)``
            pairs to score.
        nim_urls: Optional resolved NIM URL list. When ``None``, URLs are
            resolved from current module configuration.

    Returns:
        List of ``(original_index, score)`` tuples (NaN on failure).
    """
    import asyncio

    import aiohttp

    base_urls = nim_urls if nim_urls is not None else _resolve_nim_urls()
    n_urls = len(base_urls)
    urls = [f"{u}{_NIM_PREDICT_ENDPOINT}" for u in base_urls]
    # Semaphore per NIM instance to avoid overloading a single worker
    semaphores = [asyncio.Semaphore(_NIM_CONCURRENCY) for _ in range(n_urls)]
    timeout = aiohttp.ClientTimeout(total=_NIM_TIMEOUT)
    results: list[tuple[int, float]] = []

    async def _predict_one(
        session: aiohttp.ClientSession,
        idx: int,
        smi: str,
        url_idx: int,
    ) -> tuple[int, float]:
        """Score a single molecule against one NIM instance with retries.

        On failure, retries rotate to a *different* NIM URL so that a
        dead instance does not consume all retry attempts.
        """
        payload = {
            "polymers": [{"id": "A", "molecule_type": "protein", "sequence": sequence}],
            "ligands": [{"id": "B", "smiles": smi, "predict_affinity": True}],
            **_NIM_PREDICT_PARAMS,
        }

        last_error: Exception | None = None
        for attempt in range(_NIM_MAX_RETRIES + 1):
            # First attempt: use assigned URL. Retries: rotate to next URL.
            # With 2 URLs and 3 attempts: idx→other→idx (guarantees trying both).
            current_idx = url_idx if attempt == 0 else (url_idx + attempt) % n_urls
            async with semaphores[current_idx]:
                try:
                    async with session.post(urls[current_idx], json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return idx, _parse_nim_affinity(data, smi)
                        body = (await resp.text())[:500]
                        # 422 with "No GPU available" is a queue timeout — retryable
                        # True client errors (400, 404, etc.) are not retryable
                        if resp.status >= 400 and resp.status < 500 and "GPU available" not in body:
                            logger.error("NIM HTTP {} for SMILES {}: {}", resp.status, smi, body)
                            return idx, float("nan")
                        last_error = Exception(f"HTTP {resp.status}: {body}")
                        logger.warning(
                            "NIM HTTP {} for SMILES {} (attempt {}/{}, url={}): {}",
                            resp.status,
                            smi,
                            attempt + 1,
                            _NIM_MAX_RETRIES + 1,
                            base_urls[current_idx],
                            body[:100],
                        )
                except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                    last_error = exc
                    logger.warning(
                        "NIM async request failed for SMILES {} (attempt {}/{}, url={}): {}",
                        smi,
                        attempt + 1,
                        _NIM_MAX_RETRIES + 1,
                        base_urls[current_idx],
                        exc,
                    )

            if attempt < _NIM_MAX_RETRIES:
                await asyncio.sleep(1.0 * (attempt + 1))

        logger.error("NIM exhausted {} retries for SMILES {}: {}", _NIM_MAX_RETRIES + 1, smi, last_error)
        return idx, float("nan")

    connector = aiohttp.TCPConnector(limit=_NIM_CONCURRENCY * n_urls, limit_per_host=_NIM_CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for i, (idx, smi) in enumerate(smiles_indices):
            url_idx = i % n_urls  # round-robin across NIM instances
            tasks.append(_predict_one(session, idx, smi, url_idx))

        results = await asyncio.gather(*tasks)

    return list(results)


def _score_batch_internal_nim(
    target_id: str,  # noqa: ARG001
    smiles_list: list[str],
    sequence: str,
    msa_path: Path | None,  # noqa: ARG001
    *,
    device: str | None = None,  # noqa: ARG001
) -> list[float]:
    """Score a batch of molecules via the NIM backend (async HTTP).

    Validates and neutralizes SMILES, then sends concurrent requests
    across all NIM instances with round-robin load balancing.

    Args:
        target_id: Target identifier (unused by NIM; kept for API parity).
        smiles_list: Raw SMILES strings to score.
        sequence: Protein amino-acid sequence.
        msa_path: Unused by NIM (handled server-side).
        device: Unused by NIM (GPU managed by the server).

    Returns:
        List of float scores aligned with *smiles_list*; ``NaN`` for
        invalid or failed molecules.
    """
    import asyncio

    scores = [float("nan")] * len(smiles_list)

    valid_pairs: list[tuple[int, str]] = []
    for idx, smi in enumerate(smiles_list):
        if _validate_smiles(smi):
            valid_pairs.append((idx, _neutralize_smiles(smi)))
        else:
            logger.warning("Invalid SMILES or too large (>{} atoms): {}", MAX_LIGAND_ATOMS, smi)

    if not valid_pairs:
        return scores

    nim_urls = _resolve_nim_urls()
    logger.debug(
        "NIM async batch: {} molecules, {} NIM URL(s), concurrency={}",
        len(valid_pairs),
        len(nim_urls),
        _NIM_CONCURRENCY,
    )

    # Run async event loop from sync context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    results: list[tuple[int, float]]
    if loop is not None and loop.is_running():
        # Already inside an event loop (e.g. Jupyter) — use nest_asyncio or thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(asyncio.run, _nim_predict_batch_async(sequence, valid_pairs, nim_urls=nim_urls)).result()  # type: ignore[assignment]
    else:
        results = asyncio.run(_nim_predict_batch_async(sequence, valid_pairs, nim_urls=nim_urls))

    for idx, score in results:
        scores[idx] = score

    return scores


def _score_batch_internal(
    target_id: str,
    smiles_list: list[str],
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None = None,
) -> list[float]:
    """Score a batch of molecules via a single ``boltz predict`` invocation.

    Creates a temp directory of per-molecule YAML inputs and runs the CLI.
    SMILES are neutralized; partial results are recovered after timeouts;
    failed molecules are retried individually as a fallback.

    Delegates to :func:`_score_batch_internal_nim` when the NIM backend
    is active.

    Args:
        target_id: Target identifier.
        smiles_list: Raw SMILES strings to score.
        sequence: Protein amino-acid sequence.
        msa_path: Pre-computed MSA CSV path, or ``None``.
        device: Optional CUDA device index for GPU pinning.

    Returns:
        List of float scores aligned with *smiles_list*; ``NaN`` for
        invalid or failed molecules.
    """
    if _BOLTZ_BACKEND == "nim":
        return _score_batch_internal_nim(
            target_id,
            smiles_list,
            sequence,
            msa_path,
            device=device,
        )

    scores = [float("nan")] * len(smiles_list)

    # SMILES validation
    valid_indices: list[int] = []
    for idx, smi in enumerate(smiles_list):
        if _validate_smiles(smi):
            valid_indices.append(idx)
        else:
            logger.warning("Invalid SMILES or molecule exceeds {} atoms: {}", MAX_LIGAND_ATOMS, smi)

    if not valid_indices:
        return scores

    batch_failed = False

    with tempfile.TemporaryDirectory(prefix=f"boltz_batch_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "inputs"
        out_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Create YAML for each valid molecule (FIX 1: neutralize SMILES)
        stem_to_idx: dict[str, int] = {}
        for idx in valid_indices:
            smi_neutralized = _neutralize_smiles(smiles_list[idx])
            stem = f"mol_{idx:04d}"
            yaml_path = input_dir / f"{stem}.yaml"
            _write_boltz_yaml(sequence, smi_neutralized, yaml_path, msa_path=msa_path)
            stem_to_idx[stem] = idx

        # Single boltz predict call for directory
        timeout = BOLTZ_BATCH_TIMEOUT
        try:
            result = _run_boltz_predict_dir(input_dir, out_dir, timeout=timeout, device=device)
        except subprocess.TimeoutExpired:
            # FIX 3: don't return — fall through to parse partial results
            logger.error("Boltz batch timeout ({} sec) for {} molecules, parsing partial results", timeout, len(valid_indices))
            batch_failed = True
        else:
            if result.returncode != 0:
                stderr = result.stderr or ""
                if "CUDA out of memory" in stderr:
                    logger.error("Boltz batch GPU OOM for {} molecules", len(valid_indices))
                else:
                    logger.error(
                        "Boltz batch error (rc={}) for {} molecules\nstderr: {}",
                        result.returncode,
                        len(valid_indices),
                        stderr[-2000:],
                    )
                batch_failed = True

        # Parse results (always — even after timeout/error, for partial recovery)
        results_dir = out_dir / f"boltz_results_{input_dir.name}"
        parsed_ok = 0
        for stem, idx in stem_to_idx.items():
            affinity_path = results_dir / "predictions" / stem / f"affinity_{stem}.json"
            try:
                if not affinity_path.exists():
                    continue
                data = json.loads(affinity_path.read_text())
                scores[idx] = float(data["affinity_pred_value"])
                parsed_ok += 1
            except (KeyError, json.JSONDecodeError) as exc:
                logger.debug("Failed to parse affinity for {}: {}", stem, exc)

        if batch_failed:
            logger.info(
                "Batch partial recovery: {}/{} molecules scored from failed batch",
                parsed_ok,
                len(valid_indices),
            )

    # FIX 2: retry NaN molecules individually if batch had errors
    if batch_failed:
        failed_indices = [idx for idx in valid_indices if np.isnan(scores[idx])]
        if failed_indices:
            logger.info(
                "Batch fallback: retrying {} NaN molecules individually for {}",
                len(failed_indices),
                target_id,
            )
            for idx in failed_indices:
                smi_neutralized = _neutralize_smiles(smiles_list[idx])
                scores[idx] = _score_single_internal(
                    target_id,
                    smi_neutralized,
                    sequence,
                    msa_path,
                    device=device,
                )

    return scores


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
    # SMILES validation
    if not _validate_smiles(smiles):
        logger.warning("Invalid SMILES or molecule exceeds {} atoms: {}", MAX_LIGAND_ATOMS, smiles)
        return float("nan")

    # Load sequence
    sequence = _load_sequence(target_id)

    # NIM backend: no MSA/subprocess needed, just HTTP
    if _BOLTZ_BACKEND == "nim":
        smi_neutralized = _neutralize_smiles(smiles)
        return _nim_predict_affinity(sequence, smi_neutralized)

    # Find pre-computed MSA
    msa_path = _get_msa_path(target_id)
    if msa_path is None:
        msg = (
            f"Pre-computed MSA not found for {target_id}. "
            f"Run: uv run python scripts/s1_prepare/prepare_targets.py --pair <pair> --prepare-all"
        )
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory(prefix=f"boltz_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        yaml_stem = "input"
        yaml_path = tmp_path / f"{yaml_stem}.yaml"
        out_dir = tmp_path / "output"

        smiles_neutralized = _neutralize_smiles(smiles)
        _write_boltz_yaml(sequence, smiles_neutralized, yaml_path, msa_path=msa_path)

        try:
            result = _run_boltz_predict(yaml_path, out_dir)
        except subprocess.TimeoutExpired:
            logger.error("Boltz timeout ({} sec) for SMILES: {}", BOLTZ_TIMEOUT, smiles)
            return float("nan")

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "CUDA out of memory" in stderr:
                logger.error("Boltz GPU OOM for SMILES: {}", smiles)
            else:
                logger.error(
                    "Boltz error (rc={}) for SMILES: {}\nstderr: {}",
                    result.returncode,
                    smiles,
                    stderr[-2000:],
                )
            return float("nan")

        try:
            log_ic50_um = _parse_affinity(out_dir, yaml_stem)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse affinity for SMILES {}: {}", smiles, exc)
            return float("nan")

    # Convention: log10(IC50 in uM), lower = better
    return log_ic50_um


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
    from tqdm import tqdm

    # Load sequence and MSA once
    sequence = _load_sequence(target_id)
    msa_path = _get_msa_path(target_id)
    if msa_path is None and _BOLTZ_BACKEND != "nim":
        msg = (
            f"Pre-computed MSA not found for {target_id}. "
            f"Run: uv run python scripts/s1_prepare/prepare_targets.py --pair <pair> --prepare-all"
        )
        raise FileNotFoundError(msg)

    # Compute per-target hashes fresh (no stale data risk)
    seq_hash = _sequence_content_hash(sequence)
    msa_hash = _msa_content_hash(msa_path)
    ctx = _scoring_context_hash()

    cache = _load_cache(target_id) if use_cache else {}
    scores: list[float] = [float("nan")] * len(smiles_list)
    to_score: list[tuple[int, str]] = []

    # Check cache
    for idx, smi in enumerate(smiles_list):
        key = _cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
        if key in cache:
            scores[idx] = cache[key]
        else:
            to_score.append((idx, smi))

    if not to_score:
        logger.info("All {} molecules for target {} found in cache", len(smiles_list), target_id)
        return np.array(scores, dtype=np.float64)

    logger.info(
        "Scoring {} new molecules for target {} ({} total, batch_size={})",
        len(to_score),
        target_id,
        len(smiles_list),
        batch_size,
    )

    # Group into batches
    num_batches = (len(to_score) + batch_size - 1) // batch_size
    scoring_start = time.monotonic()
    scored_so_far = 0

    for batch_idx in tqdm(range(num_batches), desc=f"Boltz-2 [{target_id}]", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(to_score))
        batch = to_score[start:end]

        # Extract SMILES for batch
        batch_smiles = [smi for _, smi in batch]

        batch_t0 = time.monotonic()
        # Score batch in a single boltz predict call
        batch_scores = _score_batch_internal(target_id, batch_smiles, sequence, msa_path, device=device)
        batch_elapsed = time.monotonic() - batch_t0

        # Record results
        batch_ok = 0
        for i, (idx, smi) in enumerate(batch):
            s = batch_scores[i]
            scores[idx] = s
            if not np.isnan(s):
                key = _cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
                cache[key] = s
                batch_ok += 1

        scored_so_far += len(batch)
        total_elapsed = time.monotonic() - scoring_start
        mol_per_sec = scored_so_far / total_elapsed if total_elapsed > 0 else 0
        remaining_mol = len(to_score) - scored_so_far
        eta_min = (remaining_mol / mol_per_sec / 60) if mol_per_sec > 0 else 0
        logger.info(
            "[Boltz-2] [{}] Batch {}/{} — {}/{} scored, {:.0f}s, {:.2f} mol/s, ETA ~{:.0f}m",
            target_id,
            batch_idx + 1,
            num_batches,
            batch_ok,
            len(batch),
            batch_elapsed,
            mol_per_sec,
            eta_min,
        )

        # Save cache after each batch (not after each molecule)
        if use_cache:
            _save_cache(target_id, cache)

    return np.array(scores, dtype=np.float64)
