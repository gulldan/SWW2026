"""Обёртка скоринга Boltz-2.

Предоставляет ``score_batch(target, smiles_list)`` с дисковым кешем.
Поддерживает два бэкенда:
- ``cli`` (по умолчанию): вызывает ``boltz predict`` через subprocess
- ``nim``: вызывает NVIDIA NIM Boltz-2 REST API (persistent GPU server)

Выбор бэкенда: ``BOLTZ_BACKEND=nim`` (env var).
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
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from dualdocking.scoring.cache_io import load_cache, save_cache

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data" / "targets"
MSA_CACHE_DIR = CACHE_DIR / "boltz_msa"

BOLTZ_TIMEOUT = 600  # секунд на одну молекулу
BOLTZ_BATCH_TIMEOUT = 3600  # секунд на батч
MAX_LIGAND_ATOMS = 128
_DEFAULT_BATCH_SIZE = 64  # молекул за один вызов boltz predict (internal batch=1, safe on any VRAM)


def _auto_batch_size() -> int:
    """Pick batch size based on GPU VRAM; override via BOLTZ_BATCH_SIZE env var."""
    env = os.getenv("BOLTZ_BATCH_SIZE")
    if env is not None:
        return int(env)

    vram_gb: float | None = None

    # Attempt 1: torch.cuda (fast, in-process)
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        pass

    # Attempt 2: nvidia-smi (works even without CUDA-enabled torch)
    if vram_gb is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_gb = float(result.stdout.strip().split("\n")[0]) / 1024
        except Exception:
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

# Boltz CLI параметры
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

# NIM prediction params — mirrors _BOLTZ_CLI_ARGS_BASE values.
# step_scale=1.5: boltz2 model default (CLI uses 1.5; NIM defaults to 1.638 if unset).
_NIM_PREDICT_PARAMS: dict = {
    "recycling_steps": 3,
    "sampling_steps": 50,
    "diffusion_samples": 1,
    "sampling_steps_affinity": 50,
    "diffusion_samples_affinity": 1,
    "step_scale": 1.5,
    "output_format": "mmcif",
}


def _resolve_accelerator() -> str:
    """Pick Boltz accelerator; GPU is required unless explicitly overridden."""
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
    """Return installed Boltz version with fallback to master_config.yaml."""
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
    """Read a dotted path from configs/master_config.yaml."""
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
    """Run a command, return stripped stdout or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _docker_digest() -> str:
    """Return FREED++ Docker image digest with fallback chain."""
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
    """SHA256[:16] of MSA file contents, or ``'no_msa'``."""
    if msa_path is None:
        return "no_msa"
    return hashlib.sha256(msa_path.read_bytes()).hexdigest()[:16]


def _sequence_content_hash(sequence: str) -> str:
    """SHA256[:16] of protein sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def _scoring_context_hash() -> str:
    """Compute (and cache) the global scoring context hash."""
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
    """Public accessor for the scoring context hash."""
    return _scoring_context_hash()


def get_scoring_context_info() -> dict:
    """Return all scoring context components (for metadata persistence)."""
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
    """Remove cache files not matching the current context hash prefix.

    Returns the count of removed files.
    """
    ctx_prefix = _scoring_context_hash()[:8]
    removed = 0
    for f in CACHE_DIR.glob("*.json"):
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

# Кеш sequence в памяти (target_id -> sequence)
_sequence_cache: dict[str, str] = {}


def _cache_key(
    target_id: str,
    smi: str,
    *,
    ctx_hash: str = "",
    seq_hash: str = "",
    msa_hash: str = "",
) -> str:
    blob = f"{target_id}||{smi}||{ctx_hash}||{seq_hash}||{msa_hash}".encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_path(target_id: str) -> Path:
    """Return context-segmented cache file path."""
    return CACHE_DIR / f"{target_id}.{_scoring_context_hash()[:8]}.json"


def _load_cache(target_id: str) -> dict[str, float]:
    return load_cache(_cache_path(target_id))


def _save_cache(target_id: str, cache: dict[str, float]) -> None:
    save_cache(_cache_path(target_id), cache)


def _load_sequence(target_id: str) -> str:
    """Читает FASTA из data/targets/{target_id}.fasta, кеширует в памяти."""
    if target_id in _sequence_cache:
        return _sequence_cache[target_id]

    fasta_path = DATA_DIR / f"{target_id}.fasta"
    if not fasta_path.exists():
        msg = f"FASTA не найден: {fasta_path}. Запустите: uv run python scripts/prep/prepare_targets.py --prepare-all"
        raise FileNotFoundError(msg)

    lines = fasta_path.read_text().strip().splitlines()
    # Пропустить заголовки FASTA (строки начинающиеся с >)
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    _sequence_cache[target_id] = seq
    return seq


def _get_msa_path(target_id: str) -> Path | None:
    """Ищет pre-computed MSA в cache/boltz_msa/{target_id}.csv."""
    msa_path = MSA_CACHE_DIR / f"{target_id}.csv"
    if msa_path.exists():
        return msa_path
    return None


def _validate_smiles(smiles: str) -> bool:
    """Проверяет SMILES через RDKit: парсинг + <= MAX_LIGAND_ATOMS атомов."""
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit недоступен, пропускаем валидацию SMILES")
        return True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= MAX_LIGAND_ATOMS


def _neutralize_smiles(smiles: str) -> str:
    """Neutralize formal charges for Boltz-2 compatibility.

    Boltz-2 v2.2.1 crashes on charged species ([NH2+], [O-], etc.).
    Strips protonation states while preserving molecular identity.
    Returns original SMILES if neutralization fails.
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
    """Генерирует входной YAML для Boltz predict."""
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
    """Build env dict for boltz subprocess, applying GPU pinning if configured.

    Parameters
    ----------
    device:
        Explicit CUDA device index (e.g. ``"0"``, ``"1"``).  When *None*,
        falls back to the ``BOLTZ_CUDA_DEVICE`` environment variable.
    """
    env = dict(os.environ)
    cuda_device = device if device is not None else os.getenv("BOLTZ_CUDA_DEVICE")
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.debug("Boltz GPU pinned to CUDA_VISIBLE_DEVICES={}", cuda_device)
    return env


def _run_boltz_predict(yaml_path: Path, out_dir: Path, *, device: str | None = None) -> subprocess.CompletedProcess[str]:
    """Запуск boltz predict через subprocess."""
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
    """Парсит affinity_pred_value из JSON результата Boltz.

    Boltz-2 возвращает log10(IC50 в микромолях), т.е. меньшие значения = сильнее связывание.
    Путь: {out_dir}/boltz_results_{stem}/predictions/{stem}/affinity_{stem}.json
    """
    affinity_path = out_dir / f"boltz_results_{stem}" / "predictions" / stem / f"affinity_{stem}.json"

    if not affinity_path.exists():
        msg = f"Affinity JSON не найден: {affinity_path}"
        raise FileNotFoundError(msg)

    data = json.loads(affinity_path.read_text())
    return float(data["affinity_pred_value"])


def _score_single_internal(
    target_id: str,
    smiles: str,
    sequence: str,
    msa_path: Path,
    *,
    device: str | None = None,
) -> float:
    """Score one molecule via individual boltz predict call (no cache)."""
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
    """Запуск boltz predict на директории YAML файлов."""
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=_boltz_env(device=device),
        check=False,
    )
    # Показываем ключевые строки Boltz stderr в логе (без шума warnings)
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
    """Check if NIM Boltz-2 server is ready."""
    url = f"{_BOLTZ_NIM_URLS[0].rstrip('/')}{_NIM_HEALTH_ENDPOINT}"
    try:
        req = urllib.request.Request(url)  # noqa: S310
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return b"true" in resp.read().lower()
    except Exception:
        return False


_NIM_MAX_RETRIES = 2  # total attempts = 1 + retries


def _nim_predict_affinity(sequence: str, smiles: str) -> float:
    """Score one molecule via NIM REST API (sync, for single-molecule use).

    Returns log10(IC50 in μM), or NaN on failure.
    NIM returns two values from two affinity models — we average them.
    Retries up to _NIM_MAX_RETRIES times on transient errors.
    """
    import time

    url = f"{_BOLTZ_NIM_URLS[0].rstrip('/')}{_NIM_PREDICT_ENDPOINT}"

    payload: dict = {
        "polymers": [{"id": "A", "molecule_type": "protein", "sequence": sequence}],
        "ligands": [{"id": "B", "smiles": smiles, "predict_affinity": True}],
        **_NIM_PREDICT_PARAMS,
    }

    data = json.dumps(payload).encode()

    last_error: Exception | None = None
    for attempt in range(_NIM_MAX_RETRIES + 1):
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
                "NIM HTTP {} for SMILES {} (attempt {}/{}): {}",
                exc.code,
                smiles,
                attempt + 1,
                _NIM_MAX_RETRIES + 1,
                body,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning("NIM request failed for SMILES {} (attempt {}/{}): {}", smiles, attempt + 1, _NIM_MAX_RETRIES + 1, exc)

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


def _parse_nim_affinity(result: dict, smiles: str) -> float:
    """Extract affinity score from NIM response JSON."""
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
) -> list[tuple[int, float]]:
    """Score molecules via NIM using async HTTP with round-robin across NIM instances.

    Sends concurrent requests to all available NIM URLs using aiohttp.
    Concurrency is bounded by ``_NIM_CONCURRENCY`` semaphore per NIM instance.
    """
    import asyncio

    import aiohttp

    n_urls = len(_BOLTZ_NIM_URLS)
    urls = [f"{u.rstrip('/')}{_NIM_PREDICT_ENDPOINT}" for u in _BOLTZ_NIM_URLS]
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
        payload = {
            "polymers": [{"id": "A", "molecule_type": "protein", "sequence": sequence}],
            "ligands": [{"id": "B", "smiles": smi, "predict_affinity": True}],
            **_NIM_PREDICT_PARAMS,
        }

        last_error: Exception | None = None
        for attempt in range(_NIM_MAX_RETRIES + 1):
            async with semaphores[url_idx]:
                try:
                    async with session.post(urls[url_idx], json=payload) as resp:
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
                            "NIM HTTP {} for SMILES {} (attempt {}/{}): {}",
                            resp.status,
                            smi,
                            attempt + 1,
                            _NIM_MAX_RETRIES + 1,
                            body[:100],
                        )
                except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                    last_error = exc
                    logger.warning(
                        "NIM async request failed for SMILES {} (attempt {}/{}): {}",
                        smi,
                        attempt + 1,
                        _NIM_MAX_RETRIES + 1,
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
    """Score a batch via NIM using async HTTP requests with round-robin.

    Sends concurrent requests to all available NIM instances (``BOLTZ_NIM_URLS``).
    Each NIM has a concurrency semaphore to prevent overloading.
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

    logger.debug(
        "NIM async batch: {} molecules, {} NIM URL(s), concurrency={}",
        len(valid_pairs),
        len(_BOLTZ_NIM_URLS),
        _NIM_CONCURRENCY,
    )

    # Run async event loop from sync context
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Already inside an event loop (e.g. Jupyter) — use nest_asyncio or thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(asyncio.run, _nim_predict_batch_async(sequence, valid_pairs)).result()
    else:
        results = asyncio.run(_nim_predict_batch_async(sequence, valid_pairs))

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
    """Скорить батч молекул за один вызов boltz predict.

    Создаёт директорию с YAML файлами и запускает boltz predict на ней.
    Модель загружается один раз, что значительно быстрее.

    Fixes applied:
    - SMILES are neutralized before scoring (Boltz-2 crashes on charged species)
    - Partial results are parsed even after timeout (not just on rc!=0)
    - Failed molecules are retried individually as fallback
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

    # Валидация SMILES
    valid_indices: list[int] = []
    for idx, smi in enumerate(smiles_list):
        if _validate_smiles(smi):
            valid_indices.append(idx)
        else:
            logger.warning("Невалидный SMILES или слишком большая молекула (>{} атомов): {}", MAX_LIGAND_ATOMS, smi)

    if not valid_indices:
        return scores

    batch_failed = False

    with tempfile.TemporaryDirectory(prefix=f"boltz_batch_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "inputs"
        out_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Создаём YAML для каждой валидной молекулы (FIX 1: neutralize SMILES)
        stem_to_idx: dict[str, int] = {}
        for idx in valid_indices:
            smi_neutralized = _neutralize_smiles(smiles_list[idx])
            stem = f"mol_{idx:04d}"
            yaml_path = input_dir / f"{stem}.yaml"
            _write_boltz_yaml(sequence, smi_neutralized, yaml_path, msa_path=msa_path)
            stem_to_idx[stem] = idx

        # Один вызов boltz predict на директорию
        timeout = BOLTZ_BATCH_TIMEOUT
        try:
            result = _run_boltz_predict_dir(input_dir, out_dir, timeout=timeout, device=device)
        except subprocess.TimeoutExpired:
            # FIX 3: don't return — fall through to parse partial results
            logger.error("Boltz batch таймаут ({} сек) для {} молекул, парсим partial results", timeout, len(valid_indices))
            batch_failed = True
        else:
            if result.returncode != 0:
                stderr = result.stderr or ""
                if "CUDA out of memory" in stderr:
                    logger.error("Boltz batch OOM GPU для {} молекул", len(valid_indices))
                else:
                    logger.error(
                        "Boltz batch ошибка (rc={}) для {} молекул\nstderr: {}",
                        result.returncode,
                        len(valid_indices),
                        stderr[-2000:],
                    )
                batch_failed = True

        # Парсим результаты (всегда — даже после timeout/error, для partial recovery)
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
                logger.debug("Не удалось распарсить affinity для {}: {}", stem, exc)

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
    """Оценить одну молекулу по мишени через Boltz-2.

    Returns:
    -------
    float
        log10(IC50 в микромолях); меньше = лучше.
        NaN при ошибке.
    """
    # Валидация SMILES
    if not _validate_smiles(smiles):
        logger.warning("Невалидный SMILES или слишком большая молекула (>{} атомов): {}", MAX_LIGAND_ATOMS, smiles)
        return float("nan")

    # Загрузить sequence
    sequence = _load_sequence(target_id)

    # NIM backend: no MSA/subprocess needed, just HTTP
    if _BOLTZ_BACKEND == "nim":
        smi_neutralized = _neutralize_smiles(smiles)
        return _nim_predict_affinity(sequence, smi_neutralized)

    # Найти pre-computed MSA
    msa_path = _get_msa_path(target_id)
    if msa_path is None:
        msg = (
            f"Pre-computed MSA не найден для {target_id}. "
            f"Запустите: uv run python scripts/prep/prepare_targets.py --pair <pair> --prepare-all"
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
            logger.error("Boltz таймаут ({} сек) для SMILES: {}", BOLTZ_TIMEOUT, smiles)
            return float("nan")

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "CUDA out of memory" in stderr:
                logger.error("Boltz OOM GPU для SMILES: {}", smiles)
            else:
                logger.error(
                    "Boltz ошибка (rc={}) для SMILES: {}\nstderr: {}",
                    result.returncode,
                    smiles,
                    stderr[-2000:],
                )
            return float("nan")

        try:
            log_ic50_um = _parse_affinity(out_dir, yaml_stem)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Не удалось распарсить affinity для SMILES {}: {}", smiles, exc)
            return float("nan")

    # Конвенция: log10(IC50 в микромолях), меньше = лучше
    return log_ic50_um


def score_batch(
    target_id: str,
    smiles_list: list[str],
    *,
    device: str | None = None,
    use_cache: bool = True,
    batch_size: int = BATCH_SIZE,
) -> NDArray[np.float64]:
    """Оценить батч SMILES по мишени *target_id*.

    Скоры кешируются по паре (target, smiles) на диске.
    Молекулы группируются в батчи и обрабатываются за один вызов boltz predict,
    что значительно ускоряет работу (модель загружается один раз на батч).

    Parameters
    ----------
    target_id:
        Идентификатор, соответствующий ключу в ``configs/pairs.yaml``.
    smiles_list:
        Канонические SMILES-строки.
    use_cache:
        Если True, пропускать уже оценённые молекулы.
    batch_size:
        Количество молекул в одном батче (по умолчанию BATCH_SIZE).

    Returns:
    -------
    scores:
        1-D массив скоров (log10(IC50 в μM)), в том же порядке, что и *smiles_list*.
    """
    from tqdm import tqdm

    # Загрузить sequence и MSA один раз
    sequence = _load_sequence(target_id)
    msa_path = _get_msa_path(target_id)
    if msa_path is None and _BOLTZ_BACKEND != "nim":
        msg = (
            f"Pre-computed MSA не найден для {target_id}. "
            f"Запустите: uv run python scripts/prep/prepare_targets.py --pair <pair> --prepare-all"
        )
        raise FileNotFoundError(msg)

    # Compute per-target hashes fresh (no stale data risk)
    seq_hash = _sequence_content_hash(sequence)
    msa_hash = _msa_content_hash(msa_path)
    ctx = _scoring_context_hash()

    cache = _load_cache(target_id) if use_cache else {}
    scores: list[float] = [float("nan")] * len(smiles_list)
    to_score: list[tuple[int, str]] = []

    # Проверяем кеш
    for idx, smi in enumerate(smiles_list):
        key = _cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
        if key in cache:
            scores[idx] = cache[key]
        else:
            to_score.append((idx, smi))

    if not to_score:
        logger.info("Все {} молекул для мишени {} найдены в кеше", len(smiles_list), target_id)
        return np.array(scores, dtype=np.float64)

    logger.info(
        "Скоринг {} новых молекул для мишени {} (из {} всего, batch_size={})",
        len(to_score),
        target_id,
        len(smiles_list),
        batch_size,
    )

    # Группируем в батчи
    num_batches = (len(to_score) + batch_size - 1) // batch_size
    scoring_start = time.monotonic()
    scored_so_far = 0

    for batch_idx in tqdm(range(num_batches), desc=f"Boltz-2 [{target_id}]", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(to_score))
        batch = to_score[start:end]

        # Извлекаем SMILES для батча
        batch_smiles = [smi for _, smi in batch]

        batch_t0 = time.monotonic()
        # Скорим батч за один вызов boltz predict
        batch_scores = _score_batch_internal(target_id, batch_smiles, sequence, msa_path, device=device)
        batch_elapsed = time.monotonic() - batch_t0

        # Записываем результаты
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

        # Сохраняем кеш после каждого батча (не после каждой молекулы)
        if use_cache:
            _save_cache(target_id, cache)

    return np.array(scores, dtype=np.float64)
