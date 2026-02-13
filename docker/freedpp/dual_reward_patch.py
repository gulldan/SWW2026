"""Dual-target reward patch for FREED++.

This patch does three things:
1) replaces ligand preparation backend (OpenBabel) with RDKit+Meeko,
2) applies z-score based rewards for single_a / single_b / double modes,
3) uses UniDock GPU batch docking (``--gpu_batch``) instead of per-ligand CPU calls.

Box parameters for the second receptor are read from ``/workspace/dual_config.json``::

    {"receptor_b": "/workspace/receptor_b.pdbqt", "center_b": "12.3,4.5,6.7", "size_b": "20,20,20"}
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
from contextlib import suppress
from multiprocessing import Pool as MPPool
from pathlib import Path
from subprocess import TimeoutExpired, run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

logger = logging.getLogger(__name__)

_MAX_Z_WARNINGS = 5
_z_warn_count = {"a": 0, "b": 0}
_err_warn_count = {"a": 0, "b": 0}
_RESULT_PATTERN = re.compile(r"RESULT[:\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# GPU batch docking (Phase 2 optimisation)
# ---------------------------------------------------------------------------
# When enabled, CacheAndPool for DockingScore pre-computes all docking scores
# via a single ``unidock --gpu_batch`` call instead of N×num_sub_proc CPU calls.
_GPU_BATCH_ENABLED = os.environ.get("GPU_BATCH_DOCK", "1") != "0"
_gpu_dock_cache: dict[tuple[str, str], float] = {}  # (receptor_path, smiles) → score
_gpu_batch_log_count = 0


def _prepare_pdbqt_worker(args: tuple[str, int, str]) -> bool:
    """Worker for parallel ligand preparation via multiprocessing.Pool.map."""
    smiles, seed, outpath = args
    return _prepare_ligand_rdkit(smiles, outpath, seed)


def _run_gpu_batch(
    vina_program: str,
    receptor: str,
    center: list[float],
    size: list[float],
    pdbqt_paths: list[str],
    output_dir: str,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    timeout: int = 600,
) -> bool:
    """Run UniDock --gpu_batch for a list of prepared PDBQT files.

    Returns True on success, False on failure.
    """
    cmd = [
        vina_program,
        "--receptor",
        receptor,
        "--gpu_batch",
        *pdbqt_paths,
        "--dir",
        output_dir,
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--seed",
        str(seed),
    ]
    try:
        result = run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("gpu_batch_dock: unidock rc=%s: %s", result.returncode, (result.stderr or "")[-500:])
            return False
        return True
    except TimeoutExpired:
        logger.warning("gpu_batch_dock: timed out after %ss", timeout)
        return False


def _precompute_gpu_batch_for_receptor(
    unseen_smiles: list[str],
    config: dict[str, Any],
    receptor: str,
    center: Any,
    size: Any,
) -> None:
    """Pre-compute docking scores for *unseen_smiles* against one receptor.

    Results are stored in ``_gpu_dock_cache[(receptor, smi)]``.
    """
    global _gpu_batch_log_count
    error_val = config["error_val"]
    seed = config["seed"]

    center_f = _coords(center) if isinstance(center, str) else [float(x) for x in center]
    size_f = _coords(size) if isinstance(size, str) else [float(x) for x in size]

    with TemporaryDirectory(prefix="gpu_batch_") as tmpdir:
        lig_dir = Path(tmpdir) / "ligands"
        out_dir = Path(tmpdir) / "output"
        lig_dir.mkdir()
        out_dir.mkdir()

        # 1 conformer per SMILES (n_conf uses identical seed → identical result)
        tasks = []
        for i, smi in enumerate(unseen_smiles):
            fname = f"lig_{i:06d}.pdbqt"
            tasks.append((smi, seed, str(lig_dir / fname)))

        # CPU-parallel ligand preparation (RDKit + Meeko)
        n_workers = min(len(tasks), os.cpu_count() or 8)
        with MPPool(processes=n_workers) as pool:
            prep_ok = pool.map(_prepare_pdbqt_worker, tasks)

        valid_items = [(i, tasks[i]) for i in range(len(tasks)) if prep_ok[i]]
        valid_paths = [t[2] for _, t in valid_items]

        # Mark failed preps as error_val
        for i in range(len(tasks)):
            if not prep_ok[i]:
                _gpu_dock_cache[(receptor, tasks[i][0])] = error_val

        if not valid_paths:
            return

        ok = _run_gpu_batch(
            config["vina_program"],
            receptor,
            center_f,
            size_f,
            valid_paths,
            str(out_dir),
            config["exhaustiveness"],
            config["num_modes"],
            seed,
        )

        if ok:
            n_cached = 0
            for _, (smi, _, fpath) in valid_items:
                stem = Path(fpath).stem
                out_path = out_dir / f"{stem}_out.pdbqt"
                parsed = _parse_result_from_pdbqt(out_path)
                if parsed is not None:
                    _gpu_dock_cache[(receptor, smi)] = parsed
                    n_cached += 1
                # Not cached: _rdkit_docking falls through to CPU
            n_corrupt = len(valid_items) - n_cached
            if _gpu_batch_log_count < 5:
                _gpu_batch_log_count += 1
                logger.info(
                    "gpu_batch_dock: %d/%d ligands scored (GPU), %d corrupt → CPU fallback, receptor %s",
                    n_cached,
                    len(tasks),
                    n_corrupt,
                    Path(receptor).stem,
                )
            if n_corrupt > 0:
                logger.warning(
                    "gpu_batch_dock: %d/%d corrupt scores for %s",
                    n_corrupt,
                    len(valid_items),
                    Path(receptor).stem,
                )
        else:
            logger.warning("gpu_batch_dock: batch failed for %s, falling back to per-ligand CPU", Path(receptor).stem)


# ---------------------------------------------------------------------------
# Invalid-docking exclusion (ENG-01 / KVA-255)
# ---------------------------------------------------------------------------

_INVALID_Z_PENALTY = float(os.environ.get("INVALID_Z_PENALTY", "4.0"))
if not (0 < _INVALID_Z_PENALTY < float("inf")):
    raise ValueError(f"INVALID_Z_PENALTY must be a positive finite number, got {_INVALID_Z_PENALTY}")

_MAX_Z_ABS = float(os.environ.get("MAX_Z_ABS", "10.0"))
if not (0 < _MAX_Z_ABS < float("inf")):
    raise ValueError(f"MAX_Z_ABS must be a positive finite number, got {_MAX_Z_ABS}")

_call_stats: dict[str, int] = {
    "total": 0,
    "invalid_a": 0,
    "invalid_b": 0,
    "timeout_a": 0,
    "timeout_b": 0,
}

# Thread-local side-channel: _rdkit_docking sets timeout status for the current
# thread, _dual_call consumes and resets it after each _original_call.
_timeout_state = threading.local()


def _set_last_docking_timed_out(value: bool) -> None:
    _timeout_state.last_docking_timed_out = value


def _consume_last_docking_timed_out() -> bool:
    value = bool(getattr(_timeout_state, "last_docking_timed_out", False))
    _timeout_state.last_docking_timed_out = False
    return value


def _get_last_docking_timed_out() -> bool:
    return bool(getattr(_timeout_state, "last_docking_timed_out", False))


_METADATA_PATH = Path(os.environ.get("REWARD_METADATA_PATH", "/workspace/reward_metadata.json"))


def _capped_zscore(
    score: float,
    error_val: float,
    mean: float,
    std: float,
    penalty_cap: float,
) -> tuple[float, bool]:
    """Compute z-score, capping at *penalty_cap* when *score* equals *error_val*.

    Returns (z_value, is_invalid).
    """
    if score == error_val:
        return penalty_cap, True
    if std == 0:
        raise ZeroDivisionError("std=0 in normalization.yaml")
    z = (score - mean) / std
    z = max(-_MAX_Z_ABS, min(_MAX_Z_ABS, z))
    return z, False


def _write_reward_metadata() -> None:
    """Write invalid-reward statistics to JSON (registered via atexit)."""
    total = _call_stats["total"]
    if total == 0:
        return
    reward_mode = os.environ.get("REWARD_MODE", "single_a")
    target_id_a = os.environ.get("TARGET_ID_A", "")
    target_id_b = os.environ.get("TARGET_ID_B", "")
    metadata = {
        "total_calls": total,
        "invalid_count_a": _call_stats["invalid_a"],
        "invalid_count_b": _call_stats["invalid_b"],
        "invalid_reward_fraction_a": _call_stats["invalid_a"] / total,
        "invalid_reward_fraction_b": _call_stats["invalid_b"] / total,
        "timeout_count_a": _call_stats["timeout_a"],
        "timeout_count_b": _call_stats["timeout_b"],
        "timeout_fraction_a": _call_stats["timeout_a"] / total,
        "timeout_fraction_b": _call_stats["timeout_b"] / total,
        "invalid_z_penalty": _INVALID_Z_PENALTY,
        "max_z_abs": _MAX_Z_ABS,
        "primary_target": target_id_b if reward_mode == "single_b" else target_id_a,
        "reward_mode": reward_mode,
    }
    max_inv = float(os.environ.get("QG_MAX_INVALID_FRACTION", "0.10"))
    max_to = float(os.environ.get("QG_MAX_TIMEOUT_FRACTION", "0.05"))
    inv_ok = all(metadata[f"invalid_reward_fraction_{label}"] <= max_inv for label in ("a", "b"))
    to_ok = all(metadata[f"timeout_fraction_{label}"] <= max_to for label in ("a", "b"))
    metadata["run_valid"] = inv_ok and to_ok

    try:
        _METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _METADATA_PATH.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("dual_reward_patch: reward metadata written to %s", _METADATA_PATH)
    except OSError as exc:
        logger.warning("dual_reward_patch: failed to write reward metadata: %s", exc)


def _load_normalization(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("PyYAML required to load normalization.yaml") from exc

    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("normalization.yaml must be a dictionary")
    return data


def _coords(value: Any) -> list[float]:
    parts = [float(x) for x in value.split(",")] if isinstance(value, str) else [float(x) for x in value]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 coordinates, got: {value}")
    return parts


def _prepare_ligand_rdkit(smiles: str, out_pdbqt: str, seed: int) -> bool:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:
        logger.error("dual_reward_patch: RDKit/Meeko unavailable: %s", exc)
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) if int(seed) >= 0 else 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False

    # Optimization can fail for some structures; embedded conformer is still usable.
    with suppress(Exception):
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        return False
    pdbqt_text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        logger.debug("dual_reward_patch: Meeko write failed: %s", err)
        return False

    with Path(out_pdbqt).open("w", encoding="utf-8") as f:
        f.write(pdbqt_text)
    return True


_SCORE_MIN, _SCORE_MAX = -20.0, 10.0


def _parse_result_from_pdbqt(path: Path) -> float | None:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if "RESULT" not in line.upper():
                    continue
                match = _RESULT_PATTERN.search(line)
                if match:
                    score = float(match.group(1))
                    if not (_SCORE_MIN <= score <= _SCORE_MAX):
                        logger.warning(
                            "corrupt docking score %.3f outside [%.0f, %.0f]: %s",
                            score,
                            _SCORE_MIN,
                            _SCORE_MAX,
                            path,
                        )
                        return None
                    return score
    except OSError:
        return None
    return None


def _apply_dual_patch() -> None:
    """Monkey-patch DockingVina to support single_a/single_b/double z-scored rewards."""
    reward_mode = os.environ.get("REWARD_MODE", "single_a")
    normalization_path = os.environ.get("NORMALIZATION_PATH", "/workspace/normalization.yaml")
    target_id_a = os.environ.get("TARGET_ID_A") or ""
    target_id_b = os.environ.get("TARGET_ID_B") or ""

    normalization = _load_normalization(normalization_path)
    targets = normalization.get("targets", {})

    if target_id_a not in targets:
        raise KeyError(f"Target A '{target_id_a}' not found in normalization.yaml")

    mean_a = float(targets[target_id_a]["mean"])
    std_a = float(targets[target_id_a]["std"])

    dual_receptor_b = os.environ.get("DUAL_RECEPTOR_B")

    # Load mean_b/std_b as closure variables for both single_b and double
    mean_b = std_b = 0.0  # sentinels; only used when mode needs B
    if reward_mode in {"single_b", "double", "linear_equal", "biased_a", "selective_a"}:
        if target_id_b not in targets:
            raise KeyError(f"Target B '{target_id_b}' not found in normalization.yaml")
        mean_b = float(targets[target_id_b]["mean"])
        std_b = float(targets[target_id_b]["std"])

    # dual_cfg needed for modes that score both targets (config swap to score B)
    _dual_modes = {"double", "linear_equal", "biased_a", "selective_a"}
    dual_cfg = None
    if reward_mode in _dual_modes:
        if not dual_receptor_b:
            raise RuntimeError("DUAL_RECEPTOR_B is required for double mode")
        config_path = Path(os.environ.get("DUAL_CONFIG", "/workspace/dual_config.json"))
        with config_path.open() as f:
            dual_cfg = json.load(f)

    # ffreed must already be on sys.path (set by run_with_patch.py)
    from ffreed.env.docking import DockingVina

    _original_call = DockingVina.__call__

    def _warn_large_z(label: str, z_value: float) -> None:
        if abs(z_value) <= 10:
            return
        if _z_warn_count[label] >= _MAX_Z_WARNINGS:
            return
        _z_warn_count[label] += 1
        logger.warning("dual_reward_patch: |z_%s| > 10 (%.2f)", label, z_value)

    def _warn_error_value(label: str, value: float, error_val: float) -> None:
        if value != error_val:
            return
        if _err_warn_count[label] >= _MAX_Z_WARNINGS:
            return
        _err_warn_count[label] += 1
        logger.warning("dual_reward_patch: docking returned error_val for target %s (%s)", label, value)

    def _rdkit_docking(  # noqa: PLR0913
        smile: str,
        *,
        vina_program: str,
        receptor: str,
        box_center: Any,
        box_size: Any,
        error_val: float,
        seed: int,
        num_modes: int,
        exhaustiveness: int,
        timeout_dock: int,
        timeout_gen3d: int,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> float:
        _set_last_docking_timed_out(value=False)

        # GPU batch cache lookup (pre-computed by _precompute_gpu_batch_for_receptor)
        cache_key = (receptor, smile)
        if cache_key in _gpu_dock_cache:
            return _gpu_dock_cache[cache_key]

        center = _coords(box_center)
        size = _coords(box_size)

        with (
            NamedTemporaryFile(mode="w", suffix=".pdbqt") as ligand,
            NamedTemporaryFile(mode="r+t", suffix=".pdbqt") as docking_file,
        ):
            if not _prepare_ligand_rdkit(smile, ligand.name, seed):
                return error_val

            cmd = [
                vina_program,
                "--receptor",
                receptor,
                "--ligand",
                ligand.name,
                "--out",
                docking_file.name,
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
                "--num_modes",
                str(num_modes),
                "--exhaustiveness",
                str(exhaustiveness),
                "--seed",
                str(seed),
            ]

            try:
                result = run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_dock,
                    check=False,
                )
            except TimeoutExpired:
                _set_last_docking_timed_out(value=True)
                return error_val

            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()[-500:]
                if tail:
                    logger.debug("dual_reward_patch: unidock rc=%s: %s", result.returncode, tail)
                return error_val

            parsed = DockingVina.parse_output(result.stdout, error_val)
            if parsed != error_val:
                if not (_SCORE_MIN <= parsed <= _SCORE_MAX):
                    logger.warning(
                        "corrupt docking score %.3f from CPU path: %s",
                        parsed,
                        smile,
                    )
                    return error_val
                return parsed

            parsed_from_file = _parse_result_from_pdbqt(Path(docking_file.name))
            return parsed_from_file if parsed_from_file is not None else error_val

    # Replace OpenBabel-dependent implementation with RDKit+Meeko backend.
    DockingVina.docking = staticmethod(_rdkit_docking)

    def _dual_call(self: Any, *args: Any, **kwargs: Any) -> float:
        """Score molecule against primary receptor."""
        _call_stats["total"] += 1

        # single_b: B is primary receptor (generator passed B as --receptor)
        if reward_mode == "single_b":
            score_b = _original_call(self, *args, **kwargs)
            timed_out_b = _consume_last_docking_timed_out()
            error_val_b = float(self.config.get("error_val", 100.0))
            z_b, invalid_b = _capped_zscore(score_b, error_val_b, mean_b, std_b, _INVALID_Z_PENALTY)
            if invalid_b:
                _call_stats["invalid_b"] += 1
                if timed_out_b:
                    _call_stats["timeout_b"] += 1
                _warn_error_value("b", score_b, error_val_b)
            else:
                _warn_large_z("b", z_b)
            return z_b

        # single_a / double: A is primary receptor
        score_a = _original_call(self, *args, **kwargs)
        timed_out_a = _consume_last_docking_timed_out()
        error_val_a = float(self.config.get("error_val", 100.0))

        z_a, invalid_a = _capped_zscore(score_a, error_val_a, mean_a, std_a, _INVALID_Z_PENALTY)
        if invalid_a:
            _call_stats["invalid_a"] += 1
            if timed_out_a:
                _call_stats["timeout_a"] += 1
            _warn_error_value("a", score_a, error_val_a)
        else:
            _warn_large_z("a", z_a)

        if reward_mode == "single_a":
            return z_a

        # double mode: also score B via config swap
        if dual_cfg is None:
            raise RuntimeError("dual_cfg missing for double mode")

        receptor_b = dual_cfg["receptor_b"]
        center_b = dual_cfg["center_b"]
        size_b = dual_cfg["size_b"]

        # NOTE: mean_b/std_b are closure vars (loaded in _apply_dual_patch)

        cfg = self.config
        orig_receptor = cfg["receptor"]
        orig_center = cfg["box_center"]
        orig_size = cfg["box_size"]

        try:
            cfg["receptor"] = receptor_b
            cfg["box_center"] = [float(x) for x in center_b.split(",")]
            cfg["box_size"] = [float(x) for x in size_b.split(",")]
            score_b = _original_call(self, *args, **kwargs)
        finally:
            cfg["receptor"] = orig_receptor
            cfg["box_center"] = orig_center
            cfg["box_size"] = orig_size

        timed_out_b = _consume_last_docking_timed_out()
        error_val_b = float(cfg.get("error_val", 100.0))

        z_b, invalid_b = _capped_zscore(score_b, error_val_b, mean_b, std_b, _INVALID_Z_PENALTY)
        if invalid_b:
            _call_stats["invalid_b"] += 1
            if timed_out_b:
                _call_stats["timeout_b"] += 1
            _warn_error_value("b", score_b, error_val_b)
        else:
            _warn_large_z("b", z_b)

        if reward_mode == "linear_equal":
            return z_a + z_b
        if reward_mode == "biased_a":
            return z_a + 0.3 * z_b
        if reward_mode == "selective_a":
            return z_a - z_b

        return max(z_a, z_b)

    DockingVina.__call__ = _dual_call
    atexit.register(_write_reward_metadata)

    # -----------------------------------------------------------------------
    # GPU batch: patch CacheAndPool to pre-compute docking scores via
    # a single ``unidock --gpu_batch`` call per epoch.
    # -----------------------------------------------------------------------
    if _GPU_BATCH_ENABLED:
        try:
            from ffreed.utils import CacheAndPool
        except (ImportError, AttributeError):
            logger.debug("dual_reward_patch: ffreed.utils.CacheAndPool not available, GPU batch disabled")
            CacheAndPool = None  # type: ignore[assignment,misc]  # noqa: N806

        if CacheAndPool is None:
            logger.info("dual_reward_patch: GPU batch docking DISABLED (CacheAndPool unavailable)")
            return

        _original_cap_call = CacheAndPool.__call__

        def _gpu_aware_cap_call(self: Any, args: list[str]) -> list[Any]:
            unseen = list(set(args).difference(self.results))
            if unseen and hasattr(self.func, "property") and isinstance(self.func.property, DockingVina):
                config = self.func.property.config
                # Pre-compute for primary receptor
                _precompute_gpu_batch_for_receptor(
                    unseen,
                    config,
                    config["receptor"],
                    config["box_center"],
                    config["box_size"],
                )
                # Pre-compute for secondary receptor (dual modes)
                if reward_mode in _dual_modes and dual_cfg:
                    _precompute_gpu_batch_for_receptor(
                        unseen,
                        config,
                        dual_cfg["receptor_b"],
                        dual_cfg["center_b"],
                        dual_cfg["size_b"],
                    )
                # Process sequentially — GPU batch already computed, each
                # _rdkit_docking call is a dict lookup (instant).
                values = list(map(self.func, unseen))
                new_results = dict(zip(unseen, values))
                self.results = {**self.results, **new_results}

                # Clean cache entries to prevent memory growth
                for smi in unseen:
                    _gpu_dock_cache.pop((config["receptor"], smi), None)
                    if dual_cfg:
                        _gpu_dock_cache.pop((dual_cfg["receptor_b"], smi), None)

                return [self.results[arg] for arg in args]

            return _original_cap_call(self, args)

        CacheAndPool.__call__ = _gpu_aware_cap_call
        logger.info("dual_reward_patch: GPU batch docking ENABLED (CacheAndPool patched)")
    else:
        logger.info("dual_reward_patch: GPU batch docking DISABLED (GPU_BATCH_DOCK=0)")

    logger.info(
        "dual_reward_patch: patched DockingVina"
        " (reward_mode=%s, ligand_backend=rdkit_meeko,"
        " invalid_z_penalty=%.1f, max_z_abs=%.1f)",
        reward_mode,
        _INVALID_Z_PENALTY,
        _MAX_Z_ABS,
    )


# Auto-apply on import
_apply_dual_patch()
