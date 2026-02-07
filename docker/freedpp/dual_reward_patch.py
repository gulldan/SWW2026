"""Dual-target reward patch for FREED++.

This patch does two things:
1) replaces ligand preparation backend (OpenBabel) with RDKit+Meeko,
2) applies z-score based rewards for single_a / single_b / double modes.

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
from pathlib import Path
from subprocess import TimeoutExpired, run
from tempfile import NamedTemporaryFile
from typing import Any

logger = logging.getLogger(__name__)

_MAX_Z_WARNINGS = 5
_z_warn_count = {"a": 0, "b": 0}
_err_warn_count = {"a": 0, "b": 0}
_RESULT_PATTERN = re.compile(r"RESULT[:\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Invalid-docking exclusion (ENG-01 / KVA-255)
# ---------------------------------------------------------------------------

_INVALID_Z_PENALTY = float(os.environ.get("INVALID_Z_PENALTY", "4.0"))
if not (0 < _INVALID_Z_PENALTY < float("inf")):
    raise ValueError(f"INVALID_Z_PENALTY must be a positive finite number, got {_INVALID_Z_PENALTY}")

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
        raise ZeroDivisionError("std=0 в normalization.yaml")
    return (score - mean) / std, False


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
        "primary_target": target_id_b if reward_mode == "single_b" else target_id_a,
        "reward_mode": reward_mode,
    }
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
        raise TypeError("normalization.yaml должен быть словарём")
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


def _parse_result_from_pdbqt(path: Path) -> float | None:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if "RESULT" not in line.upper():
                    continue
                match = _RESULT_PATTERN.search(line)
                if match:
                    return float(match.group(1))
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
        raise KeyError(f"Target A '{target_id_a}' не найден в normalization.yaml")

    mean_a = float(targets[target_id_a]["mean"])
    std_a = float(targets[target_id_a]["std"])

    dual_receptor_b = os.environ.get("DUAL_RECEPTOR_B")

    # Load mean_b/std_b as closure variables for both single_b and double
    mean_b = std_b = 0.0  # sentinels; only used when mode needs B
    if reward_mode in {"single_b", "double"}:
        if target_id_b not in targets:
            raise KeyError(f"Target B '{target_id_b}' не найден в normalization.yaml")
        mean_b = float(targets[target_id_b]["mean"])
        std_b = float(targets[target_id_b]["std"])

    # dual_cfg only needed for double mode (config swap to score B)
    dual_cfg = None
    if reward_mode == "double":
        if not dual_receptor_b:
            raise RuntimeError("DUAL_RECEPTOR_B обязателен для double режима")
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
            return -z_b

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
            return -z_a

        # double mode: also score B via config swap
        if dual_cfg is None:
            raise RuntimeError("dual_cfg отсутствует при режиме double")

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

        return -max(z_a, z_b)

    DockingVina.__call__ = _dual_call
    atexit.register(_write_reward_metadata)
    logger.info(
        "dual_reward_patch: patched DockingVina (reward_mode=%s, ligand_backend=rdkit_meeko, invalid_z_penalty=%.1f)",
        reward_mode,
        _INVALID_Z_PENALTY,
    )


# Auto-apply on import
_apply_dual_patch()
