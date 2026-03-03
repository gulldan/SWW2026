"""I/O helpers for FREED++ generation (targets, receptor prep, outputs)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml


class _LoggerLike(Protocol):
    """Minimal logger protocol for FREED++ I/O helpers."""

    def info(self, message: str, *args: object) -> None:
        """Log info message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """

    def debug(self, message: str, *args: object) -> None:
        """Log debug message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """


_RunSubprocessFn = Callable[..., object]
_ExtractBoxFn = Callable[[Path, float], tuple[str, str]]


def resolve_pdb_path(target_id: str, *, data_dir: Path) -> Path:
    """Find a target PDB path by scanning prepared target metadata.

    Args:
        target_id: Target identifier stored in ``meta.yaml``.
        data_dir: Root directory with ``<pair>/<A|B>/meta.yaml`` structure.

    Returns:
        Path to first matching ``.pdb`` file (prefers non-``.fixed``).

    Raises:
        FileNotFoundError: Target or matching PDB was not found.
    """
    for pair_dir in data_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        for role in ("A", "B"):
            role_dir = pair_dir / role
            meta_path = role_dir / "meta.yaml"
            if not meta_path.exists():
                continue

            with meta_path.open() as handle:
                meta = yaml.safe_load(handle)

            if meta.get("id") != target_id:
                continue

            pdb_files = [path for path in role_dir.glob("*.pdb") if ".fixed" not in path.name]
            if not pdb_files:
                pdb_files = list(role_dir.glob("*.pdb"))
            if pdb_files:
                return pdb_files[0]

    raise FileNotFoundError(f"PDB file for target '{target_id}' not found in {data_dir}")


def prepare_receptor(
    pdb_path: Path,
    out_dir: Path,
    *,
    image: str,
    docker_bin: str,
    padding: float,
    run_subprocess: _RunSubprocessFn,
    extract_binding_box: _ExtractBoxFn,
    log: _LoggerLike,
) -> tuple[Path, str, str]:
    """Convert receptor PDB to PDBQT and compute binding box.

    Args:
        pdb_path: Input receptor PDB path.
        out_dir: Output directory for PDBQT.
        image: Docker image containing ``prepare-receptor``.
        docker_bin: Docker binary path.
        padding: Binding-box padding in Angstroms.
        run_subprocess: Subprocess runner callback.
        extract_binding_box: Binding-box extraction callback.
        log: Logger-like instance.

    Returns:
        Tuple of ``(pdbqt_path, center_str, size_str)``.
    """
    pdbqt_path = out_dir / f"{pdb_path.stem}.pdbqt"
    if not pdbqt_path.exists():
        log.info(f"Converting PDB -> PDBQT: {pdb_path} -> {pdbqt_path}")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_subprocess(
            [
                docker_bin,
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
        log.debug(f"PDBQT already exists: {pdbqt_path}")

    center_str, size_str = extract_binding_box(pdb_path, padding)
    log.info(f"Binding box: center={center_str}, size={size_str}")
    return pdbqt_path, center_str, size_str


def parse_output_smiles(workspace: Path, *, log: _LoggerLike) -> list[str]:
    """Read generated SMILES from FREED++ output artifacts.

    Preferred format is ``sample_*.json``. Legacy CSV fallback is used only
    when JSON files are absent or empty.

    Args:
        workspace: FREED++ workspace directory.
        log: Logger-like instance.

    Returns:
        Parsed SMILES list.
    """
    mols_dir = workspace / "output" / "run" / "mols"
    all_smiles: list[str] = []

    for json_path in sorted(mols_dir.glob("sample_*.json")):
        try:
            payload = json.loads(json_path.read_text())
            if isinstance(payload, list):
                all_smiles.extend(str(item) for item in payload if item)
        except Exception as exc:
            log.warning(f"Error reading {json_path}: {exc}")

    if not all_smiles:
        for csv_path in sorted(mols_dir.glob("sample_*.csv")):
            try:
                df = pd.read_csv(csv_path)
                col = "smiles" if "smiles" in df.columns else "Smiles" if "Smiles" in df.columns else None
                if col:
                    all_smiles.extend(df[col].dropna().tolist())
            except Exception as exc:
                log.warning(f"Error reading {csv_path}: {exc}")

    if not all_smiles:
        log.warning(f"No molecule files found in {mols_dir}")

    log.info(f"Read {len(all_smiles)} SMILES from output")
    return all_smiles
