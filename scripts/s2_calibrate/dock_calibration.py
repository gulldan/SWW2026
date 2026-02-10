"""Dock calibration SMILES set against all targets with Uni-Dock GPU.

Prepares ligand PDBQT files from SMILES (via 3D embedding + Meeko),
docks each ligand against every target's receptor, and writes per-target
score CSV files for downstream normalization.
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import click
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DOCKER_BIN: str = shutil.which("docker") or "docker"


_PROGRAM_ALIASES: dict[str, str] = {
    "unidock_gpu": "unidock",
}


@dataclass
class DockingParams:
    """Docking parameters shared across calibration targets."""

    exhaustiveness: int
    num_modes: int
    energy_range: float
    timeout_seconds: int
    box_padding: float


def _parse_unidock_scores_sdf(text: str) -> list[float]:
    """Extract ENERGY scores from a UniDock SDF output.

    Args:
        text: Full SDF file content.

    Returns:
        List of parsed energy scores.
    """
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


class DockerRunner:
    """Lightweight Docker exec helper for persistent containers.

    Starts a long-lived container and provides ``exec()`` to run
    commands inside it, avoiding container startup overhead per call.
    """

    def __init__(self, image: str, workdir: Path, gpus: str | None = "all") -> None:
        """Initialize the runner without starting a container.

        Args:
            image: Docker image name/tag.
            workdir: Host directory mounted as ``/workspace``.
            gpus: Docker ``--gpus`` value (``None`` to disable GPU).
        """
        self.image = image
        self.workdir = workdir
        self.gpus = gpus
        self.container_id: str | None = None

    def start(self) -> None:
        """Start a long-lived container for repeated exec calls.

        Raises:
            RuntimeError: If the container fails to start.
        """
        cmd = [DOCKER_BIN, "run", "-d", "--rm", "--entrypoint", "sleep"]
        if self.gpus:
            cmd += ["--gpus", self.gpus]
        cmd += ["-v", f"{self.workdir}:/workspace", "-w", "/workspace", self.image, "infinity"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
        self.container_id = result.stdout.strip()
        if not self.container_id:
            raise RuntimeError("Failed to start docker container")

    def exec(self, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        """Run a command inside the running container.

        Args:
            args: Command and arguments to execute.
            timeout: Maximum seconds to wait.

        Returns:
            Completed process result.

        Raises:
            RuntimeError: If the container is not running.
        """
        if not self.container_id:
            raise RuntimeError("Docker container is not running")
        cmd = [DOCKER_BIN, "exec", self.container_id, *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603

    def stop(self) -> None:
        """Stop the running container."""
        if self.container_id:
            subprocess.run([DOCKER_BIN, "stop", self.container_id], check=False)  # noqa: S603
            self.container_id = None

    def __enter__(self) -> Self:
        """Context manager entry: start the container."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Context manager exit: stop the container."""
        self.stop()


def _load_calibration_smiles(path: Path) -> list[str]:
    """Load SMILES strings from a whitespace-delimited .smi file.

    Args:
        path: Path to the SMILES file (one SMILES per line, first token).

    Returns:
        List of SMILES strings.
    """
    smiles: list[str] = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            smiles.append(stripped.split()[0])
    return smiles


def _sample_smiles(smiles: list[str], n: int, seed: int) -> list[str]:
    """Subsample SMILES if needed.

    Args:
        smiles: Full list of SMILES strings.
        n: Desired sample size.
        seed: Random seed.

    Returns:
        At most *n* sampled SMILES.
    """
    if n >= len(smiles):
        return smiles
    rng = random.Random(seed)  # noqa: S311
    return rng.sample(smiles, n)


def _resolve_program(program: str, override: str | None = None) -> str:
    """Resolve a docking program alias to its binary name.

    Args:
        program: Logical program name.
        override: Explicit binary path override.

    Returns:
        Resolved binary name.
    """
    if override:
        return override
    return _PROGRAM_ALIASES.get(program, program)


def _parse_score(text: str) -> float:
    """Parse a docking score from log/PDBQT output text.

    Args:
        text: Output text from the docking program.

    Returns:
        Best docking score.

    Raises:
        ValueError: If no score can be parsed.
    """
    for line in text.splitlines():
        match = re.match(r"^\s*1\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            return float(match.group(1))
        match = re.search(r"REMARK VINA RESULT:\s*(-?\d+(?:\.\d+)?)", line)
        if match:
            return float(match.group(1))
    match = re.search(r"Affinity:\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    raise ValueError("Could not parse docking score")


def _docker_image_digest(image: str) -> str:
    """Query the Docker image digest (sha256) for provenance tracking.

    Args:
        image: Docker image name/tag.

    Returns:
        Digest string or ``"unknown"`` on failure.
    """
    result = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    out = (result.stdout or result.stderr or "").strip()
    if "@" in out:
        return out.split("@")[-1]
    return out or "unknown"


def _container_path(path: Path) -> str:
    """Convert a host-side path to the ``/workspace`` container equivalent.

    Args:
        path: Absolute host path under PROJECT_ROOT.

    Returns:
        Container-relative path string.
    """
    return f"/workspace/{path.relative_to(PROJECT_ROOT)}"


def _safe_unlink(path: Path, runner: DockerRunner | None) -> None:
    """Remove a file, falling back to Docker rm on PermissionError.

    Args:
        path: File to remove.
        runner: Optional Docker runner for fallback removal.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except PermissionError:
        if not runner:
            raise
        runner.exec(["rm", "-f", _container_path(path)])


def _ensure_receptor_pdbqt(
    runner: DockerRunner | None,
    pdb_path: Path,
    pdbqt_path: Path,
    timeout: int,
    overwrite: bool = False,
) -> None:
    """Ensure a receptor PDBQT file exists, preparing it from PDB if needed.

    Args:
        runner: Docker runner (or None for local execution).
        pdb_path: Input PDB file.
        pdbqt_path: Output PDBQT file.
        timeout: Subprocess timeout in seconds.
        overwrite: If True, regenerate even if the file exists.

    Raises:
        RuntimeError: If hydrogen addition or receptor preparation fails.
    """
    if pdbqt_path.exists() and not overwrite:
        return
    if pdbqt_path.exists() and overwrite:
        _safe_unlink(pdbqt_path, runner)
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_pdb = pdbqt_path.with_suffix(".fixed.pdb")
    if fixed_pdb.exists() and overwrite:
        _safe_unlink(fixed_pdb, runner)

    if not fixed_pdb.exists():
        fixer_script = PROJECT_ROOT / "docker" / "freedpp" / "pdbfixer_add_h.py"
        fixer_cmd = [
            "/opt/micromamba/envs/unidock/bin/python" if runner else sys.executable,
            _container_path(fixer_script) if runner else str(fixer_script),
            "--pdb",
            _container_path(pdb_path) if runner else str(pdb_path),
            "--out",
            _container_path(fixed_pdb) if runner else str(fixed_pdb),
        ]
        if runner:
            result = runner.exec(fixer_cmd, timeout=timeout)
        else:
            result = subprocess.run(  # noqa: S603
                fixer_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Failed to add hydrogens: {err}")
    output_basename = pdbqt_path.with_suffix("")
    cmd = [
        "mk_prepare_receptor.py",
        "--pdb",
        _container_path(fixed_pdb) if runner else str(fixed_pdb),
        "-o",
        _container_path(output_basename) if runner else str(output_basename),
        "--skip_gpf",
    ]
    if runner:
        result = runner.exec(cmd, timeout=timeout)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Failed to prepare receptor: {err}")


def _write_sdf_from_smiles(
    runner: DockerRunner | None,
    smiles: str,
    sdf_path: Path,
    seed: int,
    timeout: int,
) -> tuple[bool, str]:
    """Generate an SDF file with 3D coordinates from a SMILES string.

    Args:
        runner: Docker runner (or None for local RDKit).
        smiles: Input SMILES string.
        sdf_path: Output SDF file path.
        seed: Random seed for 3D embedding.
        timeout: Subprocess timeout in seconds.

    Returns:
        Tuple of (success, error_message).
    """
    if runner:
        cmd = [
            "python",
            "/workspace/scripts/rdkit_embed.py",
            "--smiles",
            smiles,
            "--sdf",
            _container_path(sdf_path),
            "--seed",
            str(seed),
        ]
        result = runner.exec(cmd, timeout=timeout)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            return False, err or "RDKit embed failed in container"
        return True, ""

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:  # pragma: no cover - import error is fatal
        return False, f"RDKit import failed: {exc}"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "RDKit failed to parse SMILES"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False, "RDKit 3D embedding failed"
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return True, ""


def _contains_glue_atoms(pdbqt_path: Path) -> bool:
    """Check whether a PDBQT file contains macrocycle glue atoms (e.g. G0).

    Args:
        pdbqt_path: Path to the PDBQT file.

    Returns:
        True if glue atoms are detected.
    """
    try:
        text = pdbqt_path.read_text()
    except FileNotFoundError:
        return False
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        atom_type = line[77:79].strip() if len(line) >= 79 else line.split()[-1]
        if atom_name == "G" or atom_type.startswith("G"):
            return True
    return False


def _ensure_ligand_pdbqt(
    runner: DockerRunner | None,
    smiles: str,
    sdf_path: Path,
    out_path: Path,
    timeout: int,
    seed: int,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """Ensure a ligand PDBQT file is prepared from SMILES.

    Args:
        runner: Docker runner (or None for local execution).
        smiles: Input SMILES string.
        sdf_path: Intermediate SDF file path.
        out_path: Output PDBQT file path.
        timeout: Subprocess timeout in seconds.
        seed: Random seed for 3D embedding.
        overwrite: If True, regenerate even if the file exists.

    Returns:
        Tuple of (success, error_message).
    """
    if out_path.exists() and not overwrite:
        return True, ""
    if out_path.exists() and overwrite:
        _safe_unlink(out_path, runner)
    if sdf_path.exists() and overwrite:
        _safe_unlink(sdf_path, runner)

    ok, err = _write_sdf_from_smiles(runner, smiles, sdf_path, seed=seed, timeout=timeout)
    if not ok:
        return False, err

    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        _container_path(sdf_path) if runner else str(sdf_path),
        "-o",
        _container_path(out_path) if runner else str(out_path),
        "--rigid_macrocycles",
    ]
    if runner:
        result = runner.exec(cmd, timeout=timeout)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return False, err
    if _contains_glue_atoms(out_path):
        return False, "Glue atoms (G0) detected in ligand; macrocycle handling failed"
    return True, ""


def _dock_ligand(
    runner: DockerRunner | None,
    binary: str,
    receptor: Path,
    ligand: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    log_path: Path,
    out_path: Path,
) -> float:
    """Dock a single ligand against a receptor and return the best score.

    Args:
        runner: Docker runner (or None for local execution).
        binary: Docking binary name.
        receptor: Receptor PDBQT file.
        ligand: Ligand PDBQT file.
        center: Binding box center (x, y, z) in Angstroms.
        size: Binding box size (x, y, z) in Angstroms.
        params: Docking parameters.
        log_path: Path for docking log output.
        out_path: Path for docked pose output.

    Returns:
        Best docking score (kcal/mol).

    Raises:
        RuntimeError: If the docking binary exits with an error.
    """
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    binary_name = Path(binary).name.lower()
    supports_log = "unidock" not in binary_name
    cmd = [
        binary,
        "--receptor",
        _container_path(receptor) if runner else str(receptor),
        "--ligand",
        _container_path(ligand) if runner else str(ligand),
        "--center_x",
        f"{center_x:.3f}",
        "--center_y",
        f"{center_y:.3f}",
        "--center_z",
        f"{center_z:.3f}",
        "--size_x",
        f"{size_x:.3f}",
        "--size_y",
        f"{size_y:.3f}",
        "--size_z",
        f"{size_z:.3f}",
        "--exhaustiveness",
        str(params.exhaustiveness),
        "--num_modes",
        str(params.num_modes),
        "--energy_range",
        str(params.energy_range),
        "--out",
        _container_path(out_path) if runner else str(out_path),
    ]
    if supports_log:
        cmd += [
            "--log",
            _container_path(log_path) if runner else str(log_path),
        ]

    if runner:
        result = runner.exec(cmd, timeout=params.timeout_seconds)
        if supports_log and log_path.exists():
            log_text = runner.exec(["cat", _container_path(log_path)]).stdout or ""
        else:
            log_text = result.stdout or result.stderr or ""
    else:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=params.timeout_seconds,
        )
        log_text = log_path.read_text() if supports_log and log_path.exists() else (result.stdout or result.stderr or "")

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    if not supports_log:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log_text)

    return _parse_score(log_text)


def _dock_batch_unidock(  # noqa: PLR0913
    runner: DockerRunner | None,
    binary: str,
    receptor: Path,
    ligands: list[Path],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    output_dir: Path,
    batch_id: int,
    seed: int,
    timeout: int,
) -> dict[Path, float | None]:
    """Dock a batch of ligands using UniDock's ligand_index mode.

    Args:
        runner: Docker runner (or None for local execution).
        binary: UniDock binary name.
        receptor: Receptor PDBQT file.
        ligands: List of ligand PDBQT files.
        center: Binding box center (x, y, z).
        size: Binding box size (x, y, z).
        params: Docking parameters.
        output_dir: Directory for output SDF/PDBQT files.
        batch_id: Batch index for the ligand index filename.
        seed: Random seed for docking.
        timeout: Subprocess timeout in seconds.

    Returns:
        Mapping from ligand path to best score (or None on failure).

    Raises:
        RuntimeError: If the batch fails completely (all scores are None).
    """
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / f"ligand_index_{batch_id:04d}.txt"
    index_lines = [_container_path(lig) if runner else str(lig) for lig in ligands]
    index_path.write_text("\n".join(index_lines) + "\n")

    for lig in ligands:
        _safe_unlink(output_dir / f"{lig.stem}_out.sdf", runner)

    cmd = [
        binary,
        "--receptor",
        _container_path(receptor) if runner else str(receptor),
        "--ligand_index",
        _container_path(index_path) if runner else str(index_path),
        "--center_x",
        f"{center_x:.3f}",
        "--center_y",
        f"{center_y:.3f}",
        "--center_z",
        f"{center_z:.3f}",
        "--size_x",
        f"{size_x:.3f}",
        "--size_y",
        f"{size_y:.3f}",
        "--size_z",
        f"{size_z:.3f}",
        "--exhaustiveness",
        str(params.exhaustiveness),
        "--num_modes",
        str(params.num_modes),
        "--energy_range",
        str(params.energy_range),
        "--dir",
        _container_path(output_dir) if runner else str(output_dir),
        "--seed",
        str(seed),
        "--verbosity",
        "2",
        "--keep_nonpolar_H",
    ]

    if runner:
        result = runner.exec(cmd, timeout=timeout)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
    run_error = ""
    if result.returncode != 0:
        run_error = (result.stderr or result.stdout or "").strip()

    scores: dict[Path, float | None] = {}
    for lig in ligands:
        sdf_path = output_dir / f"{lig.stem}_out.sdf"
        pdbqt_path = output_dir / f"{lig.stem}_out.pdbqt"
        if sdf_path.exists():
            scores_list = _parse_unidock_scores_sdf(sdf_path.read_text())
            scores[lig] = min(scores_list) if scores_list else None
            _safe_unlink(sdf_path, runner)
            continue
        if pdbqt_path.exists():
            try:
                scores[lig] = _parse_score(pdbqt_path.read_text())
            except Exception:
                scores[lig] = None
            _safe_unlink(pdbqt_path, runner)
            continue
        scores[lig] = None

    _safe_unlink(index_path, runner)
    if run_error and all(score is None for score in scores.values()):
        raise RuntimeError(run_error)
    if run_error:
        logger.warning("Uni-Dock batch {} completed with warnings: {}", batch_id, run_error)
    return scores


def _prep_one_ligand(
    runner: DockerRunner | None,
    smi: str,
    sdf_path: Path,
    pdbqt_path: Path,
    timeout: int,
    seed: int,
    overwrite: bool,
) -> tuple[str, Path, bool, str]:
    """Prepare one ligand PDBQT file (thread-safe).

    Args:
        runner: Docker runner (or None for local execution).
        smi: SMILES string.
        sdf_path: Intermediate SDF file path.
        pdbqt_path: Output PDBQT file path.
        timeout: Subprocess timeout in seconds.
        seed: Random seed for 3D embedding.
        overwrite: If True, regenerate even if the file exists.

    Returns:
        Tuple of (smiles, pdbqt_path, success, error_message).
    """
    ok, err = _ensure_ligand_pdbqt(
        runner,
        smi,
        sdf_path,
        pdbqt_path,
        timeout=timeout,
        seed=seed,
        overwrite=overwrite,
    )
    return smi, pdbqt_path, ok, err


def _auto_prep_workers() -> int:
    """Pick number of parallel ligand prep workers based on CPU count.

    Returns:
        Worker count (minimum of CPU count and 8).
    """
    import os as _os

    return min(_os.cpu_count() or 1, 8)


def _parse_vec(vec_str: str) -> tuple[float, float, float]:
    """Parse a comma-separated 3D vector string.

    Args:
        vec_str: String in ``"x,y,z"`` format.

    Returns:
        Tuple of three floats.

    Raises:
        ValueError: If the string does not contain exactly 3 values.
    """
    parts = [float(x) for x in vec_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got: {vec_str}")
    return parts[0], parts[1], parts[2]


def _docking_version(runner: DockerRunner | None, binary: str) -> str:
    """Query the docking binary for its version string.

    Args:
        runner: Docker runner (or None for local execution).
        binary: Docking binary name.

    Returns:
        First line of version output, or ``"unknown"``.
    """
    if runner:
        result = runner.exec([binary, "--version"])
    else:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False)  # noqa: S603
    output = (result.stdout or result.stderr or "").strip()
    return output.splitlines()[0] if output else "unknown"


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--calibration-set", default=None)
@click.option("--scores-dir", default="output/data/calibration/scores")
@click.option("--n", default=None, type=int, help="Number of ligands to dock")
@click.option("--seed", default=None, type=int, help="Sampling seed")
@click.option("--target-id", multiple=True, help="Target ID to dock (repeatable)")
@click.option("--docker-image", default="dualdocking/freedpp:latest")
@click.option("--use-docker/--no-docker", default=True)
@click.option("--gpus", default="all", help="Docker --gpus value")
@click.option("--docking-program", default="unidock_gpu")
@click.option("--docking-binary", default=None, help="Override docking binary")
@click.option("--overwrite/--skip-existing", default=False)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--prep-workers",
    default=0,
    type=int,
    show_default=True,
    help="Parallel ligand prep workers. 0=auto (min(cpu_count, 8)), 1=sequential.",
)
def main(  # noqa: PLR0913, PLR0915
    calibration_config: str,
    calibration_set: str | None,
    scores_dir: str,
    n: int | None,
    seed: int | None,
    target_id: tuple[str, ...],
    docker_image: str,
    use_docker: bool,
    gpus: str,
    docking_program: str,
    docking_binary: str | None,
    overwrite: bool,
    dry_run: bool,
    prep_workers: int,
) -> None:
    """Dock calibration SMILES set against targets to build score distributions.

    Reads the calibration SMILES file, prepares ligand PDBQT files,
    docks each against every target receptor, and writes per-target
    score CSVs and a manifest JSON.

    Args:
        calibration_config: Path to the calibration YAML config.
        calibration_set: Override path to the calibration SMILES file.
        scores_dir: Output directory for per-target score CSVs.
        n: Number of ligands to dock (from calibration config if omitted).
        seed: Sampling seed (from calibration config if omitted).
        target_id: Restrict to these target IDs (all if empty).
        docker_image: Docker image for docking tools.
        use_docker: If True, run docking inside Docker.
        gpus: Docker ``--gpus`` value.
        docking_program: Logical docking program name.
        docking_binary: Override docking binary path.
        overwrite: If True, overwrite existing score files.
        dry_run: If True, skip actual docking.
        prep_workers: Number of parallel ligand preparation workers.
    """
    import yaml

    from dualdocking.config import load_pairs
    from dualdocking.generation.freedpp import _ensure_docker_image, _resolve_pdb_path
    from dualdocking.scoring.binding_box import extract_binding_box

    cfg = yaml.safe_load((PROJECT_ROOT / calibration_config).read_text())
    calibration_path = PROJECT_ROOT / (calibration_set or cfg.get("output_path"))
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration set not found: {calibration_path}")

    sample_cfg = cfg.get("sample", {})
    n = n or int(sample_cfg.get("n", 1000))
    seed = seed or int(sample_cfg.get("seed", 137))

    docking_cfg = cfg.get("docking", {})
    params = DockingParams(
        exhaustiveness=int(docking_cfg.get("exhaustiveness", 8)),
        num_modes=int(docking_cfg.get("num_modes", 9)),
        energy_range=float(docking_cfg.get("energy_range", 3.0)),
        timeout_seconds=int(docking_cfg.get("timeout_seconds", 120)),
        box_padding=float(docking_cfg.get("box_padding", 10.0)),
    )
    batch_size = int(docking_cfg.get("batch_size", 64))
    batch_mode = bool(docking_cfg.get("batch_mode", False))

    scores_root = PROJECT_ROOT / scores_dir
    scores_root.mkdir(parents=True, exist_ok=True)
    receptors_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "receptors"
    ligands_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "ligands"
    work_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "work"
    receptors_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    smiles_all = _load_calibration_smiles(calibration_path)
    logger.info("Calibration set: {} ({} total, target n={})", calibration_path, len(smiles_all), n)

    ligands_manifest = ligands_dir / "ligands.smi"
    use_existing_manifest = ligands_manifest.exists() and not overwrite
    if use_existing_manifest:
        existing = [line.strip().split()[0] for line in ligands_manifest.read_text().splitlines() if line.strip()]
        if n is not None and len(existing) != n:
            logger.warning(
                "Ligands manifest has {} entries, requested n={}; reusing manifest",
                len(existing),
                n,
            )
        n = len(existing)
        candidates = existing
    else:
        rng = random.Random(seed)  # noqa: S311
        candidates = smiles_all.copy()
        rng.shuffle(candidates)
    smiles = []

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    target_ids: list[str] = []
    for pair in pairs.values():
        target_ids.append(pair["target_a"]["id"])
        target_ids.append(pair["target_b"]["id"])
    target_ids = sorted(set(target_ids))

    if target_id:
        target_ids = [t for t in target_ids if t in set(target_id)]
        if not target_ids:
            raise ValueError("No target IDs matched the provided --target-id values")

    binary = _resolve_program(docking_program, docking_binary)
    use_unidock_batch = batch_mode and "unidock" in Path(binary).name.lower()
    if "unidock" in Path(binary).name.lower() and not use_unidock_batch:
        logger.info("Uni-Dock batch mode disabled; using single-ligand docking")

    if dry_run:
        logger.info("Dry run enabled; no docking will be executed")
    elif use_docker:
        _ensure_docker_image(docker_image)

    runner: DockerRunner | None
    runner_ctx = DockerRunner(docker_image, PROJECT_ROOT, gpus=gpus) if use_docker else nullcontext(None)
    with runner_ctx as runner:
        if runner:
            logger.info("Using docker image: {}", docker_image)

        # Prepare ligands (parallel when workers > 1)
        ligand_paths: list[Path] = []
        failed_smiles: list[str] = []

        actual_workers = 1 if dry_run else (prep_workers if prep_workers > 0 else _auto_prep_workers())

        if actual_workers > 1 and not dry_run:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

            # Pre-assign indices; submit extra candidates to cover failures
            max_tasks = min(len(candidates), int(n * 1.2) + 50)
            # ProcessPoolExecutor for local RDKit (CPU-bound, GIL-releasing C++);
            # ThreadPoolExecutor for Docker mode (subprocess I/O, runner not picklable).
            pool_class = ProcessPoolExecutor if runner is None else ThreadPoolExecutor
            logger.info(
                "Parallel ligand prep: {} candidates on {} {} workers (need {})",
                max_tasks,
                actual_workers,
                "process" if runner is None else "thread",
                n,
            )

            future_to_idx: dict = {}
            with pool_class(max_workers=actual_workers) as pool:
                for task_idx in range(max_tasks):
                    smi = candidates[task_idx]
                    sdf_p = ligands_dir / f"ligand_{task_idx:05d}.sdf"
                    pdbqt_p = ligands_dir / f"ligand_{task_idx:05d}.pdbqt"
                    f = pool.submit(
                        _prep_one_ligand,
                        runner,
                        smi,
                        sdf_p,
                        pdbqt_p,
                        params.timeout_seconds,
                        seed + task_idx,
                        overwrite,
                    )
                    future_to_idx[f] = task_idx

                # Collect results with progress bar
                results: list[tuple[int, str, Path, bool, str]] = []
                for f in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Ligands", unit="ligand"):
                    idx = future_to_idx[f]
                    smi_r, path_r, ok_r, err_r = f.result()
                    results.append((idx, smi_r, path_r, ok_r, err_r))

            # Sort by original candidate order for reproducibility
            results.sort(key=lambda r: r[0])
            for _idx, smi_r, path_r, ok_r, err_r in results:
                if len(smiles) >= n:
                    break
                if ok_r:
                    smiles.append(smi_r)
                    ligand_paths.append(path_r)
                else:
                    failed_smiles.append(smi_r)
                    logger.warning("Ligand prep failed: {} ({})", smi_r, err_r or "no error text")
                    if use_existing_manifest:
                        raise RuntimeError("Ligand prep failed for existing manifest; rerun with --overwrite")
        else:
            # Sequential (original path -- also used for dry_run)
            for smi in tqdm(candidates, desc="Ligands", unit="ligand"):
                if len(smiles) >= n:
                    break
                ligand_path = ligands_dir / f"ligand_{len(smiles):05d}.pdbqt"
                sdf_path = ligands_dir / f"ligand_{len(smiles):05d}.sdf"
                if dry_run:
                    smiles.append(smi)
                    ligand_paths.append(ligand_path)
                    continue
                ok, err = _ensure_ligand_pdbqt(
                    runner,
                    smi,
                    sdf_path,
                    ligand_path,
                    timeout=params.timeout_seconds,
                    seed=seed + len(smiles),
                    overwrite=overwrite,
                )
                if not ok:
                    failed_smiles.append(smi)
                    logger.warning("Ligand prep failed: {} ({})", smi, err or "no error text")
                    if use_existing_manifest:
                        raise RuntimeError("Ligand prep failed for existing manifest; rerun with --overwrite")
                    continue
                smiles.append(smi)
                ligand_paths.append(ligand_path)

        if len(smiles) < n:
            raise RuntimeError(f"Only {len(smiles)} ligands prepared, expected {n}")

        if failed_smiles and not use_existing_manifest:
            (ligands_dir / "ligands_failed.smi").write_text("\n".join(failed_smiles) + "\n")

        ligands_manifest.write_text("\n".join(smiles) + "\n")

        docking_version = _docking_version(runner, binary) if not dry_run else "unknown"
        manifest = {
            "docking_program": docking_program,
            "docking_binary": binary,
            "docking_version": docking_version,
            "docker_image": docker_image,
            "docker_image_digest": _docker_image_digest(docker_image),
            "calibration_set": str(calibration_set or cfg.get("output_path")),
            "n": len(smiles),
            "seed": seed,
            "params": {
                "exhaustiveness": params.exhaustiveness,
                "num_modes": params.num_modes,
                "energy_range": params.energy_range,
                "timeout_seconds": params.timeout_seconds,
                "box_padding": params.box_padding,
                "batch_mode": use_unidock_batch,
                "batch_size": batch_size if use_unidock_batch else 1,
            },
            "targets": target_ids,
            "date": datetime.now(UTC).date().isoformat(),
        }
        (scores_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

        for target in target_ids:
            scores_path = scores_root / f"{target}.csv"
            if scores_path.exists() and not overwrite:
                logger.info("Skip existing scores: {}", scores_path)
                continue

            pdb_path = _resolve_pdb_path(target)
            receptor_path = receptors_dir / f"{target}.pdbqt"
            if not dry_run:
                _ensure_receptor_pdbqt(
                    runner,
                    pdb_path,
                    receptor_path,
                    timeout=params.timeout_seconds,
                    overwrite=overwrite,
                )

            center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
            center = _parse_vec(center_str)
            size = _parse_vec(size_str)

            scores: list[float] = []
            failures = 0
            target_work = work_dir / target
            target_work.mkdir(parents=True, exist_ok=True)

            if use_unidock_batch and not dry_run:
                batch_iter = range(0, len(ligand_paths), batch_size)
                for batch_id, start in enumerate(tqdm(batch_iter, desc=f"Dock {target}", unit="batch")):
                    batch = ligand_paths[start : start + batch_size]
                    try:
                        batch_scores = _dock_batch_unidock(
                            runner,
                            binary,
                            receptor_path,
                            batch,
                            center,
                            size,
                            params,
                            target_work,
                            batch_id=batch_id,
                            seed=seed + start,
                            timeout=params.timeout_seconds,
                        )
                    except Exception as exc:
                        failures += len(batch)
                        logger.error("Batch docking failed for {} batch {}: {}", target, batch_id, exc)
                        continue
                    for lig in batch:
                        score = batch_scores.get(lig)
                        if score is None:
                            failures += 1
                            logger.error("Docking failed for {} ligand {}", target, lig.name)
                            continue
                        scores.append(score)
            else:
                for idx, ligand_path in enumerate(tqdm(ligand_paths, desc=f"Dock {target}", unit="ligand")):
                    if dry_run:
                        scores.append(0.0)
                        continue
                    log_path = target_work / f"{target}_{idx:05d}.log"
                    out_path = target_work / f"{target}_{idx:05d}.pdbqt"
                    try:
                        score = _dock_ligand(
                            runner,
                            binary,
                            receptor_path,
                            ligand_path,
                            center,
                            size,
                            params,
                            log_path,
                            out_path,
                        )
                        scores.append(score)
                    except Exception as exc:
                        failures += 1
                        logger.error("Docking failed for {} #{}: {}", target, idx, exc)
                        continue
                    finally:
                        if log_path.exists():
                            log_path.unlink(missing_ok=True)
                        if out_path.exists():
                            out_path.unlink(missing_ok=True)

            if failures:
                logger.warning("{} docking failures for target {}", failures, target)

            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with scores_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["score"])
                for score in scores:
                    writer.writerow([f"{score:.6f}"])

            logger.info("Scores -> {} (n={})", scores_path, len(scores))


if __name__ == "__main__":
    main()
