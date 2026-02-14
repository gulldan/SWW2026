"""Shared docking utilities for UniDock calibration, redocking, and enrichment.

Provides :class:`DockerRunner` (persistent Docker container manager),
:class:`DockingParams`, receptor/ligand PDBQT preparation, and batch
docking helpers.  Extracted from ``cli/s2_calibrate/dock_calibration.py``
so that ``validate_redocking.py`` and ``enrichment_benchmark.py`` can import
these without fragile ``sys.path`` hacks.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from loguru import logger

DOCKER_BIN: str = shutil.which("docker") or "docker"

_PROGRAM_ALIASES: dict[str, str] = {
    "unidock_gpu": "unidock",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DockingParams:
    """Docking parameters shared across calibration targets."""

    exhaustiveness: int
    num_modes: int
    energy_range: float
    timeout_seconds: int
    box_padding: float


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Path / program resolution
# ---------------------------------------------------------------------------


def resolve_program(program: str, override: str | None = None) -> str:
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


def parse_vec(vec_str: str) -> tuple[float, float, float]:
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


def container_path(path: Path, project_root: Path) -> str:
    """Convert a host-side path to the ``/workspace`` container equivalent.

    Args:
        path: Absolute host path under *project_root*.
        project_root: Project root directory.

    Returns:
        Container-relative path string.
    """
    return f"/workspace/{path.relative_to(project_root)}"


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def parse_score(text: str) -> float:
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


def parse_unidock_scores_sdf(text: str) -> list[float]:
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


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def safe_unlink(path: Path, runner: DockerRunner | None, project_root: Path | None = None) -> None:
    """Remove a file, falling back to Docker rm on PermissionError.

    Args:
        path: File to remove.
        runner: Optional Docker runner for fallback removal.
        project_root: Project root (needed for container path conversion).
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except PermissionError:
        if not runner or project_root is None:
            raise
        runner.exec(["rm", "-f", container_path(path, project_root)])


def docker_image_digest(image: str) -> str:
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


def contains_glue_atoms(pdbqt_path: Path) -> bool:
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


# ---------------------------------------------------------------------------
# Receptor / ligand preparation
# ---------------------------------------------------------------------------


def ensure_receptor_pdbqt(
    runner: DockerRunner | None,
    pdb_path: Path,
    pdbqt_path: Path,
    timeout: int,
    project_root: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Ensure a receptor PDBQT file exists, preparing it from PDB if needed.

    Args:
        runner: Docker runner (or None for local execution).
        pdb_path: Input PDB file.
        pdbqt_path: Output PDBQT file.
        timeout: Subprocess timeout in seconds.
        project_root: Project root (needed for container path conversion).
        overwrite: If True, regenerate even if the file exists.

    Raises:
        RuntimeError: If hydrogen addition or receptor preparation fails.
    """
    if pdbqt_path.exists() and not overwrite:
        return
    if pdbqt_path.exists() and overwrite:
        safe_unlink(pdbqt_path, runner, project_root)
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_pdb = pdbqt_path.with_suffix(".fixed.pdb")
    if fixed_pdb.exists() and overwrite:
        safe_unlink(fixed_pdb, runner, project_root)

    _pr = project_root  # shorthand

    if not fixed_pdb.exists():
        fixer_script = (_pr or pdb_path.parent) / "docker" / "freedpp" / "pdbfixer_add_h.py"
        fixer_cmd = [
            "/opt/micromamba/envs/unidock/bin/python" if runner else sys.executable,
            container_path(fixer_script, _pr) if runner and _pr else str(fixer_script),
            "--pdb",
            container_path(pdb_path, _pr) if runner and _pr else str(pdb_path),
            "--out",
            container_path(fixed_pdb, _pr) if runner and _pr else str(fixed_pdb),
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
        container_path(fixed_pdb, _pr) if runner and _pr else str(fixed_pdb),
        "-o",
        container_path(output_basename, _pr) if runner and _pr else str(output_basename),
        "--skip_gpf",
    ]
    if runner:
        result = runner.exec(cmd, timeout=timeout)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Failed to prepare receptor: {err}")


def write_sdf_from_smiles(
    runner: DockerRunner | None,
    smiles: str,
    sdf_path: Path,
    seed: int,
    timeout: int,
    project_root: Path | None = None,
) -> tuple[bool, str]:
    """Generate an SDF file with 3D coordinates from a SMILES string.

    Args:
        runner: Docker runner (or None for local RDKit).
        smiles: Input SMILES string.
        sdf_path: Output SDF file path.
        seed: Random seed for 3D embedding.
        timeout: Subprocess timeout in seconds.
        project_root: Project root (needed for container path conversion).

    Returns:
        Tuple of (success, error_message).
    """
    if runner:
        _pr = project_root
        cmd = [
            "python",
            "/workspace/cli/s1_prepare/rdkit_embed.py",
            "--smiles",
            smiles,
            "--sdf",
            container_path(sdf_path, _pr) if _pr else str(sdf_path),
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


def ensure_ligand_pdbqt(
    runner: DockerRunner | None,
    smiles: str,
    sdf_path: Path,
    out_path: Path,
    timeout: int,
    seed: int,
    project_root: Path | None = None,
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
        project_root: Project root (needed for container path conversion).
        overwrite: If True, regenerate even if the file exists.

    Returns:
        Tuple of (success, error_message).
    """
    if out_path.exists() and not overwrite:
        return True, ""
    if out_path.exists() and overwrite:
        safe_unlink(out_path, runner, project_root)
    if sdf_path.exists() and overwrite:
        safe_unlink(sdf_path, runner, project_root)

    ok, err = write_sdf_from_smiles(runner, smiles, sdf_path, seed=seed, timeout=timeout, project_root=project_root)
    if not ok:
        return False, err

    _pr = project_root
    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        container_path(sdf_path, _pr) if runner and _pr else str(sdf_path),
        "-o",
        container_path(out_path, _pr) if runner and _pr else str(out_path),
        "--rigid_macrocycles",
    ]
    if runner:
        result = runner.exec(cmd, timeout=timeout)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return False, err
    if contains_glue_atoms(out_path):
        return False, "Glue atoms (G0) detected in ligand; macrocycle handling failed"
    return True, ""


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------


def dock_ligand(
    runner: DockerRunner | None,
    binary: str,
    receptor: Path,
    ligand: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    log_path: Path,
    out_path: Path,
    project_root: Path | None = None,
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
        project_root: Project root (needed for container path conversion).

    Returns:
        Best docking score (kcal/mol).

    Raises:
        RuntimeError: If the docking binary exits with an error.
    """
    _pr = project_root
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    binary_name = Path(binary).name.lower()
    supports_log = "unidock" not in binary_name
    cmd = [
        binary,
        "--receptor",
        container_path(receptor, _pr) if runner and _pr else str(receptor),
        "--ligand",
        container_path(ligand, _pr) if runner and _pr else str(ligand),
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
        container_path(out_path, _pr) if runner and _pr else str(out_path),
    ]
    if supports_log:
        cmd += [
            "--log",
            container_path(log_path, _pr) if runner and _pr else str(log_path),
        ]

    if runner:
        result = runner.exec(cmd, timeout=params.timeout_seconds)
        if supports_log and log_path.exists():
            log_text = runner.exec(["cat", container_path(log_path, _pr) if _pr else str(log_path)]).stdout or ""
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

    return parse_score(log_text)


def dock_batch_unidock(  # noqa: PLR0913
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
    project_root: Path | None = None,
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
        project_root: Project root (needed for container path conversion).

    Returns:
        Mapping from ligand path to best score (or None on failure).

    Raises:
        RuntimeError: If the batch fails completely (all scores are None).
    """
    _pr = project_root
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / f"ligand_index_{batch_id:04d}.txt"
    index_lines = [container_path(lig, _pr) if runner and _pr else str(lig) for lig in ligands]
    index_path.write_text("\n".join(index_lines) + "\n")

    for lig in ligands:
        safe_unlink(output_dir / f"{lig.stem}_out.sdf", runner, _pr)

    cmd = [
        binary,
        "--receptor",
        container_path(receptor, _pr) if runner and _pr else str(receptor),
        "--ligand_index",
        container_path(index_path, _pr) if runner and _pr else str(index_path),
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
        container_path(output_dir, _pr) if runner and _pr else str(output_dir),
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
            scores_list = parse_unidock_scores_sdf(sdf_path.read_text())
            scores[lig] = min(scores_list) if scores_list else None
            safe_unlink(sdf_path, runner, _pr)
            continue
        if pdbqt_path.exists():
            try:
                scores[lig] = parse_score(pdbqt_path.read_text())
            except Exception:
                scores[lig] = None
            safe_unlink(pdbqt_path, runner, _pr)
            continue
        scores[lig] = None

    safe_unlink(index_path, runner, _pr)
    if run_error and all(score is None for score in scores.values()):
        raise RuntimeError(run_error)
    if run_error:
        logger.warning("Uni-Dock batch {} completed with warnings: {}", batch_id, run_error)
    return scores


def docking_version(runner: DockerRunner | None, binary: str) -> str:
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
