"""Pipeline configuration dataclasses.

This module defines type-safe configuration for the DualDocking pipeline,
replacing ad-hoc dict-based configs with proper dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dualdocking.utils.naming import canonicalize_name

ModeType = Literal["single_a", "single_b", "double"]
LatexModeType = Literal["off", "draft", "full"]
BoltzBackendType = Literal["cli", "nim"]


@dataclass(frozen=True)
class StageTask:
    """A single pipeline task identifier.

    Attributes:
        pair: Target pair identifier (e.g., ``"CDK2_ERK2"``).
        mode: Experiment mode (``"single_a"``, ``"single_b"``, or ``"double"``).
        seed: Random seed for reproducibility.
    """

    pair: str
    mode: ModeType
    seed: int

    def __str__(self) -> str:
        """Format as ``{pair}/{mode}/seed_{seed}``."""
        return f"{self.pair}/{self.mode}/seed_{self.seed}"

    def run_dir(self, project_root: Path) -> Path:
        """Compute the output directory for this task.

        Args:
            project_root: Project root directory.

        Returns:
            Absolute path like ``{project_root}/output/results/{pair}/{mode}/seed_{seed}``.
        """
        return project_root / "output" / "results" / self.pair / self.mode / f"seed_{self.seed}"


@dataclass
class ScorerConfig:
    """Scorer backend configuration.

    Attributes:
        engine: Scoring engine name (e.g., ``"boltz2"``).
        backend: Boltz-2 backend type (``"cli"`` or ``"nim"``).
        nim_base_port: Starting port for NIM instances (one per GPU).
        nim_urls: Explicit NIM URLs (overrides auto-generation from base_port).
    """

    engine: str = "boltz2"
    backend: BoltzBackendType = "cli"
    nim_base_port: int = 8000
    nim_urls: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> ScorerConfig:
        """Parse scorer config from experiment.yaml format.

        Args:
            data: Scorer config dict with keys like ``engine``, ``backend``,
                ``nim.base_port``, etc.

        Returns:
            Parsed scorer configuration.
        """
        engine = str(data.get("engine", data.get("name", "boltz2")))
        backend_raw = str(data.get("backend", "cli")).lower()
        backend: BoltzBackendType = "nim" if backend_raw == "nim" else "cli"

        nim_cfg = data.get("nim", {}) if isinstance(data.get("nim"), dict) else {}
        base_port = int(nim_cfg.get("base_port", 8000))
        urls = data.get("nim_urls", [])

        return cls(
            engine=cls._canonical_engine(engine),
            backend=backend,
            nim_base_port=base_port,
            nim_urls=list(urls) if urls else [],
        )

    @staticmethod
    def _canonical_engine(name: str) -> str:
        """Normalize scorer engine name to canonical form.

        Args:
            name: Raw engine name from config.

        Returns:
            Canonical engine name (e.g., ``"boltz2"``).
        """
        return canonicalize_name(name, {"boltz": "boltz2", "boltz2": "boltz2"})


@dataclass
class PipelineConfig:
    """Complete pipeline configuration.

    Attributes:
        project_root: Absolute path to the project root directory.
        pairs: Mapping of pair IDs to their configuration dicts.
        modes: Experiment modes to run.
        seeds: Random seeds for reproducibility.
        scorer: Scorer backend configuration.
        parallel: Number of parallel tasks (0 = auto-detect from GPUs).
        resume: Whether to skip generation if ``generated`` file exists.
        prepare_freed_data: Whether to check/download FREED++ data deps.
        prepare_baseline: Whether to download ZINC baseline archive.
        baseline_url: Optional custom URL for baseline archive.
        latex_mode: LaTeX report generation mode.
    """

    project_root: Path
    pairs: dict[str, dict]
    modes: list[ModeType] = field(default_factory=lambda: ["single_a", "single_b", "double"])
    seeds: list[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    parallel: int = 0
    resume: bool = False
    prepare_freed_data: bool = True
    prepare_baseline: bool = False
    baseline_url: str | None = None
    latex_mode: LatexModeType = "off"

    @classmethod
    def from_config_dir(cls, config_dir: Path | str) -> PipelineConfig:
        """Load pipeline config from standard YAML files.

        Reads ``pairs.yaml`` and ``experiment.yaml`` from the config directory.

        Args:
            config_dir: Path to the directory containing config YAML files.

        Returns:
            Fully initialized pipeline configuration.
        """
        from dualdocking.config import load_experiment, load_pairs

        config_path = Path(config_dir)
        pairs_data = load_pairs(config_path)["pairs"]
        exp_data = load_experiment(config_path)

        scorer_data = exp_data.get("scorer", {})
        scorer_cfg = ScorerConfig.from_dict(scorer_data)

        # Infer project_root from config_dir (configs/ is in project root)
        project_root = config_path.resolve().parent

        return cls(
            project_root=project_root,
            pairs=pairs_data,
            modes=exp_data.get("modes", ["single_a", "single_b", "double"]),
            seeds=exp_data.get("seeds", [42, 137, 256, 512, 1024]),
            scorer=scorer_cfg,
        )

    def get_active_pairs(self) -> dict[str, dict]:
        """Filter pairs, excluding those with ``set == "blocked"``.

        Returns:
            Dictionary of active pair configs (``set != "blocked"``).
        """
        return {pair_id: pair_cfg for pair_id, pair_cfg in self.pairs.items() if pair_cfg.get("set") != "blocked"}

    def generate_tasks(
        self,
        pair_filter: str | list[str] | None = None,
        seed_filter: list[int] | None = None,
    ) -> list[StageTask]:
        """Generate all pipeline tasks based on config and optional filters.

        Args:
            pair_filter: Single pair ID, list of pair IDs, or ``None`` for all.
            seed_filter: List of seeds to run, or ``None`` for all seeds from config.

        Returns:
            List of tasks (pair, mode, seed combinations).
        """
        active_pairs = self.get_active_pairs()

        if pair_filter:
            pair_ids = [pair_filter] if isinstance(pair_filter, str) else list(pair_filter)
            active_pairs = {k: v for k, v in active_pairs.items() if k in pair_ids}

        seeds = seed_filter if seed_filter is not None else self.seeds

        tasks = [
            StageTask(pair=pair_id, mode=mode, seed=seed) for pair_id in active_pairs for mode in self.modes for seed in seeds
        ]
        return tasks
