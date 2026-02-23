"""High-level pipeline runner API.

This module provides the main ``PipelineRunner`` class for executing
DualDocking experiments programmatically (CLI and Jupyter notebooks).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from tqdm import tqdm

from dualdocking.pipeline import stages
from dualdocking.pipeline.config import PipelineConfig, StageTask
from dualdocking.pipeline.core import (
    build_env_for_task,
    check_nim_health,
    detect_gpu_count,
    log_to_file,
    run_command,
    utc_timestamp,
)
from dualdocking.pipeline.executor import TaskExecutor, run_single_task

if TYPE_CHECKING:
    from dualdocking.pipeline.config import LatexModeType


class PipelineRunner:
    """High-level pipeline orchestrator.

    Provides methods for running individual stages, full experiments,
    and post-processing pipelines.

    Attributes:
        config: Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the pipeline runner.

        Args:
            config: Pipeline configuration object.
        """
        self.config = config

    @classmethod
    def from_config_dir(cls, config_dir: Path | str = "configs/") -> PipelineRunner:
        """Create a runner from standard config directory.

        Args:
            config_dir: Path to directory containing ``pairs.yaml`` and
                ``experiment.yaml``.

        Returns:
            Initialized pipeline runner.
        """
        config = PipelineConfig.from_config_dir(Path(config_dir))
        return cls(config)

    def get_log_path(self, stage: str = "experiment") -> Path:
        """Generate log file path for a pipeline run.

        Args:
            stage: Stage name for log file prefix.

        Returns:
            Path to log file in ``report/`` directory.
        """
        run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
        log_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return self.config.project_root / "report" / f"{stage}_{run_id}_{log_ts}.log"

    def run_cmd(
        self,
        cmd: list[str],
        log_path: Path,
        gpu_id: int | None = None,
    ) -> None:
        """Execute a command with environment setup.

        Args:
            cmd: Command tokens to execute.
            log_path: Path to log file.
            gpu_id: CUDA device index, or ``None`` for default.
        """
        env = build_env_for_task(self.config.scorer, gpu_id)
        run_command(
            cmd,
            log_path,
            self.config.project_root,
            env=env,
            allow_inprocess=False,  # Always False for thread-safety
        )

    def _prepare_scorer_backend(self) -> None:
        """Validate and prepare the scorer backend (e.g., NIM health checks).

        Raises:
            RuntimeError: If NIM backend is configured but no instances are healthy.
        """
        if self.config.scorer.engine != "boltz2" or self.config.scorer.backend != "nim":
            return

        # Auto-generate NIM URLs if not explicitly provided
        if not self.config.scorer.nim_urls:
            n_gpus = detect_gpu_count()
            base_port = self.config.scorer.nim_base_port
            self.config.scorer.nim_urls = [f"http://localhost:{base_port + i}" for i in range(n_gpus)]
            logger.info(
                "Scorer: boltz2 (NIM backend, {} GPUs → {} instance(s): {})",
                n_gpus,
                len(self.config.scorer.nim_urls),
                ", ".join(self.config.scorer.nim_urls),
            )

        # Health-check all NIM URLs
        healthy_urls = check_nim_health(self.config.scorer.nim_urls)
        if not healthy_urls:
            raise RuntimeError(
                f"No NIM Boltz-2 instances are healthy. "
                f"Expected {len(self.config.scorer.nim_urls)} instance(s). "
                f"Start NIM containers first (see docs/setup.md)."
            )
        if len(healthy_urls) < len(self.config.scorer.nim_urls):
            dead = [u for u in self.config.scorer.nim_urls if u not in healthy_urls]
            logger.warning(
                "Only {}/{} NIM instances healthy. Dead: {}. Proceeding with healthy only.",
                len(healthy_urls),
                len(self.config.scorer.nim_urls),
                ", ".join(dead),
            )
        self.config.scorer.nim_urls = healthy_urls

    def run_stage_fetch(self, log_path: Path | None = None) -> None:
        """Run the fetch stage (download data dependencies).

        Args:
            log_path: Optional log file path (auto-generated if ``None``).
        """
        log_path = log_path or self.get_log_path("fetch")
        logger.info("=== Stage: fetch ===")
        self.run_cmd(stages.build_fetch_command(), log_path)
        logger.info("=== Stage fetch complete ===")

    def run_stage_prepare(self, log_path: Path | None = None) -> None:
        """Run the prepare stage (PDB cleanup, FASTA, MSA, PDBQT).

        Args:
            log_path: Optional log file path (auto-generated if ``None``).
        """
        log_path = log_path or self.get_log_path("prepare")
        logger.info("=== Stage: prepare ===")
        self.run_cmd(stages.build_prepare_command(), log_path)
        logger.info("=== Stage prepare complete ===")

    def run_stage_calibrate(self, log_path: Path | None = None) -> None:
        """Run the calibration stage (dock calibration set, compute normalization).

        Args:
            log_path: Optional log file path (auto-generated if ``None``).
        """
        log_path = log_path or self.get_log_path("calibrate")
        logger.info("=== Stage: calibrate ===")
        for cmd in stages.build_calibrate_commands():
            self.run_cmd(cmd, log_path)
        logger.info("=== Stage calibrate complete ===")

    def run_stage_analyze(self, log_path: Path | None = None) -> None:
        """Run the analysis stage (aggregate, metrics, sensitivity, correlation).

        Args:
            log_path: Optional log file path (auto-generated if ``None``).
        """
        log_path = log_path or self.get_log_path("analyze")
        logger.info("=== Stage: analyze ===")
        for cmd in stages.build_analyze_commands():
            self.run_cmd(cmd, log_path)
        logger.info("=== Stage analyze complete ===")

    def run_stage_report(
        self,
        latex_mode: LatexModeType = "off",
        log_path: Path | None = None,
    ) -> None:
        """Run the report stage (plots, rendering, optional LaTeX).

        Args:
            latex_mode: LaTeX compilation mode.
            log_path: Optional log file path (auto-generated if ``None``).
        """
        log_path = log_path or self.get_log_path("report")
        logger.info("=== Stage: report ===")
        for cmd in stages.build_report_commands(latex_mode):
            self.run_cmd(cmd, log_path)
        logger.info("=== Stage report complete ===")

    def run_stage_audit(self, log_path: Path | None = None) -> None:
        """Run the audit stage (claim traceability).

        Args:
            log_path: Optional log file path (auto-generated if ``None``).
        """
        log_path = log_path or self.get_log_path("audit")
        logger.info("=== Stage: audit ===")
        self.run_cmd(stages.build_audit_command(), log_path)
        logger.info("=== Stage audit complete ===")

    def run_generate_and_score(
        self,
        pairs: list[str] | None = None,
        mode: str | None = None,
        seeds: list[int] | None = None,
        *,
        resume: bool | None = None,
        parallel: int | None = None,
        log_path: Path | None = None,
    ) -> None:
        """Run generate + score for specified tasks.

        Args:
            pairs: Optional pair filter list (``None`` = all pairs).
            mode: Optional mode filter (``None`` = all modes).
            seeds: Optional seed filter list (``None`` = all seeds from config).
            resume: Whether to skip generation if ``generated`` exists
                (defaults to ``config.resume``).
            parallel: Number of parallel tasks (defaults to ``config.parallel``).
            log_path: Optional log file path (auto-generated if ``None``).
        """
        resume = self.config.resume if resume is None else resume
        parallel = self.config.parallel if parallel is None else parallel
        log_path = log_path or self.get_log_path("generate_score")

        # Build task list with filters
        tasks = self.config.generate_tasks(pair_filter=pairs, seed_filter=seeds)

        # Filter by mode if specified
        if mode is not None:
            tasks = [t for t in tasks if t.mode == mode]

        if not tasks:
            logger.warning("No tasks to run after applying filters")
            return

        # Auto-detect parallelism
        if parallel <= 0:
            parallel = detect_gpu_count()
            logger.info("Auto-detected {} GPU(s) → parallel={}", parallel, parallel)

        # Prepare scorer backend
        self._prepare_scorer_backend()

        # Initialize run manifest
        from dualdocking.reporting.run_manifest import (
            finalize_manifest,
            init_manifest,
            manifest_path,
        )

        results_dir = self.config.project_root / "output" / "results"
        run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
        mf_path = manifest_path(results_dir, run_id)
        init_manifest(
            mf_path,
            run_id=run_id,
            config_dir=self.config.project_root / "configs",
            tasks=[(t.pair, t.mode, t.seed) for t in tasks],
            parallel=parallel,
            resume=resume,
        )

        # Execute tasks
        log_to_file(log_path, f"[{utc_timestamp()}] START tasks={len(tasks)} parallel={parallel} resume={resume}")

        executor = TaskExecutor(n_gpus=detect_gpu_count(), max_workers=parallel)

        def task_fn(task: StageTask, gpu_id: int | None) -> None:
            """Wrapper for run_single_task with bound parameters."""
            run_single_task(
                task=task,
                project_root=self.config.project_root,
                log_path=log_path,
                resume=resume,
                manifest_path=mf_path,
                gpu_id=gpu_id,
                generate_cmd_fn=stages.build_generate_command,
                score_cmd_fn=stages.build_score_command,
                run_cmd_fn=self.run_cmd,
            )

        run_failed = False
        try:
            n_completed, n_failed = executor.execute(tasks, task_fn)
            logger.info("Tasks finished: {} completed, {} failed out of {}", n_completed, n_failed, len(tasks))
            run_failed = n_failed > 0
        except Exception:
            run_failed = True
            raise
        finally:
            finalize_manifest(mf_path, status="failed" if run_failed else "completed")

    def run_full_experiment(
        self,
        pairs: list[str] | None = None,
        seeds: list[int] | None = None,
        *,
        resume: bool | None = None,
        parallel: int | None = None,
        with_post: bool = False,
        latex_mode: LatexModeType = "off",
    ) -> None:
        """Run the complete experiment pipeline.

        Executes data preparation, calibration, generate+score, and optionally
        post-processing.

        Args:
            pairs: List of pair IDs to run (``None`` = all active pairs).
            seeds: List of seeds to run (``None`` = all seeds from config).
            resume: Whether to skip generation if ``generated`` exists.
            parallel: Number of parallel tasks (0 = auto-detect).
            with_post: Whether to run post-processing (analyze + report).
            latex_mode: LaTeX compilation mode for report stage.
        """
        resume = self.config.resume if resume is None else resume
        parallel = self.config.parallel if parallel is None else parallel
        log_path = self.get_log_path("full_experiment")

        logger.info("Starting full experiment pipeline")

        # Prepare data
        if self.config.prepare_freed_data or self.config.prepare_baseline:
            logger.info("[PREP] Checking data dependencies...")
            for cmd in stages.build_prepare_commands(
                self.config.prepare_freed_data,
                self.config.prepare_baseline,
                self.config.baseline_url,
            ):
                self.run_cmd(cmd, log_path)
            logger.info("[PREP] Data dependencies ready")

        # Ensure calibration for all pairs
        tasks = self.config.generate_tasks(pair_filter=pairs, seed_filter=seeds)
        unique_pairs = list(dict.fromkeys(t.pair for t in tasks))
        logger.info("[CALIBRATION] Checking {} pair(s): {}", len(unique_pairs), ", ".join(unique_pairs))
        for pair_id in unique_pairs:
            pair_cfg = self.config.pairs[pair_id]
            stages.ensure_calibration(pair_id, pair_cfg, self.config.project_root, log_path, self.run_cmd)
        logger.info("[CALIBRATION] All pairs calibrated")

        # Run generate + score
        self.run_generate_and_score(
            pairs=pairs,
            seeds=seeds,
            resume=resume,
            parallel=parallel,
            log_path=log_path,
        )

        # Post-processing
        if with_post:
            logger.info("[POST] Starting post-processing pipeline")
            log_to_file(log_path, f"[{utc_timestamp()}] POST-START")

            post_cmds = stages.build_analyze_commands(pairs=unique_pairs, seeds=seeds) + stages.build_report_commands(latex_mode)
            post_names = [
                "aggregate",
                "metrics",
                "threshold_sensitivity",
                "proxy_correlation",
                "summarize",
                "plots",
                "render_molecules",
            ]
            if latex_mode != "off":
                post_names.append("latex")

            for i, cmd in enumerate(tqdm(post_cmds, desc="Post-pipeline", unit="step", dynamic_ncols=True)):
                step_name = post_names[i] if i < len(post_names) else cmd[-1]
                logger.info("[POST] {} — started", step_name)
                self.run_cmd(cmd, log_path)
                logger.info("[POST] {} — done", step_name)

            log_to_file(log_path, f"[{utc_timestamp()}] POST-DONE")
            logger.info("[POST] Post-processing complete")

        logger.info("Full experiment pipeline complete")

    def run_post_pipeline(self, latex_mode: LatexModeType = "draft") -> None:
        """Run the post-processing pipeline (analyze + report).

        Args:
            latex_mode: LaTeX compilation mode.
        """
        log_path = self.get_log_path("post")
        log_to_file(log_path, f"[{utc_timestamp()}] START latex_mode={latex_mode}")

        logger.info("Starting post-processing pipeline")
        self.run_stage_analyze(log_path)
        self.run_stage_report(latex_mode, log_path)

        log_to_file(log_path, f"[{utc_timestamp()}] DONE")
        logger.info("Post-processing pipeline complete")
