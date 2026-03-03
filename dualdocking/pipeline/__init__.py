"""Pipeline orchestration for DualDocking experiments.

This module provides a programmatic API for running the DualDocking pipeline,
suitable for both CLI scripts and Jupyter notebooks.

Example::

    from dualdocking.pipeline import PipelineRunner, PipelineConfig

    # From config files
    runner = PipelineRunner.from_config_dir("configs/")
    runner.run_full_experiment(pairs=["CDK2_ERK2"], seeds=[42])

    # Programmatic configuration
    config = PipelineConfig(
        pairs={"CDK2_ERK2": {...}},
        modes=["single_a", "single_b", "double"],
        seeds=[42, 137],
    )
    runner = PipelineRunner(config)
    runner.run_generate_and_score(pairs=["CDK2_ERK2"], mode="double", seeds=[42])
"""

from dualdocking.pipeline.config import PipelineConfig, ScorerConfig, StageTask
from dualdocking.pipeline.runner import PipelineRunner

__all__ = [
    "PipelineConfig",
    "PipelineRunner",
    "ScorerConfig",
    "StageTask",
]
