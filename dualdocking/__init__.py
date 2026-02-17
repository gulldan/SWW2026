"""DualDocking: single-target vs dual-target RL molecule generation.

Core library for the dual-target docking experiment. Scripts in ``cli/``
use this library to run the pipeline.

Subpackages
-----------
scoring
    Docking objectives, Boltz-2 affinity scoring, and disk cache.
generation
    FREED++ Docker-based molecule generation.
analysis
    Statistical analysis (bootstrap, FDR, Cliff's delta, enrichment, diversity).
data
    SMILES sanitization, baseline loading, and ChEMBL validation.
reporting
    Run lifecycle tracking and LaTeX report assembly.

Configuration helpers (top-level)
---------------------------------
config.load_pairs()
    Load target pairs from ``configs/pairs.yaml`` with schema validation.
config.load_metrics()
    Load hit thresholds from ``configs/metrics.yaml``.
config.load_experiment()
    Load experiment parameters from ``configs/experiment.yaml``.
config.load_reward_modes()
    Load declarative reward modes from ``configs/reward_modes.yaml``.
"""

__version__ = "0.1.0"
