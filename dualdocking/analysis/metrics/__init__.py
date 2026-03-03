"""Metrics computation for dual-target experiment analysis.

Library modules extracted from ``cli/s5_analyze/``. Each module provides
a frozen dataclass config and a pure-function entry point.

Modules
-------
aggregation
    Aggregate scored results across pairs, modes, and seeds.

computation
    Compute per-pair summary metrics, hierarchical bootstrap tests,
    degradation guards, and sensitivity analyses.

sensitivity
    Threshold sensitivity analysis across pre-registered thresholds.

correlation
    Proxy (docking) vs final (Boltz-2) correlation analysis.
"""
