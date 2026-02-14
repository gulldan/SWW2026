"""Calibration stage: docking calibration, normalization, redocking, enrichment, power analysis.

Extracts core logic from cli/s2_calibrate/ into reusable library modules.

Modules
-------
docking
    Calibration set docking: load SMILES, prepare ligands, dock against targets.
normalization
    Compute z-score normalization statistics from calibration scores.
redocking
    Re-docking validation: dock co-crystallized ligands, compute RMSD.
enrichment
    Enrichment benchmark: actives vs decoys AUROC/BEDROC/EF metrics.
power
    Statistical power analysis for paired t-test and Cliff's delta.
"""
