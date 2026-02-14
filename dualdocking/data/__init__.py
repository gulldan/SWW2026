"""Data utilities: SMILES sanitization, baseline loading, and ChEMBL validation.

Handles molecular data ingestion and cleaning for the pipeline. Used by fetch
and preparation stages (s0, s1) and by scoring/analysis stages for validation.

Subpackages
-----------
fetch
    Data fetching: PDB/FASTA downloads, MSA computation, SMILES sources,
    Docker image builds.

prep
    Target preparation: PDB-to-PDBQT conversion, baseline sampling,
    archive-to-Parquet conversion.

Modules
-------
chem_sanitize
    SMILES validation, canonicalization, and deduplication.

    - ``SanitizeStats``              -- dataclass with sanitization summary
    - ``sanitize_smiles()``          -- validate, filter, and deduplicate SMILES
    - ``canonicalize()``             -- RDKit canonical SMILES or None if invalid

baseline
    Baseline comparison via ZINC calibration sets.

    - ``bootstrap_stability_check()`` -- check mean/std stability via bootstrap CI

chembl_validation
    ChEMBL target validation and known-active retrieval.

    - ``TARGET_CHEMBL_MAP``          -- internal target ID to ChEMBL ID mapping
    - ``fetch_actives()``            -- fetch active SMILES from ChEMBL REST API
    - ``find_dual_actives()``        -- find compounds active against both targets
    - ``compute_nearest_tanimoto()`` -- nearest-neighbor Tanimoto to known actives
    - ``enrichment_factor()``        -- enrichment factor at K
"""
