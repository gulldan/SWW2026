"""Validation: MD simulation and audit checks for top candidate molecules.

Post-generation validation using molecular dynamics and structural auditing.
Used by pipeline stages s9_md and s7_audit.

Modules
-------
md_config
    Configuration dataclasses and utility helpers for MD validation.

    - ``MDConfig``                   -- frozen dataclass of MD run parameters
    - ``MDResult``                   -- per-molecule MD analysis result

md_mdp
    MDP file writers for GROMACS simulations.

md_setup
    System preparation: PDB repair, ligand parameterization, topology assembly.

md_analysis
    Trajectory analysis and MM-GBSA.

md_pipeline
    Orchestration: molecule selection, single-molecule pipeline, summary.

    - ``run_md_validation()``        -- run full MD pipeline for a pair
    - ``run_single_molecule_md()``   -- run MD for one molecule x one target
    - ``select_top_molecules()``     -- select top-N by Boltz-2 affinity

md_docker
    Docker runner for the MD validation stage.

audit
    Claim traceability audit (SAP Section 6).

    - ``AuditConfig``               -- frozen dataclass of audit parameters
    - ``run_claim_audit()``         -- run full claim traceability audit
    - ``check_significance()``      -- re-derive significance per SAP 4.4
    - ``load_core_pairs()``         -- extract core pairs from pairs.yaml
    - ``EVIDENCE_MAP``              -- claim scenario -> evidence mapping

render_md_figures
    Publication-quality figure generation for MD validation.

    - ``render_binding_pose()``     -- PyMOL ray-traced binding pose
    - ``plot_rmsd()``               -- RMSD time series (backbone + ligand)
    - ``plot_rmsf()``               -- per-residue RMSF
    - ``plot_rg_hbonds()``          -- Rg + H-bond count
    - ``render_summary_table()``    -- styled metrics table
    - ``plot_sasa()``               -- SASA time series
    - ``plot_pca_variance()``       -- PCA eigenvalue spectrum
    - ``discover_targets()``        -- find completed MD target directories

nw_align
    Needleman-Wunsch global alignment with BLOSUM62 substitution matrix.

    - ``AlignmentResult``           -- alignment identity statistics
    - ``nw_align()``                -- affine-gap NW alignment
    - ``BLOSUM62``                  -- substitution matrix dict

pair_catalog
    HTML catalog parser for the 50-pair target catalog.

    - ``PairRecord``                -- one row from the catalog table
    - ``parse_catalog()``           -- parse HTML to list of PairRecord

domain_identity
    Domain-specific sequence identity analysis.

    - ``DomainIdentityReport``      -- domain NW identity result
    - ``select_domains_for_pair()`` -- auto-discover & select domain ranges
    - ``fetch_uniprot_domains()``   -- (in pair_verify) fetch from UniProt API

pair_verify
    Orchestrator for target pair verification (PDB, DOI, sequence, ChEMBL).
    Re-exports all public symbols from the above submodules.

    - ``verify_pair()``             -- run all requested checks for one pair
    - ``verify_pdb()``              -- RCSB PDB metadata check
    - ``verify_doi()``              -- DOI resolution check
    - ``verify_sequence_identity()``-- NW alignment on UniProt sequences
    - ``verify_chembl()``           -- ChEMBL bioactivity summary
"""
