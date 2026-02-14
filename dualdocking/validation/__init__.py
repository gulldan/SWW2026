"""Validation: MD simulation and audit checks for top candidate molecules.

Post-generation validation using molecular dynamics and structural auditing.
Used by pipeline stages s6_md and s7_audit.

Modules
-------
md
    Molecular dynamics validation with GROMACS.

    - ``MDConfig``                   -- frozen dataclass of MD run parameters
    - ``MDResult``                   -- per-molecule MD analysis result
    - ``run_md_validation()``        -- run full MD pipeline for a pair
    - ``run_single_molecule_md()``   -- run MD for one molecule x one target
    - ``select_top_molecules()``     -- select top-N by Boltz-2 affinity

audit
    Claim traceability audit (SAP Section 6).

    - ``AuditConfig``               -- frozen dataclass of audit parameters
    - ``run_claim_audit()``         -- run full claim traceability audit
    - ``check_significance()``      -- re-derive significance per SAP 4.4
    - ``load_core_pairs()``         -- extract core pairs from pairs.yaml
    - ``EVIDENCE_MAP``              -- claim scenario -> evidence mapping
"""
