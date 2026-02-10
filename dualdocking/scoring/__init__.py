"""Scoring and docking: multi-objective functions, Boltz-2 affinity, and disk cache.

This subpackage implements all scoring-related logic used during RL training
(UniDock docking) and post-hoc evaluation (Boltz-2 learned affinity).

Modules
-------
objectives
    Multi-objective scalarization functions for combining two target scores.

    - ``ObjectiveKind``          -- enum of supported scalarizations
    - ``compute_combined()``     -- dispatch to the selected objective function
    - ``combined_min()``         -- worst-case (max of two scores)
    - ``combined_harmonic()``    -- harmonic mean through utility conversion
    - ``combined_weighted_sum()`` -- linear scalarization with weight
    - ``combined_tchebycheff()`` -- Tchebycheff decomposition for non-convex fronts
    - ``pareto_mask()``          -- boolean mask of Pareto-optimal points (O(n log n))
    - ``hypervolume_contributions()`` -- per-point hypervolume contribution (2-D)
    - ``parse_objectives_config()``   -- parse objectives from experiment.yaml

boltz2
    Boltz-2 affinity scorer with disk cache and NIM backend support.

    - ``score_batch()``          -- score a list of SMILES against a target
    - ``score_single()``         -- score one SMILES against a target
    - ``get_scoring_context_info()`` -- current backend/version/accelerator info
    - ``prune_stale_cache()``    -- remove entries from outdated contexts

cache_io
    Process-safe JSON cache with atomic writes and file locking (fcntl).

    - ``load_cache()``           -- read cache file under flock
    - ``save_cache()``           -- atomic read-merge-write under flock

binding_box
    PDB binding-site extraction from co-crystallized ligands.

    - ``LigandInfo``             -- dataclass with ligand metadata and coordinates
    - ``extract_ligand_info()``  -- extract ligand residue info from PDB
    - ``extract_binding_box()``  -- compute center and size of binding pocket
    - ``compute_box()``          -- box from raw coordinates with padding
"""
