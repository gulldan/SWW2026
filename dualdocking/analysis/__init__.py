"""Statistical analysis: bootstrap inference, effect sizes, enrichment, diversity.

Post-hoc analysis of generated molecules. Used by pipeline stages s5 (analyze)
and s6 (report) to compute metrics, confidence intervals, and quality checks.

Modules
-------
statistical
    Hypothesis testing and effect-size estimation.

    - ``BootstrapResult``                -- dataclass for bootstrap summary
    - ``hierarchical_bootstrap_test()``  -- two-level bootstrap CI (seeds x molecules)
    - ``cliffs_delta()``                 -- Cliff's delta effect size
    - ``cohens_d()``                     -- Cohen's d effect size
    - ``fdr_correction()``               -- Benjamini-Hochberg FDR adjustment
    - ``bootstrap_ci()``                 -- basic percentile bootstrap CI

enrichment
    Virtual-screening enrichment metrics.

    - ``EnrichmentResult``               -- dataclass for enrichment summary
    - ``compute_auroc()``                -- area under the ROC curve
    - ``compute_bedroc()``               -- Boltzmann-enhanced BEDROC (alpha=20)
    - ``compute_enrichment_factor()``    -- enrichment factor at given percentile
    - ``compute_roc_curve()``            -- full ROC curve (FPR, TPR arrays)

diversity
    Chemical diversity of generated molecule sets.

    - ``DiversityResult``                -- dataclass for diversity summary
    - ``compute_all_diversity_metrics()`` -- compute all diversity metrics at once
    - ``uniqueness()``                   -- fraction of unique canonical SMILES
    - ``scaffold_diversity()``           -- fraction of unique Murcko scaffolds
    - ``internal_diversity_tanimoto()``  -- mean pairwise Tanimoto distance
    - ``tanimoto_snn()``                 -- mean similarity to k nearest neighbors

fairness
    Generation budget and duplication checks.

    - ``validate_fairness()``            -- check molecule count and duplication rate
    - ``validate_reward_quality()``      -- check reward signal quality

metrics (subpackage)
    Pipeline-stage metrics computation (extracted from ``cli/s5_analyze/``).

    - ``metrics.aggregation``            -- aggregate scored results across runs
    - ``metrics.computation``            -- summary metrics + bootstrap statistics
    - ``metrics.sensitivity``            -- threshold sensitivity analysis
    - ``metrics.correlation``            -- proxy vs final correlation analysis
"""
