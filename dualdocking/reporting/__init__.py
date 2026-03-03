"""Reporting: run lifecycle tracking and LaTeX manuscript assembly.

Tracks experiment provenance (configs, git state, timing) and renders the
final LaTeX report from Jinja2 templates populated with analysis results.

Modules
-------
run_manifest
    Lifecycle tracking for experiment runs (JSON manifest files).

    - ``init_manifest()``        -- create a new run manifest with environment info
    - ``update_task()``          -- record task start/completion/failure
    - ``skip_task()``            -- mark a task as skipped
    - ``finalize_manifest()``    -- close the manifest with final status
    - ``load_manifest()``        -- read a manifest file
    - ``manifest_path()``        -- compute manifest file path for a run ID

latex_report
    LaTeX report generation from Jinja2 templates.

    - ``generate_latex_report()`` -- render a Jinja2 template to a .tex file
    - ``compile_latex_pdf()``     -- compile .tex to PDF (pdflatex + bibtex)
"""
