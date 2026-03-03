"""LaTeX report generation utilities.

Renders Jinja2 templates to ``.tex`` files and optionally compiles to PDF
via ``pdflatex`` + ``bibtex``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_latex_report(
    context: dict[str, Any],
    template_path: str | Path,
    output_tex: str | Path,
) -> Path:
    """Render a Jinja2 template to a ``.tex`` file.

    Args:
        context: Template variables passed to ``template.render(**context)``.
        template_path: Path to the Jinja2 ``.tex`` template.
        output_tex: Destination path for the rendered ``.tex`` output.

    Returns:
        Resolved path to the written ``.tex`` file.
    """
    template_path = Path(template_path)
    output_tex = Path(output_tex)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(default_for_string=False, default=False),
    )
    template = env.get_template(template_path.name)
    rendered = template.render(**context)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(rendered)
    return output_tex


def compile_latex_pdf(tex_path: str | Path, mode: str = "draft") -> Path | None:
    """Compile LaTeX into PDF.

    In ``"draft"`` mode, no compilation is performed and ``None`` is returned.
    In ``"full"`` mode, runs ``pdflatex`` -> ``bibtex`` -> ``pdflatex`` x2.

    Args:
        tex_path: Path to the ``.tex`` source file.
        mode: Either ``"draft"`` (skip compilation) or ``"full"``
            (produce PDF).

    Returns:
        Path to the generated PDF, or ``None`` in draft mode or if
        compilation failed to produce a PDF.

    Raises:
        ValueError: If *mode* is not ``"draft"`` or ``"full"``.
        FileNotFoundError: If ``pdflatex`` or ``bibtex`` is missing.
        subprocess.CalledProcessError: If any LaTeX compilation step fails.
    """
    tex_path = Path(tex_path)
    if mode == "draft":
        return None
    if mode != "full":
        raise ValueError("mode must be 'draft' or 'full'")

    workdir = tex_path.parent
    name = tex_path.stem

    for exe in ("pdflatex", "bibtex"):
        if shutil.which(exe) is None:
            raise FileNotFoundError(
                f"Missing required executable: {exe}. Install a LaTeX toolchain (pdflatex + bibtex) or run with mode='draft'."
            )

    def _run(cmd: list[str]) -> None:
        subprocess.run(cmd, cwd=workdir, check=True)  # noqa: S603

    _run(["pdflatex", "-interaction=nonstopmode", tex_path.name])
    _run(["bibtex", name])
    _run(["pdflatex", "-interaction=nonstopmode", tex_path.name])
    _run(["pdflatex", "-interaction=nonstopmode", tex_path.name])

    pdf_path = workdir / f"{name}.pdf"
    return pdf_path if pdf_path.exists() else None
