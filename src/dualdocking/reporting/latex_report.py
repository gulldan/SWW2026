"""LaTeX report generation utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_latex_report(
    context: dict[str, Any],
    template_path: str | Path,
    output_tex: str | Path,
) -> Path:
    """Render Jinja2 template to a .tex file."""
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
    """Compile LaTeX into PDF (full mode)."""
    tex_path = Path(tex_path)
    if mode == "draft":
        return None
    if mode != "full":
        raise ValueError("mode must be 'draft' or 'full'")

    workdir = tex_path.parent
    name = tex_path.stem

    def _run(cmd: list[str]) -> None:
        subprocess.run(cmd, cwd=workdir, check=True, capture_output=True)  # noqa: S603

    _run(["pdflatex", "-interaction=nonstopmode", tex_path.name])
    _run(["bibtex", name])
    _run(["pdflatex", "-interaction=nonstopmode", tex_path.name])
    _run(["pdflatex", "-interaction=nonstopmode", tex_path.name])

    pdf_path = workdir / f"{name}.pdf"
    return pdf_path if pdf_path.exists() else None
