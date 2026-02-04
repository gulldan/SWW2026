"""Generate LaTeX report from metrics.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import shutil

import click
from loguru import logger

from dualdocking.latex_report import compile_latex_pdf, generate_latex_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@click.command()
@click.option("--mode", type=click.Choice(["draft", "full"]), default="draft")
@click.option("--metrics", default="report/metrics.json")
@click.option("--template", default="templates/manuscript.tex.j2")
@click.option("--output", default="report/manuscript.tex")
def main(mode: str, metrics: str, template: str, output: str) -> None:
    metrics_path = PROJECT_ROOT / metrics
    metrics_data = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    context = {
        "metrics": metrics_data,
        "date": datetime.now(UTC).date().isoformat(),
    }

    tex_path = generate_latex_report(
        context=context,
        template_path=PROJECT_ROOT / template,
        output_tex=PROJECT_ROOT / output,
    )
    logger.info("LaTeX draft -> {}", tex_path)

    bib_src = (PROJECT_ROOT / template).with_name("references.bib")
    if bib_src.exists():
        bib_dst = tex_path.parent / "references.bib"
        if bib_src.resolve() != bib_dst.resolve():
            shutil.copyfile(bib_src, bib_dst)
            logger.info("Bibliography -> {}", bib_dst)

    pdf_path = compile_latex_pdf(tex_path, mode=mode)
    if pdf_path:
        logger.info("PDF generated -> {}", pdf_path)


if __name__ == "__main__":
    main()
