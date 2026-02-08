"""Generate modular LaTeX manuscript from experiment artifacts."""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import yaml
from loguru import logger

from dualdocking.latex_report import compile_latex_pdf, generate_latex_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _load_manifest(path: Path) -> dict[str, Any]:
    fallback: dict[str, Any] = {
        "title": "Dual-Target Docking: Modular Research Report",
        "author": "Automated Pipeline",
        "abstract_file": "00_abstract.tex",
        "keywords": ["dual-target docking", "FREED++", "Boltz-2", "reproducibility"],
        "sections": [
            {"id": "introduction", "title": "Introduction", "file": "01_introduction.tex"},
            {"id": "related_work", "title": "Related Work", "file": "02_related_work.tex"},
            {"id": "methods", "title": "Methods", "file": "03_methods_notes.tex"},
            {
                "id": "protocol",
                "title": "Experimental Protocol and Fairness Controls",
                "file": "04_experimental_protocol.tex",
            },
            {"id": "results_scope", "title": "Result Scope and Decision Criteria", "file": "05_results_scope.tex"},
            {"id": "limitations", "title": "Limitations", "file": "06_limitations.tex"},
            {"id": "conclusion", "title": "Conclusion and Next Steps", "file": "07_conclusion.tex"},
        ],
    }
    loaded = _load_yaml(path)
    if not loaded:
        return fallback
    merged = {**fallback, **loaded}
    if "sections" not in merged or not isinstance(merged["sections"], list):
        merged["sections"] = fallback["sections"]
    return merged


def _load_sections(chapters_dir: Path, manifest: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    abstract_file = str(manifest.get("abstract_file", "00_abstract.tex"))
    abstract_path = chapters_dir / abstract_file
    abstract = abstract_path.read_text() if abstract_path.exists() else ""

    sections: list[dict[str, str]] = []
    for idx, section in enumerate(manifest.get("sections", [])):
        sec = section if isinstance(section, dict) else {}
        sec_id = str(sec.get("id", f"section_{idx + 1}"))
        title = str(sec.get("title", sec_id.replace("_", " ").title()))
        filename = str(sec.get("file", ""))
        content = ""
        if filename:
            path = chapters_dir / filename
            if path.exists():
                content = path.read_text()
            elif bool(sec.get("required", False)):
                logger.warning("Missing required chapter file: {}", path)
        sections.append({"id": sec_id, "title": title, "content": content})
    return abstract, sections


def _pair_order(pairs_cfg: dict[str, Any], metrics: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    order = list(pairs_cfg.get("pairs", {}).keys())
    for source in (metrics, stats):
        for pair_id in source:
            if pair_id not in order:
                order.append(pair_id)
    return order


def _build_metric_rows(pairs_cfg: dict[str, Any], metrics: dict[str, Any], pair_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        pair_meta = pairs_cfg.get("pairs", {}).get(pair_id, {})
        target_a = pair_meta.get("target_a", {}).get("id", "A")
        target_b = pair_meta.get("target_b", {}).get("id", "B")
        per_mode = metrics.get(pair_id, {}) if isinstance(metrics.get(pair_id, {}), dict) else {}
        for mode in ("single_a", "single_b", "double"):
            data = per_mode.get(mode)
            if not isinstance(data, dict):
                continue
            row: dict[str, Any] = {
                "pair_id": pair_id,
                "target_a": target_a,
                "target_b": target_b,
                "mode": mode,
                "hr_a_mean": data.get("hr_a_mean"),
                "hr_a_sem": data.get("hr_a_sem"),
                "hr_b_mean": data.get("hr_b_mean"),
                "hr_b_sem": data.get("hr_b_sem"),
                "hr_both_mean": data.get("hr_both_mean"),
                "hr_both_sem": data.get("hr_both_sem"),
                "balance_mean": data.get("balance_mean"),
                "balance_sem": data.get("balance_sem"),
            }
            row.update({key: val for key, val in data.items() if key.startswith("combined_") and key.endswith(("_mean", "_sem"))})
            rows.append(row)
    return rows


def _objective_columns_from_metrics(metrics: dict[str, Any]) -> list[str]:
    """Extract unique objective column base names from metrics data."""
    cols: list[str] = []
    seen: set[str] = set()
    for pair_data in metrics.values():
        if not isinstance(pair_data, dict):
            continue
        for mode_data in pair_data.values():
            if not isinstance(mode_data, dict):
                continue
            for key in mode_data:
                if key.startswith("combined_") and key.endswith("_mean"):
                    base = key[: -len("_mean")]
                    if base not in seen:
                        cols.append(base)
                        seen.add(base)
    return cols


def _build_stats_rows(stats: dict[str, Any], pair_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        tests = stats.get(pair_id, {})
        if not isinstance(tests, dict):
            continue
        for test_id, entry in sorted(tests.items()):
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "pair_id": pair_id,
                    "test_id": test_id,
                    "comparison": entry.get("comparison"),
                    "metric": entry.get("metric"),
                    "observed_diff": entry.get("observed_diff"),
                    "p_value_adjusted": entry.get("p_value_adjusted"),
                    "cliffs_delta": entry.get("cliffs_delta"),
                    "ci_lower": entry.get("ci_lower"),
                    "ci_upper": entry.get("ci_upper"),
                    "significant": bool(entry.get("significant", False)),
                }
            )
    return rows


def _figure_paths(figures_dir: Path) -> list[str]:
    if not figures_dir.exists():
        return []
    return [str(p.relative_to(PROJECT_ROOT)) for p in sorted(figures_dir.glob("*.png"))]


def _summary_excerpt(summary_path: Path, max_chars: int = 2000) -> str:
    if not summary_path.exists():
        return ""
    text = summary_path.read_text().strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[...]"


@click.command()
@click.option("--mode", type=click.Choice(["draft", "full"]), default="draft")
@click.option("--metrics", default="report/metrics.json")
@click.option("--stats", default="report/stats.json")
@click.option("--summary", "summary_path", default="report/summary.txt")
@click.option("--template", default="templates/manuscript.tex.j2")
@click.option("--output", default="report/manuscript.tex")
@click.option("--chapters-dir", default="report/chapters")
@click.option("--chapters-manifest", default="report/chapters/manifest.yaml")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--master-config", default="configs/master_config.yaml")
def main(
    mode: str,
    metrics: str,
    stats: str,
    summary_path: str,
    template: str,
    output: str,
    chapters_dir: str,
    chapters_manifest: str,
    pairs_config: str,
    master_config: str,
) -> None:
    metrics_data = _load_json(PROJECT_ROOT / metrics)
    stats_data = _load_json(PROJECT_ROOT / stats)
    pairs_cfg = _load_yaml(PROJECT_ROOT / pairs_config)
    master_cfg = _load_yaml(PROJECT_ROOT / master_config)

    chapters_path = PROJECT_ROOT / chapters_dir
    manifest = _load_manifest(PROJECT_ROOT / chapters_manifest)
    abstract_text, sections = _load_sections(chapters_path, manifest)

    pair_ids = _pair_order(pairs_cfg, metrics_data, stats_data)
    metric_rows = _build_metric_rows(pairs_cfg, metrics_data, pair_ids)
    stats_rows = _build_stats_rows(stats_data, pair_ids)
    figures = _figure_paths(PROJECT_ROOT / "report" / "figures")
    run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
    keywords = manifest.get("keywords", [])
    keyword_text = ", ".join(str(k) for k in keywords if str(k).strip())

    objective_columns = _objective_columns_from_metrics(metrics_data)

    context = {
        "title": str(manifest.get("title", "Dual-Target Docking: Modular Research Report")),
        "author": str(manifest.get("author", "Automated Pipeline")),
        "abstract_text": abstract_text,
        "sections": sections,
        "keyword_text": keyword_text,
        "metrics": metrics_data,
        "stats": stats_data,
        "metric_rows": metric_rows,
        "stats_rows": stats_rows,
        "objective_columns": objective_columns,
        "figures": figures,
        "summary_excerpt": _summary_excerpt(PROJECT_ROOT / summary_path),
        "master_config": master_cfg,
        "run_id": run_id,
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
