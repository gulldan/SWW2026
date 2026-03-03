"""Tests for LaTeX report rendering and compilation helpers."""
# ruff: noqa: S101

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from dualdocking.reporting import latex_report


def test_generate_latex_report_writes_output(tmp_path: Path) -> None:
    """Render a minimal template into a .tex file."""
    template = tmp_path / "template.tex.j2"
    template.write_text("Hello {{ name }}")

    out = latex_report.generate_latex_report(
        context={"name": "world"},
        template_path=template,
        output_tex=tmp_path / "manuscript.tex",
    )

    assert out == tmp_path / "manuscript.tex"
    assert out.read_text() == "Hello world"


def test_compile_latex_pdf_draft_returns_none(tmp_path: Path) -> None:
    """Draft mode should skip compilation."""
    tex_path = tmp_path / "manuscript.tex"
    tex_path.write_text("dummy")

    assert latex_report.compile_latex_pdf(tex_path, mode="draft") is None


def test_compile_latex_pdf_invalid_mode_raises(tmp_path: Path) -> None:
    """Unknown mode should raise ValueError."""
    tex_path = tmp_path / "manuscript.tex"
    tex_path.write_text("dummy")

    with pytest.raises(ValueError, match="mode must be 'draft' or 'full'"):
        latex_report.compile_latex_pdf(tex_path, mode="invalid")


def test_compile_latex_pdf_missing_executable_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full mode should fail fast when pdflatex/bibtex is unavailable."""
    tex_path = tmp_path / "manuscript.tex"
    tex_path.write_text("dummy")

    monkeypatch.setattr(latex_report.shutil, "which", lambda _exe: None)

    with pytest.raises(FileNotFoundError, match="Missing required executable: pdflatex"):
        latex_report.compile_latex_pdf(tex_path, mode="full")


def test_compile_latex_pdf_runs_expected_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full mode should run pdflatex/bibtex in the expected order."""
    tex_path = tmp_path / "manuscript.tex"
    tex_path.write_text("dummy")
    expected_pdf = tmp_path / "manuscript.pdf"

    calls: list[list[str]] = []

    monkeypatch.setattr(latex_report.shutil, "which", lambda _exe: "/usr/bin/fake")

    def _fake_run(cmd: list[str], cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert cwd == tmp_path
        assert check is True
        expected_pdf.write_text("%PDF")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(latex_report.subprocess, "run", _fake_run)

    pdf_path = latex_report.compile_latex_pdf(tex_path, mode="full")

    assert pdf_path == expected_pdf
    assert calls == [
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ["bibtex", "manuscript"],
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
    ]


def test_template_figure_path_does_not_add_whitespace() -> None:
    """Template should trim figure placeholder to avoid path corruption."""
    template_path = Path("report/templates/manuscript.tex.j2")
    text = template_path.read_text()

    assert r"\includegraphics[width=0.88\linewidth]{ {{ fig }} }" not in text
    assert r"\includegraphics[width=0.88\linewidth]{ {{- fig -}} }" in text
