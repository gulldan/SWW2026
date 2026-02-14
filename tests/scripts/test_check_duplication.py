"""Tests for cli/s7_audit/check_duplication.py."""

from __future__ import annotations

from pathlib import Path

import cli.s7_audit.check_duplication as mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_exact_clone_group_detected(tmp_path: Path) -> None:
    """Exact structural clone groups should be detected across files."""
    _write(
        tmp_path / "a.py",
        """
def alpha(x, y):
    value = x + y
    return value * 2
""",
    )
    _write(
        tmp_path / "b.py",
        """
def beta(x, y):
    value = x + y
    return value * 2
""",
    )

    files = mod._collect_python_files(tmp_path, include_tests=False)
    records = mod._collect_function_records(files, tmp_path, min_lines=3, token_k=3)
    groups = mod._group_records(records, "exact_key")

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_structural_clone_group_detected(tmp_path: Path) -> None:
    """Structural clones should match after strong normalization."""
    _write(
        tmp_path / "a.py",
        """
def score(a, b):
    temp = (a + 10) * 3
    if temp > b:
        return temp - b
    return b - temp
""",
    )
    _write(
        tmp_path / "b.py",
        """
def evaluate(x, y):
    q = (x + 42) * 5
    if q > y:
        return q - y
    return y - q
""",
    )

    files = mod._collect_python_files(tmp_path, include_tests=False)
    records = mod._collect_function_records(files, tmp_path, min_lines=5, token_k=3)
    groups = mod._group_records(records, "structural_key")
    structural_only = [group for group in groups if len({rec.exact_key for rec in group}) > 1]

    assert len(structural_only) == 1
    assert len(structural_only[0]) == 2


def test_token_near_clone_pair_detected(tmp_path: Path) -> None:
    """Token near-clones should be detected when similarity is high."""
    _write(
        tmp_path / "a.py",
        """
def build_rows(items):
    rows = []
    for item in items:
        if item % 2 == 0:
            rows.append(item * 2)
        else:
            rows.append(item + 1)
    return rows
""",
    )
    _write(
        tmp_path / "b.py",
        """
def build_entries(values):
    out = []
    for value in values:
        if value % 2 == 0:
            out.append(value * 2)
        else:
            out.append(value + 3)
    return out
""",
    )

    files = mod._collect_python_files(tmp_path, include_tests=False)
    records = mod._collect_function_records(files, tmp_path, min_lines=6, token_k=4)
    pairs = mod._detect_token_pairs(records, near_threshold=0.6, line_delta=4)

    assert len(pairs) == 1
    assert pairs[0].similarity >= 0.6
