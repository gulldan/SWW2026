#!/usr/bin/env python3
"""Lint the project documentation against the code and configs it describes.

Checks that prose claims still match reality: pair identifiers, target
calibration coverage, PDB structures, repo paths, relative links, counted
claims, navigation-map coverage, mermaid syntax and cross-document duplication.

Usage:
    python3 scripts/lint_docs.py [--json] [--root PATH]

Exit code is 1 when any finding is reported, 0 when the docs are clean, so the
script can gate CI.

A file may opt out of the history-sensitive checks (PAIR-STALE, PDB-MISMATCH)
by carrying the marker ``<!-- lint: historical-record -->``; use it only for
frozen protocol records whose whole point is to state what was true earlier.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

HISTORICAL_MARKER = "<!-- lint: historical-record -->"
HISTORICAL_CHECKS = {"PAIR-STALE", "PDB-MISMATCH"}

# Paths referenced in prose that the linter resolves against the repo.
PATH_PREFIXES = ("cli/", "dualdocking/", "configs/", "scripts/", "tests/", "docker/", "slurm/")


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    line: int
    message: str


def _run_git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _load_yaml(path: Path) -> dict:
    """Load YAML, preferring PyYAML and falling back to a flat-key scan."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        # Fallback: collect top-level-ish ``key:`` names so the linter still
        # runs without PyYAML. Nested values are not reconstructed.
        return {"__keys__": re.findall(r"^\s{0,4}([A-Za-z][\w-]*):", text, re.M)}


def collect_docs(root: Path) -> list[Path]:
    docs = sorted((root / "docs").glob("*.md")) if (root / "docs").is_dir() else []
    roots = [root / n for n in ("README.md", "AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md")]
    return [p for p in [*roots, *docs] if p.is_file()]


def build_vocabulary(root: Path) -> dict:
    """Pair ids, target ids, PDB ids and calibration keys, read from configs."""
    pairs_path = root / "configs" / "pairs.yaml"
    norm_path = root / "configs" / "normalization.yaml"

    pair_ids: set[str] = set()
    targets: set[str] = set()
    pdb_of: dict[str, str] = {}

    if pairs_path.is_file():
        data = _load_yaml(pairs_path)
        pairs = data.get("pairs", data) if isinstance(data, dict) else {}
        if isinstance(pairs, dict):
            for pid, cfg in pairs.items():
                if not isinstance(cfg, dict):
                    continue
                pair_ids.add(pid)
                for role in ("target_a", "target_b"):
                    t = cfg.get(role)
                    if isinstance(t, dict) and t.get("id"):
                        targets.add(str(t["id"]))
                        if t.get("pdb_id"):
                            pdb_of[str(t["id"])] = str(t["pdb_id"]).upper()

    calibrated: set[str] = set()
    if norm_path.is_file():
        data = _load_yaml(norm_path)
        block = data.get("targets", data) if isinstance(data, dict) else {}
        if isinstance(block, dict):
            calibrated = {k for k in block if not k.startswith("__")}

    return {
        "pair_ids": pair_ids,
        "targets": targets,
        "pdb_of": pdb_of,
        "calibrated": calibrated,
        # Any symbol the project has ever used as a target name.
        "known_targets": targets | calibrated,
    }


def iter_lines(text: str):
    for i, line in enumerate(text.split("\n"), start=1):
        yield i, line


def fenced_spans(text: str) -> set[int]:
    """Line numbers (1-based) that sit inside a fenced code block."""
    inside, spans = False, set()
    for i, line in enumerate(text.split("\n"), start=1):
        if line.lstrip().startswith("```"):
            inside = not inside
            spans.add(i)
            continue
        if inside:
            spans.add(i)
    return spans


def check_pairs(doc: Path, text: str, vocab: dict, rel: str) -> list[Finding]:
    """Pair-shaped tokens that configs/pairs.yaml no longer defines."""
    out = []
    token = re.compile(r"\b([A-Za-z][A-Za-z0-9]{1,9})_([A-Za-z][A-Za-z0-9]{1,9})\b")
    seen: set[tuple[str, int]] = set()
    for lineno, line in iter_lines(text):
        for m in token.finditer(line):
            whole, a, b = m.group(0), m.group(1), m.group(2)
            if whole in vocab["pair_ids"]:
                continue
            # Only consider it a pair reference when both halves look like
            # target symbols and at least one is a name the project knows.
            if a not in vocab["known_targets"] and b not in vocab["known_targets"]:
                continue
            if a not in vocab["known_targets"] or b not in vocab["known_targets"]:
                continue
            if (whole, lineno) in seen:
                continue
            seen.add((whole, lineno))
            out.append(Finding("PAIR-STALE", rel, lineno,
                               f"`{whole}` is not a pair in configs/pairs.yaml"))
    return out


def check_pdb(doc: Path, text: str, vocab: dict, rel: str) -> list[Finding]:
    """A PDB id named next to a target that pairs.yaml maps elsewhere.

    A candidate must be written the way the PDB writes them -- a leading digit
    then uppercase alphanumerics, with at least one letter. That excludes bare
    years (``2025``) and lowercase dosages (``10mg``), both of which share the
    four-character shape. The id is attributed to the nearest preceding target
    name so that ``FGFR1 5EW8 + FGFR2 2PSQ`` maps each id to its own target.
    """
    out = []
    pdb_re = re.compile(r"\b([1-9][A-Z0-9]{3})\b")
    window = 24  # characters after a target name that still count as "next to"

    for lineno, line in iter_lines(text):
        candidates = [
            (m.start(), m.group(1))
            for m in pdb_re.finditer(line)
            if re.search(r"[A-Z]", m.group(1))
        ]
        if not candidates:
            continue
        for target, expected in vocab["pdb_of"].items():
            for tm in re.finditer(rf"\b{re.escape(target)}\b", line):
                near = [
                    pid for pos, pid in candidates
                    if 0 <= pos - tm.end() <= window
                ]
                if near and expected not in near:
                    out.append(Finding(
                        "PDB-MISMATCH", rel, lineno,
                        f"{target} shown with {near[0]} but pairs.yaml specifies {expected}"))
                    break
    return out


def check_paths(doc: Path, text: str, root: Path, rel: str) -> list[Finding]:
    """Repo paths quoted in prose that do not exist on disk."""
    out, seen = [], set()
    path_re = re.compile(r"[`\(]([A-Za-z0-9_\-./]+\.(?:py|yaml|yml|sh|toml|tex|json))[`\)]")
    for lineno, line in iter_lines(text):
        for m in path_re.finditer(line):
            cand = m.group(1).lstrip("./")
            if not cand.startswith(PATH_PREFIXES):
                continue
            if "*" in cand or "{" in cand or cand in seen:
                continue
            if not (root / cand).exists():
                seen.add(cand)
                out.append(Finding("PATH-MISSING", rel, lineno,
                                   f"`{cand}` does not exist"))
    return out


def github_anchor(heading: str) -> str:
    """GitHub's heading-to-anchor rule: lowercase, drop punctuation, spaces to hyphens."""
    s = heading.strip().lower()
    s = re.sub(r"[`*_~]", "", s)                 # inline markup never reaches the anchor
    s = re.sub(r"[^\w\s-]", "", s, flags=re.U)  # punctuation is dropped
    return re.sub(r"\s+", "-", s.strip())


def heading_anchors(path: Path) -> set[str]:
    """Every anchor a markdown file exposes, including duplicate-suffixed ones."""
    if not path.is_file():
        return set()
    anchors: set[str] = set()
    for h in re.findall(r"^#{1,6}\s+(.+?)\s*$",
                        path.read_text(encoding="utf-8", errors="replace"), re.M):
        base = github_anchor(h)
        if base in anchors:
            n = 1
            while f"{base}-{n}" in anchors:
                n += 1
            anchors.add(f"{base}-{n}")
        else:
            anchors.add(base)
    return anchors


def check_links(doc: Path, text: str, root: Path, rel: str) -> list[Finding]:
    """Relative markdown links whose target file -- or heading anchor -- is absent."""
    out = []
    link_re = re.compile(r"\[[^\]]*\]\(([^)#\s]*)(#[^)\s]*)?\)")
    anchor_cache: dict[Path, set[str]] = {}
    for lineno, line in iter_lines(text):
        for m in link_re.finditer(line):
            target, frag = m.group(1), (m.group(2) or "")[1:]
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (doc.parent / target).resolve() if target else doc
            if not resolved.exists():
                out.append(Finding("LINK-BROKEN", rel, lineno,
                                   f"link target `{target}` not found"))
                continue
            # A fragment must match a heading in the file it points at.
            if frag and resolved.suffix == ".md":
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = heading_anchors(resolved)
                if frag not in anchor_cache[resolved]:
                    where = target or resolved.name
                    out.append(Finding("ANCHOR-BROKEN", rel, lineno,
                                       f"`{where}#{frag}` has no matching heading"))
    return out


def check_counts(root: Path, docs: list[Path]) -> list[Finding]:
    """Counted claims about the repo that no longer hold."""
    out = []
    n_docs = len(list((root / "docs").glob("*.md"))) if (root / "docs").is_dir() else 0
    n_tests = 0
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        for p in tests_dir.rglob("test_*.py"):
            n_tests += len(re.findall(r"^\s*def test_", p.read_text(encoding="utf-8", errors="replace"), re.M))

    doc_claim = re.compile(r"Documentation \((\d+) files?")
    test_claim = re.compile(r"~?([\d,]+)\+?\s+tests\b", re.I)
    for doc in docs:
        rel = str(doc.relative_to(root))
        text = doc.read_text(encoding="utf-8", errors="replace")
        for lineno, line in iter_lines(text):
            for m in doc_claim.finditer(line):
                if int(m.group(1)) != n_docs:
                    out.append(Finding("COUNT-CLAIM", rel, lineno,
                                       f"claims {m.group(1)} doc files, docs/ holds {n_docs}"))
            for m in test_claim.finditer(line):
                claimed = int(m.group(1).replace(",", ""))
                if n_tests and abs(claimed - n_tests) > max(50, 0.15 * n_tests):
                    out.append(Finding("COUNT-CLAIM", rel, lineno,
                                       f"claims {claimed} tests, found {n_tests} test functions"))
    return out


def check_dates(root: Path, docs: list[Path]) -> list[Finding]:
    """An ``Updated:`` stamp older than the file's last commit."""
    out = []
    stamp = re.compile(r"Updated:\s*(\d{4}-\d{2}-\d{2})")
    for doc in docs:
        rel = str(doc.relative_to(root))
        text = doc.read_text(encoding="utf-8", errors="replace")
        m = stamp.search(text)
        if not m:
            continue
        committed = _run_git(root, "log", "-1", "--format=%cs", "--", rel)
        if committed and committed > m.group(1):
            lineno = text[: m.start()].count("\n") + 1
            out.append(Finding("STALE-DATE", rel, lineno,
                               f"stamped {m.group(1)} but last commit is {committed}"))
    return out


def check_map(root: Path, docs: list[Path]) -> list[Finding]:
    """docs/*.md that the navigation map never links."""
    out = []
    index = root / "docs" / "README.md"
    if not index.is_file():
        return out
    index_text = index.read_text(encoding="utf-8", errors="replace")
    for doc in docs:
        if doc.parent.name != "docs" or doc.name == "README.md":
            continue
        if f"({doc.name})" not in index_text and f"`{doc.name}`" not in index_text:
            out.append(Finding("ORPHAN-DOC", "docs/README.md", 0,
                               f"`{doc.name}` is not linked from the documentation map"))
    return out


def check_mermaid(doc: Path, text: str, rel: str) -> list[Finding]:
    """Mermaid blocks whose edges dangle at end of line (GitHub renders nothing)."""
    out = []
    lines = text.split("\n")
    inside = False
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if s.startswith("```mermaid"):
            inside = True
            continue
        if inside and s.startswith("```"):
            inside = False
            continue
        if inside and re.search(r"(-->|---|==>)\s*$", s):
            out.append(Finding("MERMAID-BROKEN", rel, i,
                               "edge arrow ends the line; mermaid needs source and target together"))
    return out


def check_duplication(root: Path, docs: list[Path], min_lines: int = 3) -> list[Finding]:
    """Identical multi-line blocks repeated across different documents."""
    blocks: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for doc in docs:
        rel = str(doc.relative_to(root))
        lines = doc.read_text(encoding="utf-8", errors="replace").split("\n")
        buf: list[str] = []
        start = 0
        for i, line in enumerate(lines, start=1):
            if line.strip():
                if not buf:
                    start = i
                buf.append(" ".join(line.split()))
            else:
                if len(buf) >= min_lines:
                    blocks["\n".join(buf)].append((rel, start))
                buf = []
        if len(buf) >= min_lines:
            blocks["\n".join(buf)].append((rel, start))

    out = []
    for body, places in blocks.items():
        files = {rel for rel, _ in places}
        if len(files) < 2:
            continue
        head = body.split("\n")[0][:60]
        where = ", ".join(f"{rel}:{ln}" for rel, ln in sorted(places))
        out.append(Finding("DUPLICATE-BLOCK", sorted(files)[0], places[0][1],
                           f"{len(body.splitlines())} identical lines in {len(files)} files "
                           f"({where}) starting \"{head}\""))
    return out


def lint(root: Path) -> list[Finding]:
    vocab = build_vocabulary(root)
    docs = collect_docs(root)
    findings: list[Finding] = []

    for doc in docs:
        rel = str(doc.relative_to(root))
        text = doc.read_text(encoding="utf-8", errors="replace")
        historical = HISTORICAL_MARKER in text
        per_file = [
            *check_pairs(doc, text, vocab, rel),
            *check_pdb(doc, text, vocab, rel),
            *check_paths(doc, text, root, rel),
            *check_links(doc, text, root, rel),
            *check_mermaid(doc, text, rel),
        ]
        if historical:
            per_file = [f for f in per_file if f.check not in HISTORICAL_CHECKS]
        findings += per_file

    findings += check_counts(root, docs)
    findings += check_dates(root, docs)
    findings += check_map(root, docs)
    findings += check_duplication(root, docs)

    # Repo-level: reward calibration coverage for the configured panel.
    missing = sorted(vocab["targets"] - vocab["calibrated"])
    if missing:
        findings.append(Finding(
            "TARGET-UNCALIBRATED", "configs/normalization.yaml", 0,
            f"{len(missing)} of {len(vocab['targets'])} configured targets have no "
            f"z-score entry: {', '.join(missing)}"))

    return sorted(findings, key=lambda f: (f.check, f.path, f.line))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, help="repo root (default: script's parent)")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    findings = lint(root)

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2, ensure_ascii=False))
    else:
        by_check: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            by_check[f.check].append(f)
        for check in sorted(by_check):
            group = by_check[check]
            print(f"\n{check}  ({len(group)})")
            for f in group:
                where = f"{f.path}:{f.line}" if f.line else f.path
                print(f"  {where}: {f.message}")
        print(f"\n{len(findings)} finding(s) across {len(collect_docs(root))} documents")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
