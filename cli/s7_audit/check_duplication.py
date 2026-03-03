"""Detect duplicate and near-duplicate Python logic across the repository.

Clone taxonomy implemented by this checker:
1. Exact structural clones (Type-1/2): identical AST after function-name normalization.
2. Structural near-clones: AST-equal after stronger identifier/literal normalization.
3. Token near-clones (Type-3): high Jaccard similarity over normalized token shingles.

Intended use:
    uv run python cli/s7_audit/check_duplication.py --top 30
    uv run python cli/s7_audit/check_duplication.py --json-out report/duplication_audit.json
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

_EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "output",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass(frozen=True, slots=True)
class FunctionRecord:
    """Function-level record used in clone detection."""

    path: Path
    name: str
    start: int
    end: int
    n_lines: int
    source: str
    exact_key: str
    structural_key: str
    token_shingles: frozenset[int]

    @property
    def uid(self) -> str:
        """Stable identifier for cross-referencing records."""
        return f"{self.path}:{self.start}:{self.name}"

    def to_payload(self) -> dict[str, Any]:
        """Serialize record to JSON-friendly dict."""
        return {
            "path": str(self.path),
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "n_lines": self.n_lines,
        }


@dataclass(frozen=True, slots=True)
class TokenPair:
    """Pair of token-near-clone functions with similarity score."""

    a_uid: str
    b_uid: str
    similarity: float


class _NearCloneNormalizer(ast.NodeTransformer):
    """Normalize identifiers/literals to detect structural near-clones."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Replace variable names with a canonical placeholder."""
        return ast.copy_location(ast.Name(id="v", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Replace argument names with a canonical placeholder."""
        node.arg = "a"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Collapse non-boolean literals to a single placeholder."""
        if node.value in (True, False, None):
            return node
        return ast.copy_location(ast.Constant(value="K"), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Normalize attribute names while preserving receiver structure."""
        node = self.generic_visit(node)
        if isinstance(node, ast.Attribute):
            node.attr = "attr"
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        """Normalize import aliases."""
        return ast.copy_location(ast.alias(name="mod", asname=None), node)


def _collect_python_files(root: Path, include_tests: bool) -> list[Path]:
    """Collect Python files from root with standard exclusions."""
    files: list[Path] = []
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & _EXCLUDED_DIR_NAMES:
            continue
        if not include_tests and "tests" in parts:
            continue
        files.append(path)
    return sorted(files)


def _extract_source_segment(source: str, node: ast.AST) -> str:
    """Extract source text for an AST node with robust fallback."""
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    lines = source.splitlines()
    return "\n".join(lines[start - 1 : end])


def _parse_function_source(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Parse source containing one function definition."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return node


def _strip_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Remove leading docstring statement from function body."""
    if not node.body:
        return
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        node.body = node.body[1:]


def _normalize_signature_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Normalize function/argument names for clone-key generation."""
    node.name = "FUNC"
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.arg = "a"
    if node.args.vararg is not None:
        node.args.vararg.arg = "a"
    if node.args.kwarg is not None:
        node.args.kwarg.arg = "a"


def _exact_clone_key(func_source: str) -> str:
    """Build exact structural clone key (Type-1/2)."""
    node = _parse_function_source(func_source)
    if node is None:
        return ""
    _strip_docstring(node)
    _normalize_signature_names(node)
    return ast.dump(node, include_attributes=False)


def _structural_clone_key(func_source: str) -> str:
    """Build strong-normalized structural clone key (Type-3 AST)."""
    node = _parse_function_source(func_source)
    if node is None:
        return ""
    _strip_docstring(node)
    _normalize_signature_names(node)
    normalized = _NearCloneNormalizer().visit(node)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _normalize_tokens(func_source: str) -> list[str]:
    """Normalize function tokens for token-Jaccard analysis."""
    out: list[str] = []
    stream = io.StringIO(func_source).readline
    ignored = {
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.COMMENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
    try:
        tokens = tokenize.generate_tokens(stream)
    except tokenize.TokenError:
        return out

    for tok in tokens:
        if tok.type in ignored:
            continue
        if tok.type == tokenize.NAME:
            out.append(tok.string if keyword.iskeyword(tok.string) else "NAME")
        elif tok.type == tokenize.NUMBER:
            out.append("NUM")
        elif tok.type == tokenize.STRING:
            out.append("STR")
        else:
            out.append(tok.string)
    return out


def _token_shingles(tokens: list[str], k: int) -> frozenset[int]:
    """Compute hashed token shingles for near-clone detection."""
    if k <= 0 or len(tokens) < k:
        return frozenset()
    vals: set[int] = set()
    for idx in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[idx : idx + k]).encode("utf-8")
        digest = hashlib.blake2b(shingle, digest_size=8).digest()
        vals.add(int.from_bytes(digest, byteorder="big", signed=False))
    return frozenset(vals)


def _collect_function_records(
    files: list[Path],
    root: Path,
    *,
    min_lines: int,
    token_k: int,
) -> list[FunctionRecord]:
    """Parse files and build function records above a line-count threshold."""
    records: list[FunctionRecord] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            n_lines = end - start + 1
            if n_lines < min_lines:
                continue
            func_source = _extract_source_segment(source, node)
            exact_key = _exact_clone_key(func_source)
            structural_key = _structural_clone_key(func_source)
            if not exact_key and not structural_key:
                continue
            tokens = _normalize_tokens(func_source)
            shingles = _token_shingles(tokens, token_k)
            records.append(
                FunctionRecord(
                    path=path.relative_to(root),
                    name=node.name,
                    start=start,
                    end=end,
                    n_lines=n_lines,
                    source=func_source,
                    exact_key=exact_key,
                    structural_key=structural_key,
                    token_shingles=shingles,
                )
            )
    return records


def _group_records(records: list[FunctionRecord], key_attr: str) -> list[list[FunctionRecord]]:
    """Group records by key attribute and keep only duplicate groups."""
    groups_map: dict[str, list[FunctionRecord]] = defaultdict(list)
    for rec in records:
        key = getattr(rec, key_attr)
        if key:
            groups_map[key].append(rec)
    groups = [group for group in groups_map.values() if len(group) > 1]
    groups.sort(
        key=lambda group: (len(group), sum(rec.n_lines for rec in group)),
        reverse=True,
    )
    return groups


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    """Compute Jaccard similarity between shingle sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union


def _detect_token_pairs(
    records: list[FunctionRecord],
    *,
    near_threshold: float,
    line_delta: int,
) -> list[TokenPair]:
    """Detect token near-clone pairs after filtering exact/structural clones."""
    candidates = [rec for rec in records if rec.token_shingles]
    candidates.sort(key=lambda rec: rec.n_lines)
    pairs: list[TokenPair] = []

    for idx, rec_a in enumerate(candidates):
        for rec_b in candidates[idx + 1 :]:
            if rec_b.n_lines - rec_a.n_lines > line_delta:
                break
            if rec_a.exact_key == rec_b.exact_key:
                continue
            if rec_a.structural_key == rec_b.structural_key:
                continue
            sim = _jaccard(rec_a.token_shingles, rec_b.token_shingles)
            if sim >= near_threshold:
                left, right = sorted((rec_a.uid, rec_b.uid))
                pairs.append(TokenPair(a_uid=left, b_uid=right, similarity=sim))

    pairs.sort(key=lambda pair: pair.similarity, reverse=True)
    return pairs


def _build_token_clusters(token_pairs: list[TokenPair]) -> list[list[str]]:
    """Build connected components from token near-clone pairs."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in token_pairs:
        for uid in (pair.a_uid, pair.b_uid):
            parent.setdefault(uid, uid)
        union(pair.a_uid, pair.b_uid)

    groups: dict[str, list[str]] = defaultdict(list)
    for uid in parent:
        groups[find(uid)].append(uid)

    clusters = [sorted(uids) for uids in groups.values() if len(uids) > 1]
    clusters.sort(key=len, reverse=True)
    return clusters


def _group_payload(groups: list[list[FunctionRecord]]) -> list[dict[str, Any]]:
    """Serialize clone groups for reporting."""
    return [
        {
            "size": len(group),
            "total_lines": sum(rec.n_lines for rec in group),
            "functions": [rec.to_payload() for rec in group],
        }
        for group in groups
    ]


def _token_pairs_payload(
    pairs: list[TokenPair],
    uid_to_rec: dict[str, FunctionRecord],
) -> list[dict[str, Any]]:
    """Serialize token near-clone pairs for reporting."""
    return [
        {
            "similarity": round(pair.similarity, 4),
            "a": uid_to_rec[pair.a_uid].to_payload(),
            "b": uid_to_rec[pair.b_uid].to_payload(),
        }
        for pair in pairs
    ]


def _cluster_payload(
    clusters: list[list[str]],
    uid_to_rec: dict[str, FunctionRecord],
) -> list[dict[str, Any]]:
    """Serialize token near-clone clusters for reporting."""
    return [
        {
            "size": len(cluster),
            "functions": [uid_to_rec[uid].to_payload() for uid in cluster],
        }
        for cluster in clusters
    ]


def _print_summary(report: dict[str, Any], *, top: int) -> None:
    """Print human-readable summary to stdout."""
    click.echo("=== Duplicate Audit ===")
    click.echo(f"Files scanned: {report['stats']['files_scanned']}")
    click.echo(f"Functions scanned: {report['stats']['functions_scanned']}")
    click.echo(f"Exact clone groups: {report['stats']['exact_clone_groups']}")
    click.echo(f"Structural clone groups: {report['stats']['structural_clone_groups']}")
    click.echo(f"Token near-clone pairs: {report['stats']['token_near_pairs']}")
    click.echo(f"Token near-clone clusters: {report['stats']['token_near_clusters']}")

    exact_groups = report["exact_clone_groups"][:top]
    if exact_groups:
        click.echo("\nTop exact-clone groups:")
        for idx, group in enumerate(exact_groups, start=1):
            click.echo(f"{idx}. size={group['size']}, total_lines={group['total_lines']}")
            for item in group["functions"]:
                click.echo(f"   - {item['path']}:{item['start']}:{item['name']}")

    token_pairs = report["token_near_pairs"][:top]
    if token_pairs:
        click.echo("\nTop token near-clone pairs:")
        for idx, pair in enumerate(token_pairs, start=1):
            click.echo(
                f"{idx}. sim={pair['similarity']:.4f} "
                f"{pair['a']['path']}:{pair['a']['start']} <-> {pair['b']['path']}:{pair['b']['start']}"
            )


@click.command()
@click.option("--root", default=".", show_default=True, help="Repository root to scan.")
@click.option("--include-tests", is_flag=True, default=False, help="Include tests/ in analysis.")
@click.option("--min-lines", default=12, show_default=True, type=int, help="Minimum function length to analyze.")
@click.option("--token-k", default=5, show_default=True, type=int, help="Token shingle size for near-clone detection.")
@click.option(
    "--near-threshold",
    default=0.85,
    show_default=True,
    type=float,
    help="Jaccard threshold for token near-clones.",
)
@click.option(
    "--line-delta",
    default=12,
    show_default=True,
    type=int,
    help="Maximum allowed line-count difference when comparing token pairs.",
)
@click.option("--top", default=20, show_default=True, type=int, help="Number of top groups/pairs to print.")
@click.option("--json-out", default=None, type=click.Path(path_type=Path), help="Optional JSON report output path.")
@click.option(
    "--fail-on-exact-groups",
    default=None,
    type=int,
    help="Exit with code 1 if exact clone groups exceed this number.",
)
@click.option(
    "--fail-on-token-pairs",
    default=None,
    type=int,
    help="Exit with code 1 if token near-clone pairs exceed this number.",
)
def main(
    root: str,
    include_tests: bool,
    min_lines: int,
    token_k: int,
    near_threshold: float,
    line_delta: int,
    top: int,
    json_out: Path | None,
    fail_on_exact_groups: int | None,
    fail_on_token_pairs: int | None,
) -> None:
    """Run duplicate-code audit and emit summary/report."""
    root_path = Path(root).resolve()
    files = _collect_python_files(root_path, include_tests)
    records = _collect_function_records(files, root_path, min_lines=min_lines, token_k=token_k)
    uid_to_rec = {rec.uid: rec for rec in records}

    exact_groups = _group_records(records, "exact_key")
    structural_groups = [
        group for group in _group_records(records, "structural_key") if len({rec.exact_key for rec in group}) > 1
    ]
    token_pairs = _detect_token_pairs(records, near_threshold=near_threshold, line_delta=line_delta)
    token_clusters = _build_token_clusters(token_pairs)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "root": str(root_path),
        "params": {
            "include_tests": include_tests,
            "min_lines": min_lines,
            "token_k": token_k,
            "near_threshold": near_threshold,
            "line_delta": line_delta,
        },
        "stats": {
            "files_scanned": len(files),
            "functions_scanned": len(records),
            "exact_clone_groups": len(exact_groups),
            "structural_clone_groups": len(structural_groups),
            "token_near_pairs": len(token_pairs),
            "token_near_clusters": len(token_clusters),
        },
        "exact_clone_groups": _group_payload(exact_groups),
        "structural_clone_groups": _group_payload(structural_groups),
        "token_near_pairs": _token_pairs_payload(token_pairs, uid_to_rec),
        "token_near_clusters": _cluster_payload(token_clusters, uid_to_rec),
    }

    _print_summary(report, top=top)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        click.echo(f"\nJSON report written: {json_out}")

    if fail_on_exact_groups is not None and len(exact_groups) > fail_on_exact_groups:
        raise SystemExit(1)
    if fail_on_token_pairs is not None and len(token_pairs) > fail_on_token_pairs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
