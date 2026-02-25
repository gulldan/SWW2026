"""Update target_pairs_catalog.html with verification-derived values.

The current catalog layout is a fixed 12-column table. This updater fills
ChEMBL-A/B cells from verification JSON and (if needed) direct ChEMBL queries.
It maps rows by pair names (target A/B), not by pair number.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from dualdocking.data.http_utils import get_with_retry
from dualdocking.validation.pair_verify import set_cache_dir, verify_chembl
from dualdocking.validation.pocket_similarity import PocketSimilarityReport, compute_pocket_similarity

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
CATALOG = PROJECT / "target_pairs_catalog.html"
CACHE_DIR = PROJECT / "output" / "cache" / "verify"
POCKET_CACHE_DIR = CACHE_DIR / "pocket_similarity"
VERIFY_CANDIDATES = [
    PROJECT / "report" / "data" / "verify_all_catalog_full.json",
    PROJECT / "report" / "data" / "verify_all_catalog_chembl.json",
    PROJECT / "report" / "data" / "verify_all.json",
]

_GREEK_TO_ASCII = str.maketrans(
    {
        "α": "a",
        "β": "b",
        "γ": "g",
        "δ": "d",
        "Α": "A",
        "Β": "B",
        "Γ": "G",
        "Δ": "D",
    }
)


def _pick_verify_path() -> Path:
    for path in VERIFY_CANDIDATES:
        if path.exists():
            return path
    msg = "Verification JSON not found. Checked: " + ", ".join(str(p) for p in VERIFY_CANDIDATES)
    raise FileNotFoundError(msg)


def _norm_token(text: str) -> str:
    txt = text.translate(_GREEK_TO_ASCII)
    return re.sub(r"[^A-Za-z0-9]+", "", txt).upper()


def _pair_keys(a: str, b: str) -> tuple[str, str]:
    return f"{_norm_token(a)}::{_norm_token(b)}", f"{_norm_token(b)}::{_norm_token(a)}"


def _strip_tags(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&alpha;", "α").replace("&beta;", "β").replace("&gamma;", "γ")
    text = text.replace("&times;", "×")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_pair_text(pair_text: str) -> tuple[str, str]:
    if "/" not in pair_text:
        return pair_text.strip(), ""
    a, b = pair_text.split("/", 1)
    return a.strip(), b.strip()


def _extract_chembl_id(td_html: str) -> str:
    href_match = re.search(r"/target_report_card/(CHEMBL\d+)/", td_html)
    if href_match:
        return href_match.group(1)
    text_match = re.search(r"\b(CHEMBL\d+)\b", _strip_tags(td_html))
    return text_match.group(1) if text_match else ""


def _fmt_k(n: int) -> str:
    if n >= 1000:
        val = n / 1000
        if val >= 10:
            return f"{val:.0f}K"
        return f"{val:.1f}K"
    return str(n)


def _replace_td_inner(td_html: str, inner_html: str) -> str:
    m = re.match(r"(<td[^>]*>).*?(</td>)$", td_html, flags=re.DOTALL)
    if not m:
        return td_html
    return f"{m.group(1)}{inner_html}{m.group(2)}"


def _fetch_pdb_primary_doi(pdb_id: str, doi_cache: dict[str, str]) -> str:
    key = pdb_id.strip().upper()
    if not key:
        return ""
    if key in doi_cache:
        return doi_cache[key]

    url = f"https://data.rcsb.org/rest/v1/core/entry/{key}"
    doi = ""
    try:
        resp = get_with_retry(url, timeout=20)
        data = resp.json()
        doi = str(data.get("rcsb_primary_citation", {}).get("pdbx_database_id_doi", "")).strip()
    except Exception as exc:
        logger.warning("Failed to fetch DOI for %s: %s", key, exc)
        doi = ""

    doi_cache[key] = doi
    return doi


def _link_for_pdb_or_doi(pdb_id: str, doi: str) -> str:
    if doi:
        return f'<a href="https://doi.org/{doi}">{doi}</a>'
    if pdb_id:
        return f'<a href="https://www.rcsb.org/structure/{pdb_id}">{pdb_id} (RCSB)</a>'
    return "n/a"


def _build_doi_cell_inner(pdb_a: str, pdb_b: str, doi_cache: dict[str, str]) -> str:
    doi_a = _fetch_pdb_primary_doi(pdb_a, doi_cache) if pdb_a else ""
    doi_b = _fetch_pdb_primary_doi(pdb_b, doi_cache) if pdb_b else ""
    link_a = _link_for_pdb_or_doi(pdb_a, doi_a)
    link_b = _link_for_pdb_or_doi(pdb_b, doi_b)
    return f"A: {link_a}<br/>B: {link_b}"


def _format_pocket_similarity_cell(report: PocketSimilarityReport) -> str:
    return (
        f"{float(report.identity_pct):.1f}% computed pocket "
        f"(5A; {int(report.pocket_len_a)}/{int(report.pocket_len_b)} res)"
    )


def _format_pocket_failure_cell(pdb_a: str, pdb_b: str, exc: Exception) -> str | None:
    msg = str(exc)
    if "No ligand detected" in msg:
        m = re.search(r"/([A-Za-z0-9]{4})\.pdb", msg)
        missing_pdb = m.group(1).upper() if m else pdb_a
        if missing_pdb not in {pdb_a, pdb_b}:
            missing_pdb = pdb_a
        return f"n/a (no co-crystal ligand in {missing_pdb})"
    if "No protein residues within" in msg:
        return "n/a (no protein-ligand contacts within 5A)"
    return None


def _format_fullseq_cell(rec: dict[str, Any]) -> str | None:
    verification = rec.get("verification", {})
    seq = verification.get("sequence_identity", {})
    aln = seq.get("alignment") or {}
    pct = aln.get("identity_pct")
    matches = aln.get("matches")
    aligned = aln.get("aligned_length")
    if pct is None or matches is None or aligned in (None, 0):
        return None
    return f"{float(pct):.1f}% {int(matches)}/{int(aligned)}"


def _format_domain_cell(rec: dict[str, Any]) -> str | None:
    verification = rec.get("verification", {})
    di = verification.get("domain_identity", {})
    aln = di.get("alignment") or {}
    pct = aln.get("identity_pct")
    if pct is None:
        seq = verification.get("sequence_identity", {})
        seq_aln = seq.get("alignment") or {}
        seq_pct = seq_aln.get("identity_pct")
        seq_matches = seq_aln.get("matches")
        seq_aligned = seq_aln.get("aligned_length")
        if seq_pct is None:
            return None
        domain_type = str(di.get("domain_type", "")).strip()
        if seq_matches is not None and seq_aligned not in (None, 0):
            if domain_type:
                return (
                    f"{float(seq_pct):.1f}% full-seq fallback"
                    f" ({domain_type}; {int(seq_matches)}/{int(seq_aligned)})"
                )
            return f"{float(seq_pct):.1f}% full-seq fallback ({int(seq_matches)}/{int(seq_aligned)})"
        if domain_type:
            return f"{float(seq_pct):.1f}% full-seq fallback ({domain_type})"
        return f"{float(seq_pct):.1f}% full-seq fallback"

    range_a = str(di.get("range_a", "")).strip()
    range_b = str(di.get("range_b", "")).strip()
    seq_a_len = aln.get("seq_a_len")
    seq_b_len = aln.get("seq_b_len")

    domain_a = str(di.get("domain_a", "")).strip()
    domain_b = str(di.get("domain_b", "")).strip()
    name_a = domain_a.split("(")[0].strip() if domain_a else ""
    name_b = domain_b.split("(")[0].strip() if domain_b else ""
    domain_name = name_a if name_a and name_a == name_b else f"{name_a}/{name_b}".strip("/")

    parts = [f"{float(pct):.1f}%"]
    if domain_name:
        parts.append(domain_name)
    if range_a and range_b:
        parts.append(f"{range_a}/{range_b}")
    if seq_a_len is not None and seq_b_len is not None:
        parts.append(f"{int(seq_a_len)}/{int(seq_b_len)} aa")
    return " ".join(parts)


def _chembl_report_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return {
        "chembl_id": getattr(report, "chembl_id", ""),
        "total_activities": int(getattr(report, "total_activities", 0)),
        "ic50_count": int(getattr(report, "ic50_count", 0)),
        "ki_count": int(getattr(report, "ki_count", 0)),
        "kd_count": int(getattr(report, "kd_count", 0)),
    }


def _build_chembl_cell_inner(chembl_id: str, report: dict[str, Any]) -> str:
    total = int(report.get("total_activities", 0))
    ic50 = int(report.get("ic50_count", 0))
    ki = int(report.get("ki_count", 0))
    kd = int(report.get("kd_count", 0))

    return (
        f'<a href="https://www.ebi.ac.uk/chembl/target_report_card/{chembl_id}/">{chembl_id}</a>'
        f'<span class="sub">{_fmt_k(ic50)} IC<sub>50</sub><br/>{_fmt_k(ki)} K<sub>i</sub>, {_fmt_k(kd)} K<sub>d</sub>'
        f"<br/><b>{_fmt_k(total)}</b> total</span>"
    )


def _update_chembl_cell(
    td_html: str,
    existing_report: dict[str, Any] | None,
    resolved_id: str,
    chembl_cache: dict[str, dict[str, Any]],
) -> str:
    chembl_id = resolved_id or _extract_chembl_id(td_html)
    if not chembl_id:
        return td_html

    report = _chembl_report_dict(existing_report) if existing_report else None
    if report is None:
        if chembl_id in chembl_cache:
            report = chembl_cache[chembl_id]
        else:
            fetched = _chembl_report_dict(verify_chembl(chembl_id))
            chembl_cache[chembl_id] = fetched
            report = fetched

    return _replace_td_inner(td_html, _build_chembl_cell_inner(chembl_id, report))


def _update_row(
    row_html: str,
    verify_by_key: dict[str, dict[str, Any]],
    chembl_cache: dict[str, dict[str, Any]],
    pocket_cache: dict[str, PocketSimilarityReport],
    doi_cache: dict[str, str],
) -> tuple[str, bool]:
    cells = re.findall(r"<td[^>]*>.*?</td>", row_html, flags=re.DOTALL)
    if len(cells) < 12:
        return row_html, False

    pair_text = _strip_tags(cells[1])
    target_a, target_b = _split_pair_text(pair_text)
    if not target_a or not target_b:
        return row_html, False

    key_ab, key_ba = _pair_keys(target_a, target_b)
    rec = verify_by_key.get(key_ab) or verify_by_key.get(key_ba)
    if rec is None:
        return row_html, False

    verification = rec.get("verification", {})
    resolved = rec.get("resolved_ids", {})

    new_cells = list(cells)
    old_pocket = new_cells[4]
    old_domain = new_cells[5]
    old_full = new_cells[6]
    old_a = new_cells[10]
    old_b = new_cells[11]
    old_doi = new_cells[12] if len(new_cells) > 12 else ""

    pdb_a = str(rec.get("pdb_a", "")).strip().upper()
    pdb_b = str(rec.get("pdb_b", "")).strip().upper()
    if pdb_a and pdb_b:
        cache_key = f"{pdb_a}::{pdb_b}"
        cache_key_rev = f"{pdb_b}::{pdb_a}"
        pocket_report = pocket_cache.get(cache_key) or pocket_cache.get(cache_key_rev)
        if pocket_report is None:
            try:
                pocket_report = compute_pocket_similarity(
                    pdb_a,
                    pdb_b,
                    cache_dir=POCKET_CACHE_DIR,
                    radius_angstrom=5.0,
                )
                pocket_cache[cache_key] = pocket_report
                pocket_cache[cache_key_rev] = pocket_report
            except Exception as exc:
                logger.warning("Pocket similarity failed for %s / %s: %s", pdb_a, pdb_b, exc)
                fallback_cell = _format_pocket_failure_cell(pdb_a, pdb_b, exc)
                if fallback_cell is not None:
                    new_cells[4] = _replace_td_inner(old_pocket, fallback_cell)
                pocket_report = None
        if pocket_report is not None:
            new_cells[4] = _replace_td_inner(old_pocket, _format_pocket_similarity_cell(pocket_report))

    domain_cell = _format_domain_cell(rec)
    fullseq_cell = _format_fullseq_cell(rec)
    if domain_cell is not None:
        new_cells[5] = _replace_td_inner(old_domain, domain_cell)
    if fullseq_cell is not None:
        new_cells[6] = _replace_td_inner(old_full, fullseq_cell)

    new_cells[10] = _update_chembl_cell(
        old_a,
        verification.get("chembl_a"),
        str(resolved.get("chembl_a", "")),
        chembl_cache,
    )
    new_cells[11] = _update_chembl_cell(
        old_b,
        verification.get("chembl_b"),
        str(resolved.get("chembl_b", "")),
        chembl_cache,
    )

    doi_inner = _build_doi_cell_inner(pdb_a, pdb_b, doi_cache)
    if len(new_cells) > 12:
        new_cells[12] = _replace_td_inner(old_doi, doi_inner)
    else:
        new_cells.append(f"<td>{doi_inner}</td>")

    changed = (
        (new_cells[4] != old_pocket)
        or (new_cells[5] != old_domain)
        or (new_cells[6] != old_full)
        or (new_cells[10] != old_a)
        or (new_cells[11] != old_b)
        or (len(cells) != len(new_cells))
        or (len(new_cells) > 12 and old_doi and new_cells[12] != old_doi)
    )
    if not changed:
        return row_html, False

    new_row = row_html
    for old, new in zip(cells, new_cells[: len(cells)], strict=True):
        new_row = new_row.replace(old, new, 1)
    if len(new_cells) > len(cells):
        new_row = new_row.replace("</tr>", f"{new_cells[-1]}</tr>", 1)

    return new_row, True


def _update_note(html_text: str) -> str:
    old = (
        "ChEMBL mini-stats under each target show counts of activity records by standard type"
        " (IC50, Ki, Kd) and total; newly-added rows currently show <b>n/a</b> placeholders"
        " where counts haven’t been pulled yet."
    )
    new = (
        "ChEMBL mini-stats under each target show counts of activity records by standard type"
        " (IC50, Ki, Kd) and total; values refreshed from verification cache/API on 2026-02-25."
    )
    html_text = html_text.replace(old, new)
    html_text = html_text.replace("<th>Pocket similarity (est.)</th>", "<th>Pocket similarity (computed, 5A)</th>")
    if "<th>DOI (PDB A/B)</th>" not in html_text:
        html_text = html_text.replace(
            "<th>ChEMBL-A</th>\n<th>ChEMBL-B</th>",
            "<th>ChEMBL-A</th>\n<th>ChEMBL-B</th>\n<th>DOI (PDB A/B)</th>",
        )
    html_text = html_text.replace(
        "If you want, the next “hardening” step is to compute pocket-identity strictly from the listed PDB co-crystals"
        " (e.g., residues within 5 Å of the bound ligand) and replace the “est.” pocket similarity notes with computed values.",
        "Pocket similarity is now computed from co-crystal PDB structures using residues within 5 A of the bound ligand."
        " Values are sequence-identity percentages of those ligand-contacting pocket residues.",
    )
    return html_text


def main() -> None:
    verify_path = _pick_verify_path()
    verify_data = json.loads(verify_path.read_text(encoding="utf-8"))
    if isinstance(verify_data, dict):
        verify_data = [verify_data]

    verify_by_key: dict[str, dict[str, Any]] = {}
    for rec in verify_data:
        ta = str(rec.get("target_a", "")).strip()
        tb = str(rec.get("target_b", "")).strip()
        if not ta or not tb:
            continue
        key_ab, key_ba = _pair_keys(ta, tb)
        verify_by_key[key_ab] = rec
        verify_by_key[key_ba] = rec

    set_cache_dir(CACHE_DIR)

    html_text = CATALOG.read_text(encoding="utf-8")
    na_before = len(re.findall(r"n/a\s+IC<sub>50</sub>", html_text, flags=re.IGNORECASE))

    changed_rows = 0
    chembl_cache: dict[str, dict[str, Any]] = {}
    pocket_cache: dict[str, PocketSimilarityReport] = {}
    doi_cache: dict[str, str] = {}

    def _replacer(match: re.Match[str]) -> str:
        nonlocal changed_rows
        updated, changed = _update_row(
            match.group(0),
            verify_by_key,
            chembl_cache,
            pocket_cache,
            doi_cache,
        )
        if changed:
            changed_rows += 1
        return updated

    html_text = re.sub(
        r"<tr>\s*<td class=\"c\">\d+</td>.*?</tr>",
        _replacer,
        html_text,
        flags=re.DOTALL,
    )
    html_text = _update_note(html_text)

    na_after = len(re.findall(r"n/a\s+IC<sub>50</sub>", html_text, flags=re.IGNORECASE))

    CATALOG.write_text(html_text, encoding="utf-8")
    logger.info("Updated %s", CATALOG)
    logger.info("Verify source: %s", verify_path)
    logger.info("Rows updated: %d", changed_rows)
    logger.info("n/a IC50 placeholders: %d -> %d", na_before, na_after)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
