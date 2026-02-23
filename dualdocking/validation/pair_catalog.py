"""HTML catalog parser for the 50-pair target catalog.

Extracts structured :class:`PairRecord` objects from
``target_pairs_catalog.html``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PairRecord:
    """One row from the catalog table."""

    pair_number: int = 0
    category: str = ""
    target_family: str = ""
    target_a: str = ""
    target_b: str = ""
    pdb_a: str = ""
    pdb_b: str = ""
    pdb_a_desc: str = ""
    pdb_b_desc: str = ""
    doi: str = ""
    similarity_level: str = ""
    similarity_pct: str = ""
    selectivity_ratio: str = ""
    selectivity_ref: str = ""
    rationale: str = ""
    drug_status: str = ""


# ---------------------------------------------------------------------------
# HTML Catalog Parser
# ---------------------------------------------------------------------------

_LVL_MAP: dict[str, str] = {
    "lvl-vh": "Very High",
    "lvl-h": "High",
    "lvl-mh": "Mod-High",
    "lvl-m": "Moderate",
    "lvl-l": "Low",
    "lvl-vl": "Very Low",
    "lvl-n": "None",
}


class _CatalogParser(HTMLParser):
    """Stateful HTML parser that extracts pair rows from the catalog."""

    def __init__(self) -> None:
        super().__init__()
        self.pairs: list[PairRecord] = []
        self.current_category: str = ""

        # Row-level state
        self._in_table = False
        self._in_header_row = False
        self._in_row = False
        self._cells: list[str] = []
        self._cell_data: dict[str, Any] = {}
        self._cell_idx = -1
        self._in_cell = False
        self._text_buf: list[str] = []
        self._in_pdb = False
        self._pdb_text: list[str] = []
        self._pdb_ids: list[str] = []
        self._in_pair = False
        self._pair_text: list[str] = []
        self._in_doi_link = False
        self._in_lvl = False
        self._lvl_class = ""
        self._in_sr = False
        self._sr_text: list[str] = []
        self._sr_href = ""
        self._in_drug = False
        self._drug_text: list[str] = []
        self._in_doi_cell = False
        self._doi_href = ""
        self._in_sim_cell = False
        self._sim_texts: list[str] = []
        self._in_sr_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "") or ""

        if tag == "h2":
            self._text_buf = []
            self._in_cell = True  # reuse text collection

        if tag == "table":
            self._in_table = True
            self._in_header_row = True

        if tag == "tr" and self._in_table:
            if self._in_header_row:
                pass  # skip header
            else:
                self._in_row = True
                self._cells = []
                self._cell_idx = -1
                self._cell_data = {}

        if tag == "th":
            pass  # ignore header cells

        if tag == "td" and self._in_row:
            self._cell_idx += 1
            self._in_cell = True
            self._text_buf = []
            if self._cell_idx == 3:
                self._in_doi_cell = True
                self._doi_href = ""
            if self._cell_idx == 4:
                self._in_sim_cell = True
                self._sim_texts = []
                self._sr_text = []
                self._sr_href = ""

        if tag == "br" and self._in_cell:
            self._text_buf.append("\n")

        if tag == "span":
            if "pair" in cls:
                self._in_pair = True
                self._pair_text = []
            if "pdb" in cls:
                self._in_pdb = True
                self._pdb_text = []
            if "drug" in cls:
                self._in_drug = True
                self._drug_text = []
            if "sr" in cls:
                self._in_sr = True
                self._sr_text = []
                self._sr_href = ""
            for lvl_cls, lvl_name in _LVL_MAP.items():
                if lvl_cls in cls:
                    self._in_lvl = True
                    self._lvl_class = lvl_name
                    break

        if tag == "a":
            href = attr_dict.get("href", "") or ""
            if self._in_doi_cell and not self._in_sr:
                self._doi_href = href
                self._in_doi_link = True
            if self._in_sr:
                self._sr_href = href
                self._in_sr_link = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._in_cell = False
            text = "".join(self._text_buf).strip()
            if text:
                # Strip trailing " (N pairs)"
                cat = re.sub(r"\s*\(\d+\s+pairs?\)\s*$", "", text)
                self.current_category = cat

        if tag == "table":
            self._in_table = False
            self._in_header_row = False

        if tag == "tr" and self._in_table:
            if self._in_header_row:
                self._in_header_row = False
            elif self._in_row:
                self._in_row = False
                self._finalize_row()

        if tag == "td" and self._in_cell:
            self._in_cell = False
            text = "".join(self._text_buf).strip()
            self._cells.append(text)
            if self._in_doi_cell:
                self._in_doi_cell = False
            if self._in_sim_cell:
                self._in_sim_cell = False

        if tag == "span":
            if self._in_pdb:
                self._in_pdb = False
                pdb_id = "".join(self._pdb_text).strip()
                if pdb_id:
                    self._pdb_ids.append(pdb_id)
            if self._in_pair:
                self._in_pair = False
            if self._in_drug:
                self._in_drug = False
                self._cell_data["drug_status"] = "".join(self._drug_text).strip()
            if self._in_lvl:
                self._in_lvl = False
            if self._in_sr:
                self._in_sr = False
                self._cell_data["selectivity_ratio"] = "".join(self._sr_text).strip()
                self._cell_data["selectivity_ref"] = self._sr_href

        if tag == "a":
            self._in_doi_link = False
            self._in_sr_link = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._text_buf.append(data)
        if self._in_pdb:
            self._pdb_text.append(data)
        if self._in_pair:
            self._pair_text.append(data)
        if self._in_drug:
            self._drug_text.append(data)
        if self._in_sr:
            self._sr_text.append(data)
        if self._in_lvl:
            self._cell_data["similarity_level"] = self._lvl_class

    def handle_entityref(self, name: str) -> None:
        char_map = {
            "alpha": "\u03b1",
            "beta": "\u03b2",
            "gamma": "\u03b3",
            "delta": "\u03b4",
            "Aring": "\u00c5",
            "ndash": "\u2013",
            "mdash": "\u2014",
            "rarr": "\u2192",
            "le": "\u2264",
            "ge": "\u2265",
            "lt": "<",
            "gt": ">",
            "amp": "&",
            "micro": "\u00b5",
            "times": "\u00d7",
            "sup2": "\u00b2",
            "sup3": "\u00b3",
            "kappa": "\u03ba",
        }
        ch = char_map.get(name, f"&{name};")
        if self._in_cell:
            self._text_buf.append(ch)
        if self._in_drug:
            self._drug_text.append(ch)
        if self._in_sr:
            self._sr_text.append(ch)

    def handle_charref(self, name: str) -> None:
        try:
            ch = chr(int(name, 16) if name.startswith("x") else int(name))
        except ValueError:
            ch = f"&#{name};"
        if self._in_cell:
            self._text_buf.append(ch)
        if self._in_drug:
            self._drug_text.append(ch)
        if self._in_sr:
            self._sr_text.append(ch)

    def _finalize_row(self) -> None:
        if len(self._cells) < 6:
            return

        rec = PairRecord(category=self.current_category)

        # Cell 0: pair number
        try:
            rec.pair_number = int(self._cells[0].strip())
        except ValueError:
            return  # Not a data row

        # Cell 1: target family
        rec.target_family = self._cells[1].strip()

        # Cell 2: pair names + PDB IDs
        pair_text = self._cells[2]
        self._parse_pair_cell(pair_text, rec, self._pdb_ids)
        self._pdb_ids = []

        # Cell 3: DOI
        if self._doi_href:
            doi = self._doi_href
            doi = doi.removeprefix("https://doi.org/")
            rec.doi = doi

        # Cell 4: similarity
        rec.similarity_level = self._cell_data.get("similarity_level", "")
        # The full sim text minus the selectivity ratio
        sim_full = self._cells[4] if len(self._cells) > 4 else ""
        # Extract the selectivity ratio from cell_data
        sr = self._cell_data.get("selectivity_ratio", "")
        # similarity_pct = sim text without the level label and without the selectivity
        sim_clean = sim_full
        for lvl_name in _LVL_MAP.values():
            sim_clean = sim_clean.replace(lvl_name, "").strip()
        if sr:
            sim_clean = sim_clean.replace(sr, "").strip()
        rec.similarity_pct = sim_clean
        rec.selectivity_ratio = sr
        rec.selectivity_ref = self._cell_data.get("selectivity_ref", "")

        # Cell 5: rationale
        rec.rationale = self._cells[5].strip() if len(self._cells) > 5 else ""
        rec.drug_status = self._cell_data.get("drug_status", "")

        self.pairs.append(rec)

    @staticmethod
    def _parse_pair_cell(text: str, rec: PairRecord, pdb_ids: list[str]) -> None:
        """Parse the 'Pair (PDB A / PDB B)' cell."""
        # First line (before <br>): "EGFR / HER2"
        lines = text.split("\n")
        top = lines[0].strip() if lines else text.strip()

        # Pair names from the first line
        pair_match = re.match(r"^(.+?)\s*/\s*(.+?)$", top)
        if pair_match:
            rec.target_a = pair_match.group(1).strip()
            rec.target_b = pair_match.group(2).strip()
        else:
            rec.target_a = top

        # PDB IDs from .pdb spans (collected by parser)
        if len(pdb_ids) >= 2:
            rec.pdb_a = pdb_ids[0].upper()
            rec.pdb_b = pdb_ids[1].upper()
        elif len(pdb_ids) == 1:
            rec.pdb_a = pdb_ids[0].upper()
            rec.pdb_b = pdb_ids[0].upper()

        # PDB descriptions: text in parentheses (from the second line onward)
        rest = "\n".join(lines[1:]) if len(lines) > 1 else text
        descs = re.findall(r"\(([^)]+)\)", rest)
        if len(descs) >= 2:
            rec.pdb_a_desc = descs[0].strip()
            rec.pdb_b_desc = descs[1].strip()
        elif len(descs) == 1:
            rec.pdb_a_desc = descs[0].strip()


def parse_catalog(html_path: str | Path) -> list[PairRecord]:
    """Parse the target pairs catalog HTML file.

    Args:
        html_path: Path to ``target_pairs_catalog.html``.

    Returns:
        List of :class:`PairRecord` dataclasses, one per catalog row.
    """
    html_path = Path(html_path)
    html_text = html_path.read_text(encoding="utf-8")
    parser = _CatalogParser()
    parser.feed(html_text)
    logger.info("Parsed {} pairs from {}", len(parser.pairs), html_path.name)
    return parser.pairs
