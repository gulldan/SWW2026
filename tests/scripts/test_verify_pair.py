"""Tests for dualdocking.validation.pair_verify and CLI wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dualdocking.validation.pair_verify import (
    PairRecord,
    _assess_match_quality,
    _extract_catalog_pct,
    _parse_catalog_domain_type,
    fetch_uniprot_domains,
    nw_align,
    parse_catalog,
    resolve_chembl_from_uniprot,
    resolve_uniprot_from_pdb,
    select_domains_for_pair,
    verify_doi,
    verify_pair,
    verify_pdb,
)

CATALOG_PATH = Path(__file__).resolve().parent.parent.parent / "target_pairs_catalog.html"
_HAVE_CATALOG = CATALOG_PATH.exists()


# ---------------------------------------------------------------------------
# NW Alignment
# ---------------------------------------------------------------------------


class TestNWAlign:
    def test_identical_seqs(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = nw_align(seq, seq)
        assert result.identity_pct == 100.0
        assert result.matches == len(seq)
        assert result.gaps == 0

    def test_similar_short_seqs(self):
        # Two similar short peptides
        result = nw_align("ACDEFGH", "ACDXFGH")
        assert result.identity_pct > 70.0
        assert result.aligned_length >= 7

    def test_empty_seq(self):
        result = nw_align("", "ACDEF")
        assert result.identity_pct == 0.0
        assert result.aligned_length == 0

    def test_single_residue(self):
        result = nw_align("A", "A")
        assert result.identity_pct == 100.0
        assert result.matches == 1

    def test_completely_different(self):
        # Very different peptides should have low identity
        result = nw_align("AAAAAAA", "WWWWWWW")
        assert result.identity_pct < 50.0

    def test_gap_insertion(self):
        # seq_b has insertion
        result = nw_align("ACDEFGH", "ACDXXXEFGH")
        assert result.gaps > 0
        assert result.aligned_length > 7

    def test_known_short_alignment(self):
        # Manually verifiable: HEAGAWGHEE vs PAWHEAE
        result = nw_align("HEAGAWGHEE", "PAWHEAE")
        assert result.aligned_length >= 7
        assert 0 < result.identity_pct < 100


# ---------------------------------------------------------------------------
# HTML Catalog Parsing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_CATALOG, reason="target_pairs_catalog.html not found")
class TestParseCatalog:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.pairs = parse_catalog(CATALOG_PATH)

    def test_count(self):
        assert len(self.pairs) == 50

    def test_pair1_fields(self):
        p1 = self.pairs[0]
        assert p1.pair_number == 1
        assert p1.target_a == "EGFR"
        assert p1.target_b == "HER2"
        assert p1.pdb_a == "1XKK"
        assert p1.pdb_b == "3PP0"
        assert "10.1158" in p1.doi
        assert p1.similarity_level == "Very High"
        assert "Lapatinib" in p1.drug_status

    def test_pair50_exists(self):
        p50 = [p for p in self.pairs if p.pair_number == 50]
        assert len(p50) == 1
        assert "SIRT" in p50[0].target_a or "SIRT" in p50[0].target_b

    def test_categories(self):
        categories = {p.category for p in self.pairs}
        # Should have multiple categories
        assert len(categories) >= 5

    def test_all_have_pdb(self):
        for p in self.pairs:
            assert len(p.pdb_a) == 4, f"Pair #{p.pair_number} missing PDB A"
            assert len(p.pdb_b) == 4, f"Pair #{p.pair_number} missing PDB B"

    def test_all_have_doi(self):
        for p in self.pairs:
            assert p.doi, f"Pair #{p.pair_number} missing DOI"

    def test_similarity_levels(self):
        valid_levels = {"Very High", "High", "Mod-High", "Moderate", "Low", "Very Low", "None"}
        for p in self.pairs:
            assert p.similarity_level in valid_levels, f"Pair #{p.pair_number}: '{p.similarity_level}'"


# ---------------------------------------------------------------------------
# PDB Verification (mocked)
# ---------------------------------------------------------------------------


class TestVerifyPdbMocked:
    def test_verify_pdb(self, monkeypatch):
        entry_json = {
            "rcsb_entry_info": {
                "resolution_combined": [2.4],
                "experimental_method": "X-RAY DIFFRACTION",
            },
            "struct": {"title": "Crystal structure of EGFR with lapatinib"},
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": ["1"],
                "non_polymer_entity_ids": ["2"],
            },
        }
        polymer_json = {
            "rcsb_entity_source_organism": [
                {
                    "ncbi_scientific_name": "Homo sapiens",
                    "rcsb_gene_name": [{"value": "EGFR"}],
                }
            ]
        }
        nonpoly_json = {"pdbx_entity_nonpoly": {"comp_id": "FMM"}}

        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            if "core/entry/" in url:
                mock_resp.json.return_value = entry_json
            elif "nonpolymer_entity/" in url:
                mock_resp.json.return_value = nonpoly_json
            elif "polymer_entity/" in url:
                mock_resp.json.return_value = polymer_json
            mock_resp.status_code = 200
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)

        report = verify_pdb("1XKK")
        assert report.pdb_id == "1XKK"
        assert report.resolution == 2.4
        assert report.organism == "Homo sapiens"
        assert "EGFR" in report.gene_names
        assert "FMM" in report.ligands
        assert call_count == 3


# ---------------------------------------------------------------------------
# DOI Verification (mocked)
# ---------------------------------------------------------------------------


class TestVerifyDoiMocked:
    def test_doi_resolves(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "https://cancerres.aacrjournals.org/content/64/18/6652"}

        monkeypatch.setattr("requests.head", lambda *a, **kw: mock_resp)

        report = verify_doi("10.1158/0008-5472.CAN-04-1168")
        assert report.resolves is True
        assert report.status_code == 302
        assert "aacrjournals" in report.redirect_url

    def test_doi_fails(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {}

        monkeypatch.setattr("requests.head", lambda *a, **kw: mock_resp)

        report = verify_doi("10.9999/nonexistent")
        assert report.resolves is False
        assert report.status_code == 404


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_CATALOG, reason="target_pairs_catalog.html not found")
class TestCLI:
    def test_help(self):
        from click.testing import CliRunner

        from cli.s7_audit.verify_pair import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--pair" in result.output
        assert "--catalog" in result.output

    def test_parse_pair1(self, tmp_path):
        from click.testing import CliRunner

        from cli.s7_audit.verify_pair import main

        out_json = str(tmp_path / "pair1.json")
        runner = CliRunner()
        result = runner.invoke(main, [
            "--pair", "1",
            "--catalog", str(CATALOG_PATH),
            "--output-json", out_json,
        ])
        assert result.exit_code == 0
        assert "EGFR" in result.output

        data = json.loads(Path(out_json).read_text())
        assert data["pair_number"] == 1
        assert data["pdb_a"] == "1XKK"

    def test_no_pair_error(self):
        from click.testing import CliRunner

        from cli.s7_audit.verify_pair import main

        runner = CliRunner()
        result = runner.invoke(main, ["--catalog", str(CATALOG_PATH)])
        assert result.exit_code != 0

    def test_full_flag(self):
        """--full should appear in --help and enable all checks."""
        from click.testing import CliRunner

        from cli.s7_audit.verify_pair import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--full" in result.output
        assert "--verify-sequence" in result.output
        assert "--verify-chembl" in result.output


# ---------------------------------------------------------------------------
# Auto-resolution: resolve_uniprot_from_pdb (mocked)
# ---------------------------------------------------------------------------


class TestResolveUniprotFromPdb:
    def test_resolves_single(self, monkeypatch):
        entry_json = {
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": ["1"],
            },
        }
        polymer_json = {
            "rcsb_polymer_entity_container_identifiers": {
                "uniprot_ids": ["P00533"],
            },
        }

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            if "core/entry/" in url:
                mock_resp.json.return_value = entry_json
            elif "polymer_entity/" in url:
                mock_resp.json.return_value = polymer_json
            mock_resp.status_code = 200
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        result = resolve_uniprot_from_pdb("1XKK")
        assert result == ["P00533"]

    def test_resolves_multiple_entities(self, monkeypatch):
        entry_json = {
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": ["1", "2"],
            },
        }

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            if "core/entry/" in url:
                mock_resp.json.return_value = entry_json
            elif "polymer_entity/1XKK/1" in url:
                mock_resp.json.return_value = {
                    "rcsb_polymer_entity_container_identifiers": {
                        "uniprot_ids": ["P00533"],
                    },
                }
            elif "polymer_entity/1XKK/2" in url:
                mock_resp.json.return_value = {
                    "rcsb_polymer_entity_container_identifiers": {
                        "uniprot_ids": ["Q9876"],
                    },
                }
            mock_resp.status_code = 200
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        result = resolve_uniprot_from_pdb("1XKK")
        assert result == ["P00533", "Q9876"]

    def test_no_uniprot(self, monkeypatch):
        entry_json = {
            "rcsb_entry_container_identifiers": {
                "polymer_entity_ids": ["1"],
            },
        }
        polymer_json = {
            "rcsb_polymer_entity_container_identifiers": {},
        }

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            if "core/entry/" in url:
                mock_resp.json.return_value = entry_json
            elif "polymer_entity/" in url:
                mock_resp.json.return_value = polymer_json
            mock_resp.status_code = 200
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        result = resolve_uniprot_from_pdb("XXXX")
        assert result == []


# ---------------------------------------------------------------------------
# Auto-resolution: resolve_chembl_from_uniprot (mocked)
# ---------------------------------------------------------------------------


class TestResolveChemblFromUniprot:
    def test_resolves(self, monkeypatch):
        api_json = {
            "targets": [
                {"target_chembl_id": "CHEMBL203"},
            ],
        }

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_json
            mock_resp.status_code = 200
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        result = resolve_chembl_from_uniprot("P00533")
        assert result == "CHEMBL203"

    def test_not_found(self, monkeypatch):
        api_json = {"targets": []}

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_json
            mock_resp.status_code = 200
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        result = resolve_chembl_from_uniprot("XXXXX")
        assert result is None


# ---------------------------------------------------------------------------
# verify_pair auto-resolution (mocked)
# ---------------------------------------------------------------------------


class TestVerifyPairAutoResolve:
    def test_auto_resolves_uniprot_and_chembl(self, monkeypatch):
        """do_sequence + do_chembl without explicit IDs triggers auto-resolution."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        # Mock resolve_uniprot_from_pdb
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.resolve_uniprot_from_pdb",
            lambda pdb_id: {"1XKK": ["P00533"], "3PP0": ["P04626"]}.get(pdb_id, []),
        )
        # Mock resolve_chembl_from_uniprot
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.resolve_chembl_from_uniprot",
            lambda uid: {"P00533": "CHEMBL203", "P04626": "CHEMBL1824"}.get(uid),
        )
        # Mock _fetch_uniprot_fasta (short seqs for fast NW)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: "ACDEFGHIKLMNPQRSTVWY" if uid == "P00533" else "ACDEFGHIKLMNPQRSTVWY",
        )
        # Mock fetch_uniprot_domains
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: [],
        )
        # Mock verify_chembl
        from dualdocking.validation.pair_verify import ChemblReport

        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.verify_chembl",
            lambda cid: ChemblReport(chembl_id=cid, total_activities=100, ic50_count=50),
        )

        pair = PairRecord(pair_number=1, target_a="EGFR", target_b="HER2", pdb_a="1XKK", pdb_b="3PP0")
        result = verify_pair(pair, do_sequence=True, do_chembl=True)

        assert "resolved_ids" in result
        assert result["resolved_ids"]["uniprot_a"] == "P00533"
        assert result["resolved_ids"]["uniprot_b"] == "P04626"
        assert result["resolved_ids"]["chembl_a"] == "CHEMBL203"
        assert result["resolved_ids"]["chembl_b"] == "CHEMBL1824"

        v = result["verification"]
        assert "sequence_identity" in v
        assert "domain_identity" in v
        assert "chembl_a" in v
        assert "chembl_b" in v
        assert v["chembl_a"]["chembl_id"] == "CHEMBL203"

    def test_explicit_ids_skip_resolution(self, monkeypatch):
        """Passing explicit IDs should not trigger auto-resolution."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        resolve_called = False

        def mock_resolve_uniprot(pdb_id):
            nonlocal resolve_called
            resolve_called = True
            return []

        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.resolve_uniprot_from_pdb",
            mock_resolve_uniprot,
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: "ACDEFGHIKLMNPQRSTVWY",
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: [],
        )

        pair = PairRecord(pair_number=1, target_a="EGFR", target_b="HER2", pdb_a="1XKK", pdb_b="3PP0")
        result = verify_pair(pair, do_sequence=True, uniprot_a="P00533", uniprot_b="P04626")

        assert not resolve_called
        assert "resolved_ids" not in result
        assert "sequence_identity" in result["verification"]
        assert "domain_identity" in result["verification"]


# ---------------------------------------------------------------------------
# fetch_uniprot_domains (mocked)
# ---------------------------------------------------------------------------


class TestFetchUniprotDomains:
    def test_parses_domain_features(self, monkeypatch):
        uniprot_json = {
            "features": [
                {
                    "type": "Domain",
                    "description": "Protein kinase",
                    "location": {"start": {"value": 712}, "end": {"value": 979}},
                },
                {
                    "type": "Domain",
                    "description": "Receptor L domain 1",
                    "location": {"start": {"value": 1}, "end": {"value": 165}},
                },
                {
                    "type": "Transmembrane",
                    "description": "Helical",
                    "location": {"start": {"value": 646}, "end": {"value": 668}},
                },
            ]
        }

        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json.return_value = uniprot_json
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        domains = fetch_uniprot_domains("P00533")
        assert len(domains) == 2  # only "Domain" type, not Transmembrane
        assert domains[0]["name"] == "Protein kinase"
        assert domains[0]["start"] == 712
        assert domains[0]["end"] == 979
        assert domains[1]["name"] == "Receptor L domain 1"

    def test_empty_features(self, monkeypatch):
        def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"features": []}
            return mock_resp

        monkeypatch.setattr("dualdocking.validation.pair_verify.get_with_retry", mock_get)
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        domains = fetch_uniprot_domains("Q99999")
        assert domains == []


# ---------------------------------------------------------------------------
# _parse_catalog_domain_type
# ---------------------------------------------------------------------------


class TestParseCatalogDomainType:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("~80% kinase domain ID; RMSD 1.1Å", "kinase"),
            ("~75% CR3 kinase domain ID", "kinase"),
            ("~70% JH1 kinase domain ID", "kinase"),
            ("~90% kinase domain ID (highest in set)", "kinase"),
            ("~46% kinase domain ID; similar DFG-out", "kinase"),
            ("49% kinase domain / 77% ATP-site ID", "kinase"),
            ("~25–30% (PIKK superfamily); shared ATP-cleft", "kinase"),
            ("~40–45% (split-kinase RTKs)", "kinase"),
            ("Homologous TKL kinase domains; divergent gatekeeper", "kinase"),
            ("Protein Tyr kinase vs lipid kinase; shared general kinase fold", "kinase"),
            ("~52% binding site ID; different gatekeeper", "kinase"),
            ("~33% (different RTK families); no single dual TKI", "kinase"),
            ("~56% LBD ID; only 2 aa differ", "LBD"),
            ("~51% LBD ID; 79% DBD ID; same NR3C", "LBD"),
            ("60–70% LBD ID; single Tyr/His difference", "LBD"),
            ("94% SET domain ID; 63% overall", "SET"),
            ("85–92% BD1 ID; ~95% binding pocket", "BD1"),
            ("~33% BH3 groove ID; superimposable", "BH3"),
            ("~25–35% inter-family JmjC domain ID", "JmjC"),
            ("~40–60% catalytic core (Class I sirtuins)", "sirtuin"),
            ("~46% CAT domain ID; conserved HYE", "catalytic"),
            ("<30% catalytic domain ID; conserved AdoMet", "catalytic"),
            ("~25–30% (Class I vs IIb); conserved Zn2+ catalytic mechanism", "catalytic"),
            ("~25–50% catalytic domain ID; conserved M-site", "catalytic"),
            ("~80% overall protein ID; conserved allosteric", "overall"),
            ("72% overall protein ID; conserved allosteric", "overall"),
            ("~65% overall seq ID; conserved gorge", "overall"),
            ("~70% overall seq ID; 10/16 substrate-cavity", "overall"),
            ("84% active-site residue identity; unique JAK3 Cys909", "active-site"),
            ("~87% active-site; single Ile→Val", "active-site"),
            ("Different protein classes (GTPase vs PTP)", "none"),
            ("Different folds (α/β-hydrolase vs bis-oxygenase)", "none"),
            ("No structural homology (serine hydrolase vs FAD oxidase)", "none"),
            ("Different folds (non-heme Fe vs heme enzyme)", "none"),
            ("Nuclear receptor vs GPCR; different folds", "none"),
            ("Different families (NLR sensor NACHT domain vs Ser/Thr kinase)", "none"),
            ("14-TM transporter vs serine protease; orthogonal", "none"),
            ("SDR dehydrogenase (Rossmann fold) vs nuclear receptor", "none"),
            ("Different Cys protease clans (PA vs CA)", "none"),
        ],
    )
    def test_domain_type(self, text, expected):
        assert _parse_catalog_domain_type(text) == expected


# ---------------------------------------------------------------------------
# _extract_catalog_pct
# ---------------------------------------------------------------------------


class TestExtractCatalogPct:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("~80% kinase domain ID", "~80%"),
            ("49% kinase domain / 77% ATP-site", "49%"),
            ("85–92% BD1 ID", "85–92%"),
            ("<30% catalytic domain ID", "<30%"),
            ("~25–30% (PIKK superfamily)", "~25–30%"),
            ("Different protein classes", ""),
        ],
    )
    def test_pct_extraction(self, text, expected):
        assert _extract_catalog_pct(text) == expected


# ---------------------------------------------------------------------------
# _assess_match_quality
# ---------------------------------------------------------------------------


class TestAssessMatchQuality:
    def test_exact_match(self):
        assert _assess_match_quality(80.0, "~80%") == "match"

    def test_close_match(self):
        assert _assess_match_quality(68.0, "~80%") == "close"

    def test_mismatch(self):
        assert _assess_match_quality(40.0, "~80%") == "mismatch"

    def test_range_match(self):
        assert _assess_match_quality(87.0, "85-92%") == "match"

    def test_range_close(self):
        assert _assess_match_quality(78.0, "85-92%") == "close"

    def test_empty_claim(self):
        assert _assess_match_quality(50.0, "") == "n/a"

    def test_no_number(self):
        assert _assess_match_quality(50.0, "Different protein classes") == "n/a"


# ---------------------------------------------------------------------------
# select_domains_for_pair
# ---------------------------------------------------------------------------


class TestSelectDomainsForPair:
    def test_kinase_pair(self, monkeypatch):
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: {
                "P00533": [{"name": "Protein kinase", "start": 712, "end": 979}],
                "P04626": [{"name": "Protein kinase", "start": 720, "end": 987}],
            }.get(uid, []),
        )

        pair = PairRecord(
            pair_number=1, target_a="EGFR", target_b="HER2",
            similarity_pct="~80% kinase domain ID",
        )
        ra, rb, na, nb, dtype = select_domains_for_pair(pair, "P00533", "P04626")
        assert ra == "712-979"
        assert rb == "720-987"
        assert na == "Protein kinase"
        assert nb == "Protein kinase"
        assert dtype == "kinase"

    def test_none_similarity(self):
        pair = PairRecord(
            pair_number=14, target_a="KRAS", target_b="SHP2",
            similarity_pct="Different protein classes (GTPase vs PTP)",
        )
        ra, rb, na, nb, dtype = select_domains_for_pair(pair, "P01116", "Q06124")
        assert ra == ""
        assert rb == ""
        assert dtype == "none"

    def test_overall_returns_empty(self):
        pair = PairRecord(
            pair_number=5, target_a="MEK1", target_b="MEK2",
            similarity_pct="~80% overall protein ID; conserved allosteric pocket",
        )
        ra, rb, na, nb, dtype = select_domains_for_pair(pair, "Q02750", "P36507")
        assert ra == ""
        assert rb == ""
        assert dtype == "overall"

    def test_active_site_returns_empty(self):
        pair = PairRecord(
            pair_number=33, target_a="JAK1", target_b="JAK3",
            similarity_pct="84% active-site residue identity",
        )
        ra, rb, na, nb, dtype = select_domains_for_pair(pair, "P23458", "P52333")
        assert ra == ""
        assert rb == ""
        assert dtype == "active-site"

    def test_polyprotein_domain_override(self, monkeypatch):
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)

        pair = PairRecord(
            pair_number=47, target_a="Mpro", target_b="PLpro",
            similarity_pct="Different Cys protease clans (PA vs CA)",
        )
        # This pair has similarity_level "None" → domain type = "none"
        ra, rb, na, nb, dtype = select_domains_for_pair(pair, "P0DTD1", "P0DTC1")
        assert ra == ""
        assert rb == ""
        assert dtype == "none"

    def test_lbd_pair(self, monkeypatch):
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: {
                "P03372": [{"name": "NR LBD", "start": 302, "end": 553}],
                "Q92731": [{"name": "NR LBD", "start": 257, "end": 504}],
            }.get(uid, []),
        )

        pair = PairRecord(
            pair_number=46, target_a="ERα", target_b="ERβ",
            similarity_pct="~56% LBD ID; only 2 aa differ",
        )
        ra, rb, na, nb, dtype = select_domains_for_pair(pair, "P03372", "Q92731")
        assert ra == "302-553"
        assert rb == "257-504"
        assert dtype == "LBD"


# ---------------------------------------------------------------------------
# UniProt override application in verify_pair
# ---------------------------------------------------------------------------


class TestUniprotOverrides:
    def test_pair2_cdk4_override(self, monkeypatch):
        """Pair #2 should override uniprot_a to P11802 (CDK4)."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.resolve_uniprot_from_pdb",
            lambda pdb_id: {"2W96": ["P24385"], "5L2I": ["Q00534"]}.get(pdb_id, []),
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: "ACDEFGHIKLMNPQRSTVWY",
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: [],
        )

        pair = PairRecord(
            pair_number=2, target_a="CDK4", target_b="CDK6",
            pdb_a="2W96", pdb_b="5L2I",
            similarity_pct="~71% overall protein ID",
        )
        result = verify_pair(pair, do_sequence=True)
        assert result["resolved_ids"]["uniprot_a"] == "P11802"

    def test_pair19_ezh1_override(self, monkeypatch):
        """Pair #19 should override uniprot_b to Q92800 (EZH1)."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.resolve_uniprot_from_pdb",
            lambda pdb_id: ["Q15910"],
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: "ACDEFGHIKLMNPQRSTVWY",
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: [],
        )

        pair = PairRecord(
            pair_number=19, target_a="EZH2", target_b="EZH1",
            pdb_a="5LS6", pdb_b="5LS6",
            similarity_pct="94% SET domain ID",
        )
        result = verify_pair(pair, do_sequence=True)
        assert result["resolved_ids"]["uniprot_b"] == "Q92800"

    def test_pair31_gpcr_override(self, monkeypatch):
        """Pair #31 should override both UniProt IDs for A2A/D2."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.resolve_uniprot_from_pdb",
            lambda pdb_id: {"4EIY": ["P0ABE7"], "6CM4": ["P00720"]}.get(pdb_id, []),
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: "ACDEFGHIKLMNPQRSTVWY",
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: [],
        )

        pair = PairRecord(
            pair_number=31, target_a="A2A", target_b="D2",
            pdb_a="4EIY", pdb_b="6CM4",
            similarity_pct="<25% (different GPCR families)",
        )
        result = verify_pair(pair, do_sequence=True)
        assert result["resolved_ids"]["uniprot_a"] == "P29274"
        assert result["resolved_ids"]["uniprot_b"] == "P14416"


# ---------------------------------------------------------------------------
# Domain identity in verify_pair output
# ---------------------------------------------------------------------------


class TestDomainIdentityInVerifyPair:
    def test_kinase_domain_identity_computed(self, monkeypatch):
        """verify_pair with kinase pair should compute domain identity."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        # Long enough sequences to slice
        seq_a = "A" * 1000
        seq_b = "A" * 1000
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: seq_a if uid == "P00533" else seq_b,
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: {
                "P00533": [{"name": "Protein kinase", "start": 712, "end": 979}],
                "P04626": [{"name": "Protein kinase", "start": 720, "end": 987}],
            }.get(uid, []),
        )

        pair = PairRecord(
            pair_number=1, target_a="EGFR", target_b="HER2",
            similarity_pct="~80% kinase domain ID",
        )
        result = verify_pair(pair, do_sequence=True, uniprot_a="P00533", uniprot_b="P04626")

        v = result["verification"]
        assert "domain_identity" in v
        di = v["domain_identity"]
        assert di["domain_type"] == "kinase"
        assert di["range_a"] == "712-979"
        assert di["range_b"] == "720-987"
        assert di["alignment"] is not None
        assert di["catalog_claim_pct"] == "~80%"

    def test_none_pair_no_domain_alignment(self, monkeypatch):
        """Pairs with 'None' similarity should have n/a domain identity."""
        monkeypatch.setattr("dualdocking.validation.pair_verify._CACHE_DIR", None)
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify._fetch_uniprot_fasta",
            lambda uid: "ACDEFGHIKLMNPQRSTVWY",
        )
        monkeypatch.setattr(
            "dualdocking.validation.pair_verify.fetch_uniprot_domains",
            lambda uid: [],
        )

        pair = PairRecord(
            pair_number=14, target_a="KRAS", target_b="SHP2",
            similarity_pct="Different protein classes (GTPase vs PTP)",
        )
        result = verify_pair(pair, do_sequence=True, uniprot_a="P01116", uniprot_b="Q06124")

        v = result["verification"]
        assert "domain_identity" in v
        di = v["domain_identity"]
        assert di["match_quality"] == "n/a"
        assert di["alignment"] is None
