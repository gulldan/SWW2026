"""Tests for molecule sanitization."""

from dualdocking.data.chem_sanitize import canonicalize, sanitize_smiles


def test_canonicalize_valid():
    assert canonicalize("c1ccccc1") is not None
    assert canonicalize("C(=O)O") is not None


def test_canonicalize_invalid():
    assert canonicalize("not_a_smiles") is None
    assert canonicalize("") is None


def test_sanitize_dedup():
    smiles = ["c1ccccc1", "C1=CC=CC=C1", "CCO"]  # first two are benzene
    clean, stats = sanitize_smiles(smiles)
    assert stats.total == 3
    assert stats.valid == 3
    assert stats.unique == 2  # benzene deduplicated
    assert len(clean) == 2


def test_sanitize_filters_invalid():
    smiles = ["c1ccccc1", "INVALID", "CCO"]
    _clean, stats = sanitize_smiles(smiles)
    assert stats.valid == 2
    assert 1 in stats.invalid_indices
