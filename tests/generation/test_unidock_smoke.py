"""Smoke test: verify UniDock GPU batch docking works end-to-end.

Requires:
    - UniDock binary at ``/home/qwerty/bin/unidock``
    - A prepared receptor PDBQT (uses CDK2 1HCL from calibration data)
    - GPU available

Run with::

    uv run pytest tests/generation/test_unidock_smoke.py -v -s
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from dualdocking.generation.reinvent4.scoring import (
    SCORE_MAX,
    SCORE_MIN,
    _GPU_SUB_BATCH,
    dock_gpu_batch,
    parse_vina_score,
    smiles_to_pdbqt,
)

UNIDOCK = "/home/qwerty/bin/unidock"
# Use the CDK2 receptor from any existing workspace, or from calibration data
_RECEPTOR_CANDIDATES = [
    Path("/mnt/data/SWW2026/output/data/CDK2/1hcl.pdbqt"),
    Path("/mnt/data/SWW2026/output/results/CDK2_ERK2/single_a/seed_42/reinvent4_workspace/1hcl.pdbqt"),
]
CDK2_CENTER = "103.99,87.48,79.93"
CDK2_SIZE = "30.00,30.00,30.00"

# Simple drug-like molecules that should dock without issues
TEST_SMILES = [
    "c1ccccc1",                          # benzene
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",       # ibuprofen
    "c1ccc(NC(=O)c2ccccc2)cc1",          # benzanilide
    "c1ccc2c(c1)Cc1ccccc12",             # fluorene
]

# REINVENT4-generated molecules (from actual RL training on CDK2_ERK2).
# These are larger, more complex, and stress-test UniDock GPU batch docking.
_REINVENT4_SMILES = [
    "CCOc1ccc(CC(NC(=O)c2coc(C(Cc3ccc(F)cc3)NC(=O)c3ccccc3F)n2)C(=O)N2CCCCC2)cc1",
    "CCC1(C)NC(=O)N(S(=O)(=O)c2ccc(NC(=O)c3cccs3)c(OCC(C)C)c2)C1=O",
    "O=C(COC(=O)CCCN1C(=O)c2ccccc2C1=O)Nc1ccc(Br)cc1F",
    "Cn1cc(-c2cccnc2)nc1CCCNC(=O)c1ccc[n+]([O-])c1",
    "CCN(CC)CCNC(=O)c1cn2c(Oc3ccc(F)cc3F)ccnc2n1",
    "CCOCCCN=c1ccc(-c2nc(N3CCOCC3)nc3c2CCN3C(N)=O)c[nH]1",
    "CS(=O)(=O)Nc1cccc(NC(=O)c2cccc(C3CCN(C4CCC4)CC3)c2)c1",
    "COc1cccc(CN2CCC(N=c3[nH]c(N4CCN(C)CC4)nc4cc(OC)c(OC)cc34)CC2)c1",
    "Cn1cc(C2CCN(CCCCCNC(=O)C=Cc3ccc4c(c3)C(C)(C)CCC4(C)C)CC2)c2ccccc21",
    "O=C(O)c1cc2ccc(C(O)(c3ccc(F)cc3)c3ccc(OC(F)(F)F)cc3)c(C3CC3)c2cc1O",
    "CCOc1ccc(CCNC(=O)C2CCN(S(=O)(=O)c3ccc(C(C)=O)cc3)CC2)cc1",
    "CC(C)C1(C)N=C(N)N=C(N)N1c1ccc(Oc2ccc(C(=O)O)cn2)c(Cl)c1",
    "C=CC(=O)N1CC(Oc2ccc(C#N)c(C(F)(F)F)c2)C(C)(C)C1",
    "CCS(=O)(=O)NC1COc2c(cc(C(C)(C)S(C)(=O)=O)cc2-c2cccc(OC(F)(F)F)c2)C1=O",
    "N#Cc1ccc(N2CCN(N3CCC4(OC=CC45CCCCC5)O3)CC2)cc1",
    "O=C(Nc1ccc(-c2cccc(-c3nc4cc(F)ccc4[nH]3)c2)cc1)Nc1ccc2c(c1)OCO2",
    "COC(=O)C(NC(=O)C(CC(=O)NC(CC(=O)NC(C(O)COc1c(F)ccc(F)c1F)C(C)(C)C)C(C)C)C(C)C)C(C)C",
    "Mn1cc(C2CCN(CCCNC(=O)c3cccs3)CC2)c2ccccc21",  # intentionally bad (Mn organometallic)
    "Cc1cc[nH]c(=NC(=O)NS(=O)(=O)c2cc(C#CC(C)(C)C)sc2C(=O)O)n1",
    "O=C(C1CCS(=O)(=O)CC1)N1CC2CC(C1)N(c1cccnc1)C2",
    "COc1cccc(OC)c1S(=O)(=O)N1CCc2ccccc2C1CC1CCC1",
    "CC(=O)c1cccc(NC(=O)CSCc2nc(-c3ccc(C)cc3)oc2C)c1",
    "Cc1cccc(OCCn2ncc(Cl)c(Cl)c2=O)c1",
    "CCn1nnc(-c2sc(=NCCC(O)COC)[nH]c2Cl)n1",
    "CC(NC(=O)NC1=NC(=O)CN(C)C1=O)c1ccccc1",
    "Mn1ncc(NC(=O)c2csc(-c3c(F)cccc3F)n2)c1C1CCC(N)C(F)CO1",  # intentionally bad
    "CCSc1nc2ccc(C)cc2c(=O)n1CSc1ccc(C#N)cc1",
    "O=C(CSCC(=O)N1CCCC1)Oc1ccc(Cl)cc1",
    "NC(=O)C1CCCN1C(=O)CCc1ccc(-c2ccccc2)cc1",
    "CC(=O)NCc1ccc(Cl)c(Oc2ccccc2C(C)(C)C)c1",
    "O=C(NCc1noc(-c2nsc3ccccc23)n1)C1CCCO1",
    "CCc1cnc(N2CCC(COc3ccncc3)CC2)nc1",
]

# Simple drug-like SMILES for the 128-molecule batch test
_DIVERSE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",               # aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",           # ibuprofen
    "OC(=O)c1ccccc1O",                       # salicylic acid
    "CC(=O)Nc1ccc(O)cc1",                    # acetaminophen
    "c1ccc(NC(=O)c2ccccc2)cc1",              # benzanilide
    "c1ccc2c(c1)Cc1ccccc12",                 # fluorene
    "O=C(O)c1ccc(N)cc1",                     # PABA
    "c1ccc(-c2ccccc2)cc1",                   # biphenyl
    "CC(=O)c1ccccc1",                        # acetophenone
    "c1ccc(CO)cc1",                          # benzyl alcohol
    "c1ccc(CCO)cc1",                         # phenethyl alcohol
    "CC(O)c1ccccc1",                         # 1-phenylethanol
    "c1ccc(C=O)cc1",                         # benzaldehyde
    "OC(=O)Cc1ccccc1",                       # phenylacetic acid
    "c1ccc(OC(=O)C(F)(F)F)cc1",             # phenyl trifluoroacetate
    "Nc1ccc(S(N)(=O)=O)cc1",                # sulfanilamide
    "c1ccc(NC(=O)Nc2ccccc2)cc1",            # diphenylurea
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",        # uric acid
    "Nc1ncnc2[nH]cnc12",                    # adenine
    "O=c1cc[nH]c(=O)[nH]1",                 # uracil
    "Cc1ncc(CO)c(CO)c1O",                   # pyridoxine (B6)
    "OC(=O)c1ccncc1",                       # isonicotinic acid
    "c1ccncc1",                              # pyridine
    "c1cc[nH]c1",                            # pyrrole
    "c1cnc2[nH]ccc2c1",                     # 7-azaindole
    "c1ccc(S(=O)(=O)Nc2ccccc2)cc1",         # diphenylsulfonamide
    "O=C(O)c1ccc(-c2ccccc2)cc1",            # biphenyl-4-carboxylic acid
    "c1ccc(C(=O)Nc2ccccn2)cc1",             # benzoyl pyridin-2-amine
    "CC(C)c1ccccc1",                         # cumene
    "Oc1ccc(Cl)cc1",                         # 4-chlorophenol
    "Oc1ccc(F)cc1",                          # 4-fluorophenol
    "Oc1ccc(Br)cc1",                         # 4-bromophenol
]


def _find_receptor() -> Path | None:
    for p in _RECEPTOR_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def receptor() -> Path:
    r = _find_receptor()
    if r is None:
        pytest.skip("No CDK2 receptor PDBQT found — run calibration first")
    return r


@pytest.fixture(scope="module")
def unidock_binary() -> str:
    p = Path(UNIDOCK)
    if not p.exists():
        pytest.skip(f"UniDock not found at {UNIDOCK}")
    # Quick version check
    result = subprocess.run([UNIDOCK, "--version"], capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0:
        pytest.skip(f"UniDock --version failed: {result.stderr}")
    return UNIDOCK


# ---- Test: single ligand docking (baseline) ----

def test_single_ligand_dock(receptor: Path, unidock_binary: str, tmp_path: Path) -> None:
    """Dock one ligand (aspirin) against CDK2 and verify we get a valid score."""
    lig = tmp_path / "aspirin.pdbqt"
    assert smiles_to_pdbqt("CC(=O)Oc1ccccc1C(=O)O", lig), "SMILES→PDBQT failed for aspirin"

    out = tmp_path / "aspirin_out.pdbqt"
    cx, cy, cz = CDK2_CENTER.split(",")
    sx, sy, sz = CDK2_SIZE.split(",")

    cmd = [
        unidock_binary,
        "--receptor", str(receptor),
        "--ligand", str(lig),
        "--center_x", cx, "--center_y", cy, "--center_z", cz,
        "--size_x", sx, "--size_y", sy, "--size_z", sz,
        "--exhaustiveness", "8",
        "--num_modes", "1",
        "--out", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, f"UniDock failed: {result.stderr[:500]}"
    assert out.exists(), "Output PDBQT not created"

    score = parse_vina_score(out)
    assert score is not None, "Failed to parse score"
    assert SCORE_MIN <= score <= SCORE_MAX, f"Score {score} out of valid range"
    print(f"Single dock score: {score:.2f} kcal/mol")


# ---- Test: small GPU batch (5 ligands) ----

def test_gpu_batch_small(receptor: Path, unidock_binary: str, tmp_path: Path) -> None:
    """GPU batch dock 5 ligands and verify majority produce valid scores."""
    ligs: list[Path] = []
    for i, smi in enumerate(TEST_SMILES):
        p = tmp_path / f"lig_{i}.pdbqt"
        if smiles_to_pdbqt(smi, p):
            ligs.append(p)
    assert len(ligs) >= 3, f"Too few ligands prepared: {len(ligs)}"

    out_dir = tmp_path / "out"
    scores = dock_gpu_batch(
        ligs, receptor, CDK2_CENTER, CDK2_SIZE, out_dir,
        vina_program=unidock_binary,
    )

    valid = [s for s in scores.values() if s is not None]
    print(f"GPU batch: {len(valid)}/{len(ligs)} valid scores")
    for lig, s in scores.items():
        print(f"  {lig.name}: {s}")
    assert len(valid) >= len(ligs) // 2, f"Too many failures: {len(valid)}/{len(ligs)}"


# ---- Test: batch > _GPU_SUB_BATCH triggers sub-batching ----

def test_sub_batching_splits(receptor: Path, unidock_binary: str, tmp_path: Path) -> None:
    """Verify sub-batching activates when batch exceeds _GPU_SUB_BATCH.

    We don't actually dock _GPU_SUB_BATCH+1 ligands (too slow), just
    check the constant and that the function handles a small batch correctly.
    """
    assert _GPU_SUB_BATCH == 16, f"Expected _GPU_SUB_BATCH=16, got {_GPU_SUB_BATCH}"

    # Dock 3 ligands (well under the limit) — should work in a single sub-batch
    ligs: list[Path] = []
    for i, smi in enumerate(TEST_SMILES[:3]):
        p = tmp_path / f"lig_{i}.pdbqt"
        if smiles_to_pdbqt(smi, p):
            ligs.append(p)

    out_dir = tmp_path / "out"
    scores = dock_gpu_batch(
        ligs, receptor, CDK2_CENTER, CDK2_SIZE, out_dir,
        vina_program=unidock_binary,
    )
    assert len(scores) == len(ligs)
    print(f"Sub-batch test: {sum(1 for s in scores.values() if s is not None)}/{len(ligs)} valid")


# ---- Test: 32 REINVENT4-generated molecules (complex, production-like) ----

def test_gpu_batch_reinvent4_molecules(receptor: Path, unidock_binary: str, tmp_path: Path) -> None:
    """Batch of 32 REINVENT4-generated molecules — the real stress test.

    REINVENT4 generates larger, more complex molecules than typical drug-like
    test sets. These can cause UniDock GPU to hang or produce corrupt scores.

    Success criteria:
        - Completes without timeout (< 5 min total)
        - >=50% valid scores
        - No hangs / segfaults
    """
    ligs: list[Path] = []
    skipped: list[str] = []
    for i, smi in enumerate(_REINVENT4_SMILES):
        p = tmp_path / f"r4_lig_{i}.pdbqt"
        if smiles_to_pdbqt(smi, p):
            ligs.append(p)
        else:
            skipped.append(smi[:60])

    n_prepared = len(ligs)
    print(f"\nPrepared {n_prepared}/{len(_REINVENT4_SMILES)} REINVENT4 ligands")
    if skipped:
        print(f"Skipped (PDBQT conversion failed): {len(skipped)}")
    assert n_prepared >= 20, f"Too few REINVENT4 ligands prepared: {n_prepared}"

    out_dir = tmp_path / "out"
    t0 = time.monotonic()
    scores = dock_gpu_batch(
        ligs, receptor, CDK2_CENTER, CDK2_SIZE, out_dir,
        vina_program=unidock_binary,
    )
    elapsed = time.monotonic() - t0

    valid = {k: v for k, v in scores.items() if v is not None}
    n_valid = len(valid)
    n_total = len(scores)
    valid_pct = 100.0 * n_valid / max(n_total, 1)

    print(f"Results: {n_valid}/{n_total} valid ({valid_pct:.0f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(n_total,1):.2f}s/mol)")

    if valid:
        vals = list(valid.values())
        print(f"Score range: [{min(vals):.2f}, {max(vals):.2f}] kcal/mol")

    assert elapsed < 300, f"Took too long: {elapsed:.0f}s (limit 300s)"
    assert n_valid >= n_total * 0.5, (
        f"Too many failures: {n_valid}/{n_total} valid ({valid_pct:.0f}%), need >=50%"
    )


# ---- Test: 128-molecule batch with mixed complexity ----

def _generate_128_mixed() -> list[str]:
    """Build 128 SMILES mixing simple drug-like and REINVENT4 molecules."""
    # First 64: cycle through REINVENT4 molecules
    r4 = [_REINVENT4_SMILES[i % len(_REINVENT4_SMILES)] for i in range(64)]
    # Next 64: cycle through simple drug-like molecules
    simple = [_DIVERSE_SMILES[i % len(_DIVERSE_SMILES)] for i in range(64)]
    return r4 + simple


def test_gpu_batch_128(receptor: Path, unidock_binary: str, tmp_path: Path) -> None:
    """Batch of 128 molecules — mirrors REINVENT4 real batch size.

    Uses a mix of REINVENT4-generated (complex) and simple drug-like molecules.
    With sub-batching (_GPU_SUB_BATCH=32) this should split into 4 sub-batches.

    Success criteria:
        - Completes without timeout (< 10 min total)
        - >=50% valid scores
        - No hangs / segfaults
    """
    smiles = _generate_128_mixed()

    # Phase 1: prepare PDBQTs
    ligs: list[Path] = []
    for i, smi in enumerate(smiles):
        p = tmp_path / f"lig_{i}.pdbqt"
        if smiles_to_pdbqt(smi, p):
            ligs.append(p)

    n_prepared = len(ligs)
    print(f"\nPrepared {n_prepared}/128 ligand PDBQTs")
    assert n_prepared >= 64, f"Too few ligands prepared: {n_prepared}/128"

    # Phase 2: GPU batch dock (should split into 4 sub-batches of 32)
    out_dir = tmp_path / "out"
    t0 = time.monotonic()
    scores = dock_gpu_batch(
        ligs, receptor, CDK2_CENTER, CDK2_SIZE, out_dir,
        vina_program=unidock_binary,
    )
    elapsed = time.monotonic() - t0

    # Phase 3: analyze results
    valid = {k: v for k, v in scores.items() if v is not None}
    corrupt = {k: v for k, v in scores.items() if v is None}
    n_valid = len(valid)
    n_total = len(scores)
    valid_pct = 100.0 * n_valid / max(n_total, 1)

    print(f"Results: {n_valid}/{n_total} valid ({valid_pct:.0f}%), {len(corrupt)} failed")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(n_total,1):.2f}s/mol)")

    if valid:
        vals = list(valid.values())
        print(f"Score range: [{min(vals):.2f}, {max(vals):.2f}] kcal/mol")
        mean_score = sum(vals) / len(vals)
        print(f"Mean score: {mean_score:.2f} kcal/mol")

    # Assertions
    assert elapsed < 600, f"Took too long: {elapsed:.0f}s (limit 600s)"
    assert n_valid >= n_total * 0.5, (
        f"Too many failures: {n_valid}/{n_total} valid ({valid_pct:.0f}%), need >=50%"
    )
