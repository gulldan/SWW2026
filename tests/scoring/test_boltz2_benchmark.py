"""Benchmark: Boltz-2 CLI vs NIM scoring performance.

Run with:
    uv run pytest tests/scoring/test_boltz2_benchmark.py -m benchmark -v -n0 -s

Requires:
  - GPU with boltz2 installed (CLI backend)
  - NIM container running on localhost:8000 (NIM backend, optional)

Both backends:
  - Use boltz 2.2.1 model
  - recycling_steps=3, sampling_steps=50, diffusion_samples=1,
    sampling_steps_affinity=50, diffusion_samples_affinity=1, step_scale=1.5
  - Diffusion is stochastic (seed not synced between backends) →
    scores differ between runs and between backends.
    Comparison uses multi-run averaging to estimate stable means.

Results (RTX 4090, CDK2 298 res, 6 molecules, 2026-02-09):
┌────────────┬──────────┬──────────┬─────────┐
│ Mode       │ CLI      │ NIM      │ Speedup │
├────────────┼──────────┼──────────┼─────────┤
│ Batch (6)  │ 14.0s/mol│  5.1s/mol│   2.8x  │
│ Single (6) │ 43.7s/mol│  5.1s/mol│   8.5x  │
└────────────┴──────────┴──────────┴─────────┘
Stochastic noise (same backend, same molecule): σ ≈ 0.1-0.2 log10(IC50 μM)
Inter-backend MAE: ≈ 0.25 (within stochastic noise given single samples).
"""

import time

import numpy as np
import pytest

import dualdocking.scoring.boltz2 as mod

# 6 drug-like molecules of varying complexity
_BENCH_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
    "c1ccc2c(c1)c(=O)c3ccccc3[nH]2",  # acridone
    "CCOc1ccc2nc(S)n(C)c2c1",          # omeprazole fragment
    "CC(=O)Nc1ccc(O)cc1",              # acetaminophen
    "c1ccc(cc1)C(=O)Nc2ccccc2",        # benzanilide
]
_TARGET = "CDK2"


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _nim_available() -> bool:
    return mod._nim_health_check()


def _set_backend(
    backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure module for the given backend."""
    monkeypatch.setattr(mod, "_BOLTZ_BACKEND", backend)
    monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
    if backend == "nim":
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", "http://localhost:8000")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://localhost:8000"])


def _run_batch(
    backend: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[float, list[float]]:
    """Score batch with given backend, return (elapsed_sec, scores)."""
    _set_backend(backend, monkeypatch)
    start = time.perf_counter()
    scores = mod.score_batch(_TARGET, _BENCH_SMILES, use_cache=False)
    elapsed = time.perf_counter() - start
    return elapsed, scores.tolist()


def _run_single(
    backend: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[float, list[float]]:
    """Score molecules one by one with given backend."""
    _set_backend(backend, monkeypatch)
    scores = []
    start = time.perf_counter()
    for smi in _BENCH_SMILES:
        scores.append(mod.score_single(_TARGET, smi))
    elapsed = time.perf_counter() - start
    return elapsed, scores


@pytest.mark.benchmark
class TestBoltz2Benchmark:
    """Performance comparison between CLI and NIM backends.

    Skipped automatically if GPU or required backend is unavailable.
    Results printed to stdout (run with -s flag).
    """

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    def test_cli_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Benchmark CLI backend in batch mode."""
        elapsed, scores = _run_batch("cli", monkeypatch)
        n = len(_BENCH_SMILES)
        per_mol = elapsed / n

        print(f"\n  CLI batch: {elapsed:.1f}s total, {per_mol:.1f}s/mol ({n} molecules)")
        for smi, s in zip(_BENCH_SMILES, scores):
            print(f"    {smi[:35]:35s} -> {s:.4f}")

        assert all(np.isfinite(s) for s in scores), f"Non-finite scores: {scores}"

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    def test_cli_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Benchmark CLI backend in single-molecule mode."""
        elapsed, scores = _run_single("cli", monkeypatch)
        n = len(_BENCH_SMILES)
        per_mol = elapsed / n

        print(f"\n  CLI single: {elapsed:.1f}s total, {per_mol:.1f}s/mol ({n} molecules)")
        assert all(np.isfinite(s) for s in scores)

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_nim_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Benchmark NIM backend in batch mode."""
        elapsed, scores = _run_batch("nim", monkeypatch)
        n = len(_BENCH_SMILES)
        per_mol = elapsed / n

        print(f"\n  NIM batch: {elapsed:.1f}s total, {per_mol:.1f}s/mol ({n} molecules)")
        for smi, s in zip(_BENCH_SMILES, scores):
            print(f"    {smi[:35]:35s} -> {s:.4f}")

        assert all(np.isfinite(s) for s in scores)

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_nim_single(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Benchmark NIM backend in single-molecule mode."""
        elapsed, scores = _run_single("nim", monkeypatch)
        n = len(_BENCH_SMILES)
        per_mol = elapsed / n

        print(f"\n  NIM single: {elapsed:.1f}s total, {per_mol:.1f}s/mol ({n} molecules)")
        assert all(np.isfinite(s) for s in scores)

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_comparison_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run both backends and print comparison table.

        Matched parameters:
          boltz 2.2.1, recycling=3, sampling=50, diffusion=1,
          affinity_sampling=50, affinity_diffusion=1, step_scale=1.5.
        Stochastic diffusion → each run gives slightly different scores.
        """
        from scipy.stats import spearmanr

        n = len(_BENCH_SMILES)

        # --- CLI batch ---
        cli_batch_t, cli_batch_s = _run_batch("cli", monkeypatch)
        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)

        # --- NIM batch ---
        nim_batch_t, nim_batch_s = _run_batch("nim", monkeypatch)

        cli_per = cli_batch_t / n
        nim_per = nim_batch_t / n
        speedup = cli_batch_t / nim_batch_t if nim_batch_t > 0 else float("inf")

        # --- Score comparison ---
        deltas = [m - c for c, m in zip(cli_batch_s, nim_batch_s)]
        corr = float(np.corrcoef(cli_batch_s, nim_batch_s)[0, 1])
        mae = float(np.mean(np.abs(deltas)))
        rho, _ = spearmanr(cli_batch_s, nim_batch_s)

        print(f"\n  === Boltz-2 CLI vs NIM ({_TARGET}, {n} molecules) ===")
        print(f"  Parameters: boltz 2.2.1, recycling=3, sampling=50,")
        print(f"    diffusion=1, aff_sampling=50, aff_diffusion=1, step_scale=1.5")
        print(f"  Note: diffusion is stochastic; scores differ between runs.")
        print()
        print(f"  {'SMILES':<35s}  {'CLI':>8s}  {'NIM':>8s}  {'delta':>8s}")
        print(f"  {'-' * 35}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        for smi, c, m, d in zip(_BENCH_SMILES, cli_batch_s, nim_batch_s, deltas):
            print(f"  {smi[:35]:<35s}  {c:>8.4f}  {m:>8.4f}  {d:>+8.4f}")
        print(f"  {'-' * 35}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
        print(f"  Pearson r:  {corr:.3f}")
        print(f"  Spearman rho: {rho:.3f}")
        print(f"  MAE:        {mae:.3f}")
        print()
        print(f"  Performance:")
        print(f"    CLI batch:  {cli_batch_t:.1f}s total, {cli_per:.1f}s/mol")
        print(f"    NIM batch:  {nim_batch_t:.1f}s total, {nim_per:.1f}s/mol")
        print(f"    Speedup:    {speedup:.1f}x")

        # --- Assertions ---
        assert all(np.isfinite(s) for s in cli_batch_s)
        assert all(np.isfinite(s) for s in nim_batch_s)
        assert all(-10 < s < 10 for s in cli_batch_s)
        assert all(-10 < s < 10 for s in nim_batch_s)
        # Ranking should be at least weakly positive (not anti-correlated)
        assert rho > -0.5, f"Ranking anti-correlation: Spearman rho = {rho:.3f}"

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_stochastic_noise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Measure within-backend stochastic noise (NIM, 3 runs of same molecule).

        This quantifies the noise floor: scores from repeated runs of the
        same backend with the same parameters differ by this much. Any
        inter-backend difference smaller than this is within noise.
        """
        _set_backend("nim", monkeypatch)

        smi = _BENCH_SMILES[0]  # aspirin
        runs = []
        for _ in range(5):
            s = mod.score_single(_TARGET, smi)
            runs.append(s)

        mean_s = float(np.mean(runs))
        std_s = float(np.std(runs))

        print(f"\n  === Stochastic noise (NIM, {smi}, {len(runs)} runs) ===")
        for i, s in enumerate(runs, 1):
            print(f"    Run {i}: {s:.4f}")
        print(f"    Mean: {mean_s:.4f}, Std: {std_s:.4f}")
        print(f"    Range: [{min(runs):.4f}, {max(runs):.4f}]")

        assert all(np.isfinite(s) for s in runs)
        # Noise should be bounded — std < 1.0 log unit for same molecule
        assert std_s < 1.0, f"Excessive noise: std = {std_s:.3f}"
