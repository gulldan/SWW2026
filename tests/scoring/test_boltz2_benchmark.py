"""Benchmark: Boltz-2 CLI vs NIM scoring performance.

Run with:
    uv run pytest tests/scoring/test_boltz2_benchmark.py -m benchmark -v -n0 -s

Requires:
  - GPU with boltz2 installed (CLI backend)
  - NIM container running on localhost:8000 (NIM backend, optional)

Both backends:
  - Use boltz 2.2.1 model
  - recycling_steps=3, sampling_steps=50, diffusion_samples=1,
    sampling_steps_affinity=50, diffusion_samples_affinity=1, step_scale=1.638
  - Diffusion is stochastic (seed not synced between backends) →
    scores differ between runs and between backends.
    Multi-run averaging separates stochastic noise from systematic differences.

Configurable via environment:
  BENCH_RUNS=3       Number of timed runs per backend (default 3)
  BENCH_NOISE_REPS=10 Repetitions for stochastic noise test (default 10)
"""

from __future__ import annotations

import math
import os
import struct
import time

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

import dualdocking.scoring.boltz2 as mod

# ---------------------------------------------------------------------------
# Molecule set: 20 diverse drug-like SMILES with metadata
# ---------------------------------------------------------------------------
_MOLECULES: list[tuple[str, str, int]] = [
    # (SMILES, name, heavy_atom_count)
    # --- Small fragments (5-15 heavy atoms) ---
    ("c1ccncc1", "pyridine", 6),
    ("c1ccc2[nH]ccc2c1", "indole", 9),
    ("CC(=O)Nc1ccc(O)cc1", "acetaminophen", 11),
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin", 13),
    ("Oc1ccc(cc1)C(O)=O", "4-hydroxybenz-acid", 10),
    # --- Medium drugs (15-25 heavy atoms) ---
    ("c1ccc2c(c1)c(=O)c3ccccc3[nH]2", "acridone", 15),
    ("c1ccc(cc1)C(=O)Nc2ccccc2", "benzanilide", 15),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ibuprofen", 15),
    ("CCOc1ccc2nc(S)n(C)c2c1", "omeprazole-frag", 13),
    ("c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "pyrene", 16),
    ("O=C(O)c1ccc(NCc2ccccc2)cc1", "4-benzylaminobenz", 17),
    ("COc1ccc(CC(=O)O)cc1OC", "homoveratric-acid", 14),
    ("Cc1ccc2nc(NC(=O)c3ccccc3)sc2c1", "benzamidothiazole", 18),
    ("CC(=O)Nc1nnc(s1)S(=O)(=O)c1ccccc1", "acetazolamide-analog", 17),
    ("O=c1[nH]c(=O)c2c(cccc2Cl)n1C1CCCCC1", "cyclohexylquinaz", 19),
    # --- Larger molecules (20-35 heavy atoms) ---
    ("COc1cc(ccc1O)C=NNC(=O)c1ccc(cc1)S(=O)(=O)N", "sulfonamide-hydraz", 22),
    ("CC(C)c1nc(cs1)C(=O)Nc1ccc(cc1)N1CCOCC1", "morpholino-thiazole", 23),
    ("O=C(c1ccc(NC(=O)c2ccncc2)cc1)N1CCN(CC1)c1ccccc1", "piperazinyl-diamide", 27),
    ("CC(C)(C)c1ccc(cc1)C(O)(c1ccccc1)c1ccc(cc1)C(C)(C)C", "bisphenol-ap", 28),
    ("COc1cc(ccc1OC)C(=O)NC(Cc1ccc(O)cc1)C(=O)NC", "dipeptide-mimic", 25),
]

_BENCH_SMILES = [m[0] for m in _MOLECULES]
_TARGET = "CDK2"

# Configurable run counts
N_RUNS = int(os.getenv("BENCH_RUNS", "3"))
N_NOISE_REPS = int(os.getenv("BENCH_NOISE_REPS", "10"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _nim_available() -> bool:
    return mod._nim_health_check()


def _get_gpu_info() -> str:
    """Return GPU name string, or 'unknown' if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{name} ({mem_gb:.0f}GB)"
    except Exception:
        pass
    return "unknown"


def _get_cuda_version() -> str:
    try:
        import torch

        return torch.version.cuda or "N/A"
    except Exception:
        return "N/A"


def _set_backend(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure module for the given backend."""
    monkeypatch.setattr(mod, "_BOLTZ_BACKEND", backend)
    monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
    if backend == "nim":
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", "http://localhost:8000")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://localhost:8000"])


def _run_batch_timed(
    backend: str,
    smiles: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[float, list[float]]:
    """Score batch with given backend, return (elapsed_sec, scores)."""
    _set_backend(backend, monkeypatch)
    start = time.perf_counter()
    scores = mod.score_batch(_TARGET, smiles, use_cache=False)
    elapsed = time.perf_counter() - start
    return elapsed, scores.tolist()


def _multirun(
    backend: str,
    smiles: list[str],
    n_runs: int,
    monkeypatch: pytest.MonkeyPatch,
    *,
    warmup: bool = False,
) -> tuple[list[float], list[list[float]]]:
    """Run scoring n_runs times, return (times, all_scores).

    Args:
        backend: "cli" or "nim".
        smiles: Molecules to score.
        n_runs: Number of timed runs.
        monkeypatch: Pytest monkeypatch fixture.
        warmup: If True, run one untimed warmup pass first (NIM JIT priming).

    Returns:
        Tuple of (list of elapsed times, list of score lists).
    """
    if warmup:
        _run_batch_timed(backend, smiles, monkeypatch)

    times: list[float] = []
    all_scores: list[list[float]] = []
    for _ in range(n_runs):
        elapsed, scores = _run_batch_timed(backend, smiles, monkeypatch)
        times.append(elapsed)
        all_scores.append(scores)
    return times, all_scores


def _is_bf16_grid(value: float) -> bool:
    """Check if a float value aligns to the BF16 mantissa grid.

    BF16 has 8-bit mantissa (7 explicit + 1 implicit), giving step sizes
    that double per power-of-two range. A value is on the BF16 grid if
    round-tripping through BF16 does not change it.
    """
    if not math.isfinite(value):
        return True  # inf/nan are representable
    # Pack as float32, truncate mantissa to BF16 (top 16 bits of float32)
    f32_bytes = struct.pack(">f", value)
    bf16_bytes = f32_bytes[:2] + b"\x00\x00"
    (bf16_val,) = struct.unpack(">f", bf16_bytes)
    return bf16_val == value


def _bf16_step(value: float) -> float:
    """Return the BF16 quantization step size at the given value.

    BF16 step = 2^(exponent - 7) where exponent is the biased exponent.
    """
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    # Extract exponent from float: |value| = mantissa * 2^exp
    _, exp = math.frexp(abs(value))
    # BF16 has 7 explicit mantissa bits → step = 2^(exp - 8)
    return math.ldexp(1.0, exp - 8)


def _mean_scores(all_scores: list[list[float]]) -> list[float]:
    """Average scores across runs, per molecule."""
    arr = np.array(all_scores)  # shape (n_runs, n_mols)
    return np.nanmean(arr, axis=0).tolist()


def _std_scores(all_scores: list[list[float]]) -> list[float]:
    """Std of scores across runs, per molecule."""
    arr = np.array(all_scores)
    return np.nanstd(arr, axis=0).tolist()


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBoltz2Benchmark:
    """Performance and agreement benchmarks for Boltz-2 backends.

    Skipped automatically if GPU or required backend is unavailable.
    Results printed to stdout (run with -s flag).
    """

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    def test_cli_batch_multirun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI backend: N timed runs, report mean +/- std s/mol."""
        n = len(_BENCH_SMILES)
        times, all_scores = _multirun("cli", _BENCH_SMILES, N_RUNS, monkeypatch)

        per_mol = [t / n for t in times]
        mean_t = float(np.mean(per_mol))
        std_t = float(np.std(per_mol))

        print(f"\n=== CLI Batch Benchmark ({N_RUNS} runs, {n} molecules) ===")
        print(f"  Hardware: {_get_gpu_info()}, CUDA {_get_cuda_version()}")
        for i, t in enumerate(times, 1):
            valid = sum(1 for s in all_scores[i - 1] if np.isfinite(s))
            print(f"  Run {i}: {t:.1f}s total, {t / n:.2f}s/mol ({valid}/{n} valid)")
        print(f"  Mean: {mean_t:.2f} ± {std_t:.2f} s/mol")

        # All runs should produce finite scores
        for run_scores in all_scores:
            assert any(np.isfinite(s) for s in run_scores), "All scores NaN"

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_nim_batch_multirun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NIM backend: 1 warmup + N timed runs, report mean +/- std s/mol."""
        n = len(_BENCH_SMILES)
        times, all_scores = _multirun(
            "nim", _BENCH_SMILES, N_RUNS, monkeypatch, warmup=True
        )

        per_mol = [t / n for t in times]
        mean_t = float(np.mean(per_mol))
        std_t = float(np.std(per_mol))

        print(f"\n=== NIM Batch Benchmark (1 warmup + {N_RUNS} runs, {n} molecules) ===")
        print(f"  Hardware: {_get_gpu_info()}, CUDA {_get_cuda_version()}")
        for i, t in enumerate(times, 1):
            valid = sum(1 for s in all_scores[i - 1] if np.isfinite(s))
            print(f"  Run {i}: {t:.1f}s total, {t / n:.2f}s/mol ({valid}/{n} valid)")
        print(f"  Mean: {mean_t:.2f} ± {std_t:.2f} s/mol")

        for run_scores in all_scores:
            assert any(np.isfinite(s) for s in run_scores), "All scores NaN"

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_comparison_multirun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both backends: N runs each, averaged scores, full comparison report.

        Matched parameters:
          boltz 2.2.1, recycling=3, sampling=50, diffusion=1,
          affinity_sampling=50, affinity_diffusion=1, step_scale=1.638.
        Stochastic diffusion → each run gives slightly different scores.
        Multi-run averaging reduces noise for fairer comparison.
        """
        n = len(_BENCH_SMILES)

        # --- CLI multi-run ---
        cli_times, cli_all = _multirun("cli", _BENCH_SMILES, N_RUNS, monkeypatch)

        # --- NIM multi-run (with warmup) ---
        nim_times, nim_all = _multirun(
            "nim", _BENCH_SMILES, N_RUNS, monkeypatch, warmup=True
        )

        # Averaged scores
        cli_mean = _mean_scores(cli_all)
        cli_std = _std_scores(cli_all)
        nim_mean = _mean_scores(nim_all)
        nim_std = _std_scores(nim_all)

        # Timing
        cli_per_mol = [t / n for t in cli_times]
        nim_per_mol = [t / n for t in nim_times]
        cli_t_mean = float(np.mean(cli_per_mol))
        cli_t_std = float(np.std(cli_per_mol))
        nim_t_mean = float(np.mean(nim_per_mol))
        nim_t_std = float(np.std(nim_per_mol))
        speedup = cli_t_mean / nim_t_mean if nim_t_mean > 0 else float("inf")

        # Score agreement (on averaged scores, excluding NaN)
        mask = [
            np.isfinite(c) and np.isfinite(m) for c, m in zip(cli_mean, nim_mean)
        ]
        cli_valid = [c for c, ok in zip(cli_mean, mask) if ok]
        nim_valid = [m for m, ok in zip(nim_mean, mask) if ok]
        n_valid = len(cli_valid)

        if n_valid >= 3:
            r_val, _ = pearsonr(cli_valid, nim_valid)
            rho_val, _ = spearmanr(cli_valid, nim_valid)
            deltas = [m - c for c, m in zip(cli_valid, nim_valid)]
            mae = float(np.mean(np.abs(deltas)))
        else:
            r_val = rho_val = mae = float("nan")

        # BF16 grid analysis
        nim_flat = [s for run in nim_all for s in run if np.isfinite(s)]
        n_bf16 = sum(1 for s in nim_flat if _is_bf16_grid(s))
        n_total_nim = len(nim_flat)
        bf16_pct = 100.0 * n_bf16 / n_total_nim if n_total_nim > 0 else 0.0

        # Max BF16 step in observed score range
        if nim_flat:
            score_range = (min(nim_flat), max(nim_flat))
            max_step = max(_bf16_step(score_range[0]), _bf16_step(score_range[1]))
        else:
            score_range = (0.0, 0.0)
            max_step = 0.0

        # --- Report ---
        print(f"\n{'=' * 64}")
        print(f"  Boltz-2 Benchmark: CLI vs NIM")
        print(f"{'=' * 64}")
        print(f"  Hardware: {_get_gpu_info()}, CUDA {_get_cuda_version()}")
        print(f"  Target: {_TARGET} (298 residues)")
        print(f"  Molecules: {n}")
        print()
        print("  Parameters (matched):")
        print("    recycling_steps=3, sampling_steps=50, diffusion_samples=1")
        print("    sampling_steps_affinity=50, diffusion_samples_affinity=1")
        print("    step_scale=1.638 (boltz2 default)")
        print()
        print("  Per molecule: 1 structure prediction + 1 affinity prediction")
        print(f"  Per batch of {n}: {n} structure + {n} affinity predictions")
        print(f"  Runs: {N_RUNS} (+ 1 warmup for NIM)")
        print()

        # Timing
        print(f"  --- Timing (mean ± std over {N_RUNS} runs) ---")
        print(f"    CLI batch:  {cli_t_mean:.2f} ± {cli_t_std:.2f} s/mol  "
              f"(total {float(np.mean(cli_times)):.1f}s)")
        print(f"    NIM batch:  {nim_t_mean:.2f} ± {nim_t_std:.2f} s/mol  "
              f"(total {float(np.mean(nim_times)):.1f}s)")
        print(f"    Speedup:    {speedup:.1f}x")
        print()

        # Per-molecule score table
        print(f"  --- Score Agreement (averaged over {N_RUNS} runs) ---")
        hdr = (
            f"  {'SMILES':<40s}  {'HA':>3s}  {'CLI_mean±std':>14s}  "
            f"{'NIM_mean±std':>14s}  {'delta':>8s}"
        )
        print(hdr)
        print(f"  {'-' * 40}  {'-' * 3}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
        for (smi, name, ha), cm, cs, nm, ns in zip(
            _MOLECULES, cli_mean, cli_std, nim_mean, nim_std
        ):
            delta = nm - cm if np.isfinite(cm) and np.isfinite(nm) else float("nan")
            cli_str = f"{cm:+.3f}±{cs:.3f}" if np.isfinite(cm) else "NaN"
            nim_str = f"{nm:+.3f}±{ns:.3f}" if np.isfinite(nm) else "NaN"
            delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "NaN"
            print(f"  {smi[:40]:<40s}  {ha:>3d}  {cli_str:>14s}  {nim_str:>14s}  {delta_str:>8s}")
        print(f"  {'-' * 40}  {'-' * 3}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
        print(f"  Pearson r:  {r_val:.3f}")
        print(f"  Spearman ρ: {rho_val:.3f}")
        print(f"  MAE:        {mae:.3f}")
        print(f"  Valid pairs: {n_valid}/{n}")
        print()

        # BF16
        print("  --- BF16 Quantization ---")
        print(f"    NIM scores on BF16 grid: {n_bf16}/{n_total_nim} ({bf16_pct:.0f}%)")
        print(f"    Score range: [{score_range[0]:.3f}, {score_range[1]:.3f}]")
        print(f"    Max quantization step in score range: {max_step:.4f}")
        print()

        # --- Assertions ---
        assert n_valid >= n // 2, f"Too few valid pairs: {n_valid}/{n}"
        if n_valid >= 3:
            assert rho_val > -0.5, f"Ranking anti-correlation: Spearman ρ = {rho_val:.3f}"

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_stochastic_noise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Measure within-backend stochastic noise.

        Same molecule scored M times per backend (no caching).
        Reports per-backend σ — the noise floor that inter-backend
        differences must exceed to be considered systematic.
        """
        smi = _BENCH_SMILES[0]  # pyridine (small, fast)
        results: dict[str, list[float]] = {}

        for backend in ("cli", "nim"):
            _set_backend(backend, monkeypatch)
            runs: list[float] = []
            for _ in range(N_NOISE_REPS):
                s = mod.score_single(_TARGET, smi)
                runs.append(s)
            results[backend] = runs

        print(f"\n=== Stochastic Noise ({smi}, {N_NOISE_REPS} reps per backend) ===")
        for backend, runs in results.items():
            finite = [s for s in runs if np.isfinite(s)]
            if finite:
                mean_s = float(np.mean(finite))
                std_s = float(np.std(finite))
                print(f"\n  {backend.upper()} ({len(finite)}/{len(runs)} finite):")
                for i, s in enumerate(runs, 1):
                    marker = "" if np.isfinite(s) else " [NaN]"
                    print(f"    Rep {i:2d}: {s:.4f}{marker}")
                print(f"    Mean: {mean_s:.4f}, Std: {std_s:.4f}")
                print(f"    Range: [{min(finite):.4f}, {max(finite):.4f}]")
            else:
                print(f"\n  {backend.upper()}: all NaN")

        # At least one backend should have bounded noise
        for backend, runs in results.items():
            finite = [s for s in runs if np.isfinite(s)]
            if len(finite) >= 3:
                std_s = float(np.std(finite))
                assert std_s < 1.0, f"{backend} noise too high: σ = {std_s:.3f}"

    @pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
    @pytest.mark.skipif(not _nim_available(), reason="NIM container not running")
    def test_scaling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Measure throughput vs batch size (5, 10, 20 molecules).

        Tests whether per-molecule cost decreases with batch size
        (amortized subprocess/model loading overhead).
        """
        batch_sizes = [5, 10, 20]

        print("\n=== Scaling: Throughput vs Batch Size ===")
        print(f"  Hardware: {_get_gpu_info()}")
        print(f"  {'Backend':<6s}  {'N':>3s}  {'Total(s)':>9s}  {'s/mol':>7s}")
        print(f"  {'-' * 6}  {'-' * 3}  {'-' * 9}  {'-' * 7}")

        for backend in ("cli", "nim"):
            for size in batch_sizes:
                if size > len(_BENCH_SMILES):
                    continue
                subset = _BENCH_SMILES[:size]
                elapsed, scores = _run_batch_timed(backend, subset, monkeypatch)
                per_mol = elapsed / len(subset)
                valid = sum(1 for s in scores if np.isfinite(s))
                print(
                    f"  {backend.upper():<6s}  {size:>3d}  {elapsed:>9.1f}  "
                    f"{per_mol:>7.2f}  ({valid}/{size} valid)"
                )

        # Basic sanity — 20-molecule batch should finish
        elapsed, scores = _run_batch_timed("nim", _BENCH_SMILES[:20], monkeypatch)
        assert any(np.isfinite(s) for s in scores), "No valid scores in 20-mol batch"
