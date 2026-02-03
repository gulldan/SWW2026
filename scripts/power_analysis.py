"""Post-hoc power analysis (paired t-test approximation)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import click
from loguru import logger
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _power_two_sided(d: float, n: int, alpha: float = 0.05) -> float:
    if n <= 1 or not math.isfinite(d):
        return float("nan")
    df = n - 1
    ncp = d * math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Two-sided power
    return float(1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))


@click.command()
@click.option("--stats", "stats_path", default="report/stats.json")
@click.option("--output", "output_path", default="report/power_analysis.txt")
@click.option("--n", default=5, type=int, help="Number of paired seeds")
@click.option("--alpha", default=0.05, type=float)
def main(stats_path: str, output_path: str, n: int, alpha: float) -> None:
    stats_file = PROJECT_ROOT / stats_path
    if not stats_file.exists():
        raise FileNotFoundError(f"stats.json не найден: {stats_file}")

    stats_data = json.loads(stats_file.read_text())

    lines: list[str] = []
    lines.append("Post-hoc power analysis (paired t-test approximation)")
    lines.append(f"n={n}, alpha={alpha}")
    lines.append("Caveat: This is a rough estimate; actual method is hierarchical bootstrap")
    lines.append("")

    for pair_id, tests in stats_data.items():
        lines.append(f"Pair: {pair_id}")
        for test_id, entry in tests.items():
            d = float(entry.get("cohens_d", float("nan")))
            power = _power_two_sided(d, n=n, alpha=alpha)
            lines.append(f"  {test_id}: d={d:.3f} -> power={power:.3f}")
        lines.append("")

    out_path = PROJECT_ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Power analysis -> {}", out_path)


if __name__ == "__main__":
    main()
