#!/usr/bin/env python
# ruff: noqa: T201
"""Map SLURM_ARRAY_TASK_ID to (pair, mode, seed) using PipelineConfig.

Uses the same ``generate_tasks()`` order as ``PipelineRunner``, so the
task matrix is guaranteed to be consistent with the Python orchestrator.

Usage::

    # Print full task matrix
    uv run python slurm/task_matrix.py --list

    # Get total task count
    uv run python slurm/task_matrix.py --count

    # Get a single task as shell exports (for eval in sbatch)
    uv run python slurm/task_matrix.py --index 0 --format env
    # Output: export TASK_PAIR=CDK2_ERK2 TASK_MODE=single_a TASK_SEED=42

    # Get a single task as tab-separated values
    uv run python slurm/task_matrix.py --index 0 --format tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_tasks() -> list[tuple[str, str, int]]:
    """Load task list from pipeline config, returning (pair, mode, seed) tuples."""
    # Ensure project root is on sys.path for imports
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from dualdocking.pipeline.config import PipelineConfig

    config = PipelineConfig.from_config_dir(project_root / "configs")
    tasks = config.generate_tasks()
    return [(t.pair, t.mode, t.seed) for t in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(description="SLURM array task matrix for DualDocking.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print full task matrix (TSV).")
    group.add_argument("--count", action="store_true", help="Print total task count.")
    group.add_argument("--index", type=int, metavar="N", help="Get task at array index N.")
    parser.add_argument(
        "--format",
        choices=["env", "tsv"],
        default="env",
        help="Output format for --index (default: env).",
    )
    args = parser.parse_args()

    tasks = _load_tasks()

    if args.count:
        print(len(tasks))
    elif args.list:
        print("index\tpair\tmode\tseed")
        for i, (pair, mode, seed) in enumerate(tasks):
            print(f"{i}\t{pair}\t{mode}\t{seed}")
    elif args.index is not None:
        if args.index < 0 or args.index >= len(tasks):
            print(f"Error: index {args.index} out of range [0, {len(tasks)})", file=sys.stderr)
            sys.exit(1)
        pair, mode, seed = tasks[args.index]
        if args.format == "env":
            print(f"export TASK_PAIR={pair} TASK_MODE={mode} TASK_SEED={seed}")
        else:
            print(f"{pair}\t{mode}\t{seed}")


if __name__ == "__main__":
    main()
