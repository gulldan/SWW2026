"""Migrate existing CSV data files to Parquet format.

Converts pipeline output CSVs to Parquet while preserving originals.
Idempotent: safe to run multiple times.

Usage::

    uv run python cli/migrate_csv_to_parquet.py --dry-run
    uv run python cli/migrate_csv_to_parquet.py --verify --keep-csv
    uv run python cli/migrate_csv_to_parquet.py --verify --remove-csv
    uv run python cli/migrate_csv_to_parquet.py --verify --force
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Glob patterns for CSVs to migrate (relative to PROJECT_ROOT).
# Does NOT include: boltz_msa/*.csv, alert_collections.csv, train_*.csv, *.smi
#
# Both "output/..." (local dev layout) and flat (server layout without
# the output/ prefix) are listed so the script works on either structure.
MIGRATION_PATTERNS = [
    # Stage 2: Generation
    "output/results/**/generated.csv",
    "results/**/generated.csv",
    # Stage 3: Scoring
    "output/results/**/scores_A.csv",
    "output/results/**/scores_B.csv",
    "output/results/**/scored_joined.csv",
    "results/**/scores_A.csv",
    "results/**/scores_B.csv",
    "results/**/scored_joined.csv",
    # Stage 1: Calibration
    "output/data/calibration/scores/*.csv",
    "data/calibration/scores/*.csv",
    # Stage 1: Enrichment validation
    "output/data/enrichment/scores/*_actives.csv",
    "data/enrichment/scores/*_actives.csv",
    # Stage 4: Analysis
    "report/data/threshold_sensitivity.csv",
    "report/data/proxy_final_corr.csv",
    "report/data/all_results.csv",
    # Stage 1: Validation
    "report/data/redocking_metrics.csv",
    "report/data/enrichment_metrics.csv",
]

# Patterns to exclude even if they match above
EXCLUDE_PATTERNS = {
    "boltz_msa",
    "alert_collections",
    "train_",
    "freedpp_workspace",
}


def _should_exclude(path: Path) -> bool:
    """Check if a path should be excluded from migration.

    Args:
        path: The file path to check against exclusion patterns.

    Returns:
        True if the path contains any of the EXCLUDE_PATTERNS substrings,
        False otherwise.
    """
    path_str = str(path)
    return any(excl in path_str for excl in EXCLUDE_PATTERNS)


def _find_csvs(root: Path) -> list[Path]:
    """Find all CSV files matching migration patterns.

    Searches under the project root for CSV files that match the
    MIGRATION_PATTERNS globs, excluding any paths that match
    EXCLUDE_PATTERNS.  Duplicate paths (possible when both ``output/…``
    and flat patterns match the same resolved file) are removed.

    Args:
        root: The project root directory to search under.

    Returns:
        A deduplicated list of Path objects for CSV files eligible for
        migration, in discovery order.
    """
    seen: set[Path] = set()
    found: list[Path] = []
    for pattern in MIGRATION_PATTERNS:
        for csv_path in sorted(root.glob(pattern)):
            resolved = csv_path.resolve()
            if resolved not in seen and not _should_exclude(csv_path):
                seen.add(resolved)
                found.append(csv_path)
    return found


def _migrate_one(
    csv_path: Path,
    *,
    force: bool = False,
    verify: bool = False,
    remove_csv: bool = False,
    dry_run: bool = False,
) -> str:
    """Migrate a single CSV file to Parquet format.

    Reads the CSV, converts eligible object columns to numeric types,
    writes the result as a Parquet file alongside the original, and
    optionally verifies the output and backs up the source CSV.

    Args:
        csv_path: Path to the source CSV file.
        force: If True, overwrite an existing Parquet file even when
            shapes match. Defaults to False.
        verify: If True, read back the written Parquet and validate
            that shapes and numeric column sums match. On failure the
            Parquet file is deleted. Defaults to False.
        remove_csv: If True, rename the original CSV to a ``.csv.bak``
            backup after successful conversion. Defaults to False.
        dry_run: If True, report what would happen without writing
            any files. Defaults to False.

    Returns:
        A status string indicating the outcome. Possible values are
        ``'converted'``, ``'would_convert'``, ``'skipped:exists'``,
        ``'skipped:mismatch'``, or an error string prefixed with
        ``'error:'``.
    """
    parquet_path = csv_path.with_suffix(".parquet")

    if dry_run:
        if parquet_path.exists() and not force:
            return "skipped:exists"
        return "would_convert"

    # Check if Parquet already exists
    if parquet_path.exists() and not force:
        try:
            existing_pq = pd.read_parquet(parquet_path)
            csv_df = pd.read_csv(csv_path)
            if existing_pq.shape == csv_df.shape:
                return "skipped:exists"
            logger.warning(
                "Shape mismatch {}: parquet={} csv={} (use --force to overwrite)",
                csv_path.name,
                existing_pq.shape,
                csv_df.shape,
            )
            return "skipped:mismatch"
        except Exception:
            # Can't read existing parquet, skip with warning
            return "skipped:exists"

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error("Failed to read {}: {}", csv_path, exc)
        return f"error:read:{exc}"

    # Convert numeric columns
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only convert if most values are numeric
            if converted.notna().sum() > df[col].notna().sum() * 0.5:
                # Check if original had non-numeric strings
                orig_non_null = df[col].dropna()
                conv_non_null = converted.dropna()
                if len(conv_non_null) == len(orig_non_null):
                    df[col] = converted

    # Write Parquet
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logger.error("Failed to write {}: {}", parquet_path, exc)
        return f"error:write:{exc}"

    # Verify
    if verify:
        try:
            pq_df = pd.read_parquet(parquet_path)
            if pq_df.shape != df.shape:
                logger.error(
                    "Verification failed for {}: shape {} vs {}",
                    parquet_path,
                    pq_df.shape,
                    df.shape,
                )
                parquet_path.unlink()
                return "error:verify:shape"
            # Check numeric column sums
            for col in df.select_dtypes(include="number").columns:
                csv_sum = df[col].sum()
                pq_sum = pq_df[col].sum()
                if pd.notna(csv_sum) and pd.notna(pq_sum) and abs(csv_sum - pq_sum) > 1e-6 * max(abs(csv_sum), 1.0):
                    logger.error(
                        "Verification failed for {} col {}: sum {} vs {}",
                        parquet_path,
                        col,
                        csv_sum,
                        pq_sum,
                    )
                    parquet_path.unlink()
                    return f"error:verify:sum:{col}"
        except Exception as exc:
            logger.error("Verification read failed for {}: {}", parquet_path, exc)
            return f"error:verify:{exc}"

    # Handle CSV removal
    if remove_csv:
        backup_path = csv_path.with_suffix(".csv.bak")
        csv_path.rename(backup_path)
        logger.debug("Backed up {} -> {}", csv_path.name, backup_path.name)

    return "converted"


@click.command()
@click.option("--project-root", default=".", help="Project root directory")
@click.option("--dry-run", is_flag=True, help="Show what would be done")
@click.option("--keep-csv/--remove-csv", default=True, help="Keep CSV after conversion (default: keep)")
@click.option("--verify", is_flag=True, help="Verify Parquet after conversion")
@click.option("--force", is_flag=True, help="Overwrite existing Parquet files")
def main(
    project_root: str,
    dry_run: bool,
    keep_csv: bool,
    verify: bool,
    force: bool,
) -> None:
    """Migrate CSV pipeline outputs to Parquet format.

    Finds all CSV files matching known pipeline output patterns and
    converts them to Parquet. Original CSVs are preserved by default.
    Idempotent: skips files that already have a matching Parquet.

    Args:
        project_root: Path to the project root directory. Resolved to
            an absolute path before use.
        dry_run: If True, only report what would be done without
            writing any files.
        keep_csv: If True, keep original CSV files after conversion.
            If False, rename them to ``.csv.bak``.
        verify: If True, read back each written Parquet file and
            validate shapes and numeric sums against the source CSV.
        force: If True, overwrite existing Parquet files even when
            they already match the CSV.
    """
    root = Path(project_root).resolve()
    csvs = _find_csvs(root)

    if not csvs:
        logger.info("No CSV files found to migrate")
        return

    logger.info("Found {} CSV file(s) to process", len(csvs))

    stats = {"converted": 0, "skipped": 0, "errors": 0, "would_convert": 0}

    for csv_path in csvs:
        rel = csv_path.relative_to(root)
        status = _migrate_one(
            csv_path,
            force=force,
            verify=verify,
            remove_csv=not keep_csv,
            dry_run=dry_run,
        )

        if status == "converted":
            logger.info("[CONVERTED] {}", rel)
            stats["converted"] += 1
        elif status == "would_convert":
            logger.info("[WOULD CONVERT] {}", rel)
            stats["would_convert"] += 1
        elif status.startswith("skipped"):
            logger.debug("[SKIPPED] {} ({})", rel, status)
            stats["skipped"] += 1
        elif status.startswith("error"):
            logger.error("[ERROR] {} ({})", rel, status)
            stats["errors"] += 1

    # Summary
    logger.info("--- Migration Summary ---")
    if dry_run:
        logger.info("Would convert: {}", stats["would_convert"])
        logger.info("Already exist: {}", stats["skipped"])
    else:
        logger.info("Converted: {}", stats["converted"])
        logger.info("Skipped: {}", stats["skipped"])
        logger.info("Errors: {}", stats["errors"])

    if stats["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
