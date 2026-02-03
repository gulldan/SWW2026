#!/usr/bin/env bash
set -euo pipefail

# Fetch ZINC baseline subset (seed=42) and compute SHA256.
# Usage: ZINC_BASELINE_URL="https://..." bash scripts/fetch_zinc_baseline.sh

BASELINE_DIR="data/baseline"
OUT_FILE="${BASELINE_DIR}/zinc_druglike_baseline_1000_seed42.smi"
URL="${ZINC_BASELINE_URL:-}"

if [[ -z "$URL" ]]; then
  echo "ERROR: Set ZINC_BASELINE_URL to the download URL." >&2
  exit 1
fi

mkdir -p "$BASELINE_DIR"

echo "Downloading baseline from: $URL"
curl -L "$URL" -o "$OUT_FILE"

SHA_FILE="${OUT_FILE}.sha256"
sha256sum "$OUT_FILE" | awk '{print $1}' > "$SHA_FILE"

echo "Saved: $OUT_FILE"
echo "SHA256: $(cat "$SHA_FILE")"
