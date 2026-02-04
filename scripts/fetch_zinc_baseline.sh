#!/usr/bin/env bash
set -euo pipefail

# Fetch ZINC baseline subset (seed=42) and compute SHA256.
# Usage:
#   ZINC_BASELINE_URL="https://..." bash scripts/fetch_zinc_baseline.sh
#   ZINC_BASELINE_HF_DATASET="Pixelatory/ZINC20-Druglike" bash scripts/fetch_zinc_baseline.sh

BASELINE_DIR="data/baseline"
N="${ZINC_BASELINE_N:-1000}"
SEED="${ZINC_BASELINE_SEED:-42}"
OUT_FILE="${BASELINE_DIR}/zinc_druglike_baseline_${N}_seed${SEED}.smi"
URL="${ZINC_BASELINE_URL:-}"
HF_DATASET="${ZINC_BASELINE_HF_DATASET:-Pixelatory/ZINC20-Druglike}"
HF_SPLIT="${ZINC_BASELINE_HF_SPLIT:-train}"
HF_COLUMN="${ZINC_BASELINE_HF_COLUMN:-}"
HF_BUFFER="${ZINC_BASELINE_HF_BUFFER:-0}"

mkdir -p "$BASELINE_DIR"

if [[ -n "$URL" ]]; then
  echo "Downloading baseline from URL: $URL"
  curl -L "$URL" -o "$OUT_FILE"
else
  echo "Downloading baseline from Hugging Face: ${HF_DATASET} (${HF_SPLIT})"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv}" \
  ZINC_BASELINE_HF_DATASET="$HF_DATASET" \
  ZINC_BASELINE_HF_SPLIT="$HF_SPLIT" \
  ZINC_BASELINE_HF_COLUMN="$HF_COLUMN" \
  ZINC_BASELINE_HF_BUFFER="$HF_BUFFER" \
  ZINC_BASELINE_N="$N" \
  ZINC_BASELINE_SEED="$SEED" \
  ZINC_BASELINE_OUT="$OUT_FILE" \
  uv run python - <<'PY'
import os
import random
from datasets import load_dataset

dataset = os.environ["ZINC_BASELINE_HF_DATASET"]
split = os.environ["ZINC_BASELINE_HF_SPLIT"]
column = os.environ.get("ZINC_BASELINE_HF_COLUMN") or None
buffer_size = int(os.environ.get("ZINC_BASELINE_HF_BUFFER", "100000"))
n = int(os.environ.get("ZINC_BASELINE_N", "1000"))
seed = int(os.environ.get("ZINC_BASELINE_SEED", "42"))
out_path = os.environ["ZINC_BASELINE_OUT"]

ds = load_dataset(dataset, split=split, streaming=False)
rng = random.Random(seed)
if buffer_size > 0:
    # Optional shuffle (uses RAM); keep 0 for low-memory streaming.
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

if column is None:
    for name in ds.features:
        if name.lower() == "smiles":
            column = name
            break

if column is None:
    first = next(iter(ds))
    for key, value in first.items():
        if isinstance(value, str):
            column = key
            break

if column is None:
    raise RuntimeError(f"Could not infer SMILES column for dataset {dataset}")

sample = []
seen = 0
for row in ds:
    val = row.get(column)
    if not val:
        continue
    smi = str(val).strip()
    if not smi:
        continue
    seen += 1
    if len(sample) < n:
        sample.append(smi)
        continue
    j = rng.randint(0, seen - 1)
    if j < n:
        sample[j] = smi

if len(sample) < n:
    raise RuntimeError(f"Only collected {len(sample)} SMILES from {dataset} ({split}); got {len(sample)}")

with open(out_path, "w") as f:
    f.write("\n".join(sample) + "\n")

print(f"Saved {len(sample)} SMILES -> {out_path} (column={column})")
PY
fi

SHA_FILE="${OUT_FILE}.sha256"
sha256sum "$OUT_FILE" | awk '{print $1}' > "$SHA_FILE"

echo "Saved: $OUT_FILE"
echo "SHA256: $(cat "$SHA_FILE")"
