#!/usr/bin/env bash
# Pull a Kaggle kernel notebook into notebooks/kaggle/
# Usage: ./scripts/kaggle-pull.sh owner/kernel-slug
# Example: ./scripts/kaggle-pull.sh gusthema/spaceship-titanic-with-tfdf

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
OUT="$ROOT/notebooks/kaggle"

if [[ ! -x "$KAGGLE" ]]; then
  echo "Kaggle CLI not found. Run: $ROOT/.venv/bin/pip install kaggle"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 owner/kernel-slug"
  echo "Example: $0 gusthema/spaceship-titanic-with-tfdf"
  exit 1
fi

KERNEL="$1"
mkdir -p "$OUT"
cd "$OUT"

echo "Pulling kernel: $KERNEL -> $OUT"
"$KAGGLE" kernels pull "$KERNEL" -m
echo "Done. Files in: $OUT"
