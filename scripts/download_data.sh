#!/usr/bin/env bash
set -euo pipefail

DATASET="ealaxi/paysim1"
DEST="data/raw"

echo "Installing kaggle CLI..."
pip install --quiet kaggle

echo "Downloading PaySim dataset..."
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

CSV_FILE="$DEST/PS_20174392719_1491204439457_log.csv"
if [[ -f "$CSV_FILE" ]]; then
    ROW_COUNT=$(wc -l < "$CSV_FILE")
    echo "Download complete: $ROW_COUNT rows (including header)"
else
    echo "ERROR: Expected file not found at $CSV_FILE"
    ls "$DEST/"
    exit 1
fi
