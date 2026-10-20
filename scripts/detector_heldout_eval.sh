#!/usr/bin/env bash
# Evaluates the frozen failure detector on a merged held-out run: world extraction, attempt
# labels, detector evaluation. Usage: detector_heldout_eval.sh <run_dir> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1
OUT=${2:-docs/results/heldout/detector}
PY=${PYTHON:-python}
FROZEN=docs/results/baseline/detector/frozen_detector.json

mkdir -p "$OUT"
"$PY" scripts/extract_world.py "$RUN" --jobs 3
"$PY" scripts/label_attempts.py "$RUN" --out "$OUT"
"$PY" scripts/run_detector.py evaluate "$RUN" --attempts "$OUT/attempts.csv" --frozen "$FROZEN" --out "$OUT"
