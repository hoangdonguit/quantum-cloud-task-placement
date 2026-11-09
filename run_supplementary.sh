#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
OUT="results/supplementary_$(date +%Y%m%d_%H%M%S)"
printf 'Output: %s\n' "$OUT"
python -u supplementary.py --out "$OUT" "$@"
