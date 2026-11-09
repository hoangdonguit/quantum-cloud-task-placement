#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS="${JOBS:-4}"
python experiments/official.py --check
mkdir -p results/official
python -m pip freeze > results/official/environment.txt
python tools/check_data.py --out results/official/split_manifest.json
python experiments/official.py --jobs "$JOBS" 2>&1 | tee -a results/official/pipeline.log
