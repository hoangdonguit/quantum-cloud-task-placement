#!/usr/bin/env bash
# Separate supplementary outputs; fail fast, preserve failed artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN_ROOT="results/post_submission_$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN_ROOT"
printf 'OUTPUT: %s\n' "$RUN_ROOT"
python -u scripts/ood_rate04.py --smoke --bootstrap 100 --out "$RUN_ROOT/ood_smoke" 2>&1 | tee "$RUN_ROOT/ood_smoke.log"
python -u scripts/feature_retraining.py --smoke --out "$RUN_ROOT/ablation_smoke" 2>&1 | tee "$RUN_ROOT/ablation_smoke.log"
python -u scripts/quantum_state_check.py --smoke --out "$RUN_ROOT/quantum_smoke" 2>&1 | tee "$RUN_ROOT/quantum_smoke.log"
python -u scripts/ood_rate04.py --out "$RUN_ROOT/ood04" 2>&1 | tee "$RUN_ROOT/ood04.log"
python -u scripts/feature_retraining.py --out "$RUN_ROOT/ablation" 2>&1 | tee "$RUN_ROOT/ablation.log"
python -u scripts/quantum_state_check.py --out "$RUN_ROOT/quantum" 2>&1 | tee "$RUN_ROOT/quantum.log"
printf 'DONE: %s\n' "$RUN_ROOT"
find "$RUN_ROOT" -name COMPLETE.json -print
