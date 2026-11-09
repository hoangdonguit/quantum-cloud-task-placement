#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git rev-parse --show-toplevel >/dev/null
# Do not silently include files staged before this command.
if ! git diff --cached --quiet; then
  echo 'Index already contains staged changes. Review/commit them first; no index changes made.' >&2
  exit 1
fi
python - <<'PY'
from pathlib import Path
import subprocess
roots=['scripts','algorithms','configs','src','experiments','tools','tests']
files=['README.md','.gitignore','requirements.txt','run_experiments.sh','supplementary.py','run_supplementary.sh','commit_update.sh','repair_quantum.py','data/calibration.json','data/synds_800_sub_60_trainset.csv','data/synds_200_sub_60_testset.csv','results_summary']
paths=[p for p in roots+files if Path(p).exists()]
for root in paths:
    ps=[Path(root)] if Path(root).is_file() else Path(root).rglob('*')
    for p in ps:
        if p.is_file() and p.stat().st_size>45*1024**2:
            raise SystemExit(f'Review large file before staging: {p}')
subprocess.run(['git','add','--',*paths],check=True)
# Refuse model/archive/credential artifacts already tracked inside these paths.
for name in subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACM'],text=True).splitlines():
    p=Path(name)
    if p.suffix in {'.pt','.pth','.zip'} or p.name.startswith('.env') or p.stat().st_size>45*1024**2:
        raise SystemExit(f'Unexpected staged artifact: {name}. Review index; no commit/push made.')
PY
git diff --cached --stat
if git diff --cached --quiet; then echo 'No changes to commit.'; exit 0; fi
git commit -m "Document official experiments and add post-training diagnostics"
git log -1 --oneline
# Push only when explicitly requested. Never force push.
if [[ "${1:-}" == '--push' ]]; then
  branch="$(git branch --show-current)"
  [[ -n "$branch" ]] || { echo 'Detached HEAD: push skipped'; exit 1; }
  git push -u origin "$branch"
fi
