# Repository layout

Root: README.md, requirements.txt and Git configuration.

- src/, algorithms/, experiments/, tools/: frozen core from the official study. Do not relocate Python here while preserving the historical fingerprint.
- scripts/: launchers, saved-model demo, supplementary evaluation, beta study, Aer repair and commit helper.
- docs/: report, system diagram, HTML template and demo guide.
- data/, configs/: model inputs and experiment configuration.
- results/: complete local raw outputs and checkpoints, ignored by Git.
- results_summary/: curated tables and figures tracked in Git.
- logs/: local console logs, ignored by Git.

Launch scripts from the repo root. Example: python scripts/demo_saved_models.py; bash scripts/run_supplementary.sh.
Generated HTML remains at its original results/demo_.../demo.html path and can still be opened.
Aer repair retains original bytes and operates relative to the repo working directory. Its original timeout/provenance limitations still apply.
Do not rerun historical full training merely to test layout.
