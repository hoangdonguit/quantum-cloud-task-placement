# Ideal PPO experiment

Standalone extraction of the original ideal quantum-cloud placement experiment
from `quantum_cloud_simulator`. Compares **MaskablePPO**, **FastestNode**,
**LeastQueue**, and **BestFidelity** using the same test workloads.
Noise is disabled and fidelity is always 1. Timing is synthetic.

## Run on the server

Open a terminal in this `ppo` directory. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the original experiment (training and evaluation):

```bash
python run_experiments.py --recipe scaling_config.json
```

Settings in `scaling_config.json`: 100/500/1000 tasks per episode, arrival rates
0.25/0.75, training seeds 0–4, 100,000 requested steps per model, and 30 test
workloads (seeds 100–129). This trains 30 models sequentially with live terminal
output. PPO uses CPU, as in the original configuration. Each case trains fresh
models under its own workload distribution.

To skip completed models after an interrupted run:

```bash
python run_experiments.py --recipe scaling_config.json --resume
```

An unfinished model restarts from scratch. Results and checkpoints are saved
under `results/ideal_<timestamp>/`. Read `policy_summary.csv` for aggregate
metrics and `paired_comparisons.csv` for PPO-versus-baseline comparisons.
`mean_completion_time` means flow time: waiting plus execution time.

## Small single-model run

`config.json` defaults to 10 tasks per episode and 10,000 training steps:

```bash
python train_ppo.py
python evaluate.py --model models/ppo_seed0.zip
```

Keep each model's `.zip` and `.config.json` together. Evaluation must use the
same environment configuration as training.

## Checks without training

```bash
python run_experiments.py --recipe scaling_config.json --dry-run
python -m unittest discover -s tests -v
```

This folder is self-contained. Existing checkpoints/results, stability studies,
OOD evaluation, feature ablations, and the later MinCompletion baseline are
not included.
