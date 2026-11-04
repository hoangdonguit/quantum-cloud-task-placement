"""Danh gia heuristic va model tren cac workload held-out (split 'test')."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.env import QCloudSim
from src.heuristics import REGISTRY


def run_episode(sim, workload, act_fn=None, policy=None, pol_seed=0):
    obs = sim.reset(workload)
    if policy is not None:
        policy.reset(sim, pol_seed)
    done = False
    while not done:
        a = policy(sim) if policy is not None else act_fn(obs, sim.action_masks())
        obs, _, done, _ = sim.step(a)
    return sim.metrics()


def eval_workloads(env_cfg, split, seeds, act_fn=None, policy_cls=None):
    sim = QCloudSim(env_cfg, split)
    rows = []
    for s in seeds:
        pol = policy_cls() if policy_cls is not None else None
        m = run_episode(sim, sim.make_workload(s), act_fn=act_fn, policy=pol, pol_seed=s)
        rows.append({"workload_seed": int(s), **m})
    return rows


def eval_seeds(cfg):
    e = cfg["eval"]
    return list(range(e["seed_start"], e["seed_start"] + e["n_workloads"]))


def load_agent(algo_name, run_dir, cfg):
    kind = cfg["algos"][algo_name]["kind"]
    run_dir = Path(run_dir)
    if kind == "ppo":
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(run_dir / "model.zip", device="cpu")
        return lambda o, m: int(model.predict(o, action_masks=m, deterministic=True)[0])
    from algorithms.dqn import DQNAgent
    agent = DQNAgent.load(run_dir / "model.pt")
    return agent.predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.case_config)
    names = args.policies or list(REGISTRY)
    rows = []
    for name in names:
        for r in eval_workloads(cfg["env"], "test", eval_seeds(cfg), policy_cls=REGISTRY[name]):
            rows.append({"policy": name, "kind": "heuristic", "train_seed": -1, **r})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "heuristics_per_workload.csv", index=False)
    df = pd.DataFrame(rows).groupby("policy")[["mean_T", "mean_F", "mean_frel", "resched", "score"]].mean()
    print(df.sort_values("score", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
