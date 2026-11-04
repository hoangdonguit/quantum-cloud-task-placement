"""Loi the cua agent co con khi phan phoi thay doi?"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import deep_update
from src.heuristics import REGISTRY
from experiments.evaluate import eval_seeds, eval_workloads, load_agent


def scenarios(env):
    names = env["node_names"]
    return {
        "in_distribution": {},
        "rate_x0.25": {"arrival_rate": env["arrival_rate"] * 0.25},
        "rate_x2": {"arrival_rate": env["arrival_rate"] * 2.0},
        "tasks_x2": {"n_tasks": env["n_tasks"] * 2},
        "wider_circuits": {"max_width": 50},
        "bursty": {"arrival_mode": "bursty"},
        "calib_drift_x2": {"node_mods": {"err_scale": 2.0}},
        "node_slowdown": {"node_mods": {"slow": {names[1]: 3.0}}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", default="base")
    ap.add_argument("--algos", nargs="*", default=["PPO", "DQN", "DDQN3"])
    args = ap.parse_args()
    case_dir = Path(args.root) / args.case
    cfg = json.loads((case_dir / "case_config.json").read_text())
    seeds = eval_seeds(cfg)
    agents = {}
    for algo in args.algos:
        for run_dir in sorted(case_dir.glob(f"{algo}_seed*")):
            if (run_dir / "run_meta.json").exists():
                agents[(algo, int(run_dir.name.rsplit("seed", 1)[1]))] = load_agent(algo, run_dir, cfg)
    rows = []
    for sname, mod in scenarios(cfg["env"]).items():
        env = deep_update(cfg["env"], mod)
        for pname, cls in REGISTRY.items():
            for r in eval_workloads(env, "test", seeds, policy_cls=cls):
                rows.append({"scenario": sname, "policy": pname, "kind": "heuristic", "train_seed": -1, **r})
        for (algo, ts), act in agents.items():
            for r in eval_workloads(env, "test", seeds, act_fn=act):
                rows.append({"scenario": sname, "policy": algo, "kind": "drl", "train_seed": ts, **r})
        print("xong kich ban", sname)
    df = pd.DataFrame(rows)
    out = Path(args.root) / "tables"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"ood_per_workload_{args.case}.csv", index=False)
    per_run = df.groupby(["scenario", "policy", "kind", "train_seed"])[["score", "mean_T", "mean_frel", "resched"]].mean().reset_index()
    summ = per_run.groupby(["scenario", "policy", "kind"])[["score", "mean_T", "mean_frel", "resched"]].mean().reset_index()
    std = per_run[per_run.kind == "drl"].groupby(["scenario", "policy"])["score"].std().rename("score_std_over_seeds").reset_index()
    summ = summ.merge(std, on=["scenario", "policy"], how="left")
    best_h = summ[summ.kind == "heuristic"].groupby("scenario")["score"].max().rename("best_heuristic_score")
    summ = summ.merge(best_h, on="scenario")
    summ["gap_vs_best_heuristic"] = summ["score"] - summ["best_heuristic_score"]
    summ.to_csv(out / f"ood_summary_{args.case}.csv", index=False)
    piv = summ[summ.policy.isin(args.algos + ["GreedyScore", "HybridEfficiencyRatio", "HybridFidelityQueue", "GreedyFlow"])]
    print(piv.pivot(index="scenario", columns="policy", values="score").round(3).to_string())
    print("\nGap so voi heuristic tot nhat cua tung kich ban (duong = DRL thang):")
    print(summ[summ.kind == "drl"].pivot(index="scenario", columns="policy", values="gap_vs_best_heuristic").round(3).to_string())


if __name__ == "__main__":
    main()
