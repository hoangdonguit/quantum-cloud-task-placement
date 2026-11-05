"""Thong ke chat che hon analyze.py (khong can train lai):
  1) Don vi lap lai = SEED huan luyen (n=5), khong phai workload  -> t-test mot mau + CI 95% so voi diem cua doi chung.
  2) Doi chung A: heuristic tot nhat CHON TREN TAP VALIDATION (seed 5000-5029), khong chon tren test (tranh thien lech chon loc).
  3) Doi chung B: GreedyScore co 2 tham so (m, lam) duoc TINH CHINH tren validation (luoi 30 to hop) - baseline 'well-tuned' ma de bai yeu cau.
  4) Hieu chinh da so sanh Holm tren toan bo so kiem dinh.
  python experiments/stats_v2.py --root results/ladder_v1
Dau ra: <root>/tables/stats_v2.csv"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.heuristics import REGISTRY, Policy, _argmax
from experiments.evaluate import eval_seeds, eval_workloads

GRID = [(m, lam) for m in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0) for lam in (0.0, 0.25, 0.5, 1.0, 2.0)]
VAL_SEEDS = list(range(5000, 5030))


class GreedyScoreTuned(Policy):
    """argmax [ frel - m*beta*min(T/t_max,1) - lam*exec/t_max ]. m>1 va lam>0 = 'gia bong' cua viec chiem dung node (ngoai ung)."""
    name = "GreedyScoreTuned"

    def __init__(self, m=1.0, lam=0.0):
        self.m, self.lam = m, lam

    def __call__(self, sim):
        v = sim.view()
        T = v["wait"] + v["exec"] + (sim.cur_arr - sim.arrival[sim.i])
        sc = v["frel"] - self.m * sim.beta * np.minimum(T / sim.t_max, 1.0) - self.lam * np.where(v["feas"], v["exec"], 0.0) / sim.t_max
        return _argmax(sc, v["feas"])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mean_score(env, split, seeds, cls):
    return float(np.mean([r["score"] for r in eval_workloads(env, split, seeds, policy_cls=cls)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for case_dir in sorted(p for p in root.iterdir() if (p / "case_config.json").exists()):
        cfg = json.loads((case_dir / "case_config.json").read_text())
        env, name = cfg["env"], case_dir.name
        heur = pd.read_csv(case_dir / "heuristics_per_workload.csv").groupby("policy")["score"].mean()
        val = {n: mean_score(env, "val", VAL_SEEDS, cls) for n, cls in REGISTRY.items()}
        best_val = max(val, key=val.get)
        gbest = max(GRID, key=lambda g: mean_score(env, "val", VAL_SEEDS, lambda g=g: GreedyScoreTuned(*g)))
        tuned_test = float(np.mean([r["score"] for r in eval_workloads(env, "test", eval_seeds(cfg), policy_cls=lambda: GreedyScoreTuned(*gbest))]))
        comps = {"best_heur_by_val": (best_val, float(heur[best_val])), "greedy_tuned": (f"m={gbest[0]},lam={gbest[1]}", tuned_test)}
        for run_algo in sorted({p.name.rsplit("_seed", 1)[0] for p in case_dir.glob("*_seed*")}):
            sc = np.array([pd.read_csv(p / "eval_per_workload.csv")["score"].mean() for p in sorted(case_dir.glob(f"{run_algo}_seed*"))])
            n = len(sc)
            for cname, (label, ref) in comps.items():
                diff = sc.mean() - ref
                se = sc.std(ddof=1) / np.sqrt(n)
                t = diff / se if se > 0 else np.sign(diff) * np.inf
                p = 2 * stats.t.sf(abs(t), n - 1)
                tc = stats.t.ppf(0.975, n - 1)
                rows.append({"case": name, "algo": run_algo, "comparator": cname, "comparator_detail": label, "comparator_test_score": ref,
                             "drl_mean": sc.mean(), "drl_seed_std": sc.std(ddof=1), "n_seeds": n, "diff": diff,
                             "ci95_lo": diff - tc * se, "ci95_hi": diff + tc * se, "p_seed": p,
                             "seeds_above": int((sc > ref).sum())})
        print(f"{name:12s} best_by_val={best_val:22s} tuned={gbest} tuned_test={tuned_test:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for cname in df.comparator.unique():  # Holm rieng cho tung ho doi chung
        m = df.comparator == cname
        df.loc[m, "p_holm"] = holm(df.loc[m, "p_seed"].values)
    out = root / "tables" / "stats_v2.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    for cname in df.comparator.unique():
        d = df[df.comparator == cname]
        print(f"\n== DRL vs {cname}: diff>0 = DRL thang | {len(d)} kiem dinh ==")
        print(f"  thang p<.05 (chua hieu chinh): {((d.p_seed < .05) & (d['diff'] > 0)).sum()}   sau Holm: {((d.p_holm < .05) & (d['diff'] > 0)).sum()}"
              f"   | thua p<.05: {((d.p_seed < .05) & (d['diff'] < 0)).sum()}   sau Holm: {((d.p_holm < .05) & (d['diff'] < 0)).sum()}")
        print(d.pivot(index="case", columns="algo", values="diff").round(4).to_string())
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
