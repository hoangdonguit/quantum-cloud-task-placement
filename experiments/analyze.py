"""Tong hop ket qua ladder: bang tung run, mean+-std qua seed, kiem dinh ghep cap vs heuristic tot nhat, do thi."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["score", "mean_T", "total_T", "p95_T", "mean_wait", "mean_F", "mean_frel", "resched", "makespan"]


def load(root):
    frames, curves = [], []
    for case_dir in sorted(p for p in root.iterdir() if (p / "case_config.json").exists()):
        cfg = json.loads((case_dir / "case_config.json").read_text())
        e = cfg["env"]
        meta = {"case": case_dir.name, "nodes": len(e["node_names"]), "max_width": e["max_width"], "n_tasks": e["n_tasks"],
                "rate": e["arrival_rate"], "steps": cfg["train"]["timesteps"]}
        paths = list(case_dir.glob("*/eval_per_workload.csv")) + list(case_dir.glob("heuristics_per_workload.csv"))
        for p in paths:
            df = pd.read_csv(p)
            for k, v in meta.items():
                df[k] = v
            frames.append(df)
        for p in case_dir.glob("*/train_log.csv"):
            c = pd.read_csv(p)
            algo, seed = p.parent.name.rsplit("_seed", 1)
            c["case"], c["policy"], c["train_seed"] = case_dir.name, algo, int(seed)
            curves.append(c)
    return pd.concat(frames, ignore_index=True), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    (root / "tables").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    df, curves = load(root)
    meta_cols = ["case", "nodes", "max_width", "n_tasks", "rate", "steps"]

    per_run = df.groupby(meta_cols + ["policy", "kind", "train_seed"])[METRICS].mean().reset_index()
    per_run.to_csv(root / "tables" / "per_run.csv", index=False)

    rows = []
    for (case, pol), g in per_run.groupby(["case", "policy"]):
        kind = g["kind"].iloc[0]
        if kind == "drl":
            row = {"case": case, "policy": pol, "kind": kind, "n": len(g)}
            for m in METRICS:
                row[m], row[m + "_std"] = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else np.nan
        else:
            w = df[(df.case == case) & (df.policy == pol)]
            row = {"case": case, "policy": pol, "kind": kind, "n": len(w)}
            for m in METRICS:
                row[m], row[m + "_std"] = w[m].mean(), w[m].std(ddof=1)
        rows.append(row)
    summ = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(df["case"].drop_duplicates())}
    summ = summ.sort_values(by=["case", "score"], key=lambda s: s.map(order) if s.name == "case" else -s).reset_index(drop=True)
    summ.to_csv(root / "tables" / "summary_by_case.csv", index=False)

    tests, rng = [], np.random.default_rng(0)
    for case, g in summ.groupby("case", sort=False):
        heur = g[g.kind == "heuristic"].iloc[0]["policy"]
        hs = df[(df.case == case) & (df.policy == heur)].set_index("workload_seed")["score"].sort_index()
        for algo in g[g.kind == "drl"]["policy"]:
            d = df[(df.case == case) & (df.policy == algo)].groupby("workload_seed")["score"].mean().sort_index()
            diff = (d - hs).dropna().to_numpy()
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            try:
                p = stats.wilcoxon(diff).pvalue
            except ValueError:
                p = np.nan
            runs = per_run[(per_run.case == case) & (per_run.policy == algo)]["score"]
            tests.append({"case": case, "algo": algo, "best_heuristic": heur, "mean_diff_score": diff.mean(),
                          "ci95_lo": np.percentile(boot, 2.5), "ci95_hi": np.percentile(boot, 97.5),
                          "wilcoxon_p": p, "win_rate_workloads": float((diff > 0).mean()),
                          "seeds_beating_best_heuristic": f"{int((runs > hs.mean()).sum())}/{len(runs)}"})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["significant_p<0.05"] = tests["wilcoxon_p"] < 0.05
    tests.to_csv(root / "tables" / "paired_tests.csv", index=False)

    if len(curves):
        for case, g in curves.groupby("case"):
            plt.figure(figsize=(6.5, 4))
            for algo, ga in g.groupby("policy"):
                m = ga.groupby("timesteps")["val_score"].agg(["mean", "std"]).fillna(0)
                plt.plot(m.index, m["mean"], label=algo)
                plt.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.2)
            best = summ[(summ.case == case) & (summ.kind == "heuristic")].iloc[0]
            plt.axhline(best["score"], ls="--", c="k", lw=1, label=f"best heuristic ({best['policy']}, test)")
            plt.xlabel("timesteps"); plt.ylabel("score (val)"); plt.title(f"Learning curve - {case}")
            plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(root / "figures" / f"learning_{case}.png", dpi=130)
            plt.close()

    axis_of = {"width": "max_width", "nodes": "nodes", "tasks": "n_tasks", "rate": "rate", "steps": "steps"}
    base = per_run[per_run.case == "base"].iloc[0][list(axis_of.values())] if (per_run.case == "base").any() else None
    top_h = summ[summ.kind == "heuristic"].groupby("policy")["score"].mean().sort_values(ascending=False).index[:4].tolist()
    show = ["PPO", "DQN", "DDQN3"] + top_h
    for axis, col in axis_of.items():
        if base is None:
            break
        others = [c for c in per_run.case.unique() if c.startswith(axis + "_")]
        sel = per_run[per_run.case.isin(others + ["base"])].copy()
        fixed = [v for k, v in axis_of.items() if k != axis]
        sel = sel[np.all([sel[v] == base[v] for v in fixed], axis=0)]
        if sel[col].nunique() < 2:
            continue
        plt.figure(figsize=(6.5, 4))
        for pol in show:
            g = sel[sel.policy == pol].groupby(col)["score"].agg(["mean", "std"]).fillna(0).sort_index()
            if len(g):
                plt.errorbar(g.index.astype(str), g["mean"], yerr=g["std"], marker="o", capsize=2, label=pol,
                             ls="-" if pol in ("PPO", "DQN", "DDQN3") else ":")
        plt.xlabel(col); plt.ylabel("score (test)"); plt.title(f"Ladder: {axis}")
        plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(root / "figures" / f"ladder_{axis}.png", dpi=130)
        plt.close()

    pd.set_option("display.width", 200)
    print("\n== Policy tot nhat moi case (score cao hon = tot hon) ==")
    print(summ.groupby("case", sort=False).head(3)[["case", "policy", "kind", "n", "score", "score_std", "mean_T", "mean_frel", "resched"]]
          .round(3).to_string(index=False))
    if len(tests):
        print("\n== DRL vs heuristic tot nhat (ghep cap theo workload) ==")
        print(tests.round(4).to_string(index=False))
    print(f"\nDa ghi: {root/'tables'}  va  {root/'figures'}")


if __name__ == "__main__":
    main()
