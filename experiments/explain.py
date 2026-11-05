"""Agent hoc duoc gi? Permutation importance + agreement voi heuristic."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.env import QCloudSim, obs_groups
from src.heuristics import REGISTRY
from experiments.evaluate import eval_seeds, load_agent


def rollout(sim, seed, act_fn, perturb=None, rng=None, shuffle_cols=None):
    obs = sim.reset(sim.make_workload(seed))
    done = False
    while not done:
        mask = sim.action_masks()
        o = obs
        if perturb is not None:
            cols, pool = perturb
            o = obs.copy()
            o[cols] = pool[rng.integers(len(pool))][cols]
        if shuffle_cols is not None:
            # hoan vi GIUA CAC NODE
            o = obs.copy()
            blk = o[shuffle_cols].reshape(sim.M, -1)
            o[shuffle_cols] = blk[rng.permutation(sim.M)].ravel()
        obs, _, done, _ = sim.step(act_fn(o, mask))
    return sim.metrics()["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", default="base")
    ap.add_argument("--algos", nargs="*", default=["PPO", "DQN", "DDQN3"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    
    case_dir = Path(args.root) / args.case
    cfg = json.loads((case_dir / "case_config.json").read_text())
    sim = QCloudSim(cfg["env"], "test")
    groups = obs_groups(sim.M)
    seeds = eval_seeds(cfg)
    
    # Tao pool quan sat tu tap validation
    vsim = QCloudSim(cfg["env"], "val")
    pool = []
    for s in range(7000, 7010):
        pol = REGISTRY["GreedyScore"]()
        obs = vsim.reset(vsim.make_workload(s))
        pol.reset(vsim, s)
        done = False
        while not done:
            pool.append(obs)
            obs, _, done, _ = vsim.step(pol(vsim))
    pool = np.array(pool)
    
    rng = np.random.default_rng(0)
    imp_rows, agr_rows = [], []
    
    for algo in args.algos:
        for run_dir in sorted(case_dir.glob(f"{algo}_seed*")):
            if not (run_dir / "run_meta.json").exists():
                continue
            seed_tr = int(run_dir.name.rsplit("seed", 1)[1])
            act = load_agent(algo, run_dir, cfg)
            
            base = np.mean([rollout(sim, s, act) for s in seeds])
            
            for g, cols in groups.items():
                if g == "node_static":
                    # Vá lỗi: Hoán vị giữa các node thay vì zero-mask/perturb
                    sc = np.mean([rollout(sim, s, act, shuffle_cols=cols, rng=rng) for s in seeds for _ in range(args.repeats)])
                else:
                    # Các nhóm khác giữ nguyên logic cũ
                    sc = np.mean([rollout(sim, s, act, perturb=(cols, pool), rng=rng) for s in seeds for _ in range(args.repeats)])
                
                imp_rows.append({
                    "algo": algo, "train_seed": seed_tr, "group": g, 
                    "base_score": base, "perturbed_score": sc, "score_drop": base - sc
                })
                
            hs = {n: cls() for n, cls in REGISTRY.items()}
            match = {n: [] for n in hs}
            for s in seeds[:10]:
                obs = sim.reset(sim.make_workload(s))
                for h in hs.values():
                    h.reset(sim, s)
                done = False
                while not done:
                    a = act(obs, sim.action_masks())
                    for n, h in hs.items():
                        match[n].append(h(sim) == a)
                    obs, _, done, _ = sim.step(a)
            for n, m in match.items():
                agr_rows.append({"algo": algo, "train_seed": seed_tr, "heuristic": n, "agreement": float(np.mean(m))})
                
    out = Path(args.root) / "tables"
    out.mkdir(exist_ok=True)
    imp = pd.DataFrame(imp_rows)
    agr = pd.DataFrame(agr_rows)
    imp.to_csv(out / f"explain_importance_{args.case}.csv", index=False)
    agr.to_csv(out / f"explain_agreement_{args.case}.csv", index=False)
    
    print("== Permutation importance ==")
    print(imp.groupby(["algo", "group"])["score_drop"].agg(["mean", "std"]).round(4).to_string())
    print("\n== Agreement voi heuristic ==")
    print(agr.groupby(["algo", "heuristic"])["agreement"].mean().unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()