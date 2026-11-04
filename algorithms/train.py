"""Huan luyen MOT run (case x algo x seed) roi danh gia tren workload test."""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from src.config import load_config
from src.env import QCloudGymEnv, QCloudSim
from experiments.evaluate import eval_seeds, eval_workloads, load_agent


def make_val_fn(env_cfg, cfg):
    seeds = list(range(5000, 5000 + int(cfg["train"]["n_val"])))

    def fn(act_fn):
        rows = eval_workloads(env_cfg, "val", seeds, act_fn=act_fn)
        df = pd.DataFrame(rows).mean()
        return {"val_score": df["score"], "val_mean_T": df["mean_T"], "val_frel": df["mean_frel"], "val_resched": df["resched"]}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-config", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.case_config)
    env_cfg, tcfg = cfg["env"], cfg["train"]
    steps = args.timesteps or int(tcfg["timesteps"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(cfg["algos"][args.algo])
    kind = spec.pop("kind")
    env = QCloudGymEnv(env_cfg, "train")
    val_fn = make_val_fn(env_cfg, cfg)
    curve, t0 = [], time.time()

    def log(t, res):
        curve.append({"timesteps": t, **res})

    if kind == "ppo":
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback

        class ValCb(BaseCallback):
            def __init__(self):
                super().__init__()
                self.nxt = tcfg["eval_every"]

            def _on_step(self):
                if self.num_timesteps >= self.nxt:
                    log(self.num_timesteps, val_fn(lambda o, m: int(self.model.predict(o, action_masks=m, deterministic=True)[0])))
                    self.nxt += tcfg["eval_every"]
                return True

        net = spec.pop("net_arch", [128, 128])
        model = MaskablePPO("MlpPolicy", env, seed=args.seed, verbose=0, device="cpu",
                            policy_kwargs=dict(net_arch=net), **spec)
        model.learn(total_timesteps=steps, callback=ValCb())
        model.save(out / "model.zip")
    else:
        from algorithms.dqn import DQNAgent
        agent = DQNAgent(env.sim.obs_dim, env.sim.M, spec, seed=args.seed)
        agent.learn(env, steps, eval_fn=val_fn, eval_every=tcfg["eval_every"], log=log)
        agent.save(out / "model.pt")

    train_time = time.time() - t0
    curve.append({"timesteps": steps, **val_fn(load_agent(args.algo, out, cfg))})
    pd.DataFrame(curve).to_csv(out / "train_log.csv", index=False)

    rows = eval_workloads(env_cfg, "test", eval_seeds(cfg), act_fn=load_agent(args.algo, out, cfg))
    df = pd.DataFrame(rows)
    df.insert(0, "train_seed", args.seed)
    df.insert(0, "kind", "drl")
    df.insert(0, "policy", args.algo)
    df.to_csv(out / "eval_per_workload.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps({"algo": args.algo, "seed": args.seed, "timesteps": steps,
                                                   "train_seconds": round(train_time, 1),
                                                   "test_score": float(df["score"].mean())}, indent=2))
    print(f"[{args.algo} seed={args.seed}] steps={steps} train={train_time:.0f}s test_score={df['score'].mean():.4f}")


if __name__ == "__main__":
    main()
