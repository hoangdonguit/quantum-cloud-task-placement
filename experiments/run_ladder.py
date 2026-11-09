"""Chay 'ladder' thuc nghiem: cac muc (case) tu thap -> cao; moi muc: danh gia heuristic + train {algo} x {seed}."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.provenance import run_fingerprint
from src.config import build_ladder, deep_update, load_config
from src.heuristics import REGISTRY
from experiments.evaluate import eval_seeds, eval_workloads


def run_one(case_dir, algo, seed):
    run_dir = case_dir / f"{algo}_seed{seed}"
    if (run_dir / "run_meta.json").exists():
        meta = json.loads((run_dir / "run_meta.json").read_text())
        cfg = load_config(case_dir / "case_config.json")
        if meta.get("fingerprint") != run_fingerprint(cfg):
            raise RuntimeError(f"Incompatible existing run: {run_dir}; use a new output directory")
        model = run_dir / ("model.zip" if cfg["algos"][algo]["kind"] == "ppo" else "model.pt")
        if not model.exists() or not (run_dir / "eval_per_workload.csv").exists():
            raise RuntimeError(f"Incomplete existing run: {run_dir}")
        return run_dir.name, "skip_verified"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / "algorithms" / "train.py"), "--case-config", str(case_dir / "case_config.json"),
           "--algo", algo, "--seed", str(seed), "--out-dir", str(run_dir)]
    # PATCH: them OPENBLAS_NUM_THREADS=1
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(run_dir / "train.log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT).returncode
    return run_dir.name, "ok" if rc == 0 else f"FAILED(rc={rc}) -> xem {run_dir/'train.log'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--axes", nargs="*", default=None, help="vd: base width rate (mac dinh: tat ca)")
    ap.add_argument("--algos", nargs="*", default=["PPO", "DQN", "DDQN3"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = deep_update(cfg, {"train": {"timesteps": 3000, "eval_every": 1000}, "eval": {"n_workloads": 5}})
        args.seeds = args.seeds[:2]
    cases = build_ladder(cfg, args.axes)
    out = Path(args.out)
    print(f"{len(cases)} case x {len(args.algos)} algo x {len(args.seeds)} seed = "
          f"{len(cases) * len(args.algos) * len(args.seeds)} run huan luyen")
    for c in cases:
        print(f"  - {c['name']:12s} nodes={len(c['cfg']['env']['node_names'])} width<={c['cfg']['env']['max_width']} "
              f"tasks={c['cfg']['env']['n_tasks']} rate={c['cfg']['env']['arrival_rate']} steps={c['cfg']['train']['timesteps']}")
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    # PATCH: pre-populate cache o process chinh
    from src.data import get_pool
    for split in ("train", "val", "test"):
        get_pool(split)
    t_all = time.time()
    for c in cases:
        case_dir = out / c["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        ccfg = copy.deepcopy(c["cfg"])
        ccfg["case"] = c["name"]
        config_path = case_dir / "case_config.json"
        if config_path.exists() and load_config(config_path) != ccfg:
            raise RuntimeError(f"Config mismatch: {case_dir}; use a new output directory")
        marker = case_dir / "fingerprint.txt"
        fp = run_fingerprint(ccfg)
        if marker.exists() and marker.read_text() != fp:
            raise RuntimeError(f"Source/data/environment changed: {case_dir}; use a new output directory")
        if not marker.exists() and any(case_dir.glob("*_seed*")):
            raise RuntimeError(f"Legacy output without fingerprint: {case_dir}")
        config_path.write_text(json.dumps(ccfg, indent=2))
        marker.write_text(fp)
        t0 = time.time()
        if not (case_dir / "heuristics_per_workload.csv").exists():
            rows = []
            for name, cls in REGISTRY.items():
                for r in eval_workloads(ccfg["env"], "test", eval_seeds(ccfg), policy_cls=cls):
                    rows.append({"policy": name, "kind": "heuristic", "train_seed": -1, **r})
            pd.DataFrame(rows).to_csv(case_dir / "heuristics_per_workload.csv", index=False)
        print(f"\n=== {c['name']} === heuristic xong ({time.time() - t0:.0f}s); huan luyen {len(args.algos) * len(args.seeds)} run, jobs={args.jobs}")
        tasks = [(case_dir, a, s) for a in args.algos for s in args.seeds]
        with ThreadPoolExecutor(args.jobs) as ex:
            for name, status in ex.map(lambda t: run_one(*t), tasks):
                print(f"   {name:16s} {status}")
                if status.startswith("FAILED"):
                    raise RuntimeError(status)
        frames = [pd.read_csv(p) for p in case_dir.glob("*/eval_per_workload.csv")] + [pd.read_csv(case_dir / "heuristics_per_workload.csv")]
        df = pd.concat(frames)
        per_run = df.groupby(["policy", "train_seed"])["score"].mean().reset_index()
        summ = per_run.groupby("policy")["score"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
        print(summ.round(4).head(8).to_string())
    print(f"\nHoan tat toan bo sau {(time.time() - t_all) / 60:.1f} phut. Tiep theo: python experiments/report.py --root {out}")


if __name__ == "__main__":
    main()
