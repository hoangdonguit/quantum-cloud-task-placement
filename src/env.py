"""Moi truong mo phong dat tac vu luong tu len cac QNode (quantum cloud task placement).

Mo hinh bam theo 3 bai bao + framework goc:
  * QSimPy (Nguyen, Usman, Buyya - arXiv:2405.01021): QNode (qubits, CLOPS -> D1CPS = CLOPS*19.95), QTask lay tu dataset
    MQT Bench da transpile san, hang doi FIFO moi node (uoc luong waiting qua next_available_time), reschedule khi sai rang buoc qubit.
  * DRLQ (arXiv:2407.02748): t_exec = depth * shots / D1CPS ; completion = wait + exec ; phat khi reschedule.
  * QFOR (arXiv:2508.04974): fidelity = prod_g (1 - eps_g) tren mach da transpile, dung calibration IBM; reward = F - beta*T.

Dispatcher observes model-derived estimates, not future tasks/noise outcomes. Fidelity is an analytical proxy, not measured state fidelity."""
import bisect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import MODELS, SPLIT_ID, get_pool

ROOT = Path(__file__).resolve().parent.parent
NODE_FEATS = ["feasible", "qubits", "clops", "e_2q", "e_ro", "t2", "wait", "qlen", "exec", "frel"]
TASK_FEATS = ["width", "orig_depth", "cx127", "shots", "resched"]
T_OBS = np.log1p(600.0)


def time_cost(T, scale, mode):
    x = np.asarray(T) / scale
    if mode == "legacy_clip": return np.minimum(x, 1.0)
    if mode == "linear": return x
    if mode == "log1p": return np.log1p(x) / np.log(2.0)
    raise ValueError(mode)


def _lognorm(x):
    return np.log1p(x) / T_OBS


def obs_groups(n_nodes):
    nt, nf = len(TASK_FEATS), len(NODE_FEATS)
    col = lambda names: np.array([nt + j * nf + NODE_FEATS.index(f) for j in range(n_nodes) for f in names])
    return {"task": np.arange(nt), "feasible": col(["feasible"]),
            "node_static": col(["qubits", "clops", "e_2q", "e_ro", "t2"]),
            "queue": col(["wait", "qlen"]), "estimates": col(["exec", "frel"])}


def build_nodes(env_cfg):
    cal = json.loads((ROOT / "data" / "calibration.json").read_text())["nodes"]
    mods = env_cfg.get("node_mods", {}) or {}
    nodes = []
    for name in env_cfg["node_names"]:
        n = dict(cal[name])
        n["key"] = name
        scale = float(mods.get("err_scale", 1.0))
        for k in ("e_2q", "e_1q", "e_ro"):
            n[k] = min(n[k] * scale, 0.5)
        n["d1cps"] = n["d1cps"] / float(mods.get("slow", {}).get(name, 1.0))
        nodes.append(n)
    return nodes


class QCloudSim:
    def __init__(self, env_cfg, split="train"):
        self.cfg = env_cfg
        self.split = split
        self.pool = get_pool(split, env_cfg.get("data_protocol", "workload"))
        self.nodes = build_nodes(env_cfg)
        self.M = len(self.nodes)
        self.shots = float(env_cfg["shots"])
        self.beta, self.t_max = float(env_cfg["beta"]), float(env_cfg["t_max"])
        self.fail_penalty, self.resched_delay = float(env_cfg["fail_penalty"]), float(env_cfg["resched_delay"])
        self.max_resched = int(env_cfg["max_resched"])
        mi = np.array([MODELS.index(n["model"]) for n in self.nodes])
        p = self.pool
        qubits = np.array([n["qubits"] for n in self.nodes])
        d1cps = np.array([n["d1cps"] for n in self.nodes])
        e2, e1, ero = (np.array([n[k] for n in self.nodes]) for k in ("e_2q", "e_1q", "e_ro"))
        self.feas = (p.depth[:, mi] >= 0) & (p.width[:, None] <= qubits[None, :])
        self.exec = np.where(self.feas, p.depth[:, mi] / d1cps[None, :] * self.shots, np.inf)
        logf = (p.cx[:, mi] * np.log1p(-e2) + p.g1[:, mi] * np.log1p(-e1) + p.meas[:, mi] * np.log1p(-ero))
        self.logF = np.where(self.feas, logf, -np.inf)
        self.time_penalty = lambda T: time_cost(T, self.t_max, self.cfg.get("reward_mode", "log1p"))
        self.static = np.stack([qubits / 127.0, np.array([n["clops"] for n in self.nodes]) / 3000.0,
                                np.minimum(e2 / 0.03, 1.0), np.minimum(ero / 0.2, 1.0),
                                np.minimum(np.array([n["t2_us"] for n in self.nodes]) / 300.0, 1.0)], axis=1)
        self.biggest = int(np.argmax(qubits))
        ok = self.feas.any(axis=1) & (p.width <= int(env_cfg["max_width"]))
        self.allowed = np.flatnonzero(ok)
        self.obs_dim = len(TASK_FEATS) + self.M * len(NODE_FEATS)

    def make_workload(self, seed):
        c = self.cfg
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), SPLIT_ID[self.split]]))
        n = int(c["n_tasks"])
        idx = rng.choice(self.allowed, n, replace=True)
        rate = float(c["arrival_rate"])
        if c.get("arrival_mode", "poisson") == "bursty":
            factor = np.where((np.arange(n) // 10) % 2 == 0, 3.0, 1.0 / 3.0)
            factor *= np.mean(1.0 / factor)
            rate = rate * factor
        return idx, np.cumsum(rng.exponential(1.0 / rate, n) if np.ndim(rate) == 0 else rng.exponential(1.0 / rate))

    def reset(self, workload):
        self.idx, self.arrival = workload
        self.n, self.i = len(self.idx), 0
        self.next_free = np.zeros(self.M)
        self.fin = [[] for _ in range(self.M)]
        self.cur_arr = self.arrival[0]
        self.kappa, self.n_fail, self.penalty = 0, 0, 0.0
        self.rec = []
        self._prep()
        return self.obs()

    def _prep(self):
        if self.i >= self.n:
            self.v = None
            return
        r = self.idx[self.i]
        feas, ex, lf = self.feas[r], self.exec[r], self.logF[r]
        wait = np.maximum(0.0, self.next_free - self.cur_arr)
        qlen = np.array([len(f) - bisect.bisect_right(f, self.cur_arr) for f in self.fin], dtype=float)
        best = lf[feas].max()
        frel = np.where(feas, np.exp(np.where(feas, lf - best, 0.0)), 0.0)
        self.v = dict(row=r, feas=feas, exec=ex, wait=wait, qlen=qlen, logF=lf, frel=frel, F=np.exp(np.where(feas, lf, -700.0)),
                      d1cps=np.array([n["d1cps"] for n in self.nodes]))

    def view(self):
        return self.v

    def action_masks(self):
        return self.v["feas"].copy() if self.v is not None else np.ones(self.M, bool)

    def obs(self):
        if self.v is None:
            return np.zeros(self.obs_dim, np.float32)
        v, p, r = self.v, self.pool, self.v["row"]
        task = np.array([p.width[r] / 50.0, np.log1p(p.orig_depth[r]) / np.log1p(2000),
                         np.log1p(p.cx[r, -1]) / np.log1p(5000), self.shots / 2048.0,
                         self.kappa / max(self.max_resched, 1)])
        ex = np.where(v["feas"], _lognorm(np.where(v["feas"], v["exec"], 0.0)), 1.0)
        nodes = np.concatenate([v["feas"][:, None].astype(float), self.static, _lognorm(v["wait"])[:, None],
                                _lognorm(v["qlen"])[:, None], ex[:, None], v["frel"][:, None]], axis=1)
        return np.concatenate([task, nodes.ravel()]).astype(np.float32)

    def step(self, a):
        j, v = int(a), self.v
        if not v["feas"][j]:
            if self.kappa < self.max_resched:
                r = -self.fail_penalty * (1.0 + 0.1 * self.kappa)
                self.kappa += 1
                self.n_fail += 1
                self.penalty += -r
                self.cur_arr += self.resched_delay
                self._prep()
                return self.obs(), r, False, {"placed": False}
            j = self.biggest
        start = max(self.cur_arr, self.next_free[j])
        ex = v["exec"][j]
        fin = start + ex
        T = fin - self.arrival[self.i]
        score = float(v["frel"][j]) - self.beta * self.time_penalty(T)
        self.next_free[j] = fin
        self.fin[j].append(fin)
        self.rec.append((T, start - self.arrival[self.i], ex, float(v["F"][j]), float(v["frel"][j]), self.kappa, score, j,
                         int(self.pool.width[v["row"]])))
        self.i += 1
        self.kappa = 0
        done = self.i >= self.n
        if not done:
            self.cur_arr = max(self.cur_arr, self.arrival[self.i])
        self._prep()
        return self.obs(), score, done, {"placed": True, "T": T}

    def metrics(self):
        a = np.array(self.rec, dtype=float)
        T = a[:, 0]
        return {"mean_T": T.mean(), "total_T": T.sum(), "p95_T": np.percentile(T, 95), "mean_wait": a[:, 1].mean(),
                "mean_exec": a[:, 2].mean(), "p99_T": np.percentile(T, 99), "time_over_scale": float((T >= self.t_max).mean()), "mean_F": a[:, 3].mean(), "mean_frel": a[:, 4].mean(),
                "resched": float(self.n_fail), "makespan": float((a[:, 0] + (self.arrival[:len(a)])).max() - self.arrival[0]),
                "score": (a[:, 6].sum() - self.penalty) / len(a)}


class QCloudGymEnv(gym.Env):
    def __init__(self, env_cfg, split="train"):
        super().__init__()
        self.sim = QCloudSim(env_cfg, split)
        self.use_mask = bool(env_cfg.get("mask_infeasible", True))
        self.action_space = spaces.Discrete(self.sim.M)
        self.observation_space = spaces.Box(0.0, np.inf, (self.sim.obs_dim,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        wseed = int(self.np_random.integers(0, 2 ** 31 - 1))
        return self.sim.reset(self.sim.make_workload(wseed)), {}

    def step(self, action):
        o, r, done, info = self.sim.step(action)
        if done:
            info["episode_metrics"] = self.sim.metrics()
        return o, float(r), done, False, info

    def action_masks(self):
        return self.sim.action_masks() if self.use_mask else np.ones(self.sim.M, bool)
