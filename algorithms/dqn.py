"""DQN co the cau hinh: vanilla <-> Double + Dueling + n-step, co action masking, seed day du."""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


class QNet(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=(128, 128), dueling=False):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.adv = nn.Linear(d, n_act)
        self.dueling = dueling
        self.val = nn.Linear(d, 1) if dueling else None

    def forward(self, x):
        h = self.body(x)
        a = self.adv(h)
        return self.val(h) + a - a.mean(1, keepdim=True) if self.dueling else a


class Replay:
    def __init__(self, cap, obs_dim, n_act):
        self.cap, self.n, self.p = cap, 0, 0
        self.s = np.zeros((cap, obs_dim), np.float32)
        self.s2 = np.zeros((cap, obs_dim), np.float32)
        self.m2 = np.ones((cap, n_act), bool)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.g = np.zeros(cap, np.float32)

    def add(self, s, a, r, s2, m2, d, g):
        i = self.p
        self.s[i], self.a[i], self.r[i], self.s2[i], self.m2[i], self.d[i], self.g[i] = s, a, r, s2, m2, d, g
        self.p = (self.p + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, rng, batch):
        ix = rng.integers(0, self.n, batch)
        return self.s[ix], self.a[ix], self.r[ix], self.s2[ix], self.m2[ix], self.d[ix], self.g[ix]


class DQNAgent:
    def __init__(self, obs_dim, n_act, hp, seed):
        self.obs_dim, self.n_act, self.hp, self.seed = obs_dim, n_act, dict(hp), seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        h = tuple(hp.get("hidden", (128, 128)))
        self.net = QNet(obs_dim, n_act, h, hp.get("dueling", False))
        self.tgt = QNet(obs_dim, n_act, h, hp.get("dueling", False))
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=hp["lr"])

    @torch.no_grad()
    def predict(self, obs, mask):
        q = self.net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
        q = q.masked_fill(~torch.as_tensor(mask), -1e9)
        return int(q.argmax())

    def _update(self, buf):
        hp = self.hp
        s, a, r, s2, m2, d, g = (torch.as_tensor(x) for x in buf.sample(self.rng, hp["batch_size"]))
        q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if hp.get("double", False):
                a2 = self.net(s2).masked_fill(~m2, -1e9).argmax(1, keepdim=True)
                q2 = self.tgt(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.tgt(s2).masked_fill(~m2, -1e9).max(1).values
            y = r + (1.0 - d) * g * q2
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()

    def _emit(self, nbuf, buf, gamma):
        s, a = nbuf[0][0], nbuf[0][1]
        R, g = 0.0, 1.0
        for (_, _, r, _, _, _) in nbuf:
            R += g * r
            g *= gamma
        _, _, _, s2, m2, d = nbuf[-1]
        buf.add(s, a, R, s2, m2, float(d), g)

    def learn(self, env, total_timesteps, eval_fn=None, eval_every=5000, log=None):
        hp, gamma, n = self.hp, self.hp["gamma"], int(self.hp.get("n_step", 1))
        buf = Replay(int(hp["buffer_size"]), self.obs_dim, self.n_act)
        nbuf = deque()
        obs, _ = env.reset(seed=self.seed)
        mask = env.action_masks()
        decay_steps = max(1.0, hp["eps_fraction"] * total_timesteps)
        for t in range(1, total_timesteps + 1):
            eps = max(hp["eps_end"], 1.0 - (1.0 - hp["eps_end"]) * t / decay_steps)
            a = int(self.rng.choice(np.flatnonzero(mask))) if self.rng.random() < eps else self.predict(obs, mask)
            obs2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            mask2 = np.ones(self.n_act, bool) if done else env.action_masks()
            nbuf.append((obs, a, r, obs2, mask2, done))
            if len(nbuf) == n:
                self._emit(nbuf, buf, gamma)
                nbuf.popleft()
            if done:
                while nbuf:
                    self._emit(nbuf, buf, gamma)
                    nbuf.popleft()
                obs, _ = env.reset()
                mask = env.action_masks()
            else:
                obs, mask = obs2, mask2
            if t >= hp["learning_starts"] and t % hp["train_freq"] == 0 and buf.n >= hp["batch_size"]:
                self._update(buf)
            if t % hp["target_update"] == 0:
                self.tgt.load_state_dict(self.net.state_dict())
            if eval_fn is not None and t % eval_every == 0:
                log(t, eval_fn(lambda o, m: self.predict(o, m)))

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "obs_dim": self.obs_dim, "n_act": self.n_act, "hp": self.hp}, path)

    @staticmethod
    def load(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ag = DQNAgent(ck["obs_dim"], ck["n_act"], ck["hp"], seed=0)
        ag.net.load_state_dict(ck["state"])
        return ag
