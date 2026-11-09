"""Validation tuning grid, tuned heuristic and Holm adjustment."""
import numpy as np
from src.heuristics import Policy, _argmax
GRID = [(m, lam) for m in (1.0,1.5,2.0,3.0,4.0,6.0) for lam in (0.0,0.25,0.5,1.0,2.0)]
VAL_SEEDS = list(range(5000,5030))

class GreedyScoreTuned(Policy):
    """argmax [ frel - m*beta*min(T/t_max,1) - lam*exec/t_max ]. m>1 va lam>0 = 'gia bong' cua viec chiem dung node (ngoai ung)."""
    name = "GreedyScoreTuned"

    def __init__(self, m=1.0, lam=0.0):
        self.m, self.lam = m, lam

    def __call__(self, sim):
        v = sim.view()
        T = v["wait"] + v["exec"] + (sim.cur_arr - sim.arrival[sim.i])
        sc = v["frel"] - self.m * sim.beta * sim.time_penalty(T) - self.lam * np.where(v["feas"], v["exec"], 0.0) / sim.t_max
        return _argmax(sc, v["feas"])

def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj
