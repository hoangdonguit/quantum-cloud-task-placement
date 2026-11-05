"""Heuristic placement. Moi policy: f(sim, state) -> chi so node. Chi dung thong tin vat ly sim.view() (giong het agent)."""
import numpy as np


def _argmin(values, feas):
    return int(np.argmin(np.where(feas, values, np.inf)))


def _argmax(values, feas):
    return int(np.argmax(np.where(feas, values, -np.inf)))


class Policy:
    name = "Policy"
    def reset(self, sim, seed):
        self.rng = np.random.default_rng(seed)
        self.k = 0
    def __call__(self, sim):
        raise NotImplementedError


class Random(Policy):
    name = "Random"
    def __call__(self, sim):
        return int(self.rng.choice(np.flatnonzero(sim.view()["feas"])))


class RoundRobin(Policy):
    name = "RoundRobin"
    def __call__(self, sim):
        feas = sim.view()["feas"]
        for _ in range(sim.M):
            j = self.k % sim.M
            self.k += 1
            if feas[j]:
                return j


class DRLQ_GreedyWait(Policy):
    name = "DRLQ_GreedyWait"
    def __call__(self, sim):
        v = sim.view()
        # Thêm tie-break 1e-6 * exec để không chọn nhầm node chậm
        return int(np.argmin(v["wait"] + 1e-6 * v["exec"]))


class FastestNode(Policy):
    name = "FastestNode"
    def __call__(self, sim):
        v = sim.view()
        return _argmin(v["exec"], v["feas"])


class BestFidelity(Policy):
    name = "BestFidelity"
    def __call__(self, sim):
        v = sim.view()
        return _argmax(v["logF"], v["feas"])


class GreedyWait(Policy):
    name = "GreedyWait"
    def __call__(self, sim):
        v = sim.view()
        return _argmin(v["wait"] + 1e-6 * v["exec"], v["feas"])


class LeastQueue(Policy):
    name = "LeastQueue"
    def __call__(self, sim):
        v = sim.view()
        # Thêm tie-break 1e-6 * exec
        return _argmin(v["qlen"] + 1e-3 * v["wait"] + 1e-6 * v["exec"], v["feas"])


class GreedyFlow(Policy):
    name = "GreedyFlow"
    def __call__(self, sim):
        v = sim.view()
        return _argmin(v["wait"] + v["exec"], v["feas"])


class HybridQueueSpeed(Policy):
    name = "HybridQueueSpeed"
    def __call__(self, sim):
        v = sim.view()
        return _argmax(v["d1cps"] / (v["qlen"] + 1.0), v["feas"])


class HybridFidelityQueue(Policy):
    name = "HybridFidelityQueue"
    def __call__(self, sim):
        v = sim.view()
        thr = 0.9 * v["F"][v["feas"]].mean()
        good = v["feas"] & (v["F"] >= thr)
        return _argmin(v["wait"] + 1e-6 * v["exec"], good if good.any() else v["feas"])


class HybridEfficiencyRatio(Policy):
    name = "HybridEfficiencyRatio"
    def __call__(self, sim):
        v = sim.view()
        return _argmax(v["F"] / (v["wait"] + v["exec"] + 1e-9), v["feas"])


class GreedyScore(Policy):
    name = "GreedyScore"
    def __call__(self, sim):
        v = sim.view()
        T = v["wait"] + v["exec"] + (sim.cur_arr - sim.arrival[sim.i])
        return _argmax(v["frel"] - sim.beta * np.minimum(T / sim.t_max, 1.0), v["feas"])


ALL = [Random, RoundRobin, DRLQ_GreedyWait, FastestNode, BestFidelity, GreedyWait, LeastQueue, GreedyFlow,
       HybridQueueSpeed, HybridFidelityQueue, HybridEfficiencyRatio, GreedyScore]
REGISTRY = {c.name: c for c in ALL}
