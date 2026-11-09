"""Heuristic placement. Moi policy: f(sim, state) -> chi so node. Chi dung thong tin vat ly sim.view() (giong het agent).

Nhom 1 - baseline trong bai bao: Random, RoundRobin, DRLQ_GreedyWait (khong biet rang buoc qubit), FastestNode (FDF cua QFOR),
          BestFidelity (SEF cua QFOR), GreedyWait (FAN cua QFOR), LeastQueue.
Nhom 2 - hybrid cua nhom:   HybridQueueSpeed, HybridFidelityQueue, HybridEfficiencyRatio.
Nhom 3 - tham lam toi uu 1 buoc theo dung ham muc tieu: GreedyScore, GreedyFlow."""
import numpy as np


def lexmin(feas, *priorities):
    ix = np.flatnonzero(feas)
    return int(ix[np.lexsort(tuple(np.asarray(x)[ix] for x in priorities[::-1]))[0]])


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
    """Baseline 'Greedy' cua DRLQ: chon node cho ngan nhat, KHONG xet so qubit => co the bi reschedule."""
    name = "DRLQ_GreedyWait"

    def __call__(self, sim):
        return int(np.argmin(sim.view()["wait"]))


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
        return lexmin(v["feas"], v["wait"], v["exec"])


class LeastQueue(Policy):
    name = "LeastQueue"

    def __call__(self, sim):
        v = sim.view()
        return lexmin(v["feas"], v["qlen"], v["wait"], v["exec"])


class GreedyFlow(Policy):
    """Toi thieu hoa completion time cua chinh task nay: wait + exec."""
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
    """Loc cac node co fidelity >= 90% trung binh (tren cac node kha thi) roi chon node cho ngan nhat."""
    name = "HybridFidelityQueue"

    def __call__(self, sim):
        v = sim.view()
        lf = v["logF"][v["feas"]]
        best = lf.max()
        thr = best + np.log(np.exp(lf-best).mean()) + np.log(0.9)
        good = v["feas"] & (v["logF"] >= thr)
        return lexmin(good if good.any() else v["feas"], v["wait"], v["exec"])


class HybridEfficiencyRatio(Policy):
    """Toi da fidelity tren moi don vi thoi gian hoan thanh (wait + exec)."""
    name = "HybridEfficiencyRatio"

    def __call__(self, sim):
        v = sim.view()
        return _argmax(v["logF"] - np.log(v["wait"] + v["exec"] + 1e-9), v["feas"])


class GreedyScore(Policy):
    """Toi uu 1 buoc dung ham reward: frel - beta*min(T/t_max, 1). Day la baseline tham lam manh nhat ve ly thuyet."""
    name = "GreedyScore"

    def __call__(self, sim):
        v = sim.view()
        T = v["wait"] + v["exec"] + (sim.cur_arr - sim.arrival[sim.i])
        return _argmax(v["frel"] - sim.beta * sim.time_penalty(T), v["feas"])


ALL = [Random, RoundRobin, DRLQ_GreedyWait, FastestNode, BestFidelity, GreedyWait, LeastQueue, GreedyFlow,
       HybridQueueSpeed, HybridFidelityQueue, HybridEfficiencyRatio, GreedyScore]
REGISTRY = {c.name: c for c in ALL}