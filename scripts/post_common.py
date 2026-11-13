"""Supplement-only helpers; frozen historical core is untouched."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.baseline_tuning import GRID,GreedyScoreTuned
from src.heuristics import REGISTRY

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def candidates():
    p=dict(REGISTRY)
    for m,l in GRID:p[f'GreedyScoreTuned_m{m}_l{l}']=lambda m=m,l=l:GreedyScoreTuned(m,l)
    return p

def workload_hash(sim,seed):
    idx,arr=sim.make_workload(seed)
    return hashlib.sha256(idx.tobytes()+arr.tobytes()).hexdigest()

def transform(obs,groups,n_nodes):
    from src.env import obs_groups
    x=np.array(obs,copy=True)
    for group in groups:x[obs_groups(n_nodes)[group]]=0
    return x

def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False))
