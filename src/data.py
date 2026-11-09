"""Nap workload that cua QSimPy (mach MQT Bench da transpile san cho tung loai node) -> mang numpy, co cache.

Nguon du lieu: https://github.com/Cloudslab/qsimpy  (thu muc qdataset/traintest/). Moi dong = 1 QTask:
  original_width/depth, va (width, depth, gates) SAU transpile cho 4 loai node: ibmq7, ibmq16, ibmq27, ibmq127.
  width == -1 nghia la task khong vua node loai do (khong the thuc thi)."""
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS = ["ibmq7", "ibmq16", "ibmq27", "ibmq127"]
CSV = {"train": "synds_800_sub_60_trainset.csv", "test": "synds_200_sub_60_testset.csv"}
SPLIT_ID = {"train": 0, "val": 1, "test": 2}
_CACHE = {}


def _counts(cell):
    if not isinstance(cell, str) or cell == "{}":
        return 0, 0, 0
    d = ast.literal_eval(cell)
    return d.get("cx", 0), d.get("sx", 0) + d.get("x", 0), d.get("measure", 0)


def _parse_csv(csv_name):
    cache = DATA_DIR / ("cache_" + csv_name.replace(".csv", ".npz"))
    fingerprint = hashlib.sha256((DATA_DIR / csv_name).read_bytes() + b"parser-v2").hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if "fingerprint" in z and str(z["fingerprint"]) == fingerprint:
                return {k: z[k] for k in z.files if k != "fingerprint"}
    df = pd.read_csv(DATA_DIR / csv_name)
    n, k = len(df), len(MODELS)
    out = {
        "width": df["original_width"].to_numpy(np.int64),
        "orig_depth": df["original_depth"].to_numpy(np.int64),
        "algo": df["algorithm"].to_numpy().astype(str),
        "depth": np.full((n, k), -1, np.int64),
        "cx": np.zeros((n, k), np.int64), "g1": np.zeros((n, k), np.int64), "meas": np.zeros((n, k), np.int64),
    }
    for m_idx, m in enumerate(MODELS):
        out["depth"][:, m_idx] = df[f"{m}_depth"].to_numpy()
        for i, cell in enumerate(df[f"{m}_gates"]):
            out["cx"][i, m_idx], out["g1"][i, m_idx], out["meas"][i, m_idx] = _counts(cell)
    # PATCH: atomic write tranh race condition
    tmp = DATA_DIR / f"{cache.stem}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, fingerprint=fingerprint, **out)
    os.replace(tmp, cache)
    return out


class Pool:
    def __init__(self, arrays, sl=None):
        sl = sl if sl is not None else slice(None)
        for key, value in arrays.items():
            setattr(self, key, value[sl])
        self.n = len(self.width)

    def rows_with_width(self, max_width):
        return np.flatnonzero(self.width <= max_width)


def get_pool(split, protocol="workload"):
    key = (split, protocol)
    if key in _CACHE:
        return _CACHE[key]
    if protocol == "circuit":
        a, b = (_parse_csv(CSV[x]) for x in ("train", "test"))
        arrays = {k: np.concatenate([a[k], b[k]]) for k in a}
        # Group by family and width: all duplicate circuits/rows stay together.
        buckets = {}
        for family, width in set(zip(arrays["algo"], arrays["width"])):
            bucket = next(i for i, lim in enumerate((7, 16, 27, 50, 10**9)) if width <= lim)
            buckets.setdefault((str(family), bucket), []).append((str(family), int(width)))
        assignment = {}
        for groups in buckets.values():
            groups.sort(key=lambda g: hashlib.sha256(f"409:{g[0]}:{g[1]}".encode()).hexdigest())
            n = max(1, int(len(groups)*0.15)) if len(groups) >= 3 else 0
            for i, group in enumerate(groups):
                assignment[group] = "val" if i < n else ("test" if i < 2*n else "train")
        mask = np.array([assignment[(str(f), int(w))] == split for f,w in zip(arrays["algo"], arrays["width"])])
        pool = Pool(arrays, mask)
        _CACHE[key] = pool
        return pool
    if protocol != "workload":
        raise ValueError(protocol)
    
    if split == "test":
        pool = Pool(_parse_csv(CSV["test"]))
    else:
        arrays = _parse_csv(CSV["train"])
        cut = int(0.9 * len(arrays["width"]))
        pool = Pool(arrays, slice(0, cut) if split == "train" else slice(cut, None))
    _CACHE[key] = pool
    return pool
