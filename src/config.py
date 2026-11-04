"""Nap / merge cau hinh JSON va sinh 'ladder' (cac muc thuc nghiem tu thap den cao)."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def deep_update(base, upd):
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    return json.loads(Path(path).read_text())


def build_ladder(cfg, axes=None):
    ladder = cfg["ladder"]
    cases = [{"name": "base", "axis": "base", "cfg": cfg}]
    for axis, levels in ladder["axes"].items():
        if axes and axis not in axes:
            continue
        for lv in levels:
            if lv.get("is_base"):
                continue
            c = deep_update(cfg, {"env": lv.get("env", {}), "train": lv.get("train", {})})
            cases.append({"name": f"{axis}_{lv['label']}", "axis": axis, "cfg": c})
    if axes and "base" not in axes:
        cases = [c for c in cases if c["axis"] != "base"]
    return cases
