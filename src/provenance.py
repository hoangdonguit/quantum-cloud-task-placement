"""Fingerprint the exact code, inputs, configuration and installed runtime."""
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def run_fingerprint(cfg):
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    paths = []
    for folder in ('src', 'algorithms', 'experiments', 'tools'):
        paths.extend((ROOT/folder).rglob('*.py'))
    paths += list((ROOT/'data').glob('*.csv')) + [ROOT/'data/calibration.json', ROOT/'requirements.txt']
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    versions = {p: metadata.version(p) for p in ('numpy','pandas','torch','gymnasium','stable-baselines3','sb3-contrib')}
    h.update(json.dumps(versions, sort_keys=True).encode()); h.update(platform.python_version().encode())
    return h.hexdigest()
