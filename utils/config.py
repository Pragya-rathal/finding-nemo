from __future__ import annotations
import yaml
from copy import deepcopy

def _merge(a, b):
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str, overrides: dict | None = None) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
