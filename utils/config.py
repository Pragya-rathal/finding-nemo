from __future__ import annotations
import copy
import yaml
from pathlib import Path
from typing import Any, Dict

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def load_config(config_dir: str = './configs') -> Dict[str, Any]:
    cfg = {}
    for name in ['default.yaml', 'model.yaml', 'train.yaml', 'inference.yaml']:
        p = Path(config_dir) / name
        if p.exists():
            cfg = _deep_update(cfg, load_yaml(p))
    return cfg

def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_update(copy.deepcopy(cfg), overrides)
