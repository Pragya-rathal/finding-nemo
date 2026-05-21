from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(config_dir: str | Path = "./configs") -> Dict[str, Any]:
    config_dir = Path(config_dir).resolve()
    cfg: Dict[str, Any] = {}
    for name in ["default.yaml", "model.yaml", "train.yaml", "inference.yaml"]:
        path = config_dir / name
        if path.exists():
            cfg = _deep_update(cfg, load_yaml(path))
    return cfg


def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    return _deep_update(copy.deepcopy(cfg), overrides)
