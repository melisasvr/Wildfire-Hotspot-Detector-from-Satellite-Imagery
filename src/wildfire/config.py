"""Config loading helpers."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load a YAML config. A top-level `extends: other.yaml` merges over that base file."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    return cfg


def deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def path_of(cfg: dict, key: str) -> Path:
    """Return a configured directory as a Path, creating it if needed."""
    p = Path(cfg["paths"][key])
    p.mkdir(parents=True, exist_ok=True)
    return p