"""Shared helpers: config loading, seeding, device selection."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply KEY=VALUE overrides with dotted keys, e.g. model.arch=mlp seed=3 (values parsed as YAML)."""
    for kv in overrides:
        key, value = kv.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node[k]
        node[last] = yaml.safe_load(value)
    return cfg


def resolve(path: str | Path) -> Path:
    """Paths in the config are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def config_hash(*parts: dict) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
