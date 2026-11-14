"""Channel construction, normalisation (train-only statistics), patch IO."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .patches import from_uint16

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BI = {b: i for i, b in enumerate(BANDS)}
EPS = 1e-6


def load_patch(path: str | Path) -> np.ndarray:
    """Load a stored patch as float32 reflectance, shape (6, H, W)."""
    return from_uint16(np.load(path))


def index_channel(refl: np.ndarray, name: str) -> np.ndarray:
    nir, red, s1, s2 = refl[BI["B08"]], refl[BI["B04"]], refl[BI["B11"]], refl[BI["B12"]]
    if name == "ndvi":
        return (nir - red) / (nir + red + EPS)
    if name == "nbr":
        return (nir - s2) / (nir + s2 + EPS)
    if name == "ndmi":
        return (nir - s1) / (nir + s1 + EPS)
    raise ValueError(f"unknown index: {name}")


def build_channels(refl: np.ndarray, indices) -> np.ndarray:
    """Stack the 6 reflectance bands with derived indices -> (6 + k, H, W)."""
    if not indices:
        return refl
    extra = np.stack([index_channel(refl, n) for n in indices]).astype(np.float32)
    return np.concatenate([refl, extra], axis=0)


def channel_names(indices) -> list[str]:
    return list(BANDS) + list(indices or [])


def compute_norm_stats(paths, indices) -> dict:
    """Per-channel mean/std over TRAIN patches only (streaming, float64)."""
    n = 0
    s = s2 = None
    for p in paths:
        x = build_channels(load_patch(p), indices).astype(np.float64)
        flat = x.reshape(x.shape[0], -1)
        if s is None:
            s = np.zeros(flat.shape[0]); s2 = np.zeros(flat.shape[0])
        s += flat.sum(1); s2 += (flat ** 2).sum(1); n += flat.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return {
        "channels": channel_names(indices), "indices": list(indices or []),
        "mean": mean.tolist(), "std": std.tolist(),
        "n_pixels": int(n), "n_patches": len(list(paths)),
        "fit_on": "train split only",
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    return ((x - mean) / std).astype(np.float32)


def save_stats(stats: dict, path: str | Path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
