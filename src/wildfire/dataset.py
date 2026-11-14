"""Spatial/temporal splits and leakage checks (no torch dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SPLITS = ("train", "val", "test")
PRIORITY = {"train": 0, "val": 1, "test": 2}


def spatial_block_ids(df: pd.DataFrame, block_km: float) -> pd.Series:
    """Id of the square spatial block (in the analysis CRS) each patch falls in."""
    b = block_km * 1000.0
    return np.floor(df.x_a / b).astype(int).astype(str) + "_" + np.floor(df.y_a / b).astype(int).astype(str)


def assign_spatial_block_splits(df: pd.DataFrame, block_km: float, fractions: dict, seed: int) -> pd.Series:
    """Assign whole spatial blocks to splits so unseen areas end up in val/test."""
    block = spatial_block_ids(df, block_km)
    counts = block.value_counts()
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(counts))
    order = counts.to_frame("n").assign(j=jitter).sort_values(["n", "j"], ascending=[False, True]).index
    total = counts.sum()
    target = {s: fractions[s] * total for s in SPLITS}
    got = {s: 0 for s in SPLITS}
    assign = {}
    for blk in order:
        s = max(SPLITS, key=lambda k: (target[k] - got[k]) / max(target[k], 1e-9))
        assign[blk] = s
        got[s] += counts[blk]
    return block.map(assign)


def assign_temporal_splits(df: pd.DataFrame, val_start: str, test_start: str) -> pd.Series:
    t = pd.to_datetime(df["scene_datetime"], utc=True)
    split = pd.Series("train", index=df.index)
    split[t >= pd.Timestamp(val_start, tz="UTC")] = "val"
    split[t >= pd.Timestamp(test_start, tz="UTC")] = "test"
    return split


def purge_cross_split_neighbors(df: pd.DataFrame, buffer_m: float, max_days: float | None = None) -> pd.DataFrame:
    """Drop the lower-priority patch of any cross-split pair closer than buffer_m.

    With max_days set, only pairs that are also within that many days are purged.
    """
    if len(df) < 2:
        return df
    xy = df[["x_a", "y_a"]].values
    pairs = cKDTree(xy).query_pairs(buffer_m, output_type="ndarray")
    if len(pairs) == 0:
        return df
    sp = df["split"].values
    t = pd.to_datetime(df["scene_datetime"], utc=True).values.astype("datetime64[s]").astype(np.int64)
    drop = set()
    for i, j in pairs:
        if sp[i] == sp[j]:
            continue
        if max_days is not None and abs(t[i] - t[j]) > max_days * 86400:
            continue
        drop.add(i if PRIORITY[sp[i]] < PRIORITY[sp[j]] else j)
    return df.drop(df.index[sorted(drop)])


def leakage_report(df: pd.DataFrame, buffer_m: float, max_days: float | None = None) -> dict:
    """Checks that should all pass before training."""
    rep = {"duplicate_patch_ids": int(df["patch_id"].duplicated().sum())}
    mins, close = {}, {}
    t = pd.to_datetime(df["scene_datetime"], utc=True)
    for a_i, a in enumerate(SPLITS):
        for b in SPLITS[a_i + 1:]:
            A, B = df[df.split == a], df[df.split == b]
            if len(A) == 0 or len(B) == 0:
                continue
            tree = cKDTree(B[["x_a", "y_a"]].values)
            d, j = tree.query(A[["x_a", "y_a"]].values)
            mins[f"{a}-{b}"] = float(d.min())
            n_close = 0
            for k in np.where(d < buffer_m)[0]:
                if max_days is None or abs((t[A.index[k]] - t[B.index[j[k]]]).total_seconds()) <= max_days * 86400:
                    n_close += 1
            close[f"{a}-{b}"] = int(n_close)
    rep["min_center_distance_m"] = mins
    rep["pairs_within_buffer"] = close
    rep["scenes_shared_across_splits"] = int(
        (df.groupby("scene_id")["split"].nunique() > 1).sum())
    rep["datetime_range"] = {s: [str(t[df.split == s].min()), str(t[df.split == s].max())]
                             for s in SPLITS if (df.split == s).any()}
    rep["passed"] = bool(rep["duplicate_patch_ids"] == 0 and all(v == 0 for v in close.values()))
    return rep


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sc = cfg["split"]
    df = df.copy()
    if sc["strategy"] == "spatial_block":
        df["split"] = assign_spatial_block_splits(df, sc["block_size_km"], sc["fractions"], cfg["sampling"]["seed"])
        max_days = None
    elif sc["strategy"] == "temporal":
        df["split"] = assign_temporal_splits(df, sc["temporal"]["val_start"], sc["temporal"]["test_start"])
        max_days = sc["temporal"].get("buffer_days", 3)
    else:
        raise ValueError(sc["strategy"])
    df = purge_cross_split_neighbors(df, sc["buffer_m"], max_days)
    return df.reset_index(drop=True)