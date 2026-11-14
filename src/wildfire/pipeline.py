"""Dataset assembly (QC, splits, leakage purge, normalisation) and the pre-training report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import path_of
from .dataset import SPLITS, leakage_report, make_splits
from .matching import add_context_features, match_negatives
from .preprocess import compute_norm_stats
from .viz import plot_patch_grid


def build_processed_dataset(raw: pd.DataFrame, cfg: dict):
    """Re-apply QC thresholds, split, purge cross-split neighbours, fit train-only stats."""
    ic = cfg["imagery"]
    df = raw[(raw.nodata_frac <= ic["max_nodata_frac"]) & (raw.cloud_frac <= ic["max_patch_cloud_frac"])
             & (raw.water_frac <= ic["max_water_frac"])].reset_index(drop=True)
    df = make_splits(df, cfg)
    patches_dir = path_of(cfg, "patches")
    df = add_context_features(df, patches_dir)           # land-cover proxies, kept in the manifest
    mc = cfg.get("matching") or {}
    match_rep = None
    if mc.get("enabled"):
        df, match_rep = match_negatives(df, mc.get("negatives_per_positive", 2), mc.get("caliper_sd", 0.75),
                                        cfg["sampling"]["seed"])
    max_days = cfg["split"]["temporal"].get("buffer_days") if cfg["split"]["strategy"] == "temporal" else None
    rep = leakage_report(df, cfg["split"]["buffer_m"], max_days)
    if match_rep:
        rep["matching"] = match_rep
    train_paths = [patches_dir / f for f in df[df.split == "train"].patch_file]
    if not train_paths:
        raise RuntimeError("No training patches after splitting. Check the dataset report.")
    stats = compute_norm_stats(train_paths, cfg["preprocess"]["indices"])
    return df, rep, stats


def dataset_summary(m: pd.DataFrame) -> dict:
    out = {"n_total": int(len(m)), "n_pos": int(m.label.sum()), "n_neg": int((1 - m.label).sum())}
    out["class_balance_pos_fraction"] = float(m.label.mean())
    out["per_split"] = {s: {"n": int((m.split == s).sum()), "n_pos": int(m[m.split == s].label.sum()),
                            "pos_fraction": float(m[m.split == s].label.mean()) if (m.split == s).any() else None}
                        for s in SPLITS}
    out["cloud_frac_patch"] = m.cloud_frac.describe().round(4).to_dict()
    out["cloud_scene_percent"] = m.scene_cloud.describe().round(2).to_dict()
    pos = m[m.label == 1]
    out["firms_confidence_counts"] = pos.firms_confidence.value_counts().to_dict()
    out["time_gap_h_abs"] = pos.time_gap_h.abs().describe().round(3).to_dict()
    out["frp_mw"] = pos.frp.describe().round(3).to_dict()
    out["n_scenes"] = int(m.scene_id.nunique())
    return out


def write_dataset_report(m: pd.DataFrame, leak: dict, cfg: dict, out_dir: Path):
    """Everything to eyeball before training: counts, balance, clouds, splits, examples, leakage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = path_of(cfg, "patches")
    summ = dataset_summary(m)
    summ["leakage"] = leak
    (out_dir / "dataset_summary.json").write_text(json.dumps(summ, indent=2, default=str))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].hist([m[m.label == 0].cloud_frac, m[m.label == 1].cloud_frac], bins=20, label=["negative", "positive"])
    ax[0].set_xlabel("Patch cloud fraction (SCL)"); ax[0].legend()
    ax[1].hist(m[m.label == 1].time_gap_h.abs(), bins=20, color="C1")
    ax[1].set_xlabel("|FIRMS time - Sentinel-2 time| (h), positives")
    for s, c in zip(SPLITS, ("C0", "C1", "C3")):
        d = m[m.split == s]; ax[2].scatter(d.lon, d.lat, s=6, c=c, label=f"{s} ({len(d)})")
    ax[2].set_xlabel("lon"); ax[2].set_ylabel("lat"); ax[2].legend(fontsize=8); ax[2].set_title("Split map")
    fig.tight_layout(); fig.savefig(out_dir / "dataset_overview.png", dpi=130); plt.close(fig)

    rng = np.random.default_rng(0)
    for lab, name in ((1, "positive"), (0, "negative")):
        d = m[m.label == lab]
        d = d.iloc[rng.permutation(len(d))[:8]]
        plot_patch_grid(d, patches_dir, out_dir / f"examples_{name}.png", f"Random {name} patches", n=8)
    return summ