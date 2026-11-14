"""Baseline 1: spectral rule. Baseline 2: classical ML on patch statistics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import best_f1_threshold
from .preprocess import BI, build_channels, channel_names, load_patch

EPS = 1e-6


# ---------------------------------------------------------------- Baseline 1: rule
def rule_scores_one(refl: np.ndarray, ratios) -> np.ndarray:
    """Score per ratio threshold: max B12 over pixels where B12/B11 >= ratio, else 0.

    Hot flaming pixels reflect more at 2.2 um (B12) than at 1.6 um (B11), while healthy
    vegetation usually shows the opposite. No learning beyond two tuned numbers.
    """
    b11, b12 = refl[BI["B11"]], refl[BI["B12"]]
    ratio = b12 / (b11 + EPS)
    return np.array([float(b12[ratio >= r].max()) if (ratio >= r).any() else 0.0 for r in ratios])


class SpectralRule:
    def __init__(self, ratio_grid):
        self.ratio_grid = list(ratio_grid)
        self.ratio = None
        self.threshold = None

    def _score_matrix(self, manifest, patches_dir):
        return np.stack([rule_scores_one(load_patch(f"{patches_dir}/{f}"), self.ratio_grid)
                         for f in manifest.patch_file])

    def fit(self, train_df, patches_dir):
        """Tune ratio and threshold on TRAIN only (max F1)."""
        S = self._score_matrix(train_df, patches_dir)
        y = train_df.label.values
        best = (-1, None, None)
        for k, r in enumerate(self.ratio_grid):
            thr = best_f1_threshold(y, S[:, k])
            pred = S[:, k] >= thr
            tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = ((~pred) & (y == 1)).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best[0]:
                best = (f1, r, thr)
        _, self.ratio, self.threshold = best
        return self

    def score(self, manifest, patches_dir) -> np.ndarray:
        """Continuous score, squashed to [0, 1] (reflectance can be above 1 at saturation)."""
        k = self.ratio_grid.index(self.ratio)
        return np.clip(self._score_matrix(manifest, patches_dir)[:, k], 0, 1)

    def score_patch(self, refl: np.ndarray) -> float:
        return float(np.clip(rule_scores_one(refl, [self.ratio])[0], 0, 1))

    def save(self, path):
        json.dump({"ratio": self.ratio, "threshold": self.threshold, "ratio_grid": self.ratio_grid}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        o = cls(d["ratio_grid"]); o.ratio, o.threshold = d["ratio"], d["threshold"]
        return o


# ------------------------------------------------------- Baseline 2: classical ML
def patch_features(refl: np.ndarray, indices):
    """Summary statistics per channel plus a few SWIR fire cues. Returns (vector, names)."""
    x = build_channels(refl, indices)
    names, feats = [], []
    for c, cname in enumerate(channel_names(indices)):
        v = x[c].ravel()
        for stat, val in (("mean", v.mean()), ("std", v.std()), ("min", v.min()), ("max", v.max()),
                          ("p95", np.percentile(v, 95)), ("p99", np.percentile(v, 99))):
            names.append(f"{cname}_{stat}"); feats.append(val)
    b11, b12 = refl[BI["B11"]], refl[BI["B12"]]
    ratio = b12 / (b11 + EPS)
    h, w = b12.shape
    cy, cx = h // 2, w // 2
    c = slice(cy - 4, cy + 5), slice(cx - 4, cx + 5)  # 9x9 around the FIRMS centre
    extra = {
        "ratio_gt_1.2_frac": float((ratio > 1.2).mean()),
        "max_b12_minus_b11": float((b12 - b11).max()),
        "center_b12_max": float(b12[c].max()), "center_b11_max": float(b11[c].max()),
        "center_ratio_max": float(ratio[c].max()),
    }
    names += list(extra); feats += list(extra.values())
    return np.asarray(feats, dtype=np.float32), names


def feature_matrix(manifest, patches_dir, indices):
    X, names = [], None
    for f in manifest.patch_file:
        v, names = patch_features(load_patch(f"{patches_dir}/{f}"), indices)
        X.append(v)
    return np.stack(X), names


def make_classical(kind: str, seed: int = 0):
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if kind == "gradient_boosting":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                              class_weight="balanced", random_state=seed)
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    raise ValueError(kind)


def importance_table(model, Xv, yv, names, seed=0, n_repeats=10) -> pd.DataFrame:
    r = permutation_importance(model, Xv, yv, scoring="average_precision", n_repeats=n_repeats, random_state=seed)
    return (pd.DataFrame({"feature": names, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))
