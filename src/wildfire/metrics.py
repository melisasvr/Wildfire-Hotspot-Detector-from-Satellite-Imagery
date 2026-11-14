"""Metrics, thresholds, calibration, and sliced reports. sklearn/numpy only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def best_f1_threshold(y, p) -> float:
    """Threshold maximising F1. Choose it on the validation split, never on test."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    f1 = f1[:-1]
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def compute_metrics(y, p, thr: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan")
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "threshold": float(thr),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def prevalence_adjusted_precision(recall: float, fpr: float, prevalence: float) -> float:
    """Precision if the positive rate were `prevalence` (same recall and FPR).

    Illustrative only. The real-world fire/no-fire ratio for random patches is unknown.
    """
    tp, fp = recall * prevalence, fpr * (1 - prevalence)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def calibration_table(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    """Reliability table, expected calibration error, Brier score."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = p[m].mean(), y[m].mean()
        ece += m.mean() * abs(conf - acc)
        rows.append({"bin": b, "mean_confidence": conf, "fraction_positive": acc, "count": int(m.sum())})
    return pd.DataFrame(rows), float(ece), float(brier_score_loss(y, np.clip(p, 0, 1)))


def _slice_metrics(y, pred, p) -> dict:
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    out = {"n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg}
    out["recall"] = tp / n_pos if n_pos else np.nan
    out["fpr"] = fp / n_neg if n_neg else np.nan
    # precision only means something when the slice holds both classes
    out["precision"] = tp / (tp + fp) if ((tp + fp) and n_pos and n_neg) else np.nan
    return out


def slice_report(df: pd.DataFrame, p, thr: float, slices: dict) -> pd.DataFrame:
    """Metrics per slice. `slices` maps a name to a column (categorical) or (column, bins).

    Positive-only columns (FIRMS confidence, FRP) are NaN for negatives, so those slices
    report recall only. Scene-level columns (cloud, time gap) report both classes.
    """
    d = df.reset_index(drop=True).copy()
    d["_p"] = np.asarray(p, dtype=float)
    d["_pred"] = (d["_p"] >= thr).astype(int)
    rows = []
    for name, spec in slices.items():
        if isinstance(spec, tuple):
            col, bins = spec
            key = pd.cut(d[col], bins=bins, include_lowest=True).astype(str)
            key[d[col].isna()] = "nan"
        else:
            key = d[spec].astype(str)
        for k, g in d.groupby(key):
            if k == "nan" or k == "None":
                continue
            rows.append({"slice": name, "value": k, **_slice_metrics(g["label"].values, g["_pred"].values, g["_p"].values)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bootstrap
def _cluster_groups(clusters, n):
    if clusters is None:                      # plain bootstrap: every patch is its own cluster
        return [np.array([i]) for i in range(n)]
    c = np.asarray(clusters)
    return [np.where(c == u)[0] for u in np.unique(c)]


def _draw(rng, groups):
    picks = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[i] for i in picks])


def bootstrap_ci(y, p, thr: float, clusters=None, n_boot: int = 1000, seed: int = 0,
                 metrics=("precision", "recall", "f1", "pr_auc", "roc_auc")) -> pd.DataFrame:
    """95% percentile intervals for the test metrics at a FIXED threshold.

    Resamples whole clusters (spatial blocks) with replacement, because patches from the same
    fire are not independent. Captures test-set sampling noise only. It does not capture
    training randomness or the uncertainty in the threshold chosen on validation.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    groups, rng = _cluster_groups(clusters, len(y)), np.random.default_rng(seed)
    est = compute_metrics(y, p, thr)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = _draw(rng, groups)
        if y[idx].min() == y[idx].max():
            continue
        r = compute_metrics(y[idx], p[idx], thr)
        for m in metrics:
            draws[m].append(r[m])
    rows = []
    for m in metrics:
        v = np.array(draws[m], dtype=float)
        v = v[~np.isnan(v)]
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
        rows.append({"metric": m, "estimate": est[m], "ci_low": lo, "ci_high": hi, "n_valid_resamples": int(len(v))})
    return pd.DataFrame(rows)


def paired_bootstrap_diff(y, scores: dict, thrs: dict, clusters=None, n_boot: int = 1000, seed: int = 0,
                          metrics=("pr_auc", "roc_auc", "f1")) -> pd.DataFrame:
    """Do two models really differ? Same resamples for every model, then look at the differences.

    If the interval for (A minus B) comfortably includes 0, the data cannot separate them.
    `frac_resamples_a_better` is a bootstrap fraction, not a p-value.
    """
    from itertools import combinations
    y = np.asarray(y).astype(int)
    names = list(scores)
    pairs = list(combinations(names, 2))
    groups, rng = _cluster_groups(clusters, len(y)), np.random.default_rng(seed)
    full = {n: compute_metrics(y, scores[n], thrs[n]) for n in names}
    draws = {(a, b, m): [] for a, b in pairs for m in metrics}
    for _ in range(n_boot):
        idx = _draw(rng, groups)
        if y[idx].min() == y[idx].max():
            continue
        res = {n: compute_metrics(y[idx], np.asarray(scores[n])[idx], thrs[n]) for n in names}
        for a, b in pairs:
            for m in metrics:
                draws[(a, b, m)].append(res[a][m] - res[b][m])
    rows = []
    for (a, b, m), v in draws.items():
        v = np.array(v, dtype=float); v = v[~np.isnan(v)]
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
        rows.append({"model_a": a, "model_b": b, "metric": m, "diff_a_minus_b": full[a][m] - full[b][m],
                     "ci_low": lo, "ci_high": hi,
                     "frac_resamples_a_better": float((v > 0).mean()) if len(v) else np.nan,
                     "n_valid_resamples": int(len(v))})
    return pd.DataFrame(rows, columns=["model_a", "model_b", "metric", "diff_a_minus_b", "ci_low", "ci_high",
                                       "frac_resamples_a_better", "n_valid_resamples"])