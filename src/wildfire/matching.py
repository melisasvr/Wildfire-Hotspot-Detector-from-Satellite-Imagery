"""Land-cover-context matching of negatives to positives.

Why: random negatives from the same scene include lots of easy ones (farmland, towns) while
positives sit in wildland. A model can score well just by telling those landscapes apart.
Matching keeps every positive and, within each split, keeps only the negatives whose
surrounding looks most like a positive's: similar vegetation greenness and SWIR brightness.

Only fire-free context is used (median NDVI and median B11 over the patch; medians so that a
single hot pixel cannot move them). No labels beyond "positive/negative" and no model are involved.
This is a harder stress test, not "the truth": burned ground near a fire also shifts these
features, so matched negatives are deliberately similar to fire surroundings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .preprocess import BI, index_channel, load_patch

FEATURES = ("ndvi_median", "b11_median")


def add_context_features(df: pd.DataFrame, patches_dir) -> pd.DataFrame:
    """Add robust land-cover proxies (also useful as an evaluation slice)."""
    out = df.copy()
    nd, b11 = [], []
    for f in out.patch_file:
        r = load_patch(f"{patches_dir}/{f}")
        nd.append(float(np.median(index_channel(r, "ndvi"))))
        b11.append(float(np.median(r[BI["B11"]])))
    out["ndvi_median"], out["b11_median"] = nd, b11
    return out


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference between two samples (0 means well balanced)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def match_negatives(df: pd.DataFrame, k: int = 2, caliper_sd: float = 0.75, seed: int = 0,
                    features=FEATURES):
    """Keep all positives; per split keep up to k nearest-context negatives per positive.

    Greedy nearest-neighbour without replacement, in standardised feature space, with a
    caliper: negatives farther than `caliper_sd` standard deviations from every positive are
    dropped. Returns (matched_df, report). The report shows balance before and after.
    """
    d = df.reset_index(drop=True)
    X = d[list(features)].values.astype(float)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)   # unsupervised scaling
    rng = np.random.default_rng(seed)
    keep = (d.label.values == 1).copy()
    report = {"features": list(features), "negatives_per_positive": k, "caliper_sd": caliper_sd, "splits": {}}

    for s in d.split.unique():
        ip = np.where((d.split == s) & (d.label == 1))[0]
        in_ = np.where((d.split == s) & (d.label == 0))[0]
        rep = {"n_pos": int(len(ip)), "n_neg_before": int(len(in_))}
        if len(ip) == 0 or len(in_) == 0:
            keep[in_] = True
            rep.update(n_neg_after=int(len(in_)), note="nothing to match")
            report["splits"][s] = rep
            continue
        D = cdist(Z[ip], Z[in_])
        used = np.zeros(len(in_), dtype=bool)
        got = np.zeros(len(ip), dtype=int)
        for _ in range(k):
            for i in rng.permutation(len(ip)):
                for j in np.argsort(D[i]):
                    if D[i, j] > caliper_sd:
                        break
                    if not used[j]:
                        used[j] = True; got[i] += 1
                        break
        keep[in_[used]] = True
        rep.update({
            "n_neg_after": int(used.sum()),
            "positives_with_full_k_matches": int((got == k).sum()),
            "positives_with_no_match": int((got == 0).sum()),
            "smd_before": {f: _smd(X[ip, c], X[in_, c]) for c, f in enumerate(features)},
            "smd_after": {f: _smd(X[ip, c], X[in_[used], c]) for c, f in enumerate(features)},
        })
        report["splits"][s] = rep
    return d.loc[keep].reset_index(drop=True), report