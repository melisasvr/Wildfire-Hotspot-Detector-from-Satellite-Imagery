"""Plot helpers: band composites, patch grids, PR curves, confusion matrices, maps."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .preprocess import BI, load_patch


def _stretch(img, lo=2, hi=98):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        a, b = np.percentile(img[..., c], [lo, hi])
        out[..., c] = np.clip((img[..., c] - a) / max(b - a, 1e-6), 0, 1)
    return out


def true_color(refl):
    return _stretch(np.stack([refl[BI["B04"]], refl[BI["B03"]], refl[BI["B02"]]], axis=-1))


def swir_false_color(refl):
    """R=B12, G=B11, B=B08. Hot pixels glow orange/red; vegetation looks blue-green."""
    return _stretch(np.stack([refl[BI["B12"]], refl[BI["B11"]], refl[BI["B08"]]], axis=-1))


def plot_patch_grid(df, patches_dir, path, title="", scores=None, n=8):
    """Two rows per patch: true colour and SWIR false colour."""
    df = df.head(n).reset_index(drop=True)
    if len(df) == 0:  # still write a file so reports are complete and the absence is visible
        fig, ax = plt.subplots(figsize=(5, 1.6))
        ax.text(0.5, 0.5, f"{title}\nnone on this split", ha="center", va="center"); ax.axis("off")
        fig.savefig(path, dpi=130); plt.close(fig)
        return
    fig, axes = plt.subplots(2, len(df), figsize=(2.2 * len(df), 4.8), squeeze=False)
    for i, r in df.iterrows():
        refl = load_patch(f"{patches_dir}/{r.patch_file}")
        axes[0, i].imshow(true_color(refl)); axes[1, i].imshow(swir_false_color(refl))
        cap = f"y={int(r.label)}"
        if scores is not None:
            cap += f" p={scores[i]:.2f}"
        axes[0, i].set_title(cap, fontsize=8)
        for a in axes[:, i]:
            a.axis("off")
    axes[0, 0].text(-0.05, 0.5, "RGB", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    axes[1, 0].text(-0.05, 0.5, "SWIR", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pr_curves(results: dict, y, path):
    """results: name -> score array (all on the same y)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in results.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.axhline(np.mean(y), ls="--", c="grey", lw=1, label=f"prevalence {np.mean(y):.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall (test)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_confusion(m: dict, path, title=""):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["no hotspot", "hotspot"]); ax.set_yticks([0, 1], ["no hotspot", "hotspot"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (FIRMS-derived)"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_reliability(tables: dict, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    for name, (tab, ece) in tables.items():
        ax.plot(tab.mean_confidence, tab.fraction_positive, "o-", label=f"{name} (ECE {ece:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction positive"); ax.set_title("Reliability (test)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_prediction_map(df, p, thr, path):
    """Scatter of test patches in lon/lat, coloured by outcome."""
    d = df.reset_index(drop=True)
    pred = np.asarray(p) >= thr
    y = d.label.values.astype(bool)
    cat = np.where(pred & y, "TP", np.where(pred & ~y, "FP", np.where(~pred & y, "FN", "TN")))
    colors = {"TP": "#2ca02c", "FP": "#d62728", "FN": "#ff7f0e", "TN": "#c7c7c7"}
    fig, ax = plt.subplots(figsize=(6, 6))
    for k, c in colors.items():
        m = cat == k
        ax.scatter(d.lon[m], d.lat[m], s=14, c=c, label=f"{k} ({m.sum()})", alpha=.85)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Test patches by outcome")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def write_folium_map(df, p, thr, path):
    try:
        import folium
    except ImportError:
        return
    d = df.reset_index(drop=True)
    m = folium.Map(location=[d.lat.mean(), d.lon.mean()], zoom_start=8, tiles="OpenStreetMap")
    for i, r in d.iterrows():
        pred = p[i] >= thr
        out = ("TP" if pred and r.label else "FP" if pred else "FN" if r.label else "TN")
        col = {"TP": "green", "FP": "red", "FN": "orange", "TN": "gray"}[out]
        folium.CircleMarker([r.lat, r.lon], radius=4, color=col, fill=True, fill_opacity=.8,
                            tooltip=f"{out} p={p[i]:.2f} conf={r.firms_confidence} frp={r.frp}").add_to(m)
    m.save(path)
